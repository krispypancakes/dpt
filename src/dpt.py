from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import math
import tiktoken
import time
import inspect
import os
from datetime import datetime
from data.utils import EmoData, DataLoaderFine


@dataclass
class GPTConfig:
    ctx_len: int = 512 # max seq len or context length
    vocab_size: int = 50259 # n tokens: 50000 bpe merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 8
    n_head: int = 4
    n_embd: int = 256 # embedding dimension
    batch_size: int = 32
    n_kv_head = n_head // 2  # number of key/value heads


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_query_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_size = config.n_embd // config.n_head
        # projects each token's embedding into n_head query vectors
        self.q_proj = nn.Linear(config.n_embd, self.n_query_head * self.head_size)
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.n_kv_head * self.head_size) # *2: one for k, one for v
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2) # [B, T, n_head * head_size] -> [B, n_head, T, head_size]
        kv = self.kv_proj(x)
        kv = kv.view(B, T, 2, self.n_kv_head, self.head_size) # [B, T, 2 * n_kv_head * head_size] -> [B, T, 2, n_kv_head, head_size]
        k, v = kv.unbind(dim=2) # split
        k = k.transpose(1, 2) # [B, T, n_kv_head, head_size] -> [B, n_kv_head, T, head_size]
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, attn_mask=None) # self.swa_mask) # using flash attention; causal mask is handled internally here
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 2 * config.n_embd) # originally 4x, but we want to keep it light
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(2 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config) # CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class DPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.ctx_len, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme - saves us a lot of training params
        self.transformer.wte.weight = self.lm_head.weight
        # initialize params, apply of nn.Module iterates over all submodules and applies _init_weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5 # every layer has two blocks that add to the residual pathway: att and mlp
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # by default, it is initalized with uniform distr
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.ctx_len, f"cannot forward seq of len {T}, block size is only {self.config.ctx_len}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate):
        # start with all the candidate params (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. any params that is 2D will be weight decayed, otherwise no.
        # i.e all weight tensors in matmuls and embeddings decay, all biases and layernorms don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")
        # create adamW optim and use the fused version if available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    

def get_lr(it):
    # linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

def generate_emo(model: torch.nn.Module, enc: tiktoken.Encoding) -> None:
    # TODO: implement stop-tokens ?
    model.eval()
    num_return_sequences = 4
    max_length = 200
    tokens = enc.encode("<|user|> Good morning sir, what's going on? <|endoftext|> <|assistant|> "
                        , allowed_special={"<|endoftext|>", "<|assistant|>", "<|user|>"})
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to("cuda")
    sample_rng = torch.Generator(device="cuda")
    sample_rng.manual_seed(42)
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")


def main():
    PRETRAIN = os.environ.get("PRETRAIN", None)
    config = GPTConfig
    today = datetime.today().strftime("%m-%d")
    checkpoint_dir = f"data/checkpoints/dpt/checkpoint-{today}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "loss.txt")

    torch.manual_seed(420)
    torch.cuda.manual_seed(420)

    enc = tiktoken.get_encoding("gpt2")

    # we use those in get_lr
    global max_lr, min_lr, warmup_steps, max_steps
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 40 # 100 # 375e6 tokens / 2**19 -- maybe 100 is enough ??
    n_epochs = 300
    val_loss_steps = 20
    total_batch_size =  2**16 # 524288 # 2**19 (nice number), ~.5M, in number of tokens  522240
    B = config.batch_size # micro batch size
    T = config.ctx_len # sequence length

    assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = total_batch_size // (B * T)
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}") # one step is comprised of {grad_accum_steps} micro steps
    
    if PRETRAIN:
        print("PRETRAINING")
        train_loader = DataLoaderFine(B, T, data_root="data/train_fineweb_edu_369")
        val_loader = DataLoaderFine(B, T, data_root="data/val_fineweb_edu_369")
        steps_per_epoch = train_loader.get_total_tokens()
    else:
        print("FINETUNING")
        train_data = EmoData(T=T, data_path="data/train_emo")
        train_loader = iter(DataLoader(dataset=train_data, batch_size=B, num_workers=os.cpu_count()-2, pin_memory=True, drop_last=True))
        val_data = EmoData(T=T, data_path="data/val_emo")
        val_loader = iter(DataLoader(dataset=val_data, batch_size=B, num_workers=os.cpu_count()-2, pin_memory=True, drop_last=True))
        # total number of tokens in the dataset
        steps_per_epoch = round(train_data.total_token_count / total_batch_size)

    print(f"Number of steps in one Epoch: {steps_per_epoch}")
    max_steps = steps_per_epoch * n_epochs

    torch.set_float32_matmul_precision("high") # use TF32

    # get logits
    model = DPT(config)
    model.cuda()
    model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)

    epoch = 1
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        if step > 0 and step % steps_per_epoch == 0:
            epoch += 1

        # eval every 250 steps
        if step %500 == 0 or last_step:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = next(val_loader)
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
            print(f"validation loss: {val_loss_accum.item():4f}\n")
            with open(log_file, "a") as f:
                f.write(f"epoch:{epoch}|step:{step}|val loss:{val_loss_accum.item():4f}\n")
            # store checkpoints
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(checkpoint_dir, f"model_{today}_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step, 
                    'val_loss': val_loss_accum.item(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, checkpoint_path)

        # generate from the model once in a while
        if (step > 0 and step % 1000 == 0) or last_step:
            generate_emo(model, enc)

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = next(train_loader)
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM, we want a MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps # recover the additional normalizer
            loss_accum += loss.detach() # detach from the graph
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize() # finish all the gpu work
        t1 = time.time()
        dt = (t1 - t0) # time diff in seconds
        tokens_processed = B * T * grad_accum_steps
        tokens_per_sec = tokens_processed / dt
        if step % 100 == 0:
            print(f"epoch {epoch} | step {step} | loss: {loss_accum.item():.4f} | dt: {dt:.2f}s | tok/sec: {tokens_per_sec:.2f} | norm: {norm:.4f} | lr: {lr:.4e}")
            with open(log_file, "a") as f:
                f.write(f"epoch:{epoch}|step:{step}|train loss:{loss_accum.item():6f}\n")

    print(f"Done training after {step} steps.")


if __name__ == "__main__":
    main()

# 233937.97 tok/sec
# 264648.96 tok/sec with gqa
# 309907.14 tok/sec with bs 32
# 318151.16
# 316035.17 tok/sec with new loader


# total number of tokens processed: step * tokens_per_batch * grad_acc_steps


# without swa:  11052MiB /  12288MiB
# with swa:     11038MiB /  12288MiB 

# gqa, no swa just torch implementation: 9550MiB /  12288MiB and 312624.15 lol