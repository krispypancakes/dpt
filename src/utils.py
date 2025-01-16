import matplotlib.pyplot as plt
import tiktoken
import torch.nn.functional as F
import os
import torch

def show_progress(checkpoint_path: str) -> None:
    with open(checkpoint_path, "r") as f:
        losses = f.readlines()
    
    train_losses = [(l.split(" ")[0], l.split(" ")[-1].strip()) for l in losses if "train" in l]
    val_losses = [(l.split(" ")[0], l.split(" ")[-1].strip()) for l in losses if "val" in l]

    step_train = [int(s[0]) for s in train_losses]
    loss_train = [float(s[1]) for s in train_losses]

    step_val = [int(s[0]) for s in val_losses]
    loss_val = [float(s[1]) for s in val_losses]

    print(f"min train loss: {min(loss_train)}")
    print(f"min val loss: {min(loss_val)}")
    
    plt.plot(step_train, loss_train)
    plt.plot(step_val, loss_val)
    plt.legend(("train", "val"))
    plt.ylabel("loss")
    plt.xlabel("steps")
    plt.yticks(range(2,11))
    plt.show()


def generate_seq(model: torch.nn.Module) -> None:
    PRETRAIN = os.environ.get("PRETRAIN", None)
    encoding = tiktoken.get_encoding("gpt2")
    # TODO: implement stop-tokens ?
    model.eval()
    num_return_sequences = 4
    max_length = 200
    if PRETRAIN:
        tokens = encoding.encode("A lovely day for a lonely machine that looks to ")
    else:
        tokens = encoding.encode("<|user|>Good morning sir, what's going on?<|endoftext|><|assistant|> "
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
        decoded = encoding.decode(tokens)
        print(f"sample {i}: {decoded}")
