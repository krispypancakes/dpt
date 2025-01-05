import tiktoken
from datasets import load_dataset

tokenizer = tiktoken.get_encoding("gpt2")
ds = load_dataset("Estwld/empathetic_dialogues_llm")
ds = ds.select_columns("conversations")

tokenizer._special_tokens["<|user|>"] = 50257
tokenizer._special_tokens["<|assistant|>"] = 50258
tokenizer._special_tokens["<|endofconversation|>"] = 50259

def tokenize_conv(conv: list, tokenizer: tiktoken.Encoding) -> dict:
    """ tokenize one conversation """
    token_cnt = 0
    token_conv = []
    fmt_conv = ["<|endofconversation|>"] # start of conversation
    for txt in conv["conversations"]:
        fmt_conv.append("<|endoftext|>") # start of text
        if txt["role"] == "user": # assign role
            fmt_conv.append("<|user|>")
        else:
            fmt_conv.append("<|assistant|>")
        fmt_conv.append(txt["content"]) # the actual text
        conv_str = " ".join(fmt_conv)
        # tokenize and allow the special ones
        tokens = tokenizer.encode(conv_str, 
                    allowed_special={"<|user|>", "<|assistant|>", "<|endofconversation|>", "<|endoftext|>"}) 
    token_cnt += len(tokens)
    token_conv.extend(tokens)
    return {"tokens": token_conv, "token_count": token_cnt}


def main() -> None:
    ds_tok = ds.map(lambda x: tokenize_conv(x, tokenizer))
    ds_tok["train"].save_to_disk(dataset_path="data/train_emo")
    ds_tok["valid"].save_to_disk(dataset_path="data/val_emo")
    ds_tok["test"].save_to_disk(dataset_path="data/test_emo")

if __name__ == "__main__":
    main()
