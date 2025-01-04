import tiktoken
from datasets import load_dataset

tokenizer = tiktoken.get_encoding("gpt2")
ds = load_dataset("Estwld/empathetic_dialogues_llm")
ds = ds.select_columns("conversations")

tokenizer._special_tokens["<|user|>"] = 50257
tokenizer._special_tokens["<|assistant|>"] = 50258
print(tokenizer._special_tokens)

def prep_convo(convs: dict, tokenizer: tiktoken.Encoding) -> list:
    # string together one entire conversation
    fmt_convs = []
    for conv in convs['conversations']:
        if conv["role"] == "user":
            fmt_convs.append("<|user|>")
        else:
            fmt_convs.append("<|assistant|>")
        fmt_convs.append(conv["content"])
        fmt_convs.append("<|endoftext|>")
    convo_str = " ".join(fmt_convs)
    tokens = {"tokens": tokenizer.encode(convo_str, allowed_special={"<|user|>", "<|assistant|>", "<|endoftext|>"})}
    return tokens


def main() -> None:
    ds_tok = ds.map(lambda x: prep_convo(x, tokenizer))
    ds_tok["train"].save_to_disk(dataset_path="data/train_emo")
    ds_tok["valid"].save_to_disk(dataset_path="data/val_emo")
    ds_tok["test"].save_to_disk(dataset_path="data/test_emo")

if __name__ == "__main__":
    main()
