from datasets import load_from_disk, load_dataset
import tiktoken
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp


tokenizer = tiktoken.get_encoding("gpt2")
shard_size = int(1e8)  # 100M tokens per shard


def load_filter():
    fw_sample = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", cache_dir="data/raw/")
    fw_filtered = fw_sample["train"].filter(lambda x: x["scroe"] >= 3.69)

    train_test = fw_filtered.train_test_split(test_size=.1, seed=42)
    train_set = train_test["train"]
    val_set = train_test["test"]

    # name: split_dataset_score
    train_set.save_to_disk("data/train_fineweb_edu_369")
    val_set.save_to_disk("data/val_fineweb_edu_369")


def tokenize_single(text):
    tokens = [tokenizer._special_tokens['<|endoftext|>']]
    tokens.extend(tokenizer.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token values too large for uint16"
    return tokens_np.astype(np.uint16)

def write_file(filename, np_tokens):
    np.save(filename, np_tokens)

def process_chunk(chunk):
    return [tokenize_single(text) for text in chunk]

def preprocess(path: str, split: str) -> None:
    ds = load_from_disk(dataset_path=path)
    
    # Determine number of processes
    num_processes = max(1, os.cpu_count() - 1)  # Use all but one CPU core
    
    # Split the dataset into chunks
    chunk_size = len(ds) // num_processes
    chunks = [ds['text'][i:i+chunk_size] for i in range(0, len(ds), chunk_size)]
    
    token_count = 0
    shard_count = 0
    np_tokens = np.empty((shard_size,), dtype=np.uint16)
    
    with mp.Pool(num_processes) as pool:
        for chunk_tokens in tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc=f"Processing {split} data"):
            for tokens in chunk_tokens:
                if token_count + len(tokens) > shard_size:
                    np_shard = os.path.join(path, f"{split}_np_tokens_{shard_count:06d}")
                    write_file(np_shard, np_tokens[:token_count])
                    shard_count += 1
                    token_count = 0
                
                np_tokens[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
    
    # Write the last shard if there's any data left
    if token_count > 0:
        np_shard = os.path.join(path, f"{split}_np_tokens_{shard_count:06d}")
        write_file(np_shard, np_tokens[:token_count])


if __name__ == '__main__':
    preprocess("data/train_fineweb_edu_369", "train")
    preprocess("data/val_fineweb_edu_369", "val")
