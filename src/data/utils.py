from typing import Tuple
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset
import os
import math
import numpy as np


class EmoData(Dataset):
    def __init__(self, T: int, data_path: str) -> None:
        dataset = load_from_disk(dataset_path=data_path)
        self.T = T
        self.total_token_count = sum([tkcnt for tkcnt in dataset["token_count"]])
        all_tokens = []
        for tokens in dataset["tokens"]:
            all_tokens.extend(tokens)
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self) -> None: 
        return self.total_token_count - self.T - 1 # all possible training examples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.tokens[idx:idx + self.T + 1]
        x = sequence[:-1]
        y = sequence[1:]
        return x, y


class DataLoaderFine:
    def __init__(self, B: int, T: int, data_root: str) -> None:
        self.B = B
        self.T = T
        self.data_root = data_root
        # get the shard filenames
        shards = os.listdir(data_root)
        # we train on the numpy files
        shards = [s for s in shards if s.split(".")[-1]=="npy"]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(shards) > 0, f"no shards found for split"
        print(f"found {len(shards)} shards for split")
        self.reset()
    
    def __iter__(self) -> 'DataLoaderFine':
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.next_batch()

    def reset(self) -> None:
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0
        self.preload_shard()
    
    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # outputs
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            # this is actually pretty cool, circular iteration
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            # preloaded tokens
            self.tokens = self.pre_loaded_tokens
            self.current_position = 0
            self.preload_shard()

        return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
    
    def get_total_tokens(self) -> int:
        ds = load_from_disk(dataset_path=self.data_root)
        return sum([tcnt for tcnt in ds["token_count"]])
    
    def preload_shard(self) -> None:
        # keeps tokens of next shard in memory
        next_shard = (self.current_shard+2) % len(self.shards)
        self.pre_loaded_tokens = self.load_tokens(self.shards[next_shard])    

    @staticmethod
    def load_tokens(filename: str):
        npt = np.load(filename).astype(np.int32)
        return torch.tensor(npt, dtype=torch.long)

    @staticmethod
    def get_lr(it: int, warmup_steps: int, max_steps: int, min_lr: int, max_lr: int) -> float:
        # linear warmup
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
