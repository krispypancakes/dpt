from datasets import load_dataset, Dataset


def load_dump(_dump: str) -> Dataset:
    ds = load_dataset("airtrain-ai/fineweb-edu-fortified", _dump)
    return ds.filter(lambda x: x["score"]>=3, num_proc=10)

def main() -> None:
    ds = load_dump("CC-MAIN-2013-20")
    train_test = ds["train"].train_test_split(test_size=.2, seed=69)
    train_val = train_test["train"].train_test_split(test_size=.1, seed=69)
    
    train_val["train"].save_to_disk("data/train_ds")
    train_val["test"].save_to_disk("data/val_ds")
    train_test["test"].save_to_disk("data/test_ds")
    

if __name__ == "__main__":
    main()
