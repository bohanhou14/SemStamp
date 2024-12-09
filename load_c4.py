from datasets import load_dataset, Dataset
import argparse
from tqdm import trange

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20000)
    args = parser.parse_args()
    train_dataset = load_dataset("allenai/c4", "realnewslike", split=f"train[:{args.k}]")
    human_dataset = load_dataset("allenai/c4", "realnewslike", split=f"train[{args.k}:{args.k+5000}]") # take the human texts from the next 5000 samples for reference
    val_dataset = load_dataset("allenai/c4", "realnewslike", split=f"validation[:{args.k}]")
    train_dataset.save_to_disk("data/c4-train")
    human_dataset.save_to_disk("data/c4-human")
    val_dataset.save_to_disk("data/c4-val")
