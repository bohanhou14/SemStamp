import sys
from datasets import load_from_disk, Dataset
import os

ds = load_from_disk(sys.argv[1])
# split into train, val, test
train_len = int(0.8 * len(ds))
val_len = int(0.1 * len(ds))
test_len = len(ds) - train_len - val_len
train_texts = ds['text'][:train_len]
train_para_texts = ds['para_text'][:train_len]
val_texts = ds['text'][train_len:train_len + val_len]
val_para_texts = ds['para_text'][train_len:train_len + val_len]
test_texts = ds['text'][train_len + val_len:]
test_para_texts = ds['para_text'][train_len + val_len:]

if not os.path.exists(sys.argv[1]+"-split"):
    os.mkdir(sys.argv[1]+"-split")

Dataset.from_dict({'text': train_texts, "para_text": train_para_texts}).save_to_disk(sys.argv[1]+"-split/train")
Dataset.from_dict({'text': val_texts, "para_text": val_para_texts}).save_to_disk(sys.argv[1]+"-split/valid")
Dataset.from_dict({'text': test_texts, "para_text": test_para_texts}).save_to_disk(sys.argv[1]+"-split/test")