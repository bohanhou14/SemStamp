from datasets import load_dataset, Dataset

dataset = load_dataset("allenai/c4", "realnewslike")
train_texts = dataset["train"]['text']
val_texts = dataset["validation"]['text']
test_texts = dataset["test"]['text']

Dataset.from_dict({'text': train_texts}).save_to_disk("data/c4_train")
Dataset.from_dict({'text': val_texts}).save_to_disk("data/c4_val")
Dataset.from_dict({'text': test_texts}).save_to_disk("data/c4_test")
