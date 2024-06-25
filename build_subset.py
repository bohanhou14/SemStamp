import argparse
from datasets import load_from_disk, Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text column')
    parser.add_argument('--n', help='number of texts to keep', default=1000, type=int)
    args = parser.parse_args()
    dataset = load_from_disk(args.dataset_path)
    texts = dataset['text']
    texts = texts[:args.n]
    new_path = args.dataset_path + f'-{args.n}'
    Dataset.from_dict({'text': texts}).save_to_disk(new_path)

