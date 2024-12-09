import pprint
import argparse
import os
import sys
import torch.multiprocessing as mp
from datasets import load_from_disk, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sbert_lsh_model import SBERTLSHModel
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, Queue
import numpy as np
from nltk.tokenize import sent_tokenize
from sampling_utils import extract_prompt_from_text
from sampling_lsh_utils import lsh_reject_completion
from sampling_kmeans_utils import kmeans_reject_completion, load_embeds

PUNCTS = '.,!?'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'data', type=str, help='Path to Hugging Face dataset that has a column "text".')
    parser.add_argument(
        '--model', type=str, help='Model name to generate continuation. HuggingFace/OpenAI.', default="facebook/opt-1.3b")
    parser.add_argument(
        '--embedder', type=str, help='Model name to embed sentences.', default=None)
    parser.add_argument('--len_prompt', '-l', type=int, default=32,
                        help='MAX length of prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=205,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--min_new_tokens', type=int, default=195,
                        help='Minimum number of new tokens to generate.')
    parser.add_argument('--rep_p', type=float, default=1.05,
                        help='Repetition penalty.')
    parser.add_argument('--lmbd', type=float, default=0.25,
                        help='Ratio of valid sentences.')
    parser.add_argument('--delta', type=float, default=0,
                        help='Logit augmentation for baseline or margin size for LSH and KMeans.')
    parser.add_argument('--sp_mode', type=str, choices=['lsh', 'kmeans'],
                        help='Spatial mode for generation (lsh or kmeans).', default=None)
    parser.add_argument('--sp_dim', type=int, default=8,
                        help='Number of partitions in the embedding space. Default is 8.')
    parser.add_argument('--embed_path', type=str,
                        help='Path to precomputed embed for training KMeans.', default=None)
    parser.add_argument('--cc_path', type=str,
                        help='KMeans precomputed cluster centers data.', default=None)
    pp = pprint.PrettyPrinter(indent=4)
    args = parser.parse_args()
    pp.pprint(vars(args))  # Debug print for parsed arguments
    return args

def worker(rank, dataset_chunk, output_queue, args, device):
    """
    Worker function to process a dataset chunk on a single GPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, use_safetensors=False)
    model.to(device)
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=0,
        repetition_penalty=args.rep_p,
    )

    if args.sp_mode == "lsh":
        lsh_model = SBERTLSHModel(
            lsh_model_path=args.embedder, device=device, batch_size=1, lsh_dim=args.sp_dim, sbert_type='base'
        )

        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            response = lsh_reject_completion(
                prompt, model, tokenizer, gen_config, lsh_model, args.sp_dim,
                lmbd=args.lmbd, device=device, margin=args.delta
            )
            ex['text'] = response.strip()
            return ex

    elif args.sp_mode == "kmeans":
        cluster_centers = torch.load(args.cc_path)
        # print(f"Load cluster centers: {cluster_centers.shape}")

        embedder = SentenceTransformer(args.embedder, device=device)

        def text_to_generated_text(ex):
            prompt = extract_prompt_from_text(ex['text'], args.len_prompt)
            response = kmeans_reject_completion(
                prompt=prompt, model=model, tokenizer=tokenizer,gen_config=gen_config, embedder=embedder,
                cluster_centers=cluster_centers, lmbd=args.lmbd, k_dim=args.sp_dim, margin=args.delta, device=device
            )
            ex['text'] = response.strip()
            return ex
    else:
        raise NotImplementedError

    processed_chunk = dataset_chunk.map(text_to_generated_text, batch_size=1)
    output_queue.put(processed_chunk)

def parallel_generate(args):
    """
    Splits the dataset and distributes work across multiple GPUs by index.
    """
    dataset = load_from_disk(args.data)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs detected. This script requires at least one GPU.")

    print(f"Detected {num_gpus} GPU(s). Splitting dataset for parallel processing.")

    output_queue = Queue()
    processes = []

    # Create a process for each GPU and its respective dataset shard
    for rank in range(num_gpus):
        device = f"cuda:{rank}"
        # Shard the dataset by assigning a specific index to the GPU
        dataset_chunk = dataset.shard(num_shards=num_gpus, index=rank)
        p = Process(target=worker, args=(rank, dataset_chunk, output_queue, args, device))
        p.start()
        processes.append(p)

    all_results = []
    for _ in processes:
        all_results.append(output_queue.get())

    for p in processes:
        p.join()

    # Combine all results into a single dataset
    merged_dataset = Dataset.from_dict({'text': [item['text'] for chunk in all_results for item in chunk]})
    output_path = os.path.join(args.data, f"{args.sp_mode}-generated")
    os.makedirs(output_path, exist_ok=True)
    merged_dataset.save_to_disk(output_path)

if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    parallel_generate(args)
