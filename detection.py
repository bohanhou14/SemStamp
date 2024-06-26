
from transformers import AutoTokenizer, AutoModelForCausalLM
# from eval_clm import get_roc_metrics
import pandas as pd
from sbert_lsh_model import SBERTLSHModel
from tqdm import trange
from sentence_transformers import SentenceTransformer
import argparse
from datasets import load_from_disk
from nltk.tokenize import sent_tokenize
import os
import torch
import numpy as np
from detection_utils import detect_kmeans, detect_lsh, run_bert_score, evaluate_z_scores, flatten_gens_and_paras

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='hf dataset containing text and para_text columns')
    parser.add_argument('--detection_mode', choices=['kmeans', 'lsh'], help='detection mode. lsh for semstamp and kmeans for k-semstamp')
    parser.add_argument('--cc_path', type=str, help='path to cluster centers')
    parser.add_argument('--embedder', type=str, help='sentence embedder')
    parser.add_argument('--model', type=str, help='backbone LM for text generation', default='facebook/opt-1.3b')
    parser.add_argument('--sp_dim', type=int, default=3, help='dimension of the subspaces. default 3 for sstamp and 8 for ksstamp')
    parser.add_argument('--max_new_tokens', type=int, default=205)
    parser.add_argument('--lmbd', type=float, default=0.25, help='ratio of valid sentences')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = load_from_disk(args.dataset_path)
    gens = dataset['text']
    paras = None
    if 'para_text' in dataset.column_names:
        paras = dataset['para_text']
    human_texts = load_from_disk(args.human_text)['text'][:len(gens)]
    z_scores, para_scores, human_scores = [], [], []
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # ksemstamp detection
    if args.detection_mode == 'kmeans':
        cluster_centers = torch.load(args.cc_path)
        embedder = SentenceTransformer(args.embedder)
        for i in trange(0, len(gens), 1, desc='kmeans_detection'):
            gen_sents = gens[i] if type(gens[i]) == list else sent_tokenize(gens[i])
            if paras != None:
                para_sents = paras[i] if type(paras[i]) == list else sent_tokenize(paras[i])
            z_score = detect_kmeans(sents=gen_sents, embedder=embedder, lmbd=args.lmbd,
                                    k_dim=args.sp_dim, cluster_centers=cluster_centers)
            para_score = detect_kmeans(
                sents=para_sents, embedder=embedder, lmbd=args.lmbd, k_dim=args.sp_dim, cluster_centers=cluster_centers)
            z_scores.append(z_score)
            para_scores.append(para_score)

        for i in trange(0, len(human_texts), 1, desc='kmeans_human'):
            sents = sent_tokenize(human_texts[i])
            z_score = detect_kmeans(sents=sents, embedder=embedder, lmbd=args.lmbd,
                                    k_dim=args.sp_dim, cluster_centers=cluster_centers)
            human_scores.append(z_score)
    # semstamp detection
    elif args.detection_mode == 'lsh':
        lsh_model_class = SBERTLSHModel
        lsh_model = lsh_model_class(
            lsh_model_path=args.embedder, device='cuda', batch_size=1, lsh_dim=args.sp_dim, sbert_type='base')
        for i in trange(0, len(gens), 1):
            text_sents = gens[i] if type(gens[i]) == list else sent_tokenize(gens[i])
            if paras != None:
                para_sents = paras[i] if type(paras[i]) == list else sent_tokenize(paras[i])
            z_score = detect_lsh(sents=text_sents, lsh_model=lsh_model,
                                 lmbd=args.lmbd, lsh_dim=args.sp_dim)
            para_z_score = detect_lsh(
                sents=para_sents, lsh_model=lsh_model, lmbd=args.lmbd, lsh_dim=args.sp_dim)
            z_scores.append(z_score)
            para_scores.append(para_z_score)

        for i in trange(0, len(human_texts), 1):
            sents = sent_tokenize(human_texts[i])
            z_score = detect_lsh(sents=sents, lsh_model=lsh_model,
                                 lmbd=args.lmbd, lsh_dim=args.sp_dim)
            human_scores.append(z_score)

    z_score_name = os.path.join(
        args.dataset_path, "z_scores.npy")
    para_score_name = os.path.join(
        args.dataset_path, "para_z_scores.npy")
    human_score_name = os.path.join(
        args.dataset_path, "human_z_scores.npy")

    np.save(z_score_name, z_scores)
    np.save(para_score_name, para_scores)
    np.save(human_score_name, human_scores)

    results_path = os.path.join(
        args.dataset_path, f"results.csv")
    auroc, fpr1, fpr5 = evaluate_z_scores(
        args, z_scores, para_scores, human_scores, args.dataset_path)
    
    gen_sents, para_sents = flatten_gens_and_paras(gens, paras)
    bert_score = run_bert_score(gen_sents, para_sents)
    metrics = [f"{auroc:.3f}", f"{fpr1:.3f}", f"{fpr5:.3f}", f"{bert_score:.3f}"]
    columns = ["auroc", "fpr1", "fpr5", "bert_score"]
    df = pd.DataFrame(data=[metrics], columns=columns)
    df.to_csv(results_path, sep="\t", index=False)
