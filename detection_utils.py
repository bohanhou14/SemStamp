from sklearn.metrics import roc_curve, auc
import sampling_utils
from sampling_lsh_utils import get_mask_from_seed
from sampling_kmeans_utils import get_cluster_mask, get_cluster_id
import numpy as np
import torch
from bert_score import BERTScorer
import matplotlib.pyplot as plt
import os
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)
scorer = BERTScorer(model_type = "microsoft/deberta-xlarge-mnli", rescale_with_baseline=True, device=device, lang = "en")

def run_bert_score(gen_sents, para_sents):
    P, R, F1 = scorer.score(gen_sents, para_sents)
    return torch.mean(F1).item()

def flatten_gens_and_paras(gens, paras):
    new_gens = []
    new_paras = []
    for gen, para in zip(gens, paras):
        min_len = min(len(gen), len(para))
        new_gens.extend(gen[:min_len])
        new_paras.extend(para[:min_len])
    return new_gens, new_paras


def truncate_to_max_length(texts, max_length):
    new_texts = []
    for t in texts:
        t = " ".join(t.split(" ")[:max_length])
        if t[-1] not in sampling_utils.PUNCTS:
            t = t + "."
        new_texts.append(t)
    return new_texts

def detect_kmeans(sents, embedder, lmbd, k_dim, cluster_centers):
    n_sent = len(sents)
    n_watermark = 0
    curr_cluster_id = get_cluster_id(
        sents[0], embedder=embedder, cluster_centers=cluster_centers)
    cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)
    # print(f"Prompt: {sents[0]}")
    for i in range(1, n_sent):
        curr_cluster_id = get_cluster_id(
            sents[i], embedder=embedder, cluster_centers=cluster_centers)
        # print(f'sentence {i}: {sents[i]}')
        # print(f'k-means index in accept_mask: {curr_cluster_id in cluster_mask}')
        # print(f'curr_cluster_id: {curr_cluster_id}')
        if curr_cluster_id in cluster_mask:
            n_watermark += 1
        cluster_mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)
    n_test_sent = n_sent - 1  # exclude the prompt
    num = n_watermark - lmbd * (n_test_sent)
    denom = np.sqrt((n_test_sent) * lmbd * (1-lmbd))
    # print(f'n_watermark: {n_watermark}, n_test_sent: {n_test_sent}')
    return num / denom

def detect_lsh(sents, lsh_model, lmbd, lsh_dim, cutoff=None):
    if cutoff == None:
        cutoff = lsh_dim
    n_sent = len(sents)
    n_watermark = 0
    lsh_seed = lsh_model.get_hash([sents[0]])[0]
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
    for i in range(1, len(sents)):
        lsh_candidate = lsh_model.get_hash([sents[i]])[0]
        if lsh_candidate in accept_mask:
            n_watermark += 1
        lsh_seed = lsh_candidate
        accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
    n_test_sent = n_sent - 1  # exclude the prompt and the ending
    num = n_watermark - lmbd * (n_test_sent)
    denom = np.sqrt((n_test_sent) * lmbd * (1-lmbd))
    # print(f'n_watermark: {n_watermark}, n_test_sent: {n_test_sent}')
    zscore = num / denom
    # print(f"zscore: {zscore}")
    return zscore

def get_roc_metrics(labels, preds):
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_roc_metrics_from_zscores(m, mp, h, dataset_path):
    mp = np.nan_to_num(mp)
    h = np.nan_to_num(h)
    len_z = len(mp)
    mp_fpr, mp_tpr, mp_area = get_roc_metrics(
        [1] * len_z + [0] * len_z, np.concatenate((mp, h[:len_z])))
    plt.plot(mp_fpr, mp_tpr)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("ROC Curve")
    name = os.path.join(dataset_path, "roc_curve.png")
    plt.savefig(name)
    name = os.path.join(dataset_path, "fpr.npy")
    np.save(name, mp_fpr)
    name = os.path.join(dataset_path, "tpr.npy")
    np.save(name, mp_tpr)
    return mp_area, mp_fpr


def evaluate_z_scores(mz, mpz, hz, dataset_path):
    mz = np.array(mz)
    mpz = np.array(mpz)
    hz = np.array(hz)
    fpr_5_threshold = 0
    fpr_1_threshold = 0
    for z_threshold in np.arange(0, 6, 0.005):
        fp = len(hz[hz > z_threshold]) / len(hz)
        if (fp >= 0.0095 and fp <= 0.0104):
            fpr_1_threshold = z_threshold
        elif (fp >= 0.045 and fp <= 0.054):
            fpr_5_threshold = z_threshold
    mp_area, mp_fpr = get_roc_metrics_from_zscores(mz, mpz, hz, dataset_path)
    if fpr_1_threshold == 0:
        fpr_1_threshold = 2.33 # according to standard z-score table
    return mp_area, len(mpz[mpz > fpr_1_threshold]) / len(mpz), len(mpz[mpz > fpr_5_threshold]) / len(mpz)

