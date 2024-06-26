import argparse
import os
from datasets import load_from_disk
import torch
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import StoppingCriteriaList
from collections import defaultdict
import pickle
from tqdm import trange
from kmeans_pytorch import *  # maybe faiss
import sampling_utils
from sampling_utils import SentenceEndCriteria, gen_sent

rng = sampling_utils.rng
MAX_TRIALS = sampling_utils.MAX_TRIALS
hash_key = sampling_utils.hash_key

def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=-1)

    return choice_cluster.cpu()

def update_pickle(name, input_to_embed):
    with open(name, 'rb') as f:
        d = pickle.load(f)
        d.update(input_to_embed)
    f.close()
    with open(name, 'wb') as f:
        pickle.dump(d, f)
    f.close()

def embed_gen_list(dataset_path, embedder_path, load_batch_size=10000, encode_batch_size=32, device='cuda', padding='max_length'):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(
        lambda example, idx: {
            'text_unique_id': idx
        },
        with_indices=True
    )
    embedder = SentenceTransformer(embedder_path, device=device)
    embedder = embedder.eval()
    texts = dataset['text']
    sent_embeds = [];  embeds = defaultdict()
    name = os.path.join(dataset_path, "embeds.pkl")
    initial_load = False
    # divide texts and paras into batches
    text_chunks = [texts[i:i + encode_batch_size] for i in range(0, len(texts), encode_batch_size)]

    for i in trange(len(text_chunks), desc="encoding chunks"):
        text_chunk = text_chunks[i]
        text_embeds = embedder.encode(text_chunk, convert_to_tensor=True); sent_embeds.extend(text_embeds)
        if (len(sent_embeds) >= load_batch_size and len(sent_embeds) % load_batch_size == 0):
            embeds['text'] = sent_embeds
            embeds = dict(embeds)
            if initial_load == True:
                update_pickle(name, embeds)
            else:
                with open(name, 'wb') as f:
                    pickle.dump(embeds, f)
                initial_load = True
                f.close()
            del embeds; embeds = defaultdict()
    # after all dic updates, load the rest < load_batch_size data
    if (len(sent_embeds) > 0):
        embeds['text'] = sent_embeds
        embeds = dict(embeds)
        if initial_load == True:
            update_pickle(name, embeds)
        else:
            with open(name, 'wb') as f:
                pickle.dump(embeds, f)
    return name


def get_cluster_mask(curr_cluster_id, k_dim, lmbd):
    rng.manual_seed(curr_cluster_id.item() * hash_key)
    num_accept = int(k_dim * lmbd)
    mask = torch.randperm(k_dim, device='cuda', generator=rng)[:num_accept]
    return mask.to('cuda')

def kmeans_reject_overlap(text, embedder, cluster_centers, margin=0.01):
    gen_embed = embedder.encode(text, convert_to_tensor=True)
    gen_embed = gen_embed.reshape(1, -1)
    cluster_centers = torch.tensor(np.array(cluster_centers))
    dis = pairwise_cosine(gen_embed, cluster_centers, device='cuda')

    # each row of ranking corresponds to the cluster distance closeness of a generation
    ranked_dis = torch.argsort(dis, dim=-1)
    closest = ranked_dis[0]

    # second nearest cluster
    second_closest = ranked_dis[1]

    first_dis = dis[closest]

    sec_dis = dis[second_closest]

    if ((sec_dis - first_dis) > margin):
        return text, closest.clone().detach()
    else:
        return None, closest.clone().detach()


def get_cluster_id(text, cluster_centers, embedder):
    embedding = embedder.encode(text, convert_to_tensor=True)
    embedding = embedding.reshape(1, -1)
    # print(cluster_centers.shape)
    cluster_id = kmeans_predict(
        embedding,
        cluster_centers=cluster_centers,
        distance='cosine',
        device='cuda'
    )
    return cluster_id


def kmeans_reject_completion(
        prompt: str,
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, gen_config: GenerationConfig,  # gen args
        embedder: SentenceTransformer,
        lmbd: float,
        cluster_centers: torch.Tensor,
        # LSH args # watermark args. lambda is probability of accepting (i.e., green list size)
        k_dim: int,
        device='cuda',
        margin=0.01,
        **kwargs):

    sent_end_criteria = SentenceEndCriteria(tokenizer)
    curr_cluster_id = get_cluster_id(prompt, cluster_centers, embedder)
    mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)

    text = prompt
    new_text = prompt
    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = len(text_ids[0])
    sent_end_criteria.update(new_text)

    total_trials = 0
    success_trials = 0  # also include trials that maxed out MAX_TRIALS
    current_trials = 0
    maxedout_trials = 0
    debug_text_segments = [(prompt, text_ids.size(1), curr_cluster_id)]
    cluster_id_sequence = [curr_cluster_id.item()]
    
    while True:
        # input_ids = tokenizer.encode(last_step_text, return_tensors='pt').to(device)
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        new_text, new_text_ids = gen_sent(model = model, 
                tokenizer = tokenizer, 
                text_ids = text_ids,
                gen_config = gen_config,
                stopping_criteria = stopping_criteria
            )
        # print(f'NEW TEXT: {new_text}')
        if new_text == '':
            print('WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
            break

        total_trials += 1
        current_trials += 1

        accepted_text, curr_cluster_id = kmeans_reject_overlap(text=new_text, embedder=embedder, cluster_centers=cluster_centers, margin=margin)

        if (accepted_text == None and current_trials < MAX_TRIALS):
            continue
        else:
            new_text = accepted_text if accepted_text != None else new_text
        cluster_id_sequence.append(curr_cluster_id.item())
        if (curr_cluster_id in mask) or current_trials >= MAX_TRIALS:
            if current_trials >= MAX_TRIALS:
                print(
                    f'WARNING: desired semantic signature can\'t be sampled after max_trials {MAX_TRIALS}', flush=True)
                print(f"cluster_id_sequence: {cluster_id_sequence}")
                print(
                    f"cluster_ids_counter: {torch.bincount(torch.tensor(cluster_id_sequence))}")
                print(f'CONTEXT: {text}', flush=True)
                print(
                    f'NOTE: use regular (non-filtered-by-sig) continuation: {new_text}', flush=True)
                maxedout_trials += 1
            debug_text_segments.append(
                (new_text, new_text_ids.size(1) - text_ids.size(1), curr_cluster_id))
            current_trials = 0
            success_trials += 1
            mask = get_cluster_mask(curr_cluster_id, k_dim, lmbd)
            text += new_text
            text_ids = new_text_ids
            sent_end_criteria.update(text)
            if (len(text_ids[0]) - prompt_length) >= gen_config.max_new_tokens-1:
                break
    return text


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

def get_cluster_centers(embeds, k_dim, gamma=0.002):
    cluster_ids, cluster_centers = kmeans(
        embeds,
        num_clusters=k_dim,
        distance='cosine',
        device='cuda'
    )
    return cluster_ids, cluster_centers

def load_embeds(embed_path):
    with open(embed_path, 'rb') as f:
        d = pickle.load(f)
    gen_embeds = torch.stack(d['text']).to('cuda').squeeze()
    return gen_embeds


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('embedder_path', type=str)
    args = parser.parse_args()
    embed_gen_list(args.data_path, args.embedder_path)
