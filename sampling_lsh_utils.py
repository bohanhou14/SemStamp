import sampling_utils
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, StoppingCriteriaList
from sbert_lsh_model import SBERTLSHModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from sampling_utils import SentenceEndCriteria, device, gen_sent

# rng = torch.Generator()
device = "cuda" if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device)
MAX_TRIALS = sampling_utils.MAX_TRIALS
hash_key = sampling_utils.hash_key

def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))


def get_mask_from_seed(lsh_dim: int, accept_rate: float, seed: int):
    n_bins = 2**lsh_dim
    n_accept = int(n_bins * accept_rate)
    rng.manual_seed(hash_key * seed)
    vocab_permutation = torch.randperm(n_bins, device='cuda', generator=rng)
    greenlist_ids = vocab_permutation[:n_accept]
    return greenlist_ids.to(device)


def reject_close_generation(lsh_model, sents, margin, cutoff=None):
    embeds = lsh_model.get_embeddings(sents)
    embeds = torch.tensor(embeds, device='cuda')
    normals = torch.tensor(lsh_model.hasher.normals, device='cuda')
    if cutoff != None:
        normals = normals[:cutoff]

    # sims[i, j] is the cosine similarity between the ith generation and the jth normal vec
    sims = cosine_distance_matrix(embeds, normals)
    sims_abs = torch.abs(sims)
    # max_sim is the highest cosine similarity of each generation with any normal vec
    min_sims = sims_abs.min(dim=1).values
    select = []
    for i in range(len(min_sims)):
        # print(max_sims[i])
        min_sim = min_sims[i].item()
        if (abs(min_sim) >= margin):
            # print(min_sim)
            select.append(i)
    sents = np.array(sents)
    sents = sents[select]
    return list(sents), select

def lsh_reject_completion(
        prompt: str,
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, gen_config: GenerationConfig,  # gen args
        lsh_model: SBERTLSHModel, lsh_dim: int,  # LSH args
        # watermark args. lambda is probability of accepting (i.e., green list size)
        lmbd=1.0,
        device='cuda',
        margin=0.002,
        **kwargs):
    print(f"prompt: {prompt}")
    stats = {}
    sent_end_criteria = SentenceEndCriteria(tokenizer)
    lsh_seed = lsh_model.get_hash([prompt])[0]
    accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)

    text = prompt
    new_text = prompt
    text_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = len(text_ids[0])
    sent_end_criteria.update(new_text)

    total_trials = 0
    success_trials = 0  # also include trials that maxed out MAX_TRIALS
    current_trials = 0
    maxedout_trials = 0
    debug_text_segments = [(prompt, text_ids.size(1), lsh_seed)]
    while True:
        stopping_criteria = StoppingCriteriaList([sent_end_criteria])
        if "opt" in model.config._name_or_path:
            new_text, new_text_ids = gen_sent(model = model, 
                tokenizer = tokenizer, 
                text_ids = text_ids,
                gen_config = gen_config,
                stopping_criteria = stopping_criteria
            )
        else:
            raise NotImplementedError("model type not supported")
        if new_text == '':
            print('WARNING: stopped generation because generated nothing (after discarding last generated token)', flush=True)
            break
        total_trials += 1
        current_trials += 1
        accepted_text, _ = reject_close_generation(
                lsh_model, [new_text], margin=margin, cutoff=None)
        if (len(accepted_text) == 0 and current_trials < MAX_TRIALS):
            continue
        lsh_candidate = lsh_model.get_hash([new_text])[0]
        if lsh_candidate not in accept_mask:
            continue
        if (lsh_candidate in accept_mask) or current_trials >= MAX_TRIALS:
            if current_trials >= MAX_TRIALS:
                print(
                    f'WARNING: desired semantic signature can\'t be sampled after max_trials {MAX_TRIALS}', flush=True)
                print(f'CONTEXT: {text}', flush=True)
                print(
                    f'NOTE: use regular (non-filtered-by-sig) continuation: {new_text}', flush=True)
                maxedout_trials += 1
            debug_text_segments.append(
                (new_text, new_text_ids.size(1) - text_ids.size(1), lsh_candidate))
            current_trials = 0
            success_trials += 1
            # passed, proceed to next sentence
            lsh_seed = lsh_candidate
            accept_mask = get_mask_from_seed(lsh_dim, lmbd, lsh_seed)
            text += new_text
            text_ids = new_text_ids
            sent_end_criteria.update(text)
            if (len(text_ids[0]) - prompt_length) >= gen_config.max_new_tokens-1:
                break
    return text