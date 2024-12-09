import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import StoppingCriteria
from nltk.tokenize import sent_tokenize
from string import punctuation
from itertools import groupby

MAX_TRIALS = 100
if torch.cuda.is_available():
    rng = torch.Generator("cuda")
else: 
    rng = torch.Generator("cpu")
hash_key = 15485863
PUNCTS = '!.?'
device = "cuda" if torch.cuda.is_available() else "cpu"


class SentenceEndCriteria(StoppingCriteria):
    """
    ONLY WORK WITH BATCH SIZE 1

    Stop generation whenever the generated string is **more than one** sentence (i.e. one full sentence + one extra token). this is determined by nltk sent_tokenize.
    Only stop if ALL sentences in the batch is at least two sentences

    Args:
        tokenizer (PreTrainedTokenizer):
            The exact tokenizer used for generation. MUST BE THE SAME!
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0

    def update(self, current_text):
        self.current_num_sentences = len(sent_tokenize(current_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert input_ids.size(0) == 1
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return len(sent_tokenize(text)) > self.current_num_sentences + 1


def discard_final_token_in_outputs(outputs):
    # (bz, seqlen)
    return outputs[:, :-1]


def extract_prompt_from_text(text, len_prompt):
    tokens = text.split(' ')
    tokens = tokens[:len_prompt]
    new_text = ' '.join(tokens)
    prompts = []
    for p in PUNCTS:
        idx = new_text.find(p)
        if idx != -1:
            tokens = new_text[:idx + 1].split(" ")
            # has to be greater than a minimum prompt
            if len(tokens) > 3:
                prompts.append(new_text[:idx + 1])
    if len(prompts) == 0:
        prompts.append(new_text + ".")
    # select first (sub)sentence, deliminated by any of PUNCTS
    prompt = list(sorted(prompts, key=lambda x: len(x)))[0]
    return prompt

def gen_sent(model, tokenizer, text_ids, gen_config, stopping_criteria):
    outputs = model.generate(
            # input_ids,
            text_ids,
            gen_config,
            stopping_criteria=stopping_criteria,
        )
    outputs = discard_final_token_in_outputs(outputs)
    new_text_ids = outputs
    new_text = tokenizer.decode(
        new_text_ids[0, text_ids.size(1):], skip_special_tokens=True)
    # print(new_text, new_text_ids)
    return new_text, new_text_ids

def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    if end_sent and len(sent) > 0 and sent[-1] not in PUNCTS:
        sent += "."
    return clean_text(sent)

def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)

def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]
