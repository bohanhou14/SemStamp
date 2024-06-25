import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer,
)
from transformers.utils.versions import require_version
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


def encode(attention_mask, model_output=None, token_embeddings=None):
    if model_output != None:
        embedding = mean_pooling(attention_mask, model_output=model_output)
    else:
        embedding = mean_pooling(
            attention_mask, token_embeddings=token_embeddings)
    # sentence_embeddings = F.normalize(embedding, p=2, dim=1)
    return embedding

def mean_pooling(attention_mask, model_output=None, token_embeddings=None):
    if (model_output != None and token_embeddings == None):
        token_embeddings = model_output[
            0
        ].to("cuda")  # First element of model_output contains all token embeddings
        # print(f"token_embeddings: {token_embeddings.size()}")
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def cosine_distance_matrix(x, y):
    return F.cosine_similarity(
        x.view(x.size(0), 1, x.size(1))
        .expand(x.size(0), y.size(0), x.size(1))
        .contiguous()
        .view(-1, x.size(1)),
        y.expand(x.size(0), y.size(0), y.size(1)).flatten(end_dim=1),
    ).view(x.size(0), y.size(0))

class ParaphraseContrastiveTrainer(Trainer):
    def __init__(self, delta, *args, **kwargs):
        self.inaccurate = 0
        self.total = 0
        self.delta = delta
        self.sim_loss = nn.MarginRankingLoss(
            margin=self.delta, reduction='sum')
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if return_outputs:
            raise NotImplementedError

        text_outputs = model(
            input_ids=inputs["text_input_ids"],
            attention_mask=inputs["text_attention_mask"],
        )
        text_embeddings = mean_pooling(
            model_output=text_outputs, attention_mask=inputs["text_attention_mask"])

        para_outputs = model(
            input_ids=inputs["para_input_ids"],
            attention_mask=inputs["para_attention_mask"],
        )
        para_embeddings = mean_pooling(
            model_output=para_outputs, attention_mask=inputs["para_attention_mask"])

        sims = cosine_distance_matrix(text_embeddings, para_embeddings)

        pos = sims.diag()
        # neg = off_diag(sims).max(dim=1).values
        n = sims.size(0)
        mask = torch.ones(n, n, device=sims.device) - \
            torch.eye(n, device=sims.device)
        masked_sims = sims * mask - torch.eye(n, device=sims.device)
        # change to dim = 0 and see if there's any difference
        neg = masked_sims.max(dim=1).values
        # use max cos sim in batch as negative example
        loss = self.sim_loss(pos, neg, torch.ones_like(pos))
        inaccurate = 0
        for i in range(n):
            if torch.argmax(sims[i]) != i:
                self.inaccurate += 1
        self.total += n
        print(f"grand contrastive acc: {1 - self.inaccurate / self.total}")
        # print("---------------------------------")

        return (loss, text_outputs) if return_outputs else loss
    
    # import this to make sure the trainer calls the right loss func during eval and prediction
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and ovferride to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        with torch.no_grad():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)
