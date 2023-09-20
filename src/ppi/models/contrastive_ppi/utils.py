import torch
import torch.nn.functional as F
from rich.progress import Progress
from torch.optim import Adam, AdamW, SGD
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def _get_optimizer_params(model, lr, weight_decay=0.0):
    # param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


def get_optimizer(
    model,
    optimizer_name: str,
    learning_rate: float,
    eps: float,
    betas: list,
    weight_decay: float,
    **kwargs,
):
    parameter_groups = _get_optimizer_params(model, learning_rate)

    if optimizer_name.lower() == "adam":
        optimizer = Adam(
            parameter_groups,
            lr=learning_rate,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(
            parameter_groups,
            lr=learning_rate,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
        )

    if optimizer_name.lower() == "sgd":
        optimizer = SGD(
            parameter_groups,
            lr=learning_rate,
            momentum=betas[0],
            weight_decay=weight_decay,
        )

    return optimizer


def get_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    max_learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    pct_start: float,
    **kwargs,
):
    if scheduler_name.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    if scheduler_name.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    elif scheduler_name.lower() == "cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
        )

    return scheduler