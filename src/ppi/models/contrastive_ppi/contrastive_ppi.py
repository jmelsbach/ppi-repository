from collections import OrderedDict
from typing import Any
import lightning.pytorch as pl
from torchmetrics.classification import Accuracy, BinaryPrecisionRecallCurve
from torchmetrics.classification import AUROC, ROC
from utils import get_optimizer, get_scheduler


from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
import torch
import torch.nn as nn
from torch.optim import AdamW


class ContrastivePPI(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 3e-4,
        base_model: str = "Rostlab/prot_bert",
        distance: str = "cosine",
        scheduler: str = "cycle",
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 0.00001,
        weight_decay: float = 0.01,
        optimizer: str = "adamw",
        pct_start: float = 0.3,
        num_cycles: float = 0.5,
        num_warmup_steps: int = 0,

    ):
        super().__init__()
        self.save_hyperparameters()

        self.acc = Accuracy(task="binary")
        self.auc = AUROC(task="binary")
        self.aupr = BinaryPrecisionRecallCurve(thresholds=None)

        self.temperature = nn.Parameter(torch.randn(1), requires_grad=True).to(
            self.device
        )

        if base_model == "t5-base":
            self.encoder = ContrastiveT5(base_model)
        elif base_model == "Rostlab/prot_bert":
            self.encoder = ContrastiveProtBert(base_model)
        elif base_model == "facebook/esm2_t6_8M_UR50D":
            self.encoder = ContrastiveESM(base_model)

    def training_step(self, batch, batch_idx):
        a, b = batch
        bs = a["input_ids"].size(0)
        embeddings_a = self.encoder(a)
        embeddings_b = self.encoder(b)
        distance_matrix = calculate_distance(
            embeddings_a, embeddings_b, "cosine"
        ) * torch.exp(self.temperature)
        loss = nn.CrossEntropyLoss()(
            distance_matrix, torch.arange(bs, dtype=torch.long).to(self.device)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a, b, labels = batch
        embeddings_a = self.encoder(a)
        embeddings_b = self.encoder(b)
        # inner_product = torch.sum(embeddings_a * embeddings_b, dim=1)
        cosine = nn.CosineSimilarity(dim=1)(embeddings_a, embeddings_b)
        preds = torch.sigmoid(cosine)
        acc = self.acc(preds, labels)
        auc = self.auc(preds, labels)
        aupr = self.aupr(preds, labels)
        self.log("val_acc", acc, prog_bar=True)
        self.log("Validation AUC", auc, prog_bar=True)
        # self.log("Validation AUPR", aupr, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            model=self,
            optimizer_name=self.hparams.optimizer,
            learning_rate=self.hparams.learning_rate,
            eps=self.hparams.eps,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = get_scheduler(
            scheduler_name=self.hparams.scheduler,
            optimizer=optimizer,
            max_learning_rate=self.hparams.learning_rate,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=self.hparams.num_cycles,
            pct_start=self.hparams.pct_start,
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "lr",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def test_step(self, batch, batch_idx):
        a, b, labels = batch
        bs = len(a)
        logits_a = self.encoder(a)
        logits_b = self.encoder(b)
        distance_matrix = calculate_distance(logits_a, logits_b, "cosine")
        targets = torch.arange(bs)
        loss = nn.CrossEntropyLoss(distance_matrix, targets)
        self.log("train_loss", loss)


class ContrastiveT5(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.config = AutoConfig.from_pretrained(base_model)
        self.head = ContrastiveHead(self.config.hidden_size, 1)

    def forward(self, x):
        input_ids = x["input_ids"].squeeze(1)
        attention_mask = x["attention_mask"].squeeze(1)
        model_output = self.model(input_ids, attention_mask)
        embedding = self._pool(model_output, attention_mask)
        return self.head(embedding)

    def encode(self, x: str):
        return self.model(**self.tokenizer(x, return_tensors="pt"))

    def _pool(self, model_output, attention_mask):
        # https://huggingface.co/efederici/sentence-bert-base
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class ContrastiveProtBert(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.config = AutoConfig.from_pretrained(base_model)
        # self.head = ContrastiveHead(self.config.hidden_size, 1)

        self.embedding_dim = self.config.hidden_size
        self.hidden_dim = 512
        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "fc1",
                        nn.Linear(
                            in_features=self.embedding_dim,
                            out_features=self.hidden_dim // 2,
                        ),
                    ),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(0.1)),
                    (
                        "fc2",
                        nn.Linear(
                            in_features=self.hidden_dim // 2,
                            out_features=self.hidden_dim // 4,
                        ),
                    ),
                ]
            )
        )
        # self._freeze_encoder()

    def forward(self, x):
        input_ids = x["input_ids"].squeeze(1)
        attention_mask = x["attention_mask"].squeeze(1)
        model_output = self.model(input_ids, attention_mask)
        return self.head(model_output.pooler_output)

    def _freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def _pool(self, model_output, attention_mask):
        # https://huggingface.co/efederici/sentence-bert-base
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


class ContrastiveESM(nn.Module):
    def __init__(self, base_model: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.config = AutoConfig.from_pretrained(base_model)
        self.head = SkipConnectionHead(self.config.hidden_size)
        self._freeze_encoder()

    def forward(self, x):
        input_ids = x["input_ids"].squeeze(1)
        attention_mask = x["attention_mask"].squeeze(1)
        model_output = self.model(input_ids, attention_mask)
        return self.head(model_output.last_hidden_state[:, 0, :])
    
    def _freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False 

    def encode(self, x: str):
        return self.model(**self.tokenizer(x, return_tensors="pt"))


class ContrastiveHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class SkipConnectionHead(nn.Module):
    # adapted from https://github.com/qdrant/quaterion-models/blob/master/quaterion_models/heads/skip_connection_head.py
    def __init__(self, transformer_output_dim, dropout=0.0, skip_dropout=0.0):
        super().__init__()
        self.gates = nn.Parameter(torch.Tensor(transformer_output_dim))
        self.reset_parameters()
        self.skip_dropout_prob = skip_dropout
        self.dropout = nn.Dropout(p=dropout)

        self.skip_dropout = (
            nn.Dropout(p=self.skip_dropout_prob)
            if self.skip_dropout_prob > 0.0
            else nn.Identity()
        )
        self.fc = nn.Linear(transformer_output_dim, transformer_output_dim)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(self.skip_dropout(x)) * torch.sigmoid(self.gates) + x

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(
            self.gates, -4.0
        )  # -4. ensures that all vector components are disabled by default


def calculate_distance(content_embeddings, label_embeddings, distance):
    assert distance in [
        "cosine",
        "inner_product",
    ], "--distance must be either 'euclidean' or 'cosine'"
    if distance == "inner_product":
        return content_embeddings @ label_embeddings.T
    elif distance == "cosine":
        return cosine_matrix(content_embeddings, label_embeddings)


def cosine_matrix_old(a, b, eps=1e-8):
    """
    > It takes two matrices, normalizes them, and returns the cosine similarity matrix

    Args:
      a: the first matrix
      b: batch size
      eps: a small value to avoid division by zero

    Returns:
      The cosine similarity matrix between the two sets of vectors.
    """

    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def cosine_matrix(a, b, eps=1e-8):
    a_n = a.norm(dim=1).clamp(min=eps)
    b_n = b.norm(dim=1).clamp(min=eps)
    a_norm = a / a_n[:, None]
    b_norm = b / b_n[:, None]
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt