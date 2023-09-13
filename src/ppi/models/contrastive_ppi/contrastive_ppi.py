from collections import OrderedDict
from typing import Any
import lightning.pytorch as pl
from torchmetrics import Accuracy
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
import torch
from torch.optim import AdamW


class ContrastivePPI(pl.LightningModule):
    def __init__(self, base_model: str = "Rostlab/prot_bert", distance: str = "cosine"):
        super().__init__()
        self.save_hyperparameters()

        self.acc = Accuracy(task="binary")


        self.temperature = nn.Parameter(torch.randn(1), requires_grad=True).to(self.device)

        if base_model == "t5-base":
            self.encoder = ContrastiveT5(base_model)
        elif base_model == "Rostlab/prot_bert":
            self.encoder = ContrastiveProtBert(base_model)
        elif base_model == "esm1_t6_43M_UR50S":
            self.encoder = ContrastiveESM(base_model)

    def training_step(self, batch, batch_idx):
        a, b = batch
        bs = a["input_ids"].size(0)
        embeddings_a = self.encoder(a)
        embeddings_b = self.encoder(b)
        distance_matrix = calculate_distance(embeddings_a, embeddings_b, "cosine") * torch.exp(self.temperature)
        loss = nn.CrossEntropyLoss()(
            distance_matrix, torch.arange(bs, dtype=torch.long).to(self.device)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        a, b, labels = batch
        embeddings_a = self.encoder(a)
        embeddings_b = self.encoder(b)
        inner_product = torch.sum(embeddings_a * embeddings_b, dim=1)
        preds = torch.sigmoid(inner_product)
        acc = self.acc(preds, labels)
        self.log("val_acc", acc, prog_bar=True)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

    def test_step(self, batch, batch_idx):
        a, b, labels = batch
        bs = len(a)
        logits_a = self.encoder(a)
        logits_b = self.encoder(b)
        distance_matrix = calculate_distance(logits_a, logits_b, "cosine")
        loss = nn.CrossEntropyLoss(distance_matrix, torch.arange(bs))
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
        #self.head = ContrastiveHead(self.config.hidden_size, 1)

        self.embedding_dim = self.config.hidden_size
        self.hidden_dim = 512
        self.head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dim//2)),
            ('relu', nn.ReLU()),
            ("dropout", nn.Dropout(0.1)),
            ('fc2', nn.Linear(in_features=self.hidden_dim//2, out_features=self.hidden_dim//4)),
        ]))
        #self._freeze_encoder()

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
        self.head = ContrastiveHead(self.config.hidden_size, 1)

    def forward(self, x):
        return self.head(self.model(**x))

    def encode(self, x: str):
        return self.model(**self.tokenizer(x, return_tensors="pt"))


class ContrastiveHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def calculate_distance(content_embeddings, label_embeddings, distance):
    assert distance in [
        "cosine",
        "inner_product",
    ], "--distance must be either 'euclidean' or 'cosine'"
    if distance == "inner_product":
        return content_embeddings @ label_embeddings.T
    elif distance == "cosine":
        return cosine_matrix(content_embeddings, label_embeddings)


def cosine_matrix(a, b, eps=1e-8):
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
