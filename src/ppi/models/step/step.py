from collections import OrderedDict

import lightning.pytorch as pl
import torch
import torchmetrics
from torch import nn, optim
from transformers import AutoModelForMaskedLM


class StepModel(pl.LightningModule):
    def __init__(
        self, embedding_dim=1024, base_model="Rostlab/prot_bert_bfd"
    ):
        super().__init__()
        
        self.valid_acc = torchmetrics.Accuracy(task="binary")
        self.valid_auroc = torchmetrics.AUROC(task="binary")
        
        self.loss_function = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.encoder = AutoModelForMaskedLM.from_pretrained(base_model)

        self.encoder_features = embedding_dim
       
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(0.5)),
                    (
                        "dense1",
                        nn.Linear(
                            self.encoder_features * 4,
                            int(self.encoder_features * 4 / 16),
                        ),
                    ),
                    ("dropout2", nn.Dropout(0.2)),
                    (
                        "dense2",
                        nn.Linear(
                            int(self.encoder_features * 4 / 16),
                            int(self.encoder_features * 4 / (16 * 16)),
                        ),
                    ),
                    ("dropout3", nn.Dropout(0.2)),
                    (
                        "dense3",
                        nn.Linear(int(self.encoder_features * 4 / (16 * 16)), 1),
                    ),
                ]
            )
        )


    def forward(self, tokenized_protein_sequences):
        attention_mask = tokenized_protein_sequences["attention_mask"].squeeze()
        input_ids = tokenized_protein_sequences["input_ids"].squeeze()
        protein_embedding = self.encoder(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1]
        pooled_embedding = self._pool(protein_embedding, protein_embedding[:,0], attention_mask)
        return pooled_embedding

    def training_step(self, batch, batch_idx):
        protein_sequence_1, protein_sequence_2, y = batch
        embedding_1 = self.forward(protein_sequence_1)
        embedding_2 = self.forward(protein_sequence_2)
        z1 = embedding_1 * embedding_2
        logits = self.head(z1)
        logits = logits.squeeze()
        loss = self.loss_function(logits, y.float())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        protein_sequence_1, protein_sequence_2, y = batch
        embedding_1 = self.forward(protein_sequence_1)
        embedding_2 = self.forward(protein_sequence_2)
        z1 = embedding_1 * embedding_2
        logits = self.head(z1)
        logits = logits.squeeze()
        loss = self.loss_function(logits, y.float())
        preds = self.sigmoid(logits)

        self.valid_acc(preds, y)
        self.valid_auroc(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.valid_acc.compute()
        auroc = self.valid_auroc.compute()
        self.log("valid_acc", acc, on_epoch=True, prog_bar=True)
        self.log("valid_auroc", auroc, on_epoch=True, prog_bar=True)
        self.valid_acc.reset()
        self.valid_auroc.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def _pool(
        self,
        token_embeddings,
        cls_token,
        attention_mask,
        pool_cls=True,
        pool_max=True,
        pool_mean=True,
        pool_mean_sqrt=True,
    ):
        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            token_embeddings[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector