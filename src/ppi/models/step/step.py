import lightning.pytorch as pl

from collections import OrderedDict
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import (
    AUROC,
    F1Score,
    ROC,
    Accuracy,
    AveragePrecision,
    MatthewsCorrCoef,
    Precision,
    PrecisionRecallCurve,
    Recall,
    ConfusionMatrix,
)
from torchmetrics.collections import MetricCollection
from torchnlp.encoders import LabelEncoder
from transformers import BertModel, BertTokenizer, BertConfig


class StepModel(pl.LightningModule):
    """
    based on https://github.com/SCAI-BIO/STEP/blob/main/src/modeling/ProtBertPPIModel.py
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.model_name = "Rostlab/prot_bert_bfd"

        # metrics initialization
        self.__build_metrics()
        self.current_val_epoch = 0
        self.current_test_epoch = 0

        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        # freeze encoder if user mants it
        if self.hparams.nr_frozen_epochs == -1:
            self.freeze_encoder()
            self._frozen = True
        else:
            self._frozen = False

    def __build_metrics(self) -> None:
        self.train_metrics = MetricCollection(
            [
                Accuracy(task="binary"),
                Precision(task="binary"),
                Recall(task="binary"),
                F1Score(task="binary"),
                AUROC(task="binary"),
                MatthewsCorrCoef(task="binary"),
            ],
            prefix="train_",
        )
        self.valid_metrics = MetricCollection(
            [
                Accuracy(task="binary"),
                Precision(task="binary"),
                Recall(task="binary"),
                F1Score(task="binary"),
                AUROC(task="binary"),
                MatthewsCorrCoef(task="binary"),
            ],
            prefix="val_",
        )
        self.test_metrics = self.valid_metrics.clone(prefix="test_")

    def __build_model(self) -> None:
        """Init BERT model + tokenizer + classification head."""
        config = BertConfig.from_pretrained(self.model_name)
        config.gradient_checkpointing = True
        self.model = BertModel.from_pretrained(self.model_name, config=config)
        self.encoder_features = 1024

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )

        # Label Encoder
        # TODO should be the label names of the dataset
        sample = ["0", "1"]
        self.label_encoder = LabelEncoder(sample, reserved_labels=[])
        self.label_encoder.unknown_index = None

        self.classification_head = nn.Sequential(
            OrderedDict(
                [
                    ("dropout1", nn.Dropout(self.hparams.dropout_prob)),
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

        self.sigmoid = nn.Sigmoid()

    def classifier(self, model_out_A, model_out_B):
        x = model_out_A * model_out_B
        result = self.classification_head(x)
        result = result.view(-1)
        return {"logits": result}

    def __build_loss(self) -> None:
        self._loss_bce_with_integrated_sigmoid = nn.BCEWithLogitsLoss()

    def unfreeze_encoder(self) -> None:
        if self._frozen:
            for param in self.model.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self._frozen = True

    def pool_strategy(
        self,
        features,
        pool_cls=True,
        pool_max=True,
        pool_mean=True,
        pool_mean_sqrt=True,
    ):
        token_embeddings = features["token_embeddings"]
        cls_token = features["cls_token_embeddings"]
        attention_mask = features["attention_mask"]
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

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = (
                    features["token_weights_sum"]
                    .unsqueeze(-1)
                    .expand(sum_embeddings.size())
                )
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def forward(self, input_ids, token_type_ids, attention_mask):
        word_embeddings = self.model(input_ids.squeeze(), attention_mask.squeeze())[0]
        pooling = self.pool_strategy(
            {
                "token_embeddings": word_embeddings,
                "cls_token_embeddings": word_embeddings[:, 0],
                "attention_mask": attention_mask.squeeze(),
            }
        )

        return pooling

    def loss_bce_with_integrated_sigmoid(
        self, predictions: dict, targets: dict
    ) -> torch.Tensor:
        return self._loss_bce_with_integrated_sigmoid(
            predictions["logits"], targets.float()
        )

    def __single_step(self, batch):
        inputs_A, inputs_B, targets = batch
        model_out_A = self.forward(**inputs_A)
        model_out_B = self.forward(**inputs_B)
        classifier_output = self.classifier(model_out_A, model_out_B)
        loss = self.loss_bce_with_integrated_sigmoid(classifier_output, targets)

        trues = targets
        preds = classifier_output["logits"]
        preds = self.sigmoid(preds)

        return (loss, trues, preds)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> dict:
        train_loss, trues, preds = self.__single_step(batch)
        # self.train_metrics.update(preds, trues)

        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        output = OrderedDict(
            {
                "loss": train_loss,
            }
        )
        return train_loss

    def on_train_epoch_end(self) -> None:
        # result = self.train_metrics.compute()
        # self.train_metrics.reset()

        # result.pop('train_ROC', None)
        # result.pop('train_PrecisionRecallCurve', None)
        # self.log_dict(result, on_epoch=True)

        # check for unfreezing encoder
        if self.current_epoch + 1 >= self.hparams.nr_frozen_epochs:
            self.unfreeze_encoder()

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        val_loss, trues, preds = self.__single_step(batch)

        self.valid_metrics.update(preds, trues)
        output = OrderedDict(
            {
                "val_loss": val_loss,
            }
        )
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def on_validation_epoch_end(self) -> None:
        self.current_val_epoch += 1
        result = self.valid_metrics.compute()
        self.valid_metrics.reset()
        self.log_dict(result, on_epoch=True)

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        test_loss, trues, preds = self.__single_step(batch)

        # self.test_metrics.update(preds, trues)

        output = OrderedDict(
            {
                "test_loss": test_loss,
            }
        )

        return output

    @property
    def num_training_steps(self) -> int:
        """
        Total training steps inferred from datamodule and devices.

        https://github.com/PyTorchLightning/pytorch-lightning/issues/5449#issuecomment-774265729
        """
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def lr_lambda(self, current_step: int) -> float:
        num_warmup_steps = self.hparams.warmup_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - num_warmup_steps)),
        )

    def configure_optimizers(self):
        no_decay_params = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if not any(ndp in name for ndp in no_decay_params)
                ],
                "lr": self.hparams.encoder_learning_rate,
            },
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if any(ndp in name for ndp in no_decay_params)
                ],
                "weight_decay": 0.0,
                "lr": self.hparams.encoder_learning_rate,
            },
            {
                "params": self.classification_head.parameters(),
            },
        ]

        parameters = optimizer_grouped_parameters
        optimizer = optim.AdamW(
            parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_epsilon,
            # betas = self.hparams.betas
        )

        scheduler = LambdaLR(optimizer, self.lr_lambda)
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "reduce_on_plateau": False,
            "monitor": "val_loss",
            "name": "learning_rate",
        }

        return [optimizer], [scheduler_dict]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument(
            "--adam_epsilon", default=1e-08, type=float, help="adam_epsilon"
        )
        parser.add_argument(
            "--nb_trials", default=24, type=int, help="Number of trials to run"
        )
        parser.add_argument(
            "--per_device_train_batch_size",
            default=8,
            type=int,
            help="Batch size to be used for training data.",
        )
        parser.add_argument(
            "--per_device_eval_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for validation data.",
        )
        parser.add_argument(
            "--per_device_test_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for test data.",
        )
        parser.add_argument(
            "--per_device_predict_batch_size",
            default=8,
            type=int,
            help="The batch size per GPU/TPU core/CPU for test data.",
        )
        parser.add_argument(
            "--max_length",
            default=1536,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=5e-06,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--weight_decay",
            default=1e-2,
            type=float,
            help="Weight decay for AdamW.",
        )
        parser.add_argument(
            "--warmup_steps",
            default=200,
            type=int,
            help="Warm up steps for learning rate schedular.",
        )
        parser.add_argument(
            "--dropout_prob",
            default=0.5,
            type=float,
            help="Classification head dropout probability.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=1,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        parser.add_argument(
            "--gradient_checkpointing",
            default=True,
            type=bool,
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )

        return parser
