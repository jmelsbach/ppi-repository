from lightning import LightningDataModule
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from ppi.data.data import PPIDataset
import pandas as pd
from pathlib import Path


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 256,
        num_workers: int = 8,
        base_model: str = "Rostlab/prot_bert",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = Path(data_path)
        self.base_model = base_model

    def setup(self, stage: str = None):
        train_df = pd.read_csv(self.data_path / "train.txt", delimiter="\t")
        val_df = pd.read_csv(self.data_path / "valid.txt", delimiter="\t")
        test_df = pd.read_csv(self.data_path / "test.txt", delimiter="\t")

        self.train_dataset = PPIDataset(
            train_df, self.base_model, max_length=512, return_labels=True
        )
        self.val_dataset = PPIDataset(
            val_df, self.base_model, max_length=512, return_labels=True
        )
        self.test_dataset = PPIDataset(
            test_df, self.base_model, max_length=512, return_labels=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class ContrastiveDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 8,
        base_model: str = "Rostlab/prot_bert",
        max_length: int = 1024,
    ):
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            base_model=base_model,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.max_length = max_length

    def setup(self, stage: str = None):
        train_df = pd.read_csv(self.data_path / "train.txt", delimiter="\t")
        val_df = pd.read_csv(self.data_path / "valid.txt", delimiter="\t")
        test_df = pd.read_csv(self.data_path / "test.txt", delimiter="\t")

        train_df = train_df[train_df["class"] == 1]

        self.train_dataset = PPIDataset(
            train_df, self.base_model, max_length=self.max_length, return_labels=False
        )
        self.val_dataset = PPIDataset(
            val_df, self.base_model, max_length=self.max_length, return_labels=True
        )
        self.test_dataset = PPIDataset(
            test_df, self.base_model, max_length=self.max_length, return_labels=True
        )
