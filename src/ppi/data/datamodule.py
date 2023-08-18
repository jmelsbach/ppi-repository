from lightning import LightningDataModule
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from ppi.data.data import PPIDataset
import pandas as pd
from pathlib import Path


class ContrastiveDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 8,
        base_model: str = "Rostlab/prot_bert",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.data_path = Path(data_path)

    def setup(self, stage: str = None):
        train_df = pd.read_csv(self.data_path / "train.txt", delimiter="\t")
        val_df = pd.read_csv(self.data_path / "valid.txt", delimiter="\t")
        test_df = pd.read_csv(self.data_path / "test.txt", delimiter="\t")

        train_df = train_df[train_df["class"] == 1]

        self.train_dataset = PPIDataset(
            train_df, self.tokenizer, max_length=512, return_labels=False
        )
        self.val_dataset = PPIDataset(
            val_df, self.tokenizer, max_length=512, return_labels=True
        )
        self.test_dataset = PPIDataset(
            test_df, self.tokenizer, max_length=512, return_labels=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
