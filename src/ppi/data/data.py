from torch.utils.data import Dataset
from transformers import AutoTokenizer
import re


class PPIDataset(Dataset):
    def __init__(self, data, base_model, max_length=512, return_labels=True):
        self.data = data
        self.base_model = base_model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.max_length = max_length
        self.return_labels = return_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_a = self.data["sequenceA"].iloc[idx]
        sequence_b = self.data["sequenceB"].iloc[idx]
        label = self.data["class"].iloc[idx]

        if (
            "yarongef".lower() in self.base_model.lower()
            or "rostlab" in self.base_model.lower()
        ):
            # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
            # See https://huggingface.co/Rostlab/prot_bert#preprocessing
            sequence_a = " ".join(sequence_a)
            sequence_b = " ".join(sequence_b)
            sequence_a = re.sub(r"[UZOB]", "X", sequence_a)
            sequence_b = re.sub(r"[UZOB]", "X", sequence_b)

        inputs_a = self.tokenizer(
            sequence_a,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs_b = self.tokenizer(
            sequence_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.return_labels:
            return (inputs_a, inputs_b, label)
        else:
            return (inputs_a, inputs_b)
