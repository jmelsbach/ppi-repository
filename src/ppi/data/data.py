from torch.utils.data import Dataset


class PPIDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, return_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_labels = return_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_a = self.data["sequenceA"].iloc[idx]
        sequence_b = self.data["sequenceB"].iloc[idx]
        label = self.data["class"].iloc[idx]

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
