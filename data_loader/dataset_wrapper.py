# To wrap tokenized annotations and labels into a PyTorch-friendly dataset.

import torch


class DS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        """
        encodings: dict of input tensors (e.g., from a tokenizer)
        labels: optional list or array of labels (for training/validation)
        """
        self.encodings = encodings
        self.labels = labels  # can be None

    def __len__(self):
        return len(next(iter(self.encodings.values())))

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

        # Only include labels if available
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
