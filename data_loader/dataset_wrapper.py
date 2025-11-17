# To wrap tokenized annotations and labels into a PyTorch-friendly dataset.

import torch


class DS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        """
        encodings: BatchEncoding
        labels: optional list of labels (for training/validation)
        """
        self.encodings = encodings
        self.labels = labels  # can be None

    def __len__(self):
        return len(next(iter(self.encodings.values())))

    def __getitem__(self, idx):
        # item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}

        # # Only include labels if available (e.g. for training)
        # if self.labels is not None:
        #     item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        # return item
    
        item = {key: torch.as_tensor(value[idx]) for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item