import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data_loader.dataset_wrapper import DS

class BERTBasedModel:
    def __init__(self, model_dir, subfolder=None, device=None):
        self.model_dir = model_dir
        if subfolder is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                subfolder=subfolder
                )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                subfolder=subfolder
                )

        # Choose device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, X, batch_size=32):
        X = list(X)
        encodings = self.tokenizer(
            X,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        ds = DS(encodings)
        dl = DataLoader(ds, batch_size=batch_size)

        all_logits = []

        with torch.inference_mode():
            for batch in dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                all_logits.append(outputs.logits.cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_pred = logits.argmax(axis=1)
        return y_pred