import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

from data_loader.dataset_wrapper import DS


class BERTBasedModel:
    def __init__(self, model_dir):
        self.model_dir = model_dir

    def predict(self, X):
        X = list(X)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        X_enc = tokenizer(
            X,
            padding=True,
            truncation=True,
            max_length=128,
        )
        X_ds = DS(X_enc)
        trained_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )
        trainer = Trainer(model=trained_model)
        predictions = trainer.predict(X_ds)
        y_pred = np.argmax(predictions.predictions, axis=1)
        return y_pred
