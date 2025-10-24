# To run this file (while in Viral-Annotation-Classifier):
# python -m train_llm.llm_training

### Imports

import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from data_loader.annotation_data_loader import AnnotationDataLoader
from data_loader.dataset_wrapper import DS
from models.config import AnnotationLabels

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

### Define model to use and output directory

# model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# model_output_dir = "PMB_model"
# checkpoint_dir = "PMB_model_checkpoints"

# model_name = "google-bert/bert-base-uncased"
# model_output_dir = "BERT_model"
# checkpoint_dir = "BERT_model_checkpoints"

model_name = "google-bert/bert-base-uncased"
model_output_dir = "test_model"
checkpoint_dir = "test_model_checkpoints"


print(f"Model chosen: {model_name}")
print(f"The model will be saved to {model_output_dir}")

### Load data

dataloader = AnnotationDataLoader()
train_df = dataloader.get_train()
val_df = dataloader.get_validation()

print("Data is loaded")

### Label mapping (from string to int and vice-versa)

label2id = AnnotationLabels.label2id
id2label = AnnotationLabels.id2label

### Tokenize training and validation sets

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_enc = tokenizer(
    train_df["protein_annotation"].to_list(),
    padding=True,
    truncation=True,
    max_length=128,
)

val_enc = tokenizer(
    val_df["protein_annotation"].to_list(),
    padding=True,
    truncation=True,
    max_length=128,
)

print("Encoding done")

# Wrap tokenised annotations and labels in PyTorch friendly wrapper
train_ds = DS(train_enc, train_df["label"].tolist())
val_ds = DS(val_enc, val_df["label"].tolist())

### Class weights to manage imbalance

classes = np.array(sorted(train_df["label"].unique()))
weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=train_df["label"].values
)
class_weights = torch.tensor(weights, dtype=torch.float)

# Attach classification head on top of the pretrained encoder
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3, id2label=id2label, label2id=label2id
)


# Custom subclass of the Hugging Face Trainer that overrides the compute_loss method.
# We override because we want to add class weights
class WeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_uninformative": f1_score(labels, preds, average=None, labels=[0])[0],
        "f1_low": f1_score(labels, preds, average=None, labels=[1])[0],
        "f1_proper": f1_score(labels, preds, average=None, labels=[2])[0],
    }


def save(trainer, tokenizer, save_dir):
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model and Tokenizer saved to {save_dir}")


# args = TrainingArguments(
#     output_dir=checkpoint_dir,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     metric_for_best_model="macro_f1",
#     greater_is_better=True,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=32,
#     learning_rate=2e-5,
#     num_train_epochs=1,
#     weight_decay=0.01,
#     warmup_ratio=0.1,
#     logging_steps=50,
#     seed=42,
#     remove_unused_columns=True,
# )

# Use these arguments to have a very quick training and just check that the script runs fine

args = TrainingArguments(
    output_dir=checkpoint_dir,
    eval_strategy="no",  # skip evaluation for speed
    save_strategy="no",  # don't save checkpoints
    per_device_train_batch_size=2,  # smaller batch size
    per_device_eval_batch_size=2,
    learning_rate=5e-5,  # slightly higher LR for faster convergence
    num_train_epochs=0.05,  # fraction of an epoch (if supported) or 1 if not
    max_steps=20,  # stop after 20 steps regardless of dataset size
    weight_decay=0.0,
    warmup_ratio=0.0,
    logging_steps=50,
    seed=42,
    remove_unused_columns=True,
)

trainer = WeightedTrainer(
    model=base_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

save(trainer, tokenizer, model_output_dir)
