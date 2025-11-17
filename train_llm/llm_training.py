# python -m train_llm.llm_training --model pmb --run-name PMB_model

import numpy as np
from pathlib import Path
import argparse

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
    EarlyStoppingCallback
)

from data_loader.annotation_data_loader import AnnotationDataLoader
from data_loader.dataset_wrapper import DS
from models.config import AnnotationLabels
from train_llm.config import PMB_LINK, BERT_LINK, MODELS_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train sequence classification model on annotation data."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["pmb", "bert", "biolb"],
        default="bert",
        help="Which base model to use: 'pmb', 'bert', or 'biolb' (default: bert)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="BERT_model",
        help="Name for this training run (used to name the output directory).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Per-device train batch size (default: 16)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Per-device eval batch size (default: 32)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (default: 0.0, i.e. none)",
    )
    return parser.parse_args()


MODEL_LINKS = {
    "pmb": PMB_LINK,
    "bert": BERT_LINK,
}

def main():
    args = parse_args()

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))


    # Define model to use and output directory
    model_link = MODEL_LINKS[args.model]
    outdir_name = args.run_name
    outdir_path = MODELS_DIR / outdir_name
    checkpoint_dir = MODELS_DIR / f"{outdir_name}_checkpoints"

    print(f"Model chosen: {args.model} -> {model_link}")
    print(f"The model will be saved to {outdir_path}")

    # Load data
    dataloader = AnnotationDataLoader()
    train_df = dataloader.get_train()
    val_df = dataloader.get_validation()

    # Label mapping
    label2id = AnnotationLabels.label2id
    id2label = AnnotationLabels.id2label

    # Encode training and validation sets

    tokenizer = AutoTokenizer.from_pretrained(model_link)

    train_enc = tokenizer(
        train_df["protein_annotation"].to_list(),
        padding=False,
        truncation=True,
        max_length=64,
    )

    val_enc = tokenizer(
        val_df["protein_annotation"].to_list(),
        padding=False,
        truncation=True,
        max_length=64,
    )

    # Wrap encoded annotations and labels in PyTorch friendly wrapper
    train_ds = DS(train_enc, train_df["label"].tolist())
    val_ds = DS(val_enc, val_df["label"].tolist())

    # Class weights to manage imbalance
    classes = np.array(sorted(train_df["label"].unique()))
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_df["label"].values
    )
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_link, num_labels=3, id2label=id2label, label2id=label2id
    )



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

    # Training arguments

    model_args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=50,
        seed=42,
        remove_unused_columns=True,
)

    trainer = WeightedTrainer(
        model=base_model,
        args=model_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    save(trainer, tokenizer, outdir_path)


if __name__ == "__main__":
    main()
