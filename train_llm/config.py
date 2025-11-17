from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "fine_tuned_BERT_models"
PMB_LINK = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
BERT_LINK = "google-bert/bert-base-uncased"