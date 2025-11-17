from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "fine_tuned_BERT_models"

PMB_DIR = MODELS_DIR / "PMB_model"
BERT_DIR = MODELS_DIR / "BERT_model"

class AnnotationLabels(object):
        label_names = ["Uninformative", "Low", "Proper"]  # Expected label names
        label2id = {label: id for id, label in enumerate(label_names)}
        id2label = {id: label for label, id in label2id.items()}   
