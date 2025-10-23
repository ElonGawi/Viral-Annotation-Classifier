class AnnotationLabels(object):
    label_names = ["Uninformative", "Low", "Proper"]  # Expected label names
    label2id = {label: id for id, label in enumerate(label_names)}
    id2label = {id: label for label, id in label2id.items()}
