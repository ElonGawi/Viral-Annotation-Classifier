import os
import pandas as pd
from math import isclose
from sklearn.model_selection import train_test_split


# Data directory and name of the file containing the data
data_dir = "data"
data_filename = "AF50m_subset_REGEX_man_labels_5k.txt"

# Load data
data_path = os.path.join(data_dir, data_filename)
data = pd.read_csv(data_path, sep="\t")

# Extract only labelled data, and remove note column
labelled_data = data[data["manual_label"].notna()].iloc[:, :-1].copy()

# Ensure all labels are correct
label_names = ["uninformative", "low", "proper"]  # Expected label names
labelled_data["manual_label"] = labelled_data["manual_label"].str.strip().str.lower()
assert set(labelled_data["manual_label"].unique()) <= set(
    label_names
), "Unexpected label present."

# Map string labels to integers
label2id = {label: id for id, label in enumerate(label_names)}
id2label = {id: label for label, id in label2id.items()}
labelled_data["label_id"] = labelled_data["manual_label"].map(label2id)

# Split the data 3-way
train_df, val_test_df = train_test_split(
    labelled_data,
    test_size=0.3,
    random_state=42,
    stratify=labelled_data["label_id"],
)

val_df, test_df = train_test_split(
    val_test_df,
    test_size=1 / 3,
    random_state=42,
    stratify=val_test_df["label_id"],
)

print(f"Training: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test: {test_df.shape}")

### Save to files

# Define file paths
train_path = os.path.join(data_dir, "train_split.tsv")
val_path = os.path.join(data_dir, "val_split.tsv")
test_path = os.path.join(data_dir, "test_split.tsv")

# Save DataFrames to tab-separated files
train_df.to_csv(train_path, sep="\t", index=False)
val_df.to_csv(val_path, sep="\t", index=False)
test_df.to_csv(test_path, sep="\t", index=False)

print(f"Saved train/val/test files to: {data_dir}")
