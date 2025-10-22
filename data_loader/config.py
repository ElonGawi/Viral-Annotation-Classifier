import os
from pathlib import Path

module_dir = Path(__file__).parent

class DataLoaderConfig(object):
    output_dir_relative = r"data"
    # make path absolute 
    output_dir = os.path.join(module_dir, output_dir_relative)

    train_filename = r"train_split.tsv"
    val_filename = r"val_split.tsv"
    test_filename = r"test_split.csv"


