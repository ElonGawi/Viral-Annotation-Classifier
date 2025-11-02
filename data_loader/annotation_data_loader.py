import os
import pandas as pd
from .config import DataLoaderConfig


class AnnotationDataLoader(object):
    """Loading the pre-splited train, validation and test"""

    def __init__(self, config=DataLoaderConfig):
        self._config = config

    def get_train(self):
        """
        Rerturn the pre-splited training data
        """
        train_path = os.path.join(self._config.output_dir, self._config.train_filename)
        return pd.read_csv(train_path, sep="\t")

    def get_validation(self):
        """
        Rerturn the pre-splited val data
        """
        val_path = os.path.join(self._config.output_dir, self._config.val_filename)
        return pd.read_csv(val_path, sep="\t")

    def get_test(self):
        """
        Rerturn the pre-splited test data
        """

        test_path = os.path.join(self._config.output_dir, self._config.test_filename)
        return pd.read_csv(test_path, sep="\t")
