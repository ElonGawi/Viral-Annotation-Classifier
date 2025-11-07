import os
import pandas as pd
import numpy as np
from .config import DataLoaderConfig


class AnnotationDataLoader(object):
    """Loading the pre-splited train, validation and test"""

    def __init__(self, config=DataLoaderConfig):
        self._config = config

    def get_train(self, resample=False, class_ratio=None):
        """
        Rerturn the pre-splited training data
        class ratio: e.g. {0: 0.3, 
                            1: 0.3, 
                            2: 1}
        """
        train_path = os.path.join(self._config.output_dir, self._config.train_filename)
        train_df =  pd.read_csv(train_path, sep="\t")
        
        if resample:            
            from imblearn.over_sampling import RandomOverSampler
            from collections import Counter
            if class_ratio:
                classes_ref = [k for k, v in class_ratio.items() if v ==1] 

                if len(classes_ref) != 1:
                    raise Exception("ERROR classes ratio must have at ONE class with the value 1 so the rest are relatiev to it")
                    
                classes_ref = classes_ref[0]
                len_of_ref = (train_df["label"] == classes_ref).sum()

                target_counts = {}
                for k, v in class_ratio.items():
                    target_counts[k] = int(len_of_ref * v)
            
                ros = RandomOverSampler(sampling_strategy = target_counts)
            else:
                ros = RandomOverSampler()

            X_train_resampled, y_train_resampled = ros.fit_resample(np.array(train_df["protein_annotation"]).reshape(-1, 1), train_df["label"])
            train_df = pd.DataFrame({"protein_annotation" :   X_train_resampled.flatten(), "label": y_train_resampled})
            print(f"Original Training Class Distribution: {Counter(y_train_resampled)}")

        return train_df
    
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
