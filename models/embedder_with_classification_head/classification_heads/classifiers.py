import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import compute_sample_weight
from ...config import AnnotationLabels


class NeuralNetClassifier():
    def __init__(self, train_with_sample_weights=False, **model_kwargs):
        self._train_with_sample_weights = train_with_sample_weights
        self.model = MLPClassifier(**model_kwargs)
        
    def train(self, train_x, train_y):
        if self._train_with_sample_weights:
            classes = np.unique(AnnotationLabels.id2label.keys())
            
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=train_y
                )

            self.model.fit(train_x, train_y, sample_weight=sample_weights)
        else:
            self.model.fit(train_x, train_y)

    def predict(self, X):
        return self.model.predict(X) 
    