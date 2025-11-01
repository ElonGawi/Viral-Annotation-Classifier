import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import compute_sample_weight
from ...config import AnnotationLabels

class NeuralNetClassifier():
    def __init__(self, activation='relu'):
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',        # use Adam optimizer
                    learning_rate_init=0.001,
                    max_iter=500,
                    random_state=42)
        
    def train(self, train_x, train_y, class_weights=True):
        if class_weights:
            classes = np.unique(AnnotationLabels.id2label.keys())
            
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=train_y
                )

            import pdb; pdb.set_trace()
            self.model.fit(train_x, train_y, sample_weights=sample_weights)
        else:
            self.model.fit(train_x, train_y)

    def predict(self, X):
        return self.model.predict(X) 