from sklearn.neural_network import MLPClassifier
from sklearn.utils import compute_sample_weight


class NeuralNetClassifier():
    def __init__(self, train_with_sample_weights=False, **model_kwargs):
        self._train_with_sample_weights = train_with_sample_weights
        self.model = MLPClassifier(**model_kwargs)
        
    def train(self, train_x, train_y):
        if self._train_with_sample_weights:
            
            sample_weights = compute_sample_weight(
                class_weight='balanced',
                y=train_y
                )

            self.model.fit(train_x, train_y, sample_weight=sample_weights)
        else:
            self.model.fit(train_x, train_y)

    def predict(self, X):
        return self.model.predict(X)



class GenericSklearnCalssifier():
    def __init__(self, sklearn_model, model_kwargs={}):
        """_summary_

        Args:
            sklearn_model (_type_): instance of sklearn model, or a class name of the model to create
            model_kwargs (dict, optional): if type of class was passed on sklearn_model, create that model using these kwargs. Defaults to {}.
        """
        if isinstance(sklearn_model, type):
            self.model = sklearn_model(**model_kwargs)
        else:
            self.model = sklearn_model

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)
    
    def predict(self, X, probablities=False):
        """_summary_

        Args:
            X (_type_): _description_
            probablities (bool, optional): if True return the predicted probablities, . Defaults to False.

        Returns:
            _type_: _description_
        """
        if probablities:
            return self.model.predict_proba(X)
        return self.model.predict(X)