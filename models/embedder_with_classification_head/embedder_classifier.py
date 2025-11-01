from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from .classification_heads.classifiers import NeuralNetClassifier
from .embedders.embedders import AnnotationEmbedder

from sklearn.neural_network import MLPClassifier

class EmbedderClassifier(object):
    def __init__(self):
        self._embedder_model = AnnotationEmbedder()
        # self._classification_head = LogisticRegression(max_iter=1000)
        self._classification_head =  NeuralNetClassifier()

        
    def _embed(self, sentences):
        return self._embedder_model.predict(sentences) 
    
    def train(self, train_XY):
        # train embedder
        train_x = train_XY["protein_annotation"]
        train_y = train_XY["label"]
        self._embedder_model.train(train_x, train_y)

        # train classifier 
        embeddings = self._embed(train_x)
        self._classification_head.train(embeddings, train_y)

    def predict(self, X):
        embeddings = self._embed(X)
        return self._classification_head.predict(embeddings)