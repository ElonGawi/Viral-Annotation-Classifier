from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

class EmbedderClassifier(object):
    def __init__(self):
        self._embedder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self._classification_head = LogisticRegression(max_iter=1000)

    def _embed(self, sentences):
        return self._embedder_model.encode(sentences) 
    
    def train(self, train_XY):
        embeddings = self._embed(train_XY["X"])
        self._classification_head.fit(embeddings, train_XY["y"])

    def predict(self, X):
        embeddings = self._embed(X)
        return self._classification_head.predict(embeddings)