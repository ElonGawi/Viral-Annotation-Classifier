from sentence_transformers import SentenceTransformer

class AnnotationEmbedder():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def train(self, train_x, train_y):
        pass

    def predict(self, X):
        return self.model.encode(X)

    