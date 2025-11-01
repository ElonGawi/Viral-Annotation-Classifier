from sentence_transformers import SentenceTransformer



class AnnotationEmbedder():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def train(self, train_x, train_y):
        pass

    def predict(self, X):
        return self.model.encode(X)

    