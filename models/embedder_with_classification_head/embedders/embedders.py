from sentence_transformers import SentenceTransformer



class SentenceTransformerEmbedder():
    """
    Embedder taken from the SentenceTransformer in hugging face. 
    """
    def __init__(self, model_name):
        """_summary_

        Args:
            model_name (str): model name in hugging face.
        """
        self.model = SentenceTransformer(model_name)
    
    def train(self, train_x, train_y):
        pass

    def predict(self, X):
        return self.model.encode(X)

    