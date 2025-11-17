class EmbedderClassifier(object):
    def __init__(self, embedder, classifider):
        self._embedder_model = embedder
        self._classification_head =  classifider

        
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

    def predict(self, X, probablities=False):
        embeddings = self._embed(X)
        return self._classification_head.predict(embeddings, probablities=probablities)
    










