import pickle
from .classification_heads.classifiers import GenericSklearnCalssifier
from .embedders.embedders import SentenceTransformerEmbedder
import os

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
    

    def load_model(path="./models/embedder_with_classification_head/trained_model/pretrained_embedder_dict.pkl"):
        with open(path, "rb") as f:
            embdder_classidier_dict = pickle.load(f)

        classifier = GenericSklearnCalssifier(embdder_classidier_dict["classification_head"])
        embedder = SentenceTransformerEmbedder(embdder_classidier_dict["embedder_name"])

        return EmbedderClassifier(embedder, classifier)
    

    def save_to_file(self, path):
        pretrained_embedder = {}
        pretrained_embedder["classification_head"] = self._classification_head.model 
        pretrained_embedder["embedder_name"] = r"pritamdeka/S-BioBert-snli-multinli-stsb"
            
        with open(path, "wb") as f:
            pickle.dump(pretrained_embedder, f)

    










