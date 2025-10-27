import re
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB

class TFIDFClassifier:
    def __init__(self):
        self._lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
        self._clf =  RandomForestClassifier(bootstrap=True, class_weight='balanced', max_depth=10, max_features='sqrt', min_samples_leaf=2, min_samples_split=11, n_estimators=910, random_state=42, n_jobs=-1)
        self._vectorizer = TfidfVectorizer(lowercase=False, stop_words=list(ENGLISH_STOP_WORDS), ngram_range=(1,2), max_df=0.9)


    def predict(self, X):
        train = X.apply(self.clean_lemmatize)
        vectors = self._vectorizer.transform(train)
        return self._clf.predict(vectors)
    

    def fit(self, train_df):
        train = train_df["protein_annotation"].apply(self.clean_lemmatize)
        labels = train_df["label"]

        vectors = self._vectorizer.fit_transform(train)
        self._clf.fit(vectors, labels)


    def clean_lemmatize(self, text):
        text = text.lower()  # Lowercase
        text = re.sub(r"[\[\]\(\)]", "", text)  # removing brackets etc
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)  # remove non-alphanumeric characters
        text = re.sub(r"\s+", " ", text)  # remove multiple spaces

        # Remove words shorter than 2 characters
        text = re.sub(r"\b\w{1}\b", "", text)  # Removes isolated 1-character tokens
        text = re.sub(r"\s+", " ", text)  # cleans up extra spaces again

        doc = self._lemmatizer(text)
        lemmas = []
        for tok in doc:
            if tok.is_space:
                    continue
            t = tok.text
            if tok.is_alpha:
                lemmas.append(tok.lemma_.lower()) #eg binding -> bind
            elif re.compile(r'(?=.*[a-zA-Z])(?=.*\d)').search(t):
                # Keep alphanumeric tokens like asp45 or hsp70
                lemmas.append(t.lower())
            # else: skip pure numbers and punctuation (shouldn’t occur post-cleaning)

        return " ".join(lemmas)

    def save_to_file(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_from_file(path):
        with open(path, "rb") as f:
            return pickle.load(f)