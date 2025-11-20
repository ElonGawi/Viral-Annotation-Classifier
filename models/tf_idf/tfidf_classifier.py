import os
import re
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Union
import spacy
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB


class TFIDFClassifier:


    """
    Text classification model using TF-IDF features and a scikit-learn classifier.

    This class was developed to classify protein annotations into
    the categories "uninformative" = 0, "low" = 1, and "proper" = 2. 
    This is done through lemmatization using spaCy, TF-IDF vectorization,
    and classification with a scikit-learn classifier. The class provides
    methods for fitting a vectorizer and classifier on a labeled dataset,
    predicting the classes of a dataset of protein annotation, as well as
    saving and loading trained models.
    """


    def __init__(
            self,
            spacy_model: str = "en_core_web_sm",
            classifier: Optional[BaseEstimator] = None,
            vectorizer: Optional[TfidfVectorizer] = None,
            is_fitted: bool = False
        ) -> None:

        """
        Initializes a TFIDFClassifier instance. 

        Args:
            spacy_model (str, optional): 
                The name of the spaCy language model to load for lemmatization.
                Defaults to "en_core_web_sm".
            classifier (Optional[BaseEstimator], optional): 
                A scikit-learn classifier implementing `fit` and `predict`. 
                Defaults to `LogisticRegression(solver="saga", 
                                                max_iter=1000,
                                                penalty='l2', 
                                                C=17.0798, 
                                                class_weight=None)`. 
            vectorizer (Optional[TfidfVectorizer], optional):
                A TF-IDF vectorizer.
                Defaults to `TfidfVectorizer(lowercase=False,
                                            stop_words=list(ENGLISH_STOP_WORDS), 
                                            ngram_range=(1,2), 
                                            max_df=0.9)`.
            is_fitted (bool):
                Defines if the passed vectorizer and classifier are pre-fit. 
                Defaults to False. 

        Raises:
            OSError: 
                If the passed spaCy language model `spacy_model` cannot be loaded
            TypeError: 
                If the passed classifier `classifier` does not have the method `fit` nor `predict`
            TypeError: 
                If the passed TF-IDF vectorizer `vectorizer` does not have the method
                `fit_transform` nor `transform`
        """

        # Loading spaCy model
        try:
            self._lemmatizer = spacy.load(spacy_model, disable=["parser", "ner", "textcat"])
        except OSError as e:
            raise OSError(
                f"Could not load spaCy model '{spacy_model}'."
                "Make sure it is installed (e.g. `python -m spacy download en_core_web_sm`)."
            ) from e

        # Validate classifier
        if classifier is None:
            classifier =  LogisticRegression(solver="saga",
                                             max_iter=1000,
                                             penalty='l2',
                                             C=17.0798,
                                             class_weight=None)
        else:
            for required_method in ("fit", "predict"):
                if not hasattr(classifier, required_method):
                    raise TypeError(
                        f"Classifier must implement a '{required_method}' method, "
                        f"but {classifier.__class__.__name__} does not."
                    )

        self._clf = classifier

        # Validating TF-IDF vectorizer
        if vectorizer is None:
            vectorizer = TfidfVectorizer(
                lowercase=False,
                stop_words=list(ENGLISH_STOP_WORDS),
                ngram_range=(1,2),
                max_df=0.9
                )
        else:
            for required_method in ("fit_transform", "transform"):
                if not hasattr(vectorizer, required_method):
                    raise TypeError(
                        f"Vectorizer must implement '{required_method}' method, "
                        f"but {vectorizer.__class__.__name__} does not."
                    )

        self._vectorizer = vectorizer

        self._is_fitted = is_fitted

    def fit(self, train_df: pd.DataFrame) -> None:

        """
        Fits the TF-IDF vectorizer and classifier on a training DataFrame

        Args:
            train_df (pd.DataFrame): 
                Training data with two required columns:
                -"protein_annotation": text data to be used as input.
                -"label": target labels for the classifier. 

        Raises:
            TypeError: 
                If the passed `train_df` is not a pandas DataFrame
            KeyError: 
                If the passed `train_df` is lacking a required column 
                ("protein_annotation" or "label")
            ValueError: 
                If the passed `train_df` is empty
            ValueError: 
                If the column "label" in the passed `train_df` has missing values
            type:
                TypeError or ValueError
                If the column "protein_annotation" in the passed `train_df`
                contains entries which are invalid annotations 

        """

        # Checking that training data is a pandas DataFrame
        if not isinstance(train_df, pd.DataFrame):
            raise TypeError(
                f"`train_df` must be a pandas DataFrame, got {type(train_df)}."
            )

        # Checking that the required columns are present
        required_cols = {"protein_annotation", "label"}
        missing = required_cols - set(train_df.columns)
        if missing:
            raise KeyError(
                f"`train_df` is missing required columns: {', '.join(missing)}."
            )

        # Checking that the given DataFrame is not empty
        if train_df.empty:
            raise ValueError("`train_df` is empty, cannot fit classifier.")

        labels = train_df["label"]

        # Checking all rows have a label
        if labels.isnull().any():
            raise ValueError("`label` column contains missing values.")

        texts = train_df["protein_annotation"]

        # Applying cleaning and lemmatization function
        cleaned_texts: List[str] = []
        for i, t in texts.items():
            try:
                cleaned_texts.append(self._clean_lemmatize(t))
            except (TypeError, ValueError) as e:
                raise type(e)(f"Invalid `protein_annotation` at index {i}: {e}") from e

        # Fitting the TF-IDF vectorizer
        vectors = self._vectorizer.fit_transform(cleaned_texts)

        # Fitting the classifier
        self._clf.fit(vectors, labels)

        self._is_fitted = True

    def predict(self, annotations: Union[Sequence[str], pd.Series], probabilities: bool=False) -> np.ndarray:

        """
        Predicts labels for a sequence of input texts.

        Args:
            annotations (Union[Sequence[str], pd.Series]): 
                A sequence (list, tuple, or pandas Series) of text strings to classify.
            probabilities (bool):
                Flag insicating if the method passes predicted labels or predicted 
                probabilities. Defaults to False. 

        Raises:
            NotFittedError: 
                If the TF-IDF vectorizer and classifier of this instance have not 
                been fit to training data yet.
            type: 
                TypeError or ValueError
                If the passed sequence contains entries that are not valid annotations
            ValueError: 
                If the number of predictors generated are not equal to the number
                of annotations passed, or the predicted probabilities do not have
                three columns and the same length as number of annotations passed

        Returns:
            np.ndarray: 
                Numpy ndarray of predicted labels or of predicted probabilities
        """

        if not self._is_fitted:
            raise NotFittedError(
                "This TFIDFClassifier instance has not been fitted. "
                "Call `fit` with appropriate training data before using `predict`, "
                "or load a pre-trained TFIDFClassifier using `load_from_file`."
            )

        # Validate that X is a proper sequence
        raw_texts = self._validate_text_sequence(annotations)

        # Clean and lemmatize each entry, with index-aware error messages
        cleaned: List[str] = []
        for i, t in enumerate(raw_texts):
            try:
                cleaned.append(self._clean_lemmatize(t))
            except (TypeError, ValueError) as e:
                raise type(e)(f"Invalid text at index {i}: {e}") from e

        # Vectorize with TF-IDF
        vectors = self._vectorizer.transform(cleaned)

        # Predict the outcome using a classifier
        if probabilities:
            if not hasattr(self._clf, "predict_proba"):
                raise TypeError(
                    "Probabilities can only be predicted if the classifier implements the "
                    f"`predict_proba` method, {self._clf.__class__.__name__} does not."
                )
            preds = self._clf.predict_proba(vectors)
        else:
            preds = self._clf.predict(vectors)

        # Ensure output is a Numpy ndarray
        if not isinstance(preds, np.ndarray):
            preds = np.asarray(preds)

        # Check there are as many predictions as input annotations
        if probabilities:
            if preds.ndim != 2 or preds.shape[0] != len(annotations) or preds.shape[1] != 3:
                raise ValueError(
                    f"Classifier returned predictions with unexpected shape {preds.shape}, "
                    f"expected ({len(annotations)}, 3)."
                )
        else:
            if preds.ndim != 1 or preds.shape[0] != len(annotations):
                raise ValueError(
                    f"Classifier returned predictions with unexpected shape {preds.shape}, "
                    f"expected ({len(annotations)},)."
                )

        return preds

    def save_to_file(self, path: Union[str, os.PathLike]) -> None:

        """
        Saves the current class instance to a pickle file.

        Args:
            path (Union[str, os.PathLike]): 
                The path where the file is to be saved

        Raises:
            IOError:
                If the passed path has a parent directory that does not exist
            IOError: 
                If there is an error saving the class instance to a pickle file
        """

        path = Path(path)

        # Ensure parent directory exsists if given
        if path.parent and not path.parent.exists():
            raise IOError(
                f"Directory '{path.parent}' does not exist."
                "Create it before saving the classifier."
            )

        try:
            with path.open("wb") as f:
                pickle.dump(self, f)
        except (OSError, pickle.PickleError) as e:
            raise IOError(f"Error saving classifier to '{path}': {e}") from e

    @classmethod
    def load_from_file(cls, path: Union[str, os.PathLike]) -> "TFIDFClassifier":

        """
        Loads a TFIDFClassifier instance from a pickle file

        Args:
            path (Union[str, os.PathLike]): 
                The path containing the pickle file

        Raises:
            FileNotFoundError:
                If the file does not exist in the given path
            IOError: 
                If there is an error unpickling the file
            TypeError: 
                If the loaded object is not an instance of a TFIDFCLassifier

        Returns:
            TFIDFClassifier: 
                An instance of a TFIDFClassifier
        """

        path = Path(path)

        # Checking that the file exists
        if not path.exists():
            raise FileNotFoundError(f"No such file: '{path}'")

        try:
            with path.open("rb") as f:
                obj = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            raise IOError(f"Error loading classifier from '{path}': {e}") from e
        return obj

    @property
    def is_fitted(self) -> bool:

        """
        Describes if the vectorizer and classifier have been fit to training data

        Returns:
            bool:
                True if fit, else False
        """

        return self._is_fitted

    @property
    def classifier_parameters(self) -> dict:

        """
        Returns the parameters of the underlying classifier.

        Returns:
            dict: 
                A dictionary of parameter names mapped to their values
        """

        return self._clf.get_params()

    @property
    def vectorizer_parameters(self) -> dict:

        """
        Returns the parameters of the undeying TF-IDF vectorizer

        Returns:
            dict: 
                A dictionary of parameter names mapped to their values
        """

        return self._vectorizer.get_params()

    def describe(self) -> str:

        """
        A descriptive string of the TFIDFClassifier instance

        Returns:
            str: 
                String describing if the instance is fit to training data, 
                the used vectorizer class and the used classifier class. 
        """

        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"classifier={self._clf.__class__.__name__}, "
            f"vectorizer={self._vectorizer.__class__.__name__}, "
            f"status={status})"
            )

    def _clean_lemmatize(self, text: str) -> str:

        """
        Cleans a text string and lemmatize using spaCy.

        Cleaning will convert the string to lowecase, remove any brackets or prarenthases, 
        replace any non-alphanumeric characters with a blank space, remove any words with 
        only one character, and remove multiple trailing spaces. 

        Lemmatization will reduce words to their base lemma. Words with digits in them,
        such as asp45, remain intact, while pure digits are removed from the text.

        Args:
            text (str): 
                A protein annotation

        Returns:
            str: 
                The cleaned and lemmatized version of the given protein annotation
        """

        # Validating text is a string
        text = self._validate_single_text(text)

        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[\[\]\(\)]", "", text)  # Remove brackets and paranthases
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)  # Remove non-alphanumeric characters
        text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
        text = re.sub(r"\b\w{1}\b", "", text)  # Remove isolated 1-character tokens
        text = re.sub(r"\s+", " ", text)  # Cleans up extra spaces again

        # Lemmatization
        doc = self._lemmatizer(text)
        lemmas: List[str] = []

        for tok in doc:
            if tok.is_space:
                continue
            t = tok.text
            if tok.is_alpha:
                lemmas.append(tok.lemma_.lower()) # eg binding -> bind
            elif re.compile(r'(?=.*[a-zA-Z])(?=.*\d)').search(t):
                # Keep alphanumeric tokens like asp45 or hsp70
                lemmas.append(t.lower())
            # else: skip pure numbers and punctuation (shouldn’t occur post-cleaning)

        return " ".join(lemmas)

    def _validate_single_text(self, text: object) -> str:

        """
        Validation method to ensure passed annotations are string objects

        Args:
            text (object): 
                A passed protein annotation

        Raises:
            TypeError: 
                If the passed `text` is not a string object

        Returns:
            str: 
                The passed protein annotation
        """

        if not isinstance(text, str):
            raise TypeError(
                f"Expected a string for an annotation, got {type(text)}: {repr(text)}"
            )

        return text

    def _validate_text_sequence(self, sequence: Union[Sequence[str], pd.Series]) -> List[str]:

        """
        Validation method to ensure a sequence of text has a valid format

        Args:
            X (Union[Sequence[str], pd.Series]): 
                A sequence (list, tuple, or pandas Series) of text strings to validate.

        Raises:
            TypeError:
                If the passed sequence `X` is not a list, tuple, or pandas.Series
            ValueError: 
                If the passed sequence `X` is empty

        Returns:
            List[str]: 
                A list of text
        """

        if isinstance(sequence, pd.Series):
            data = sequence.tolist()
        elif isinstance(sequence, (list, tuple)):
            data = list(sequence)
        else:
            raise TypeError(
                "X must be a list, tuple, or pandas.Series"
                f"got {type(sequence)}"
            )

        if not data:
            raise ValueError("X is empty; expected at least one text string")

        return data
