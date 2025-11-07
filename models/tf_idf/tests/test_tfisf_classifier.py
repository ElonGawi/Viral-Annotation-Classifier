# tests/test_tfidf_classifier.py

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

from tfidf_classifier import TFIDFClassifier  # adjust if module name differs


# -----------------------
# Fixtures
# -----------------------

@pytest.fixture
def small_train_df() -> pd.DataFrame:
    """Small training DataFrame for smoke tests."""
    return pd.DataFrame(
        {
            "protein_annotation": [
                "binding protein domain",
                "uninformative sequence",
                "",  # empty string should be allowed
            ],
            "label": [
                "proper",
                "uninformative",
                "low",
            ],
        }
    )


@pytest.fixture
def fitted_clf(small_train_df) -> TFIDFClassifier:
    """Return a classifier fitted on a tiny dataset."""
    clf = TFIDFClassifier()
    clf.fit(small_train_df)
    return clf


# -----------------------
# __init__ tests
# -----------------------

def test_init_default_ok():
    clf = TFIDFClassifier()
    assert clf.describe() == 'TFIDFClassifier(classifier=LogisticRegression, vectorizer=TfidfVectorizer, status=not fitted)'  
    assert clf.is_fitted is False


def test_init_invalid_classifier_raises_typeerror():
    class NotAClassifier:
        pass

    with pytest.raises(TypeError):
        TFIDFClassifier(classifier=NotAClassifier())


def test_init_invalid_vectorizer_raises_typeerror():
    class DummyVectorizer:
        def transform(self, X):
            return X

    with pytest.raises(TypeError):
        TFIDFClassifier(vectorizer=DummyVectorizer())


def test_init_invalid_spacy_model_raises_oserror():
    # This assumes such a model definitely doesn't exist.
    with pytest.raises(OSError):
        TFIDFClassifier(spacy_model="this_model_does_not_exist_123")


# -----------------------
# fit tests
# -----------------------

def test_fit_sets_is_fitted_and_vocab(small_train_df):
    clf = TFIDFClassifier()
    clf.fit(small_train_df)

    assert clf.is_fitted is True


def test_fit_requires_dataframe():
    clf = TFIDFClassifier()
    with pytest.raises(TypeError):
        clf.fit("not a dataframe")  # type: ignore[arg-type]


def test_fit_requires_required_columns():
    clf = TFIDFClassifier()
    df = pd.DataFrame({"protein_annotation": ["a", "b"]})
    with pytest.raises(KeyError):
        clf.fit(df)


def test_fit_raises_on_empty_dataframe():
    clf = TFIDFClassifier()
    empty = pd.DataFrame(columns=["protein_annotation", "label"])
    with pytest.raises(ValueError):
        clf.fit(empty)


def test_fit_raises_on_missing_labels():
    clf = TFIDFClassifier()
    df = pd.DataFrame(
        {
            "protein_annotation": ["a", "b"],
            "label": ["low", None],
        }
    )
    with pytest.raises(ValueError):
        clf.fit(df)


def test_fit_raises_on_non_string_annotation():
    clf = TFIDFClassifier()
    df = pd.DataFrame(
        {
            "protein_annotation": ["valid", 12.0],  # second is float
            "label": ["low", "proper"],
        }
    )
    with pytest.raises(TypeError):
        clf.fit(df)


# -----------------------
# predict tests
# -----------------------

def test_predict_raises_notfitted_before_fit():
    clf = TFIDFClassifier()
    with pytest.raises(NotFittedError):
        clf.predict(["test protein"])


def test_predict_returns_numpy_array(fitted_clf):
    X = ["new protein annotation", "another one"]
    preds = fitted_clf.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


def test_predict_works_with_series(fitted_clf):
    X = pd.Series(["protein A", "protein B"])
    preds = fitted_clf.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (len(X),)


def test_predict_raises_on_invalid_container_type(fitted_clf):
    with pytest.raises(TypeError):
        fitted_clf.predict("not a list or series")  # type: ignore[arg-type]


def test_predict_raises_on_empty_sequence(fitted_clf):
    with pytest.raises(ValueError):
        fitted_clf.predict([])


def test_predict_raises_on_non_string_entry(fitted_clf):
    with pytest.raises(TypeError):
        fitted_clf.predict(["ok", 12.0])  # second is not a string


# -----------------------
# save / load tests
# -----------------------

def test_save_and_load_roundtrip(tmp_path: Path, fitted_clf, small_train_df):
    model_path = tmp_path / "tfidf_clf.pkl"

    # Save
    fitted_clf.save_to_file(model_path)
    assert model_path.exists()

    # Load
    loaded = TFIDFClassifier.load_from_file(model_path)
    assert isinstance(loaded, TFIDFClassifier)
    assert loaded.is_fitted is True

    # Same predictions on the same input
    X = small_train_df["protein_annotation"]
    preds_original = fitted_clf.predict(X)
    preds_loaded = loaded.predict(X)

    assert np.array_equal(preds_original, preds_loaded)


def test_save_raises_if_directory_missing(tmp_path: Path, fitted_clf):
    bad_dir = tmp_path / "nonexistent_dir"
    bad_path = bad_dir / "model.pkl"

    with pytest.raises(IOError):
        fitted_clf.save_to_file(bad_path)


def test_load_nonexistent_file_raises(tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.pkl"
    with pytest.raises(FileNotFoundError):
        TFIDFClassifier.load_from_file(missing_path)


def test_load_wrong_object_type_raises(tmp_path: Path):
    path = tmp_path / "not_a_classifier.pkl"

    # Save something that is not a TFIDFClassifier
    with path.open("wb") as f:
        pickle.dump(123, f)

    with pytest.raises(TypeError):
        TFIDFClassifier.load_from_file(path)


# -----------------------
# describe / properties
# -----------------------

def test_describe_changes_after_fit(small_train_df):
    clf = TFIDFClassifier()
    desc_before = clf.describe()
    assert "not fitted" in desc_before

    clf.fit(small_train_df)
    desc_after = clf.describe()
    assert "fitted" in desc_after
    assert "TFIDFClassifier(" in desc_after
    assert "classifier=" in desc_after
    assert "vectorizer=" in desc_after


def test_properties_expose_internal_objects(fitted_clf):
    assert fitted_clf.is_fitted is True
