"""Tests for Pipeline.partial_fit in numcompute_stream.pipeline."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import NotFittedError, OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier

RNG = np.random.default_rng(0)
X = RNG.normal(size=(200, 3))
Y = (X[:, 0] > 0).astype(int)


def make_pipe(buffer_size=100):
    return Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=3, buffer_size=buffer_size))])


def feed(pipe, size=50):
    for i in range(0, len(X), size):
        pipe.partial_fit(X[i:i + size], Y[i:i + size])
    return pipe


# --- it does what the manual loop does ---

def test_returns_self():
    pipe = make_pipe()
    assert pipe.partial_fit(X[:10], Y[:10]) is pipe


def test_steps_learn_like_the_manual_chain():
    pipe = feed(make_pipe())
    scaler, tree = StandardScaler(), DecisionTreeClassifier(max_depth=3, buffer_size=100)
    for i in range(0, 200, 50):
        scaler.partial_fit(X[i:i + 50])
        tree.partial_fit(scaler.transform(X[i:i + 50]), Y[i:i + 50])
    assert np.allclose(pipe.named_steps["scale"].mean_, scaler.mean_)
    assert np.allclose(pipe.named_steps["scale"].scale_, scaler.scale_)
    assert pipe.named_steps["model"].feature_.tolist() == tree.feature_.tolist()
    assert pipe.predict(X).tolist() == tree.predict(scaler.transform(X)).tolist()


def test_scaler_ends_up_the_same_as_fit_on_everything():
    pipe = feed(make_pipe())
    whole = StandardScaler().fit(X)
    assert np.allclose(pipe.named_steps["scale"].mean_, whole.mean_)
    assert np.allclose(pipe.named_steps["scale"].scale_, whole.scale_)


def test_predict_works_and_learns_something():
    pipe = feed(make_pipe())
    assert (pipe.predict(X) == Y).mean() > 0.9


def test_buffer_rebuilds_inside_the_pipeline():
    pipe = make_pipe(buffer_size=100)
    pipe.partial_fit(X[:50], Y[:50])
    assert not hasattr(pipe.named_steps["model"], "feature_")
    pipe.partial_fit(X[50:100], Y[50:100])
    assert pipe.named_steps["model"].n_nodes_ >= 1


def test_imputer_scaler_tree_chain_handles_nan_in_a_chunk():
    pipe = Pipeline([("fill", SimpleImputer()), ("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(buffer_size=50))])
    messy = X.copy()
    messy[::7, 1] = np.nan
    for i in range(0, 200, 50):
        pipe.partial_fit(messy[i:i + 50], Y[i:i + 50])
    assert (pipe.predict(messy) == Y).mean() > 0.9


def test_a_pipeline_without_a_model_needs_no_y():
    pipe = Pipeline([("scale", StandardScaler())])
    pipe.partial_fit(X[:50]).partial_fit(X[50:100])
    assert np.allclose(pipe.named_steps["scale"].mean_, X[:100].mean(axis=0))


def test_fit_starts_a_new_stream():
    pipe = feed(make_pipe())
    pipe.fit(X[:60], Y[:60])
    pipe.partial_fit(X[:10], Y[:10])          # the same width still works after fit


# --- empty chunk, bad input ---

def test_empty_chunk_is_ignored():
    pipe = make_pipe().partial_fit(X[:50], Y[:50])
    mean_before = pipe.named_steps["scale"].mean_.copy()
    pipe.partial_fit(np.empty((0, 3)), np.empty(0))
    assert np.allclose(pipe.named_steps["scale"].mean_, mean_before)


def test_x_and_y_must_match_and_a_model_needs_y():
    pipe = make_pipe()
    with pytest.raises(ValueError, match="same number of rows"):
        pipe.partial_fit(X[:10], Y[:9])
    with pytest.raises(ValueError, match="y_chunk is required"):
        pipe.partial_fit(X[:10])


def test_a_step_without_partial_fit_is_a_clear_type_error():
    class OnlyFit:
        def fit(self, X):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("old", OnlyFit()), ("model", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="'old' has no partial_fit"):
        pipe.partial_fit(X[:10], Y[:10])

    class NoPartialModel:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return X[:, 0]

    pipe = Pipeline([("scale", StandardScaler()), ("model", NoPartialModel())])
    with pytest.raises(TypeError, match="'model' has no partial_fit"):
        pipe.partial_fit(X[:10], Y[:10])
    assert not hasattr(pipe.named_steps["scale"], "mean_")     # nothing was learned


# --- a rejected chunk changes nothing ---

def test_error_in_the_model_undoes_the_scaler_update():
    pipe = make_pipe().partial_fit(X[:50], Y[:50])
    scaler = pipe.named_steps["scale"]
    mean_before, scale_before = scaler.mean_.copy(), scaler.scale_.copy()
    bad_y = np.array([0, 1, np.nan] + [0] * 7)
    with pytest.raises(ValueError):
        pipe.partial_fit(X[50:60], bad_y)
    assert np.array_equal(scaler.mean_, mean_before)
    assert np.array_equal(scaler.scale_, scale_before)
    assert pipe.named_steps["model"]._n_buffered == 50
    assert pipe.named_steps["scale"] is scaler                  # same object, put back in place


def test_nan_reaching_the_tree_undoes_the_scaler_update():
    pipe = make_pipe().partial_fit(X[:50], Y[:50])
    scaler = pipe.named_steps["scale"]
    mean_before = scaler.mean_.copy()
    messy = X[50:60].copy()
    messy[0, 0] = np.nan                                         # no imputer, so it reaches the tree
    with pytest.raises(ValueError, match="impute"):
        pipe.partial_fit(messy, Y[50:60])
    assert np.array_equal(scaler.mean_, mean_before)


def test_pipeline_still_works_after_a_rejected_chunk():
    pipe = make_pipe().partial_fit(X[:50], Y[:50])
    with pytest.raises(ValueError):
        pipe.partial_fit(X[50:60], Y[:5])
    pipe.partial_fit(X[50:100], Y[50:100])
    assert pipe.named_steps["scale"].mean_.shape == (3,)


# --- the number of columns must stay the same ---

def test_a_new_category_in_the_encoder_is_a_clear_error_and_changes_nothing():
    pipe = Pipeline([("enc", OneHotEncoder()), ("model", DecisionTreeClassifier())])
    pipe.partial_fit(np.array([["a"], ["b"]]), [0, 1])
    with pytest.raises(ValueError, match="'enc' now gives 3 columns but gave 2"):
        pipe.partial_fit(np.array([["c"], ["a"]]), [1, 0])
    assert pipe.named_steps["enc"].categories_[0].tolist() == ["a", "b"]
    assert pipe.named_steps["model"]._n_buffered == 2
    pipe.partial_fit(np.array([["b"], ["a"]]), [1, 0])           # known categories still work
    assert pipe.named_steps["model"]._n_buffered == 4


def test_the_encoder_is_fine_when_every_category_is_in_the_first_chunk():
    pipe = Pipeline([("enc", OneHotEncoder()), ("model", DecisionTreeClassifier())])
    pipe.partial_fit(np.array([["a"], ["b"], ["c"]]), [0, 1, 1])
    pipe.partial_fit(np.array([["c"], ["a"]]), [1, 0])
    assert pipe.predict(np.array([["a"], ["c"]])).shape == (2,)


def test_predict_before_any_chunk_is_not_fitted():
    with pytest.raises(NotFittedError):
        make_pipe().predict(X[:5])
