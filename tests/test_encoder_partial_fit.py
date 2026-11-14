"""Tests for OneHotEncoder.partial_fit (categories that grow as chunks arrive)."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import NotFittedError, OneHotEncoder


def stream(chunks, **options):
    encoder = OneHotEncoder(**options)
    for chunk in chunks:
        encoder.partial_fit(chunk)
    return encoder


def names(encoder):
    return [c.tolist() for c in encoder.categories_]


# --- same answer as fit on all the data ---

def test_chunks_give_the_same_categories_and_output_as_fit():
    X = np.array([["a", "x"], ["b", "y"], ["a", "z"], ["c", "x"], ["b", "x"], ["d", "w"]])
    streamed = stream([X[:2], X[2:3], X[3:]])
    whole = OneHotEncoder().fit(X)
    assert names(streamed) == names(whole)
    assert streamed.n_features_out_ == whole.n_features_out_
    assert np.array_equal(streamed.transform(X), whole.transform(X))


def test_numbers_work_too():
    X = np.array([[3, 10], [1, 20], [3, 30], [2, 10]])
    streamed = stream([X[:1], X[1:]])
    assert names(streamed) == names(OneHotEncoder().fit(X))
    assert np.array_equal(streamed.transform(X), OneHotEncoder().fit_transform(X))


def test_one_row_at_a_time():
    X = np.array([["a"], ["b"], ["a"], ["c"], ["b"]])
    streamed = stream([X[i:i + 1] for i in range(len(X))])
    assert names(streamed) == [["a", "b", "c"]]


def test_the_order_of_the_chunks_does_not_matter():
    X = np.array([["a", "x"], ["b", "y"], ["c", "z"], ["d", "x"]])
    forward, backward = stream([X[:2], X[2:]]), stream([X[2:], X[:2]])
    assert names(forward) == names(backward)
    assert np.array_equal(forward.transform(X), backward.transform(X))


# --- growing categories ---

def test_a_new_category_adds_an_output_column():
    encoder = stream([[["red"], ["blue"]]])
    assert names(encoder) == [["blue", "red"]] and encoder.n_features_out_ == 2
    encoder.partial_fit([["green"]])
    assert names(encoder) == [["blue", "green", "red"]] and encoder.n_features_out_ == 3


def test_columns_after_a_new_category_move_one_place_to_the_right():
    encoder = stream([[["red"], ["blue"]]])
    assert encoder.transform([["red"]]).tolist() == [[0.0, 1.0]]
    encoder.partial_fit([["green"]])                         # green sorts before red
    assert encoder.transform([["red"]]).tolist() == [[0.0, 0.0, 1.0]]


def test_a_repeated_category_adds_no_column():
    encoder = stream([[["a"], ["b"]], [["a"], ["b"]], [["b"]]])
    assert encoder.n_features_out_ == 2


def test_each_feature_grows_on_its_own():
    encoder = stream([[["a", "x"]], [["b", "x"]], [["b", "y"]]])
    assert names(encoder) == [["a", "b"], ["x", "y"]]
    assert encoder.n_features_out_ == 4


def test_an_unseen_category_is_an_error_until_it_is_learned():
    encoder = stream([[["a"], ["b"]]])
    with pytest.raises(ValueError, match="unseen categories"):
        encoder.transform([["c"]])
    encoder.partial_fit([["c"]])
    assert encoder.transform([["c"]]).tolist() == [[0.0, 0.0, 1.0]]


def test_ignore_policy_gives_zeros_for_a_category_not_learned_yet():
    encoder = stream([[["a"], ["b"]]], handle_unknown="ignore")
    assert encoder.transform([["z"]]).tolist() == [[0.0, 0.0]]


def test_strings_of_different_lengths_in_later_chunks():
    encoder = stream([[["ab"]], [["a much longer category"]]])
    assert encoder.n_features_out_ == 2
    assert encoder.transform([["ab"]]).sum() == 1.0


def test_it_can_transform_after_every_chunk():
    encoder = OneHotEncoder()
    for chunk in ([["a"], ["b"]], [["c"]], [["a"]]):
        encoder.partial_fit(chunk)
        assert encoder.transform(chunk).shape == (len(chunk), encoder.n_features_out_)


# --- empty chunks, fit, and not fitted ---

def test_empty_chunks_are_ignored():
    encoder = stream([[["a"], ["b"]]])
    encoder.partial_fit(np.zeros((0, 1), dtype=str))
    assert names(encoder) == [["a", "b"]]


def test_an_empty_first_chunk_leaves_the_encoder_not_fitted():
    encoder = OneHotEncoder().partial_fit(np.zeros((0, 2), dtype=str))
    with pytest.raises(NotFittedError):
        encoder.transform([["a", "b"]])


def test_transform_before_any_chunk_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        OneHotEncoder().transform([["a"]])


def test_fit_starts_over_so_a_later_partial_fit_begins_a_new_stream():
    encoder = stream([[["old1"], ["old2"]]])
    encoder.fit([["a"], ["b"]])
    assert names(encoder) == [["a", "b"]]
    encoder.partial_fit([["c"]])                             # a new stream: only this chunk counts
    assert names(encoder) == [["c"]]


def test_fit_itself_still_works_as_before():
    encoder = OneHotEncoder().fit([["a", "x"], ["b", "x"]])
    assert names(encoder) == [["a", "b"], ["x"]]
    assert encoder.transform([["b", "x"]]).tolist() == [[0.0, 1.0, 1.0]]


# --- mistakes ---

def test_wrong_number_of_columns_is_rejected_and_changes_nothing():
    encoder = stream([[["a", "x"]]])
    with pytest.raises(ValueError, match="3 features, but this transformer was fitted with 2"):
        encoder.partial_fit([["a", "x", "y"]])
    assert names(encoder) == [["a"], ["x"]]


def test_mixing_numbers_and_strings_is_rejected_and_changes_nothing():
    encoder = stream([[["a"]]])
    with pytest.raises(TypeError, match="number|numeric"):
        encoder.partial_fit([[1.0]])
    assert names(encoder) == [["a"]]


def test_nan_is_rejected_and_a_bad_first_chunk_leaves_the_encoder_unfitted():
    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match="NaN"):
        encoder.partial_fit([[1.0], [np.nan]])
    with pytest.raises(NotFittedError):
        encoder.transform([[1.0]])


def test_bad_inputs():
    encoder = OneHotEncoder()
    with pytest.raises(ValueError, match="2-D"):
        encoder.partial_fit(["a", "b"])
    with pytest.raises(TypeError):
        encoder.partial_fit(np.array([[object()]], dtype=object))


def test_returns_self_and_does_not_change_the_chunk():
    chunk = np.array([["a", "x"], ["b", "y"]])
    before = chunk.copy()
    encoder = OneHotEncoder()
    assert encoder.partial_fit(chunk) is encoder
    assert np.array_equal(chunk, before)


def test_non_contiguous_chunks_work():
    X = np.array([["a", "x"], ["b", "y"], ["c", "x"], ["d", "z"]])
    streamed = stream([X[::2], X[1::2]])
    assert names(streamed) == names(OneHotEncoder().fit(X))
