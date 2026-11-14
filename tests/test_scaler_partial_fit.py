"""Tests for StandardScaler.partial_fit (streaming standardisation)."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import NotFittedError, StandardScaler


def stream(chunks):
    scaler = StandardScaler()
    for chunk in chunks:
        scaler.partial_fit(chunk)
    return scaler


# --- same answer as fit on all the data ---

def test_chunks_give_the_same_mean_and_scale_as_fit():
    X = np.random.default_rng(0).normal(5, 3, size=(100, 4))
    streamed = stream(np.array_split(X, [10, 11, 60]))
    whole = StandardScaler().fit(X)
    assert np.allclose(streamed.mean_, whole.mean_)
    assert np.allclose(streamed.scale_, whole.scale_)
    assert streamed.n_features_in_ == 4


def test_transform_after_streaming_matches_fit_transform():
    X = np.random.default_rng(1).normal(size=(60, 3))
    streamed = stream(np.array_split(X, 5))
    assert np.allclose(streamed.transform(X), StandardScaler().fit_transform(X))


def test_one_row_at_a_time():
    X = np.random.default_rng(2).normal(size=(30, 2))
    streamed = stream([X[i:i + 1] for i in range(len(X))])
    whole = StandardScaler().fit(X)
    assert np.allclose(streamed.mean_, whole.mean_)
    assert np.allclose(streamed.scale_, whole.scale_)


def test_the_tiny_example_from_the_lesson():
    scaler = stream([[[1.0, 10.0], [3.0, 30.0]]])
    assert scaler.mean_.tolist() == [2.0, 20.0]
    assert scaler.scale_.tolist() == [1.0, 10.0]
    scaler.partial_fit([[5.0, 20.0]])
    assert scaler.mean_.tolist() == [3.0, 20.0]
    assert np.allclose(scaler.transform([[5.0, 20.0]]), [[2 / np.sqrt(8 / 3), 0.0]])


def test_it_can_transform_after_every_chunk():
    scaler = StandardScaler()
    for chunk in np.array_split(np.arange(12, dtype=float).reshape(6, 2), 3):
        scaler.partial_fit(chunk)
        assert scaler.transform(chunk).shape == chunk.shape


def test_big_numbers_with_a_small_spread():
    X = 1e6 + np.random.default_rng(3).normal(size=(50, 2))
    streamed = stream(np.array_split(X, 5))
    whole = StandardScaler().fit(X)
    assert np.allclose(streamed.scale_, whole.scale_, rtol=1e-6)


# --- constant columns and NaN ---

def test_constant_column_has_scale_1_and_transforms_to_exactly_zero():
    X = np.column_stack([np.full(9, 0.1), np.arange(9, dtype=float)])
    scaler = stream(np.array_split(X, 3))
    assert scaler.scale_[0] == 1.0
    assert np.all(scaler.transform(X)[:, 0] == 0.0)          # not 1e-17, exactly 0 (same rule as fit)


def test_a_single_row_is_a_constant_column_until_more_data_arrives():
    scaler = stream([[[4.0, 8.0]]])
    assert scaler.scale_.tolist() == [1.0, 1.0]
    assert scaler.transform([[4.0, 8.0]]).tolist() == [[0.0, 0.0]]
    scaler.partial_fit([[6.0, 12.0]])
    assert scaler.scale_.tolist() == [1.0, 2.0]


def test_nan_is_skipped_and_stays_nan_after_transform():
    X = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 7.0], [4.0, 9.0]])
    streamed = stream([X[:2], X[2:]])
    whole = StandardScaler().fit(X)
    assert np.allclose(streamed.mean_, whole.mean_)
    assert np.allclose(streamed.scale_, whole.scale_)
    out = streamed.transform(X)
    assert np.array_equal(np.isnan(out), np.isnan(X))


def test_a_column_with_no_numbers_yet_has_nan_mean_and_scale_1():
    scaler = stream([[[1.0, np.nan], [3.0, np.nan]]])
    assert np.isnan(scaler.mean_[1]) and scaler.scale_[1] == 1.0
    scaler.partial_fit([[5.0, 10.0], [7.0, 20.0]])           # numbers arrive later
    assert scaler.mean_[1] == 15.0 and scaler.scale_[1] == pytest.approx(5.0)


# --- empty chunks, fit, and not fitted ---

def test_empty_chunks_are_ignored():
    scaler = stream([[[1.0, 2.0], [3.0, 4.0]]])
    mean_before = scaler.mean_.copy()
    scaler.partial_fit(np.zeros((0, 2)))
    assert np.array_equal(scaler.mean_, mean_before)


def test_an_empty_first_chunk_leaves_the_scaler_not_fitted():
    scaler = StandardScaler().partial_fit(np.zeros((0, 3)))
    with pytest.raises(NotFittedError):
        scaler.transform([[1.0, 2.0, 3.0]])


def test_transform_before_any_chunk_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        StandardScaler().transform([[1.0]])


def test_fit_starts_over_so_a_later_partial_fit_begins_a_new_stream():
    scaler = stream([[[100.0], [200.0]]])
    scaler.fit([[1.0], [3.0]])
    assert scaler.mean_.tolist() == [2.0]
    scaler.partial_fit([[5.0]])                               # a new stream: only this chunk counts
    assert scaler.mean_.tolist() == [5.0]


def test_partial_fit_after_fit_changes_nothing_in_fit_itself():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
    scaler = StandardScaler().fit(X)
    assert np.allclose(scaler.transform(X), StandardScaler().fit_transform(X))


# --- mistakes ---

def test_wrong_number_of_columns_is_rejected_and_changes_nothing():
    scaler = stream([[[1.0, 2.0], [3.0, 4.0]]])
    mean_before, scale_before = scaler.mean_.copy(), scaler.scale_.copy()
    with pytest.raises(ValueError, match='3 columns but earlier chunks had 2'):
        scaler.partial_fit([[1.0, 2.0, 3.0]])
    assert np.array_equal(scaler.mean_, mean_before)
    assert np.array_equal(scaler.scale_, scale_before)


def test_a_bad_first_chunk_leaves_the_scaler_unfitted():
    scaler = StandardScaler()
    with pytest.raises(ValueError, match='infinity'):
        scaler.partial_fit([[1.0, np.inf]])
    with pytest.raises(NotFittedError):
        scaler.transform([[1.0, 2.0]])


def test_bad_inputs():
    scaler = StandardScaler()
    with pytest.raises(ValueError, match='2-D'):
        scaler.partial_fit([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        scaler.partial_fit([['a', 'b']])


def test_returns_self_and_does_not_change_the_chunk():
    chunk = np.array([[1.0, np.nan], [3.0, 4.0]])
    before = chunk.copy()
    scaler = StandardScaler()
    assert scaler.partial_fit(chunk) is scaler
    assert np.array_equal(chunk, before, equal_nan=True)


def test_non_contiguous_chunks_work():
    X = np.arange(24, dtype=float).reshape(6, 4)
    scaler = stream([X[::2], X[1::2]])
    assert np.allclose(scaler.mean_, StandardScaler().fit(X).mean_)
