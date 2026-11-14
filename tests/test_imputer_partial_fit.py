"""Tests for SimpleImputer.partial_fit (streaming missing-value filling)."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import NotFittedError, SimpleImputer


def stream(chunks, **options):
    imputer = SimpleImputer(**options)
    for chunk in chunks:
        imputer.partial_fit(chunk)
    return imputer


# --- strategy "mean" ---

def test_chunks_give_the_same_means_as_fit():
    X = np.random.default_rng(0).normal(5, 3, size=(100, 4))
    X[np.random.default_rng(1).random(X.shape) < 0.2] = np.nan
    streamed = stream(np.array_split(X, [10, 11, 60]), strategy='mean')
    assert np.allclose(streamed.statistics_, SimpleImputer('mean').fit(X).statistics_)
    assert np.allclose(streamed.transform(X), SimpleImputer('mean').fit_transform(X))


def test_one_row_at_a_time():
    X = np.random.default_rng(2).normal(size=(30, 2))
    X[::4, 0] = np.nan
    streamed = stream([X[i:i + 1] for i in range(len(X))], strategy='mean')
    assert np.allclose(streamed.statistics_, SimpleImputer('mean').fit(X).statistics_)


def test_the_tiny_example_from_the_lesson():
    imputer = stream([[[1.0, np.nan], [3.0, 4.0]]], strategy='mean')
    assert imputer.statistics_.tolist() == [2.0, 4.0]
    imputer.partial_fit([[np.nan, 8.0]])
    assert imputer.statistics_.tolist() == [2.0, 6.0]
    assert imputer.transform([[np.nan, np.nan]]).tolist() == [[2.0, 6.0]]


def test_the_replacement_follows_the_stream():
    imputer = stream([[[10.0]]], strategy='mean')
    assert imputer.transform([[np.nan]]).tolist() == [[10.0]]
    imputer.partial_fit([[20.0], [30.0]])
    assert imputer.transform([[np.nan]]).tolist() == [[20.0]]


def test_nan_is_skipped_separately_in_each_column():
    X = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 7.0], [4.0, 9.0]])
    imputer = stream([X[:2], X[2:]], strategy='mean')
    assert np.allclose(imputer.statistics_, np.nanmean(X, axis=0))


def test_big_numbers_with_a_small_spread():
    X = 1e9 + np.random.default_rng(3).normal(size=(40, 2))
    streamed = stream(np.array_split(X, 4), strategy='mean')
    assert np.allclose(streamed.statistics_, SimpleImputer('mean').fit(X).statistics_, rtol=1e-12)


# --- columns with no numbers yet, and strategy "constant" ---

def test_a_column_with_no_numbers_yet_uses_fill_value_and_never_returns_nan():
    imputer = stream([[[1.0, np.nan], [3.0, np.nan]]], strategy='mean')
    assert imputer.statistics_.tolist() == [2.0, 0.0]
    assert not np.isnan(imputer.transform([[np.nan, np.nan]])).any()
    imputer.partial_fit([[5.0, 10.0]])                       # a number arrives
    assert imputer.statistics_.tolist() == [3.0, 10.0]


def test_custom_fill_value_for_a_column_with_no_numbers_yet():
    imputer = stream([[[np.nan, 1.0]]], strategy='mean', fill_value=-1.0)
    assert imputer.statistics_.tolist() == [-1.0, 1.0]


def test_constant_strategy_always_uses_fill_value():
    imputer = stream([[[1.0, 2.0]], [[3.0, 4.0]]], strategy='constant', fill_value=9.0)
    assert imputer.statistics_.tolist() == [9.0, 9.0]
    assert imputer.transform([[np.nan, 5.0]]).tolist() == [[9.0, 5.0]]


def test_constant_strategy_still_checks_the_number_of_columns():
    imputer = stream([[[1.0, 2.0]]], strategy='constant')
    with pytest.raises(ValueError, match='3 columns but earlier chunks had 2'):
        imputer.partial_fit([[1.0, 2.0, 3.0]])


# --- empty chunks, fit, and not fitted ---

def test_empty_chunks_are_ignored():
    imputer = stream([[[1.0, 2.0], [3.0, 4.0]]], strategy='mean')
    imputer.partial_fit(np.zeros((0, 2)))
    assert imputer.statistics_.tolist() == [2.0, 3.0]


def test_an_empty_first_chunk_leaves_the_imputer_not_fitted():
    imputer = SimpleImputer('mean').partial_fit(np.zeros((0, 3)))
    with pytest.raises(NotFittedError):
        imputer.transform([[1.0, 2.0, 3.0]])


def test_transform_before_any_chunk_raises_not_fitted_error():
    with pytest.raises(NotFittedError):
        SimpleImputer('mean').transform([[1.0]])


def test_fit_starts_over_so_a_later_partial_fit_begins_a_new_stream():
    imputer = stream([[[100.0], [200.0]]], strategy='mean')
    imputer.fit([[1.0], [3.0]])
    assert imputer.statistics_.tolist() == [2.0]
    imputer.partial_fit([[10.0]])                            # a new stream: only this chunk counts
    assert imputer.statistics_.tolist() == [10.0]


def test_fit_itself_still_works_as_before():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    assert SimpleImputer('mean').fit(X).statistics_.tolist() == [2.0, 4.0]


# --- mistakes ---

def test_wrong_number_of_columns_is_rejected_and_changes_nothing():
    imputer = stream([[[1.0, 2.0], [3.0, 4.0]]], strategy='mean')
    before = imputer.statistics_.copy()
    with pytest.raises(ValueError, match='3 columns but earlier chunks had 2'):
        imputer.partial_fit([[1.0, 2.0, 3.0]])
    assert np.array_equal(imputer.statistics_, before)


def test_a_bad_first_chunk_leaves_the_imputer_unfitted():
    imputer = SimpleImputer('mean')
    with pytest.raises(ValueError, match='infinity'):
        imputer.partial_fit([[1.0, np.inf]])
    with pytest.raises(NotFittedError):
        imputer.transform([[1.0, 2.0]])


def test_bad_inputs():
    imputer = SimpleImputer('mean')
    with pytest.raises(ValueError, match='2-D'):
        imputer.partial_fit([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        imputer.partial_fit([['a', 'b']])


def test_returns_self_and_does_not_change_the_chunk():
    chunk = np.array([[1.0, np.nan], [3.0, 4.0]])
    before = chunk.copy()
    imputer = SimpleImputer('mean')
    assert imputer.partial_fit(chunk) is imputer
    assert np.array_equal(chunk, before, equal_nan=True)


def test_non_contiguous_chunks_work():
    X = np.arange(24, dtype=float).reshape(6, 4)
    X[0, 0] = np.nan
    imputer = stream([X[::2], X[1::2]], strategy='mean')
    assert np.allclose(imputer.statistics_, SimpleImputer('mean').fit(X).statistics_)
