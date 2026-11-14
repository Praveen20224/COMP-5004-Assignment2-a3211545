"""Tests for numcompute_stream.stats.RunningStats (running mean and variance)."""

import numpy as np
import pytest

from numcompute_stream.stats import RunningStats, Welford


def feed(chunks, **options):
    rs = RunningStats(**options)
    for chunk in chunks:
        rs.update(chunk)
    return rs


# --- correct answers ---

def test_one_chunk_matches_numpy():
    X = np.random.default_rng(0).normal(size=(50, 3))
    rs = feed([X])
    assert np.allclose(rs.mean, X.mean(axis=0))
    assert np.allclose(rs.variance(), X.var(axis=0))
    assert np.allclose(rs.std(), X.std(axis=0))
    assert rs.count.tolist() == [50, 50, 50]


def test_chunks_give_same_answer_as_all_at_once():
    X = np.random.default_rng(1).normal(5, 3, size=(100, 4))
    rs = feed(np.array_split(X, [7, 30, 31, 80]))      # chunks of different sizes
    assert np.allclose(rs.mean, X.mean(axis=0))
    assert np.allclose(rs.variance(), X.var(axis=0))


def test_one_row_at_a_time():
    X = np.random.default_rng(2).normal(size=(40, 2))
    rs = feed([X[i:i + 1] for i in range(len(X))])
    assert np.allclose(rs.mean, X.mean(axis=0))
    assert np.allclose(rs.variance(), X.var(axis=0))


def test_order_of_chunks_does_not_matter():
    X = np.random.default_rng(3).normal(size=(60, 3))
    chunks = np.array_split(X, 6)
    forward = feed(chunks)
    backward = feed(chunks[::-1])
    assert np.allclose(forward.mean, backward.mean)
    assert np.allclose(forward.variance(), backward.variance())


def test_sample_variance_with_ddof_1():
    X = np.random.default_rng(4).normal(size=(30, 2))
    rs = feed(np.array_split(X, 3))
    assert np.allclose(rs.variance(ddof=1), X.var(axis=0, ddof=1))
    assert np.allclose(rs.std(ddof=1), X.std(axis=0, ddof=1))


def test_the_example_from_the_lesson():
    rs = feed([[2, 4, 4], [4, 5, 5, 7, 9]])            # two 1-D chunks of one variable
    assert rs.mean.tolist() == [5.0]
    assert rs.variance().tolist() == [4.0]


def test_matches_the_group_welford_class_for_one_column():
    values = np.random.default_rng(5).normal(size=200)
    w = Welford()
    rs = RunningStats()
    for part in np.array_split(values, 8):
        w.update(part)
        rs.update(part)
    assert rs.mean[0] == pytest.approx(w.mean)
    assert rs.variance()[0] == pytest.approx(w.variance())
    assert rs.variance(ddof=1)[0] == pytest.approx(w.variance(ddof=1))


# --- input shapes and types ---

def test_1d_chunk_is_one_column_and_a_scalar_is_one_value():
    rs = feed([[1.0, 2.0, 3.0], 4.0])
    assert rs.n_features == 1
    assert rs.count.tolist() == [4]
    assert rs.mean.tolist() == [2.5]


def test_integer_and_bool_chunks_are_accepted():
    rs = feed([np.array([[1, 2], [3, 4]]), np.array([[True, False]])])
    assert np.allclose(rs.mean, [5 / 3, 2.0])


def test_non_contiguous_chunks_work():
    X = np.arange(24, dtype=float).reshape(6, 4)
    every_other_row = X[::2]
    transposed = X.T
    assert np.allclose(feed([every_other_row]).mean, every_other_row.mean(axis=0))
    assert np.allclose(feed([transposed]).mean, transposed.mean(axis=0))


def test_the_chunk_itself_is_not_changed():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    before = X.copy()
    feed([X])
    assert np.array_equal(X, before, equal_nan=True)


def test_update_returns_self_and_update_stats_is_the_same_method():
    rs = RunningStats()
    assert rs.update([[1.0]]) is rs
    other = RunningStats()
    other.update_stats([[1.0]]).update_stats([[3.0]])
    assert other.mean.tolist() == [2.0]


# --- edge cases ---

def test_before_any_chunk_everything_is_empty():
    rs = RunningStats()
    assert rs.n_features is None
    assert rs.mean.shape == (0,)
    assert rs.variance().shape == (0,)


def test_empty_chunks_change_nothing():
    rs = feed([[[1.0, 2.0], [3.0, 4.0]]])
    mean_before = rs.mean.copy()
    rs.update(np.zeros((0, 2))).update([])
    assert np.array_equal(rs.mean, mean_before)
    assert rs.count.tolist() == [2, 2]
    assert feed([np.zeros((0, 3))]).n_features is None      # an empty first chunk does not fix the width


def test_nan_is_skipped_in_each_column_separately():
    X = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 7.0], [4.0, 9.0]])
    rs = feed([X[:2], X[2:]])
    assert rs.count.tolist() == [3, 3]
    assert np.allclose(rs.mean, np.nanmean(X, axis=0))
    assert np.allclose(rs.variance(), np.nanvar(X, axis=0))


def test_a_column_that_is_all_nan_stays_nan_while_the_others_work():
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    rs = feed([X])
    assert rs.count.tolist() == [2, 0]
    assert rs.mean[0] == 2.0 and np.isnan(rs.mean[1])
    assert np.isnan(rs.variance()[1])


def test_count_not_bigger_than_ddof_gives_nan():
    rs = feed([[[5.0]]])
    assert rs.variance().tolist() == [0.0]
    assert np.isnan(rs.variance(ddof=1)[0])


def test_chunk_of_equal_values_has_zero_variance():
    rs = feed([np.full((10, 2), 7.0), np.full((5, 2), 7.0)])
    assert rs.mean.tolist() == [7.0, 7.0]
    assert rs.variance().tolist() == [0.0, 0.0]


def test_big_numbers_with_a_small_spread_stay_accurate():
    values = 1e9 + np.array([4.0, 7.0, 13.0, 16.0])
    rs = feed([values[:2], values[2:]])
    assert rs.variance()[0] == pytest.approx(22.5)          # the "mean of squares" formula gives -128 here


# --- mistakes ---

def test_nan_policy_raise_rejects_nan_and_changes_nothing():
    rs = feed([[[1.0, 2.0]]], nan_policy='raise')
    with pytest.raises(ValueError, match='NaN'):
        rs.update([[np.nan, 3.0]])
    assert rs.count.tolist() == [1, 1]
    assert rs.mean.tolist() == [1.0, 2.0]


def test_infinity_is_rejected_and_changes_nothing():
    rs = feed([[[1.0, 2.0]]])
    with pytest.raises(ValueError, match='infinity'):
        rs.update([[np.inf, 3.0]])
    assert rs.count.tolist() == [1, 1]


def test_wrong_number_of_columns_is_rejected_and_changes_nothing():
    rs = feed([[[1.0, 2.0]]])
    with pytest.raises(ValueError, match='3 columns but earlier chunks had 2'):
        rs.update([[1.0, 2.0, 3.0]])
    assert rs.count.tolist() == [1, 1]


def test_bad_inputs():
    rs = RunningStats()
    with pytest.raises(TypeError):
        rs.update([['a', 'b']])
    with pytest.raises(ValueError, match='1-D or 2-D'):
        rs.update(np.zeros((2, 2, 2)))
    assert rs.n_features is None                              # a rejected first chunk fixes nothing
    with pytest.raises(ValueError, match='at least one column'):
        rs.update(np.zeros((3, 0)))


def test_bad_options():
    with pytest.raises(ValueError, match='nan_policy'):
        RunningStats(nan_policy='propagate')
    rs = feed([[[1.0]]])
    with pytest.raises(ValueError):
        rs.variance(ddof=-1)
    with pytest.raises(TypeError):
        rs.variance(ddof=1.5)


def test_reset_forgets_everything():
    rs = feed([[[1.0, 2.0]]])
    assert rs.reset() is rs
    assert rs.n_features is None
    rs.update([[1.0, 2.0, 3.0]])                              # a different width is fine after a reset
    assert rs.n_features == 3
