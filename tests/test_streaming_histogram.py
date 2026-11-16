"""Tests for StreamingHistogram in numcompute_stream.stats."""

import numpy as np
import pytest

from numcompute_stream.stats import StreamingHistogram, histogram


def test_chunks_add_up_to_the_same_counts_as_one_big_histogram():
    data = np.random.default_rng(0).normal(size=1000)
    h = StreamingHistogram(bins=20, range=(-5, 5))
    for chunk in np.array_split(data, 7):
        h.update(chunk)
    counts, edges = histogram(data, bins=20, range=(-5, 5))
    assert h.counts.tolist() == counts.tolist()
    assert np.allclose(h.edges, edges)


def test_update_returns_self_and_update_stats_is_the_same_method():
    h = StreamingHistogram(bins=2, range=(0, 10))
    assert h.update([1]) is h
    h.update_stats([9])
    assert h.counts.tolist() == [1, 1]


def test_the_last_bin_includes_its_right_edge():
    h = StreamingHistogram(bins=2, range=(0, 10)).update([0, 10])
    assert h.counts.tolist() == [1, 1]


def test_values_outside_the_range_are_counted_separately():
    h = StreamingHistogram(bins=2, range=(0, 10)).update([-1, 5, 11, 12])
    assert h.counts.tolist() == [0, 1]
    assert h.n_outside == 3


def test_nan_is_skipped():
    h = StreamingHistogram(bins=2, range=(0, 10)).update([1.0, np.nan, 9.0])
    assert h.counts.tolist() == [1, 1] and h.n_outside == 0


def test_range_comes_from_the_first_chunk_and_then_stays():
    h = StreamingHistogram(bins=4).update([0, 8])
    assert h.edges.tolist() == [0, 2, 4, 6, 8]
    h.update([100])
    assert h.edges.tolist() == [0, 2, 4, 6, 8] and h.n_outside == 1


def test_a_first_chunk_with_one_repeated_value_still_gets_a_range():
    h = StreamingHistogram(bins=2).update([5, 5, 5])
    assert h.edges[0] == 4.5 and h.edges[-1] == 5.5
    assert h.counts.sum() == 3


def test_empty_and_all_nan_chunks_change_nothing():
    h = StreamingHistogram(bins=3)
    h.update([]).update([np.nan, np.nan])
    assert h.edges is None and h.counts.sum() == 0
    assert np.isnan(h.median())


def test_2d_chunks_are_flattened():
    h = StreamingHistogram(bins=2, range=(0, 10)).update([[1, 2], [8, 9]])
    assert h.counts.tolist() == [2, 2]


def test_rejected_chunk_changes_nothing():
    h = StreamingHistogram(bins=2, range=(0, 10)).update([1, 9])
    with pytest.raises(ValueError):
        h.update([1, np.inf])
    with pytest.raises(TypeError):
        h.update(["a"])
    assert h.counts.tolist() == [1, 1] and h.n_outside == 0


def test_reset_keeps_a_given_range_but_forgets_a_learned_one():
    given = StreamingHistogram(bins=2, range=(0, 10)).update([1]).reset()
    assert given.counts.sum() == 0 and given.edges.tolist() == [0, 5, 10]
    learned = StreamingHistogram(bins=2).update([0, 4]).reset()
    assert learned.edges is None and learned.n_outside == 0


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3"])
def test_bad_bins(bad):
    with pytest.raises(ValueError):
        StreamingHistogram(bins=bad)


@pytest.mark.parametrize("bad", [(5, 5), (6, 5), (0, np.inf), (1, 2, 3), "ab"])
def test_bad_range(bad):
    with pytest.raises(ValueError):
        StreamingHistogram(bins=3, range=bad)


# --- quantiles ---

def test_quantiles_are_within_one_bin_width_of_the_exact_ones():
    data = np.random.default_rng(1).normal(size=5000)
    h = StreamingHistogram(bins=100, range=(-6, 6))
    for chunk in np.array_split(data, 10):
        h.update(chunk)
    q = np.array([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    assert np.all(np.abs(h.quantile(q) - np.quantile(data, q)) <= 0.12)


def test_known_values():
    h = StreamingHistogram(bins=4, range=(0, 8)).update([6.5, 7.5])
    assert h.quantile([0, 0.5, 1]).tolist() == [6.0, 7.0, 8.0]
    assert h.median() == 7.0


def test_quantile_stays_inside_the_bins_that_have_values():
    # 40 and 45 are in the bin [40, 50), 50 is in the bin [50, 60)
    h = StreamingHistogram(bins=10, range=(0, 100)).update([40, 45, 50])
    values = h.quantile(np.linspace(0, 1, 11))
    assert values.min() >= 40 and values.max() <= 60 and np.all(np.diff(values) >= 0)


def test_quantile_shapes_and_nothing_counted():
    h = StreamingHistogram(bins=2, range=(0, 10))
    assert np.isnan(h.quantile(0.5)) and h.quantile([0.1, 0.9]).shape == (2,)
    h.update([1, 9])
    assert np.ndim(h.quantile(0.5)) == 0 and h.quantile([0.1, 0.5, 0.9]).shape == (3,)


@pytest.mark.parametrize("bad", [-0.1, 1.1, [], [[0.5]], [0.5, 2]])
def test_bad_q(bad):
    with pytest.raises(ValueError):
        StreamingHistogram(bins=2, range=(0, 10)).update([1]).quantile(bad)
