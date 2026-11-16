"""Tests for the streaming metrics in numcompute_stream.metrics."""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    StreamingAccuracy, StreamingAUC, StreamingConfusionMatrix, StreamingF1,
    StreamingPrecision, StreamingRecall, accuracy, auc, confusion_matrix, f1,
    precision, recall, roc_curve)

RNG = np.random.default_rng(0)
Y_TRUE = RNG.integers(0, 3, 300)
Y_PRED = np.where(RNG.random(300) < 0.6, Y_TRUE, RNG.integers(0, 3, 300))
CHUNKS = [(Y_TRUE[i:i + 50], Y_PRED[i:i + 50]) for i in range(0, 300, 50)]

BINARY_TRUE = (Y_TRUE > 0).astype(int)
BINARY_PRED = (Y_PRED > 0).astype(int)


def feed(metric, y_true, y_pred, size=37):
    for i in range(0, len(y_true), size):
        metric.update(y_true[i:i + size], y_pred[i:i + size])
    return metric


# --- the streaming answer equals the all-at-once answer ---

def test_confusion_matrix_matches_the_batch_function():
    got = feed(StreamingConfusionMatrix(), Y_TRUE, Y_PRED).result()
    assert got.tolist() == confusion_matrix(Y_TRUE, Y_PRED).tolist()


def test_accuracy_matches_the_batch_function():
    assert feed(StreamingAccuracy(), Y_TRUE, Y_PRED).result() == pytest.approx(accuracy(Y_TRUE, Y_PRED))


@pytest.mark.parametrize("average", ["micro", "macro", "weighted"])
@pytest.mark.parametrize("stream_class, batch", [
    (StreamingPrecision, precision), (StreamingRecall, recall), (StreamingF1, f1)])
def test_scores_match_the_batch_functions(stream_class, batch, average):
    got = feed(stream_class(average=average), Y_TRUE, Y_PRED).result()
    assert got == pytest.approx(batch(Y_TRUE, Y_PRED, average=average))


@pytest.mark.parametrize("stream_class, batch", [
    (StreamingPrecision, precision), (StreamingRecall, recall), (StreamingF1, f1)])
def test_binary_and_per_class_scores_match(stream_class, batch):
    got = feed(stream_class(), BINARY_TRUE, BINARY_PRED).result()
    assert got == pytest.approx(batch(BINARY_TRUE, BINARY_PRED))
    per_class = feed(stream_class(average=None), Y_TRUE, Y_PRED).result()
    assert np.allclose(per_class, batch(Y_TRUE, Y_PRED, average=None))


def test_string_labels():
    t = np.array(["cat", "dog", "cat", "bird"])
    p = np.array(["cat", "cat", "cat", "bird"])
    m = StreamingConfusionMatrix().update(t[:2], p[:2]).update(t[2:], p[2:])
    assert m.classes_.tolist() == ["bird", "cat", "dog"]
    assert m.result().tolist() == confusion_matrix(t, p).tolist()


# --- the matrix grows when a new class appears ---

def test_a_new_class_in_a_later_chunk_grows_the_matrix():
    m = StreamingConfusionMatrix().update([0, 1], [0, 1]).update([2], [0])
    assert m.classes_.tolist() == [0, 1, 2]
    assert m.result().tolist() == [[1, 0, 0], [0, 1, 0], [1, 0, 0]]


def test_a_class_that_is_only_ever_predicted_is_still_a_class():
    m = StreamingConfusionMatrix().update([0, 0], [0, 5])
    assert m.classes_.tolist() == [0, 5]


def test_growing_the_matrix_keeps_a_window_correct():
    m = StreamingConfusionMatrix(window=2)
    m.update([0, 1], [0, 1]).update([1, 1], [1, 0]).update([2, 2], [2, 2])
    expected = confusion_matrix([1, 1, 2, 2], [1, 0, 2, 2], labels=[0, 1, 2])
    assert m.result().tolist() == expected.tolist()


# --- rolling window ---

def test_window_equals_the_batch_metric_on_the_last_chunks():
    acc, cm, f = StreamingAccuracy(window=2), StreamingConfusionMatrix(window=2), StreamingF1(average="macro", window=2)
    for t, p in CHUNKS:
        for metric in (acc, cm, f):
            metric.update(t, p)
    last_t, last_p = np.concatenate([c[0] for c in CHUNKS[-2:]]), np.concatenate([c[1] for c in CHUNKS[-2:]])
    assert acc.result() == pytest.approx(accuracy(last_t, last_p))
    assert cm.result().tolist() == confusion_matrix(last_t, last_p, labels=[0, 1, 2]).tolist()
    assert f.result() == pytest.approx(f1(last_t, last_p, average="macro"))


def test_window_forgets_old_chunks():
    acc = StreamingAccuracy(window=2)
    for right in (True, True, False, False):
        acc.update([1, 1], [1, 1] if right else [0, 0])
    assert acc.result() == 0.0


def test_window_bigger_than_the_stream_counts_everything():
    assert feed(StreamingAccuracy(window=1000), Y_TRUE, Y_PRED).result() == pytest.approx(accuracy(Y_TRUE, Y_PRED))


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "2"])
def test_bad_window(bad):
    with pytest.raises(ValueError):
        StreamingAccuracy(window=bad)
    with pytest.raises(ValueError):
        StreamingAUC(window=bad)


# --- behaviour: chaining, reset, empty and bad chunks ---

def test_update_returns_self_and_reset_forgets():
    acc = StreamingAccuracy()
    assert acc.update([1], [1]) is acc
    assert acc.reset() is acc
    assert np.isnan(acc.result()) and acc.classes_ is None
    acc.update([0, 1], [0, 0])
    assert acc.result() == 0.5


def test_nothing_seen_yet():
    assert np.isnan(StreamingAccuracy().result())
    assert np.isnan(StreamingF1().result())
    assert StreamingPrecision(average=None).result().size == 0
    assert StreamingConfusionMatrix().result().shape == (0, 0)
    assert np.isnan(StreamingAUC().result())


def test_empty_chunk_is_ignored():
    acc = StreamingAccuracy(window=1).update([1, 1], [1, 0]).update([], [])
    assert acc.result() == 0.5


def test_rejected_chunk_changes_nothing():
    m = StreamingConfusionMatrix().update([0, 1], [0, 1])
    with pytest.raises(ValueError):
        m.update([0, 1, 2], [0, 1])
    with pytest.raises(ValueError):
        m.update([0, np.nan], [0, 1])
    with pytest.raises(TypeError):
        m.update(["a", "b"], ["a", "b"])        # strings after numbers
    with pytest.raises(TypeError):
        m.update([0, 1], ["a", "b"])
    assert m.result().tolist() == [[1, 0], [0, 1]] and m.classes_.tolist() == [0, 1]


def test_binary_scores_need_at_most_two_classes_and_a_seen_pos_label():
    with pytest.raises(ValueError):
        StreamingF1().update([0, 1, 2], [0, 1, 2]).result()
    with pytest.raises(ValueError):
        StreamingPrecision(pos_label=5).update([0, 1], [0, 1]).result()


def test_zero_division_is_used_when_nothing_was_predicted_positive():
    assert StreamingPrecision().update([1, 0], [0, 0]).result() == 0.0
    assert StreamingPrecision(zero_division=1).update([1, 0], [0, 0]).result() == 1.0


def test_bad_score_settings():
    with pytest.raises(ValueError):
        StreamingF1(average="mean")
    with pytest.raises(ValueError):
        StreamingF1(zero_division=2)


# --- AUC ---

def test_auc_perfect_worst_and_tied():
    assert StreamingAUC(bins=10).update([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]).result() == 1.0
    assert StreamingAUC(bins=10).update([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]).result() == 0.0
    assert StreamingAUC(bins=10).update([1, 0], [0.5, 0.5]).result() == 0.5


def test_auc_adds_up_over_chunks():
    one = StreamingAUC(bins=10).update([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.3]).result()
    two = StreamingAUC(bins=10).update([1, 0], [0.9, 0.1]).update([1, 0], [0.8, 0.3]).result()
    assert one == two == 1.0


def test_auc_is_close_to_the_exact_auc_with_many_bins():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    score = np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1)
    streaming = StreamingAUC(bins=1000)
    for i in range(0, 2000, 200):
        streaming.update(y[i:i + 200], score[i:i + 200])
    fpr, tpr, _ = roc_curve(y, score)
    assert streaming.result() == pytest.approx(auc(fpr, tpr), abs=0.002)


def test_auc_needs_both_classes():
    a = StreamingAUC().update([1, 1], [0.4, 0.6])
    assert np.isnan(a.result())
    a.update([0], [0.5])
    assert not np.isnan(a.result())


def test_auc_window_and_reset():
    a = StreamingAUC(bins=10, window=1)
    a.update([1, 0], [0.9, 0.1]).update([1, 0], [0.1, 0.9])
    assert a.result() == 0.0
    a.reset()
    assert np.isnan(a.result())


def test_auc_pos_label_and_string_labels():
    a = StreamingAUC(bins=10, pos_label="yes").update(["yes", "no", "yes"], [0.9, 0.2, 0.7])
    assert a.result() == 1.0


def test_auc_score_range():
    a = StreamingAUC(bins=4, score_range=(-2, 2)).update([1, 0], [1.5, -1.5])
    assert a.result() == 1.0
    with pytest.raises(ValueError):
        a.update([1], [2.5])


def test_auc_rejected_chunk_changes_nothing():
    a = StreamingAUC(bins=10).update([1, 0], [0.9, 0.1])
    for bad_true, bad_score in [([1, 0], [0.5]), ([1, 0], [0.5, 1.5]), ([1, 0], [0.5, np.nan]),
                                ([1, np.nan], [0.5, 0.5])]:
        with pytest.raises(ValueError):
            a.update(bad_true, bad_score)
    with pytest.raises(TypeError):
        a.update([1, 0], ["a", "b"])
    assert a.result() == 1.0
    assert a.update([], []) is a


@pytest.mark.parametrize("settings", [{"bins": 0}, {"bins": 2.5}, {"bins": True},
                                      {"score_range": (1, 1)}, {"score_range": (0, np.inf)},
                                      {"score_range": (0, 1, 2)}])
def test_bad_auc_settings(settings):
    with pytest.raises(ValueError):
        StreamingAUC(**settings)
