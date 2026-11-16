"""Tests for StreamTrainer in numcompute_stream.stream."""

import numpy as np
import pytest

from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer, _nbytes
from numcompute_stream.tree import DecisionTreeClassifier

RNG = np.random.default_rng(0)
X = RNG.normal(size=(300, 3))
Y = (X[:, 0] > 0).astype(int)
CHUNKS = [(X[i:i + 50], Y[i:i + 50]) for i in range(0, 300, 50)]


def make_pipe(buffer_size=100):
    return Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=3, buffer_size=buffer_size))])


def run(trainer, chunks=CHUNKS):
    for X_chunk, y_chunk in chunks:           # test-then-train
        trainer.score_chunk(X_chunk, y_chunk)
        trainer.fit_chunk(X_chunk, y_chunk)
    return trainer


# --- the loop and the log ---

def test_one_row_per_chunk_with_the_expected_columns():
    trainer = run(StreamTrainer(make_pipe()))
    assert len(trainer.log) == 6
    assert [row["chunk"] for row in trainer.log] == list(range(6))
    assert all(row["n_rows"] == 50 for row in trainer.log)
    assert set(trainer.log[0]) == {"chunk", "n_rows", "accuracy", "cumulative_accuracy",
                                   "fit_seconds", "retrained", "memory_bytes"}


def test_first_chunk_has_no_score_and_the_loop_still_works():
    trainer = run(StreamTrainer(make_pipe()))
    first = trainer.log[0]
    assert np.isnan(first["accuracy"]) and np.isnan(first["cumulative_accuracy"])
    assert not np.isnan(trainer.log[1]["accuracy"])


def test_scores_are_made_before_the_chunk_is_learned():
    trainer = StreamTrainer(make_pipe(buffer_size=50))
    trainer.score_chunk(*CHUNKS[0])
    trainer.fit_chunk(*CHUNKS[0])                # this chunk rebuilds the tree
    accuracy = trainer.score_chunk(*CHUNKS[1])   # a new chunk the tree has never seen
    manual = np.mean(trainer.pipeline.predict(CHUNKS[1][0]) == CHUNKS[1][1])
    assert accuracy == pytest.approx(manual)


def test_cumulative_accuracy_equals_the_streaming_metric_on_the_same_predictions():
    pipe = make_pipe()
    trainer = StreamTrainer(pipe)
    check = StreamingAccuracy()
    for X_chunk, y_chunk in CHUNKS:
        if trainer.log:
            check.update(y_chunk, pipe.predict(X_chunk))
        trainer.score_chunk(X_chunk, y_chunk)
        trainer.fit_chunk(X_chunk, y_chunk)
    assert trainer.log[-1]["cumulative_accuracy"] == pytest.approx(check.result())


def test_score_chunk_returns_the_accuracy_it_logs():
    trainer = run(StreamTrainer(make_pipe()), CHUNKS[:2])
    value = trainer.score_chunk(*CHUNKS[2])
    assert value == trainer.log[-1]["accuracy"]
    assert 0.0 <= value <= 1.0


def test_fit_chunk_returns_self():
    trainer = StreamTrainer(make_pipe())
    assert trainer.fit_chunk(*CHUNKS[0]) is trainer


def test_window_adds_a_rolling_accuracy():
    trainer = run(StreamTrainer(make_pipe(), window=2))
    assert "rolling_accuracy" in trainer.log[-1]
    assert trainer.log[1]["rolling_accuracy"] == trainer.log[1]["accuracy"]   # one scored chunk so far
    with pytest.raises(ValueError):
        StreamTrainer(make_pipe(), window=0)


def test_last_labels_and_predictions_are_kept_for_plotting():
    trainer = StreamTrainer(make_pipe())
    assert trainer.last_y_true is None and trainer.last_y_pred is None
    run(trainer)
    assert trainer.last_y_true.tolist() == CHUNKS[-1][1].tolist()
    assert trainer.last_y_pred.shape == trainer.last_y_true.shape


# --- retrain flag, time and memory ---

def test_retrained_is_true_only_on_the_chunk_that_rebuilds_the_tree():
    trainer = run(StreamTrainer(make_pipe(buffer_size=100)))
    assert [row["retrained"] for row in trainer.log] == [False, True, False, True, False, True]


def test_fit_seconds_is_logged_for_every_fitted_chunk():
    trainer = run(StreamTrainer(make_pipe()))
    assert all(row["fit_seconds"] > 0 for row in trainer.log)


def test_memory_grows_while_the_buffer_fills_and_drops_after_a_rebuild():
    trainer = run(StreamTrainer(make_pipe(buffer_size=1000)))
    sizes = [row["memory_bytes"] for row in trainer.log]
    assert sizes == sorted(sizes) and len(set(sizes)) == 6
    rebuilt = run(StreamTrainer(make_pipe(buffer_size=100)))
    assert rebuilt.log[1]["memory_bytes"] < rebuilt.log[0]["memory_bytes"]     # buffer emptied


def test_nbytes_counts_arrays_inside_lists_dicts_and_objects_once():
    class Holder:
        def __init__(self, shared):
            self.a = shared
            self.b = [shared, np.zeros(3)]
            self.c = {"x": np.zeros(2, dtype=np.int32), "name": "text", "n": 5}

    shared = np.zeros(10)                                     # 80 bytes
    assert _nbytes(Holder(shared)) == 80 + 24 + 8
    assert _nbytes("text") == 0 and _nbytes(5) == 0


# --- one row per chunk, even when used in pieces ---

def test_fit_without_score_adds_a_row_with_nan_accuracy():
    trainer = StreamTrainer(make_pipe())
    trainer.fit_chunk(*CHUNKS[0]).fit_chunk(*CHUNKS[1])
    assert len(trainer.log) == 2
    assert np.isnan(trainer.log[1]["accuracy"]) and not np.isnan(trainer.log[1]["fit_seconds"])


def test_score_without_fit_leaves_the_fit_columns_empty():
    trainer = StreamTrainer(make_pipe()).fit_chunk(*CHUNKS[0])
    trainer.score_chunk(*CHUNKS[1])
    row = trainer.log[-1]
    assert np.isnan(row["fit_seconds"]) and np.isnan(row["memory_bytes"]) and row["retrained"] is False
    trainer.fit_chunk(*CHUNKS[1])                  # completes that row instead of adding one
    assert len(trainer.log) == 2 and not np.isnan(trainer.log[-1]["fit_seconds"])


def test_empty_chunks_are_ignored():
    trainer = StreamTrainer(make_pipe())
    assert np.isnan(trainer.score_chunk(np.empty((0, 3)), np.empty(0)))
    trainer.fit_chunk(np.empty((0, 3)), np.empty(0))
    assert trainer.log == []


# --- errors ---

def test_rejected_chunks_are_not_logged():
    trainer = run(StreamTrainer(make_pipe()), CHUNKS[:2])
    rows = len(trainer.log)
    with pytest.raises(ValueError):
        trainer.score_chunk(X[:10], Y[:9])
    with pytest.raises(ValueError):
        trainer.fit_chunk(X[:10], Y[:9])
    with pytest.raises(ValueError):
        trainer.fit_chunk(np.ones((5, 7)), np.zeros(5))     # wrong number of columns
    with pytest.raises(ValueError):
        trainer.score_chunk(np.ones((5, 7)), np.zeros(5))
    assert len(trainer.log) == rows


def test_pipeline_must_be_able_to_learn_and_predict():
    with pytest.raises(TypeError):
        StreamTrainer(object())
    with pytest.raises(TypeError):
        StreamTrainer(StandardScaler())


def test_a_bigger_stream_learns():
    trainer = run(StreamTrainer(make_pipe(buffer_size=100)))
    assert trainer.log[-1]["cumulative_accuracy"] > 0.8
