import time
import numpy as np

from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.preprocessing import NotFittedError

__all__ = ["StreamTrainer"]


def _nbytes(obj, seen=None):
    # total size in bytes of the numpy arrays inside obj (looks into lists, tuples, dicts and
    # objects), each object is counted once. Numbers and strings count as 0.
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(item, seen) for item in obj)
    if isinstance(obj, dict):
        return sum(_nbytes(item, seen) for item in obj.values())
    if hasattr(obj, "__dict__"):
        return _nbytes(vars(obj), seen)
    return 0


class StreamTrainer:
    """
    Run a pipeline over a stream, chunk by chunk, and log what happens.

    Params:
        pipeline: a Pipeline with partial_fit and predict, whose last step is a model
        window: None or an int >= 1. If given, the log also has a rolling accuracy over the
                last `window` chunks.
    """

    def __init__(self, pipeline, window=None):
        if not (hasattr(pipeline, "partial_fit") and hasattr(pipeline, "predict")):
            raise TypeError("pipeline needs partial_fit and predict methods")
        self.pipeline = pipeline
        self.window = window
        self._cumulative = StreamingAccuracy()
        self._rolling = None if window is None else StreamingAccuracy(window=window)
        self.log = []
        self.last_y_true = None
        self.last_y_pred = None

    def score_chunk(self, X, y):
        """
        Predict the chunk with the pipeline as it is now, and log how well that went.

        Params:
            X: array shape (n_rows, n_features)
            y: array shape (n_rows,)

        Returns:
            float: accuracy on this chunk, NaN if the pipeline cannot predict yet (nothing
            has been learned). An empty chunk gives NaN and is not logged.

        Raises:
            ValueError: if X and y have different numbers of rows, or X is not valid for the
                        pipeline. If a chunk is rejected, nothing is logged.
        """
        self._check_rows(X, y)
        if len(X) == 0:
            return float("nan")
        y = np.asarray(y)

        try:
            y_pred = self.pipeline.predict(X)
        except NotFittedError:                       # the pipeline has not learned anything yet
            accuracy = float("nan")
        else:
            accuracy = float(np.mean(y_pred == y))
            self._cumulative.update(y, y_pred)
            if self._rolling is not None:
                self._rolling.update(y, y_pred)
            self.last_y_true, self.last_y_pred = y, y_pred

        self._new_row(len(y), accuracy)
        return accuracy

    def fit_chunk(self, X, y):
        """
        Let the pipeline learn from the chunk, and log the time and memory.

        Params:
            X: array shape (n_rows, n_features)
            y: array shape (n_rows,), the true labels

        Returns:
            self. An empty chunk is ignored and not logged.

        Raises:
            ValueError: if X and y have different numbers of rows, or whatever the pipeline
                        raises for a bad chunk (see Pipeline.partial_fit). Nothing is logged then.
        """
        self._check_rows(X, y)
        if len(X) == 0:
            return self

        model = self.pipeline.steps[-1][1]
        tree_before = getattr(model, "feature_", None)
        start = time.perf_counter()
        self.pipeline.partial_fit(X, y)
        seconds = time.perf_counter() - start
        tree_after = getattr(model, "feature_", None)   # a rebuilt tree is a new array

        if self.log and np.isnan(self.log[-1]["fit_seconds"]):
            row = self.log[-1]                          # the row that score_chunk started
        else:
            row = self._new_row(len(X), float("nan"))
        row["fit_seconds"] = seconds
        row["retrained"] = tree_after is not tree_before
        row["memory_bytes"] = _nbytes(self.pipeline)
        return self


    def _check_rows(self, X, y):
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}")

    def _new_row(self, n_rows, accuracy):
        row = {"chunk": len(self.log), "n_rows": n_rows, "accuracy": accuracy,
               "cumulative_accuracy": float(self._cumulative.result())}
        if self._rolling is not None:
            row["rolling_accuracy"] = float(self._rolling.result())
        row.update(fit_seconds=float("nan"), retrained=False, memory_bytes=float("nan"))
        self.log.append(row)
        return row
