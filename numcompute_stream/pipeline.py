import copy
import numpy as np

__all__ = ["Pipeline"]

class Pipeline:
    """
    Input:
        steps: list of (name, object) pairs

    Attributes:
        steps: list of (name, object) pairs
        named_steps: dict, name -> object

    Error Raised:
        ValueError: if steps is empty
        TypeError: if a step that is not the last one has no transform method
        (any other missing method gives python's own AttributeError)
    """
    def __init__(self, steps):

        # Validating input steps during initialzation
        if len(steps) == 0:
            raise ValueError("steps must not be empty")
        for name, step in steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"step {name!r} is not the last step, so it needs a transform method")

        self.steps = list(steps) # Better to create a copy
        self.named_steps = dict(steps)
        self._widths = None

    def fit(self, X, y=None):

        self._widths = None
        if y is not None and len(X) != len(y):
            raise ValueError(f"X and y must have the same number of rows, got {len(X)} and {len(y)}")

        data = X
        for name, step in self.steps[:-1]:
            step.fit(data)
            data = step.transform(data)

        # Getting the last step , usually predict
        last = self.steps[-1][1]
        if hasattr(last, "predict") and y is not None: #as we dont have a model yet
            last.fit(data, y)  # a model learns from y(answers)
        else:
            last.fit(data) # just fits if there is no model
        return self

    def transform(self, X):
        data = X
        for name, step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        data = X
        for name, step in self.steps[:-1]: #run every step except last , which is usually model training
            data = step.transform(data)
        return self.steps[-1][1].predict(data)

    def partial_fit(self, X_chunk, y_chunk=None):
        """
        Learn from one chunk: every step updates itself, then passes the chunk on.

        Input:
            X_chunk: array shape (n_rows, n_features). An empty chunk (0 rows) is ignored.
            y_chunk: array shape (n_rows,), needed when the last step is a model (has predict)

        Returns: self

        Error Raised:
            TypeError: if a step has no partial_fit method
            ValueError: if X_chunk and y_chunk have different numbers of rows, y_chunk is missing,
                        or a step now gives a different number of columns than for the earlier
                        chunks
        """
        for name, step in self.steps:
            if not hasattr(step, "partial_fit"):
                raise TypeError(f"step {name!r} has no partial_fit method, so the pipeline cannot learn from a stream")
        if y_chunk is not None and len(X_chunk) != len(y_chunk):
            raise ValueError(f"X and y must have the same number of rows, got {len(X_chunk)} and {len(y_chunk)}")
        last = self.steps[-1][1]
        has_model = hasattr(last, "predict")
        if has_model and y_chunk is None:
            raise ValueError("y_chunk is required when the last step is a model")
        if len(X_chunk) == 0:
            return self

        # just in case, keep a copy of what the transformers know now, so a rejected chunk can be undone
        saved = [copy.deepcopy(step.__dict__) for name, step in self.steps[:-1]]
        widths = {}
        try:
            data = X_chunk
            for name, step in self.steps[:-1]:
                step.partial_fit(data)       # learn from this chunk first
                data = step.transform(data)  # then pass it on, using what was learned from previous chunks
                widths[name] = np.shape(data)[-1]
                if self._widths is not None and self._widths[name] != widths[name]:
                    raise ValueError(f"step {name!r} now gives {widths[name]} columns but gave {self._widths[name]} for the earlier chunks; a streaming pipeline needs the same number of columns every time")
            if has_model:
                last.partial_fit(data, y_chunk)
            else:
                last.partial_fit(data)
        except Exception:
            for (name, step), state in zip(self.steps, saved):   # put all transformers back
                step.__dict__.clear()
                step.__dict__.update(state)
            raise
        self._widths = widths
        return self

