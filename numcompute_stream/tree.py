import numpy as np

from numcompute_stream.preprocessing import NotFittedError, _check_numeric_2d

__all__ = ["DecisionTreeClassifier"]

_CRITERIA = ("gini", "entropy")
_MIN_GAIN = 1e-12  # a split must beat this, so rounding noise never creates a split


# Private helpers

def _impurity(counts, criterion):
    """
    This is to show how mixed a group is, 0 when everyone agrees.
    Params:
        counts: array shape 
        criterion: "gini" or "entropy"

    Returns:
        array: shape (...), same as counts without the last dimension. An empty group is 0.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1.0)
    if criterion == "gini":
        impurity = 1.0 - (p ** 2).sum(axis=-1)
    else:
        safe = np.where(p > 0, p, 1.0)
        impurity = 0.0 - (p * np.log2(safe)).sum(axis=-1)
    return np.where(total[..., 0] > 0, impurity, 0.0)


def _best_split(X, y, n_classes, criterion, feature_ids=None):
    """
    This is to find the best question "feature <= threshold" for these rows.

    Params:
        X: array shape (n_samples, n_features)
        y: int array shape (n_samples,)
        n_classes: number of classes
        criterion: "gini" or "entropy"
        feature_ids: columns that may be used

    Returns:
        (feature, threshold, gain): the column number, the cut-off and how much the
        impurity drops. (None, None, 0.0) if no question at all helps: fewer than 2 rows, a
        pure group, columns with all the same value, or a gain that is not above zero.
    """
    n = X.shape[0]
    if feature_ids is None:
        feature_ids = np.arange(X.shape[1])
    if n < 2 or len(feature_ids) == 0:
        return None, None, 0.0

    cols = X[:, feature_ids]                                 
    order = np.argsort(cols, axis=0, kind="stable")           
    xs = np.take_along_axis(cols, order, axis=0)              

    onehot = (y[:, None] == np.arange(n_classes)).astype(float)   
    total = onehot.sum(axis=0)                                
    left = np.cumsum(onehot[order], axis=0)[:-1]              
    right = total - left                                      
    n_left = np.arange(1, n)[:, None]                         

    after = (n_left * _impurity(left, criterion) + (n - n_left) * _impurity(right, criterion)) / n
    gain = _impurity(total, criterion) - after                
    gain = np.where(xs[1:] != xs[:-1], gain, -np.inf)        
    gain = gain.T                                             

    best = gain.max()
    if not best > _MIN_GAIN:
        return None, None, 0.0
    flat = np.flatnonzero(gain.ravel() >= best - _MIN_GAIN)[0]    
    column, cut = divmod(int(flat), n - 1)

    low, high = xs[cut, column], xs[cut + 1, column]
    threshold = low / 2 + high / 2                            
    if threshold >= high:                                     
        threshold = low
    return int(feature_ids[column]), float(threshold), float(gain[column, cut])


def _check_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


class DecisionTreeClassifier:
    """
    Decision tree for numeric features that can also learn from a stream.

    The tree is stored as parallel arrays, one entry per node (node 0 is the root).
    Nodes are numbered breadth first, and a left child comes before its right child.

    Params:
        criterion: "gini" or "entropy", default "gini"
        max_depth: int >= 1, default 5. The most questions in a row. The root has depth 0
                   and a node at depth max_depth is always a leaf.
        min_samples_split: int >= 2, default 2. A group with fewer rows is not split.
        max_features: None or int >= 1, default None. None tries every column at each split.
                      A number tries that many columns, chosen at random (at most n_features).
        buffer_size: int >= 1, default 500. partial_fit rebuilds the tree once the buffer
                     holds at least this many rows.
        random_state: None or int, default None. Seed for choosing the columns, so the same
                      seed on the same data gives the same tree.
    Raises:
        TypeError: if a setting has the wrong type (for example max_depth="5" or True)
        ValueError: if a setting is out of range
    """

    def __init__(self, criterion="gini", max_depth=5, min_samples_split=2,
                 max_features=None, buffer_size=500, random_state=None):
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
        _check_int("max_depth", max_depth, 1)
        _check_int("min_samples_split", min_samples_split, 2)
        if max_features is not None:
            _check_int("max_features", max_features, 1)
        _check_int("buffer_size", buffer_size, 1)
        if random_state is not None and (isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer))):
            raise TypeError(f"random_state must be None or an integer, got {random_state!r}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.random_state = random_state
        self._reset_stream()


    def fit(self, X, y):
        """
        Train the tree from scratch on this data, and forget any stream.

        Params:
            X: array shape (n_samples, n_features)
            y: array shape (n_samples,)

        Returns:
            self (so tree.fit(X, y).predict(X) works)

        Raises:
            ValueError: X is not 2-D, has no rows or no columns, has NaN or inf, or y does
                        not have one label per row
            TypeError: X or y is not numeric (y may also be strings)
        """
        X = self._check_X(X)
        y = self._check_y(y, len(X))
        if len(X) == 0:
            raise ValueError("X must have at least one sample, got 0 rows")
        self._reset_stream()
        self.n_features_in_ = X.shape[1]
        self._grow(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """
        Add one chunk to the buffer, and rebuild the tree when the buffer is full enough.

        Params:
            X_chunk: array shape (n_rows, n_features)
            y_chunk: array shape (n_rows,)

        Returns:
            self

        Raises:
            Same as fit, and ValueError if the number of columns is different from the
            earlier chunks. If a chunk is rejected, nothing is changed.
        """
        X = self._check_X(X_chunk, "X_chunk")
        y = self._check_y(y_chunk, len(X))
        if len(X) == 0:
            return self
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but this tree was trained with {self.n_features_in_} features")

        self.n_features_in_ = X.shape[1]
        self._buffer_X.append(X)
        self._buffer_y.append(y)
        self._n_buffered += len(y)

        if self._n_buffered >= self.buffer_size:
            self._grow(np.vstack(self._buffer_X), np.concatenate(self._buffer_y))
            self._reset_stream()
        return self

    def predict(self, X):
        """
        Predict one label for each row.

        Params:
            X: array shape (n_samples, n_features), same rules as X in fit

        Returns:
            array shape (n_samples,)
        Raises:
            NotFittedError: if no row has been seen at all
            ValueError: if X has a different number of columns than the training data, or
                        the same problems as fit
            TypeError: if X is not numeric
        """
        X = self._check_X(X)
        has_tree = hasattr(self, "feature_")
        if not has_tree and self._n_buffered == 0:
            raise NotFittedError("This DecisionTreeClassifier has not seen any data yet; call fit or partial_fit first")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but this tree was trained with {self.n_features_in_} features")

        if not has_tree:
            labels, counts = np.unique(np.concatenate(self._buffer_y), return_counts=True)
            return np.full(len(X), labels[counts.argmax()])  

        node = np.zeros(len(X), dtype=int)                    
        for _ in range(self.max_depth):                       
            rows = np.flatnonzero(self.feature_[node] >= 0)   
            if len(rows) == 0:
                break
            here = node[rows]
            go_left = X[rows, self.feature_[here]] <= self.threshold_[here]
            node[rows] = np.where(go_left, self.left_[here], self.right_[here])
        return self.classes_[self.value_[node].argmax(axis=1)]


    def _check_X(self, X, name="X"):
        #To check numbers only, 2-D, no inf (from preprocessing), and no NaN (trees cannot handle it)
        X = _check_numeric_2d(X, name=name)
        if np.isnan(X).any():
            raise ValueError(f"{name} contains NaN; impute it first (for example with SimpleImputer)")
        return X

    def _check_y(self, y, n_rows):
        #To check 1-D, one label per row, numbers or strings
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if len(y) != n_rows:
            raise ValueError(f"X and y must have the same number of rows, got {n_rows} and {len(y)}")
        if y.dtype.kind not in ("b", "i", "u", "f", "U"):
            raise TypeError(f"y must contain numbers or strings, got dtype {y.dtype}")
        if y.dtype.kind == "f" and np.isnan(y).any():
            raise ValueError("y contains NaN")
        return y

    def _reset_stream(self):
        # empty the buffer
        self._buffer_X = []
        self._buffer_y = []
        self._n_buffered = 0

    def _grow(self, X, y):
        #This where thr actual build happens from ,build the tree from these (already checked) rows and store the arrays
        n_samples, n_features = X.shape
        self.classes_, codes = np.unique(y, return_inverse=True)   # labels as 0 .. n_classes-1
        n_classes = len(self.classes_)

        rng = np.random.default_rng(self.random_state)
        k = n_features if self.max_features is None else min(self.max_features, n_features)

        # one entry per node, the root is node 0 (a leaf until we find a question for it)
        feature, threshold, left, right, value = [-1], [np.nan], [-1], [-1], [None]
        todo = [(0, np.arange(n_samples), 0)]                      # (node, its rows, its depth)

        while todo:
            node, rows, depth = todo.pop(0)                        # from the front: breadth first
            value[node] = np.bincount(codes[rows], minlength=n_classes)

            if depth >= self.max_depth or len(rows) < self.min_samples_split:
                continue                         

            columns = np.sort(rng.choice(n_features, size=k, replace=False)) if k < n_features else None
            best_feature, best_threshold, _ = _best_split(X[rows], codes[rows], n_classes,self.criterion, columns)
            if best_feature is None:
                continue                                          

            go_left = X[rows, best_feature] <= best_threshold
            left_node = len(feature)                              
            right_node = left_node + 1
            for _ in range(2):
                feature.append(-1)
                threshold.append(np.nan)
                left.append(-1)
                right.append(-1)
                value.append(None)

            feature[node], threshold[node] = best_feature, best_threshold
            left[node], right[node] = left_node, right_node
            todo.append((left_node, rows[go_left], depth + 1))
            todo.append((right_node, rows[~go_left], depth + 1))

        self.feature_ = np.array(feature, dtype=int)
        self.threshold_ = np.array(threshold, dtype=float)
        self.left_ = np.array(left, dtype=int)
        self.right_ = np.array(right, dtype=int)
        self.value_ = np.array(value, dtype=int)
        self.n_nodes_ = len(feature)
