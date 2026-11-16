"""
numcompute_stream.metrics

Classification and regression metrics, numpy only.

- labels are non-empty 1-D numeric or string arrays (no NaN / inf)
- confusion matrix: rows = actual labels, columns = predicted labels
- precision / recall / f1 are binary by default (positive label 1)
"""
import warnings

import numpy as np

__all__ = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'mse',
           'roc_curve', 'auc','StreamingConfusionMatrix', 'StreamingAccuracy', 
           'StreamingPrecision', 'StreamingRecall', 'StreamingF1', 'StreamingAUC']


def _check_labels(y, name):
    arr = np.asarray(y)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f'{name} must be a nonempty 1-D array')
    if arr.dtype.kind not in 'biufU':
        raise TypeError(f'{name} must contain numeric or string labels')
    if arr.dtype.kind in 'biuf' and not np.isfinite(arr).all():
        raise ValueError(f'{name} must not contain NaN or infinity')
    return arr


def _check_pair(y_true, y_pred):
    t, p = _check_labels(y_true, 'y_true'), _check_labels(y_pred, 'y_pred')
    if t.shape != p.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if (t.dtype.kind == 'U') != (p.dtype.kind == 'U'):
        raise TypeError('actual and predicted labels must both be numbers or both strings')
    return t, p


def _encode(y_true, y_pred, labels=None):
    # Map labels to integer positions 0..C-1 so np.bincount can count them.
    t, p = _check_pair(y_true, y_pred)
    seen = np.unique(np.concatenate((t, p)))
    if labels is None:
        classes = seen
    else:
        classes = _check_labels(labels, 'labels')
        if np.unique(classes).size != classes.size:
            raise ValueError('labels must not contain duplicates')
        if not np.isin(seen, classes).all():
            raise ValueError('labels must include every observed label')
    # searchsorted needs sorted classes; `order` maps back to the user's order
    order = np.argsort(classes, kind='stable')
    t_idx = order[np.searchsorted(classes[order], t)]
    p_idx = order[np.searchsorted(classes[order], p)]
    return classes, t_idx, p_idx


def accuracy(y_true, y_pred):
    """Fraction of correct predictions, a float in [0, 1].
    Raises ValueError for empty, mismatched, non-1-D or NaN/inf labels."""
    t, p = _check_pair(y_true, y_pred)
    return float(np.mean(t == p))


def confusion_matrix(y_true, y_pred, labels=None):
    """Counts of actual (rows) vs predicted (columns) labels, shape (C, C).

    labels: optional order for the rows/columns (no duplicates, must include
    every label in the data). Default is the sorted labels found in the data.
    """
    classes, t_idx, p_idx = _encode(y_true, y_pred, labels)
    c = classes.size
    # turn each (actual, predicted) pair into one number, then count in one go
    return np.bincount(t_idx * c + p_idx, minlength=c * c).reshape(c, c)


def _score(y_true, y_pred, average, pos_label, zero_division, kind):
    # Shared by precision/recall/f1. Per class we need tp, support (how many
    # really are that class) and predicted (how many were predicted as it):
    #   precision = tp / predicted, recall = tp / support,
    #   f1 = 2tp / (support + predicted)
    if average not in ('binary', 'micro', 'macro', 'weighted', None):
        raise ValueError('average must be binary, micro, macro, weighted or None')
    if zero_division not in (0, 1):
        raise ValueError('zero_division must be 0 or 1')

    classes, t_idx, p_idx = _encode(y_true, y_pred)
    c = classes.size
    support = np.bincount(t_idx, minlength=c).astype(float)
    predicted = np.bincount(p_idx, minlength=c).astype(float)
    tp = np.bincount(t_idx[t_idx == p_idx], minlength=c).astype(float)

    if average == 'binary':
        if c > 2:
            raise ValueError('binary averaging needs at most two classes')
        mask = classes == pos_label
        if c == 2 and not mask.any():
            raise ValueError('pos_label is not one of the observed classes')
        tp, support, predicted = tp[mask].sum(), support[mask].sum(), predicted[mask].sum()
    elif average == 'micro':
        tp, support, predicted = tp.sum(), support.sum(), predicted.sum()

    top, bottom = {'precision': (tp, predicted),
                   'recall': (tp, support),
                   'f1': (2 * tp, support + predicted)}[kind]
    # divide, using zero_division wherever the bottom is 0
    result = np.full(np.shape(bottom), float(zero_division))
    np.divide(top, bottom, out=result, where=bottom != 0)

    if average is None:
        return result
    if average == 'weighted':
        return float(np.sum(result * support / support.sum()))
    return float(result.mean())  # macro (for binary/micro it's a single number)


def precision(y_true, y_pred, *, average='binary', pos_label=1, zero_division=0):
    """TP / (TP + FP). average: 'binary', 'micro', 'macro', 'weighted' or None
    (one score per class, sorted order). zero_division: 0 or 1, used when the
    denominator is 0. Returns a float, or an array when average=None."""
    return _score(y_true, y_pred, average, pos_label, zero_division, 'precision')


def recall(y_true, y_pred, *, average='binary', pos_label=1, zero_division=0):
    """TP / (TP + FN). Parameters are the same as precision()."""
    return _score(y_true, y_pred, average, pos_label, zero_division, 'recall')


def f1(y_true, y_pred, *, average='binary', pos_label=1, zero_division=0):
    """2TP / (2TP + FP + FN). Parameters are the same as precision()."""
    return _score(y_true, y_pred, average, pos_label, zero_division, 'f1')


def mse(y_true, y_pred, *, axis=None, nan_policy='raise'):
    """Mean squared error (same shapes required, no broadcasting).

    nan_policy: 'raise' (default), 'omit' (drop pairs where either value is
    NaN) or 'propagate' (NaN in -> NaN out). Too-big results come back as inf.
    """
    if nan_policy not in ('raise', 'omit', 'propagate'):
        raise ValueError('invalid nan_policy')
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.dtype.kind not in 'biuf' or p.dtype.kind not in 'biuf':
        raise TypeError('regression inputs must contain real numbers')
    if t.shape != p.shape or t.size == 0:
        raise ValueError('regression inputs must be nonempty and have equal shapes')

    # convert to float first so small int types can't wrap around
    t, p = t.astype(float), p.astype(float)
    if np.isinf(t).any() or np.isinf(p).any():
        raise ValueError('regression inputs must not contain infinity')
    if nan_policy == 'raise' and (np.isnan(t) | np.isnan(p)).any():
        raise ValueError('regression inputs contain NaN')

    # a NaN in either array makes the squared error NaN, 'omit' then skips it
    with warnings.catch_warnings(), np.errstate(over='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN mean -> NaN
        squares = (t - p) ** 2
        result = np.nanmean(squares, axis=axis) if nan_policy == 'omit' \
            else np.mean(squares, axis=axis)
    return float(result) if np.ndim(result) == 0 else result


def roc_curve(y_true, y_score, *, pos_label=1):
    """Bonus: ROC curve for binary labels.

    Returns (fpr, tpr, thresholds). The first point is (0, 0) at threshold inf,
    then one point per distinct score (tied scores share a point).
    A sample is predicted positive when score >= threshold.
    """
    t = _check_labels(y_true, 'y_true')
    score = np.asarray(y_score)
    if score.dtype.kind not in 'biuf':
        raise TypeError('y_score must be real numeric')
    score = score.astype(float)
    if score.shape != t.shape or not np.isfinite(score).all():
        raise ValueError('y_score must be finite and match y_true shape')
    if np.unique(t).size != 2 or not np.any(t == pos_label):
        raise ValueError('ROC needs two classes and an observed pos_label')

    # walk from the highest score to the lowest
    order = np.argsort(-score, kind='stable')
    sorted_scores = score[order]
    is_pos = t[order] == pos_label

    # a threshold only matters where the score changes: keep the last sample
    # of each run of equal scores
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), t.size - 1]
    tp = np.cumsum(is_pos)[ends]
    fp = ends + 1 - tp
    return (np.r_[0.0, fp / (~is_pos).sum()],
            np.r_[0.0, tp / is_pos.sum()],
            np.r_[np.inf, sorted_scores[ends]])


def auc(x, y):
    """Bonus: area under a curve (trapezoid rule). x must be non-decreasing,
    both arrays finite, 1-D, equal length, at least 2 points."""
    x, y = np.asarray(x), np.asarray(y)
    if x.dtype.kind not in 'biuf' or y.dtype.kind not in 'biuf':
        raise TypeError('curve coordinates must be real numeric')
    x, y = x.astype(float), y.astype(float)
    if x.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError('x and y must be equal-length 1-D arrays with at least two points')
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or np.any(np.diff(x) < 0):
        raise ValueError('coordinates must be finite and x nondecreasing')
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))


# Streaming metrics added
#
# Accuracy, precision, recall and F1 only need the confusion matrix, so each chunk is turned into
# a small table of counts and added to a running total. AUC needs scores, so it counts the scores
# of each class in fixed bins instead. Neither one stores the chunks themselves.

class _Windowed:
    """Keeps one table of counts per chunk, as a single running total or for the last `window` chunks.

    Params:
        window: None to count every chunk, or an int >= 1 to count only the last `window` chunks
    """

    def __init__(self, window=None):
        if window is not None and (isinstance(window, (bool, np.bool_))
                                   or not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError('window must be None or an integer >= 1')
        self.window = window
        self.reset()

    def reset(self):
        """Forget everything seen so far. Returns self."""
        self._tables = []
        return self

    def _add(self, table):
        self._tables.append(table)
        if self.window is None:
            self._tables = [sum(self._tables)]            # one running total
        else:
            self._tables = self._tables[-self.window:]    # the last `window` chunks

    def _total(self):
        return sum(self._tables)


class StreamingConfusionMatrix(_Windowed):
    """Confusion matrix that adds up over chunks. Rows are actual labels, columns are predicted labels.

    Params:
        window: None (count every chunk) or an int >= 1 (count only the last `window` chunks)
    """

    def reset(self):
        super().reset()
        self.classes_ = None
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of actual and predicted labels. Returns self so calls can be chained.

        Raises:
            ValueError: if the labels are not 1-D, differ in length, or contain NaN or inf
            TypeError: if the labels are not numbers or strings, or numbers and strings are mixed
            An empty chunk is ignored. If a chunk is rejected, nothing is changed.
        """
        if np.size(y_true_chunk) == 0 and np.size(y_pred_chunk) == 0:
            return self
        t, p = _check_pair(y_true_chunk, y_pred_chunk)
        classes = np.unique(np.concatenate((t, p)))
        if self.classes_ is not None:
            if (self.classes_.dtype.kind == 'U') != (classes.dtype.kind == 'U'):
                raise TypeError('labels must be all numbers or all strings, like the earlier chunks')
            classes = np.union1d(self.classes_, classes)
        table = confusion_matrix(t, p, labels=classes)

        # nothing can fail after this point
        if self.classes_ is not None and classes.size != self.classes_.size:
            where = np.searchsorted(classes, self.classes_)   # the old classes' places in the new list
            grown = []
            for old in self._tables:                          # copy each old table into a bigger one
                new = np.zeros((classes.size, classes.size), dtype=old.dtype)
                new[np.ix_(where, where)] = old
                grown.append(new)
            self._tables = grown
        self.classes_ = classes
        self._add(table)
        return self

    def _matrix(self):
        if not self._tables:
            return np.zeros((0, 0), dtype=np.int64)
        return self._total()

    def result(self):
        return self._matrix()


class StreamingAccuracy(StreamingConfusionMatrix):
    """Accuracy over the stream (or the last `window` chunks). Same update/reset as StreamingConfusionMatrix.

    result() is correct predictions divided by all predictions, NaN before any data.
    """

    def result(self):
        matrix = self._matrix()
        if matrix.sum() == 0:
            return float('nan')
        return float(np.trace(matrix) / matrix.sum())


class _StreamingScore(StreamingConfusionMatrix):
    # Shared by precision, recall and f1. Same maths as the functions above, but the counts
    # come from the running confusion matrix instead of from the labels.
    kind = None

    def __init__(self, *, average='binary', pos_label=1, zero_division=0, window=None):
        if average not in ('binary', 'micro', 'macro', 'weighted', None):
            raise ValueError('average must be binary, micro, macro, weighted or None')
        if zero_division not in (0, 1):
            raise ValueError('zero_division must be 0 or 1')
        self.average, self.pos_label, self.zero_division = average, pos_label, zero_division
        super().__init__(window)

    def result(self):
        matrix = self._matrix().astype(float)
        if matrix.sum() == 0:
            return np.array([]) if self.average is None else float('nan')
        classes = self.classes_
        tp, support, predicted = np.diag(matrix), matrix.sum(axis=1), matrix.sum(axis=0)

        if self.average == 'binary':
            if classes.size > 2:
                raise ValueError('binary averaging needs at most two classes')
            mask = classes == self.pos_label
            if classes.size == 2 and not mask.any():
                raise ValueError('pos_label is not one of the observed classes')
            tp, support, predicted = tp[mask].sum(), support[mask].sum(), predicted[mask].sum()
        elif self.average == 'micro':
            tp, support, predicted = tp.sum(), support.sum(), predicted.sum()

        top, bottom = {'precision': (tp, predicted),
                       'recall': (tp, support),
                       'f1': (2 * tp, support + predicted)}[self.kind]
        score = np.full(np.shape(bottom), float(self.zero_division))
        np.divide(top, bottom, out=score, where=bottom != 0)

        if self.average is None:
            return score
        if self.average == 'weighted':
            return float(np.sum(score * support / support.sum()))
        return float(score.mean())


class StreamingPrecision(_StreamingScore):
    kind = 'precision'


class StreamingRecall(_StreamingScore):
    kind = 'recall'


class StreamingF1(_StreamingScore):
    kind = 'f1'


class StreamingAUC(_Windowed):
    """Area under the ROC curve that adds up over chunks, for two classes (positive against the rest).

    Params:
        bins: int >= 1, number of score bins, default 100
        score_range: (low, high) the scores must be inside, default (0, 1)
        pos_label: the label that counts as positive, default 1
        window: None (count every chunk) or an int >= 1 (count only the last `window` chunks)
    """

    def __init__(self, *, bins=100, score_range=(0.0, 1.0), pos_label=1, window=None):
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError('bins must be a positive integer')
        r = np.asarray(score_range, dtype=float)
        if r.shape != (2,) or not np.isfinite(r).all() or r[0] >= r[1]:
            raise ValueError('score_range must be a finite (low, high) with low < high')
        self.bins, self.score_range, self.pos_label = int(bins), (r[0], r[1]), pos_label
        self.edges = np.linspace(r[0], r[1], self.bins + 1)
        super().__init__(window)

    def update(self, y_true_chunk, y_score_chunk):
        """Add one chunk of labels and scores. Returns self so calls can be chained.

        Raises:
            ValueError: if the labels are not 1-D or contain NaN or inf, the scores do not match
                        the labels in shape, or a score is NaN, inf or outside score_range
            TypeError: if the labels or scores are not numbers (labels may also be strings)
            An empty chunk is ignored. If a chunk is rejected, nothing is changed.
        """
        if np.size(y_true_chunk) == 0 and np.size(y_score_chunk) == 0:
            return self
        t = _check_labels(y_true_chunk, 'y_true')
        score = np.asarray(y_score_chunk)
        if score.dtype.kind not in 'biuf':
            raise TypeError('y_score must be real numeric')
        score = score.astype(float)
        if score.shape != t.shape or not np.isfinite(score).all():
            raise ValueError('y_score must be finite and match y_true shape')
        if score.min() < self.edges[0] or score.max() > self.edges[-1]:
            raise ValueError(f'y_score must be inside score_range {self.score_range}')

        is_pos = t == self.pos_label
        pos, _ = np.histogram(score[is_pos], bins=self.edges)
        neg, _ = np.histogram(score[~is_pos], bins=self.edges)
        self._add(np.stack([pos, neg]))
        return self

    def result(self):

        if not self._tables:
            return float('nan')
        pos, neg = self._total()
        if pos.sum() == 0 or neg.sum() == 0:
            return float('nan')
        below = np.cumsum(neg) - neg                   # other rows with a lower score bin
        return float((pos * (below + neg / 2)).sum() / (pos.sum() * neg.sum()))
