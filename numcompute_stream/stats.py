"""
numcompute_stream.stats

Descriptive statistics and a streaming mean/variance, numpy only.

- data must be numbers, inf is not allowed, empty input raises ValueError
- nan_policy: 'propagate' (NaN in -> NaN out), 'omit' (skip NaN) or 'raise'
- axis=None reduces everything, keepdims=True keeps reduced axes as length 1
"""
import warnings

import numpy as np

__all__ = ['mean', 'median', 'var', 'variance', 'std', 'min', 'max',
           'histogram', 'quantile', 'Welford', 'StreamingStats', 'RunningStats',
           'StreamingHistogram']


def _check_data(data, nan_policy):
    # check the input and return it as a float array
    if nan_policy not in ('propagate', 'omit', 'raise'):
        raise ValueError("nan_policy must be 'propagate', 'omit' or 'raise'")
    arr = np.asarray(data)
    if arr.dtype.kind not in 'biuf':
        raise TypeError('data must contain real numbers')
    arr = arr.astype(float)
    if arr.size == 0:
        raise ValueError('data must not be empty')
    if np.isinf(arr).any():
        raise ValueError('data must not contain infinity')
    if nan_policy == 'raise' and np.isnan(arr).any():
        raise ValueError('data contains NaN')
    return arr


def _check_ddof(ddof):
    # True/False count as ints in python, so check for bool first
    if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, (int, np.integer)):
        raise TypeError('ddof must be a non-negative integer')
    if ddof < 0:
        raise ValueError('ddof must be non-negative')


def _apply(name, data, axis, nan_policy, keepdims, **kwargs):
    # run a numpy function, using the nan version (np.nanmean etc.) for 'omit'
    arr = _check_data(data, nan_policy)
    func = getattr(np, ('nan' if nan_policy == 'omit' else '') + name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slice -> NaN is fine
        return func(arr, axis=axis, keepdims=keepdims, **kwargs)


def _spread(name, data, axis, ddof, nan_policy, keepdims):
    # var and std: same as numpy, but NaN when there are too few values
    _check_ddof(ddof)
    result = _apply(name, data, axis, nan_policy, keepdims, ddof=ddof)
    arr = np.asarray(data, dtype=float)
    valid = ~np.isnan(arr) if nan_policy == 'omit' else np.ones(arr.shape, dtype=bool)
    count = valid.sum(axis=axis, keepdims=keepdims)
    return np.where(count > ddof, result, np.nan)


def mean(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Arithmetic mean along axis. Raises TypeError for non-numbers and
    ValueError for empty/inf data, a bad policy, or NaN with 'raise'."""
    return _apply('mean', data, axis, nan_policy, keepdims)


def median(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Median along axis (middle two are averaged for even counts)."""
    return _apply('median', data, axis, nan_policy, keepdims)


def var(data, axis=None, *, ddof=0, nan_policy='propagate', keepdims=False):
    """Variance, divided by (count - ddof). ddof=1 gives the sample variance.
    Returns NaN when count <= ddof."""
    return _spread('var', data, axis, ddof, nan_policy, keepdims)


variance = var


def std(data, axis=None, *, ddof=0, nan_policy='propagate', keepdims=False):
    """Standard deviation, same ddof rules as var()."""
    return _spread('std', data, axis, ddof, nan_policy, keepdims)


def min(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Minimum along axis."""
    return _apply('min', data, axis, nan_policy, keepdims)


def max(data, axis=None, *, nan_policy='propagate', keepdims=False):
    """Maximum along axis."""
    return _apply('max', data, axis, nan_policy, keepdims)


def quantile(data, q, axis=None, *, interpolation='linear', nan_policy='omit',
             keepdims=False):
    """Quantiles of data. q is a fraction between 0 and 1 (not 0-100),
    either a single number or a 1-D array.

    interpolation: 'linear', 'lower', 'higher' or 'midpoint'.
    The result has shape q.shape + the reduced shape of data.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim > 1 or q.size == 0 or not np.all((q >= 0) & (q <= 1)):
        raise ValueError('q must be a nonempty scalar or 1-D array in [0, 1]')
    if interpolation not in ('linear', 'lower', 'higher', 'midpoint'):
        raise ValueError('unsupported interpolation method')
    return _apply('quantile', data, axis, nan_policy, keepdims,
                  q=q, method=interpolation)


def histogram(data, bins=10, range=None, *, nan_policy='omit', density=False):
    """Count values in bins. Data is flattened, NaN is skipped by default.

    bins: a positive int, or a 1-D array of strictly increasing edges.
    range: (low, high), only used when bins is an int.
    density=True scales the counts so the histogram integrates to 1.
    The last bin includes its right edge. Returns (counts, edges).
    """
    if nan_policy not in ('omit', 'raise'):
        raise ValueError("histogram nan_policy must be 'omit' or 'raise'")
    arr = _check_data(data, nan_policy).ravel()
    arr = arr[~np.isnan(arr)]

    if np.ndim(bins) == 0:
        # number of bins
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError('bins must be a positive integer or increasing edges')
        if arr.size == 0 and range is None:
            raise ValueError('all-NaN data needs explicit edges or a range')
    else:
        # explicit edges
        bins = np.asarray(bins, dtype=float)
        if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all() \
                or not np.all(np.diff(bins) > 0):
            raise ValueError('bin edges must be finite and strictly increasing')

    if range is not None:
        r = np.asarray(range, dtype=float)
        if r.shape != (2,) or not np.isfinite(r).all() or r[0] >= r[1]:
            raise ValueError('range must be a finite (low, high) with low < high')

    counts, edges = np.histogram(arr, bins=bins, range=range)
    if density:
        if counts.sum() == 0:
            raise ValueError('density is undefined when no values fall in the bins')
        counts = counts / counts.sum() / np.diff(edges)
    return counts, edges


class Welford:
    """Running mean and variance without keeping the data (Welford's method).

    nan_policy: 'omit' (default) skips NaN, 'raise' throws ValueError.
    update() takes a single number or a 1-D array. For several columns use
    one Welford object per column.
    """

    def __init__(self, nan_policy='omit'):
        if nan_policy not in ('omit', 'raise'):
            raise ValueError("nan_policy must be 'omit' or 'raise'")
        self.nan_policy = nan_policy
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0  # sum of squared differences from the mean

    def update(self, values):
        """Add a number or a 1-D batch. Returns self so calls can be chained.
        If the input is rejected nothing is changed."""
        arr = np.asarray(values)
        if arr.dtype.kind not in 'biuf':
            raise TypeError('values must contain real numbers')
        if arr.ndim > 1:
            raise ValueError('values must be a scalar or 1-D')
        if arr.size == 0:
            return self
        arr = _check_data(arr, self.nan_policy).ravel()
        arr = arr[~np.isnan(arr)]

        # merge the batch into the running totals
        n = arr.size
        if n == 0:
            return self
        total = self.count + n
        delta = arr.mean() - self._mean
        self._m2 += np.sum((arr - arr.mean()) ** 2) + delta ** 2 * self.count * n / total
        self._mean += delta * n / total
        self.count = total
        return self

    @property
    def mean(self):
        """Current mean, NaN if nothing was added yet."""
        return float(self._mean) if self.count else float('nan')

    def variance(self, ddof=0):
        """Variance, NaN when count <= ddof."""
        _check_ddof(ddof)
        if self.count <= ddof:
            return float('nan')
        return float(self._m2 / (self.count - ddof))

    def std(self, ddof=0):
        """Standard deviation, NaN when count <= ddof."""
        return float(np.sqrt(self.variance(ddof)))


StreamingStats = Welford

class RunningStats:
    """Running mean and variance for every column of a table, one chunk at a time.

    Params:
        nan_policy: 'omit' (default) skips NaN separately in each column,
                    'raise' throws ValueError if a chunk has NaN
    """

    def __init__(self, nan_policy='omit'):
        if nan_policy not in ('omit', 'raise'):
            raise ValueError("nan_policy must be 'omit' or 'raise'")
        self.nan_policy = nan_policy
        self.reset()

    def reset(self):
        """Forget everything. Returns self."""
        self.n_features = None
        self.count = np.zeros(0, dtype=np.int64)
        self._mean = np.zeros(0)
        self._m2 = np.zeros(0)
        return self

    def update(self, X_chunk):
        """Add a chunk. Returns self so calls can be chained.

        Params:
            X_chunk: numbers, shape (n_rows, n_features). A 1-D input is one
                     column of n_rows values, a single number is one value.
                     Empty chunks are ignored.

        Raises:
            TypeError: if X_chunk is not numeric
            ValueError: if X_chunk has more than 2 dimensions or no columns, contains inf,
                        contains NaN with nan_policy='raise', or has a different
                        number of columns than the earlier chunks
            If a chunk is rejected, nothing is changed.
        """
        X = np.asarray(X_chunk)
        if X.dtype.kind not in 'biuf':
            raise TypeError('chunk must contain real numbers')
        X = X.astype(float)
        if X.ndim > 2:
            raise ValueError('chunk must be a number, 1-D or 2-D, got shape ' + str(X.shape))
        if X.ndim == 2 and X.shape[1] == 0:
            raise ValueError('chunk must have at least one column')
        X = X.reshape(-1, X.shape[-1] if X.ndim == 2 else 1)
        if X.shape[0] == 0:
            return self                                    # an empty chunk changes nothing
        if np.isinf(X).any():
            raise ValueError('chunk must not contain infinity')
        valid = ~np.isnan(X)
        if self.nan_policy == 'raise' and not valid.all():
            raise ValueError('chunk contains NaN')
        if self.n_features is not None and X.shape[1] != self.n_features:
            raise ValueError(f'chunk has {X.shape[1]} columns but earlier chunks had {self.n_features}')

        # nothing can fail after this point, so start from zeros on the first chunk
        d = X.shape[1]
        if self.n_features is None:
            self.n_features = d
            self.count, self._mean, self._m2 = np.zeros(d, dtype=np.int64), np.zeros(d), np.zeros(d)

        # summary of this chunk alone, per column (NaN is left out)
        n_b = valid.sum(axis=0)
        mean_b = np.divide(np.where(valid, X, 0.0).sum(axis=0), n_b, out=np.zeros(d), where=n_b > 0)
        m2_b = (np.where(valid, X - mean_b, 0.0) ** 2).sum(axis=0)

        # merge it into the running totals
        total = self.count + n_b
        share = np.divide(n_b, total, out=np.zeros(d), where=total > 0)   # n_b / total
        delta = mean_b - self._mean
        self._m2 = self._m2 + m2_b + delta ** 2 * self.count * share
        self._mean = self._mean + delta * share
        self.count = total
        return self

    update_stats = update          

    @property    
    def mean(self):
        """Mean of each column, NaN where no number was seen yet."""
        return np.where(self.count > 0, self._mean, np.nan)

    def variance(self, ddof=0):
        """Variance of each column, divided by (count - ddof). NaN when count <= ddof."""
        _check_ddof(ddof)
        divisor = self.count - ddof
        return np.divide(self._m2, divisor, out=np.full(self.count.shape, np.nan), where=divisor > 0)

    def std(self, ddof=0):
        """Standard deviation of each column, same ddof rules as variance()."""
        return np.sqrt(self.variance(ddof))


class StreamingHistogram:
    """Histogram with fixed bins that is updated one chunk at a time, plus approximate quantiles.
    
    Params:
        bins: int >= 1, number of equal-width bins, default 10
        range: None or (low, high) with low < high, both finite
    """

    def __init__(self, bins=10, range=None):
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError('bins must be a positive integer')
        if range is not None:
            r = np.asarray(range, dtype=float)
            if r.shape != (2,) or not np.isfinite(r).all() or r[0] >= r[1]:
                raise ValueError('range must be a finite (low, high) with low < high')
        self.bins = int(bins)
        self.range = range
        self.reset()

    def reset(self):
        #Forget all counts
        self.edges = None if self.range is None else np.linspace(self.range[0], self.range[1], self.bins + 1)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.n_outside = 0
        return self

    def update(self, X_chunk):
        """Add a chunk of numbers (any shape, it is flattened). Returns self so calls can be chained.

        Raises:
            TypeError: if X_chunk is not numeric
            ValueError: if X_chunk contains inf
            If a chunk is rejected, nothing is changed.
        """
        values = np.asarray(X_chunk)
        if values.dtype.kind not in 'biuf':
            raise TypeError('chunk must contain real numbers')
        values = values.astype(float).ravel()
        if np.isinf(values).any():
            raise ValueError('chunk must not contain infinity')
        values = values[~np.isnan(values)]
        if values.size == 0:
            return self                                    # nothing to count

        if self.edges is None:                             # first chunk with numbers decides the range
            low, high = values.min(), values.max()
            if low == high:
                low, high = low - 0.5, high + 0.5          # all the same value: make a range around it
            self.edges = np.linspace(low, high, self.bins + 1)

        counts, _ = np.histogram(values, bins=self.edges)  # the last bin includes its right edge
        self.counts = self.counts + counts
        self.n_outside += values.size - counts.sum()
        return self

    update_stats = update          # the name used in the assignment spec

    def quantile(self, q):
        """Approximate quantile(s) from the counts. q is a fraction in [0, 1], a number or a 1-D array.

        It finds the bin where the running total passes q * total, then moves linearly inside
        that bin, so the answer is off by less than one bin width. NaN if nothing was counted.
        """
        q = np.asarray(q, dtype=float)
        if q.ndim > 1 or q.size == 0 or not np.all((q >= 0) & (q <= 1)):
            raise ValueError('q must be a nonempty scalar or 1-D array in [0, 1]')
        total = self.counts.sum()
        if total == 0:
            return np.full(q.shape, np.nan)

        running = np.cumsum(self.counts)                   # values up to and including each bin
        last = np.flatnonzero(self.counts)[-1]             # last bin that has values
        target = q * total
        i = np.minimum(np.searchsorted(running, target, side='right'), last)   # the bin holding the target
        before = running[i] - self.counts[i]               # values in earlier bins
        fraction = (target - before) / self.counts[i]      # how far through the bin
        return self.edges[i] + fraction * (self.edges[i + 1] - self.edges[i])

    def median(self):
        return float(self.quantile(0.5))
