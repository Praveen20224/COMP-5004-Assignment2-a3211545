"""
Benchmarks for NumCompute-Stream.

Table 1: loop against vectorised
    - best-split search of the tree (a Python loop with running counts, against _best_split)
    - tree predict (one row at a time, against all rows walking the tree together)
Table 2: different buffer_size values
    The same synthetic stream, whose rule changes halfway (drift), goes through
    scaler -> tree with test-then-train for each buffer_size.

Run from the repository folder:
    python benchmark/stream_benchmarks.py
    python benchmark/stream_benchmarks.py --markdown          (tables ready for the report)
    python benchmark/stream_benchmarks.py --no-plot           (do not save the image)
Times are the best of a few runs, so they change a little from run to run; the accuracies do not.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                    # save images without opening a window
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so the package is found from any folder

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier, _best_split
from numcompute_stream.visualise import compare_models

SPLIT_SIZES = (100, 1000, 10000)                  # rows for the best-split search (5 columns)
PREDICT_SIZES = (100, 1000, 10000, 100000)        # rows for predict
BUFFER_SIZES = (50, 100, 250, 500, 1000)
N_CHUNKS, CHUNK_ROWS, DRIFT_CHUNK = 40, 50, 20    # 2000 rows, the rule changes at chunk 20


# Timing

def best_time(function, repeats=3):
    # the fastest of a few runs, in seconds
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


# Table 1: loop against vectorised

def loop_best_split(X, y, n_classes):
    # the same search as _best_split (gini), but with Python loops and running counts
    n, n_features = X.shape
    total = [int(np.sum(y == c)) for c in range(n_classes)]

    def gini(counts, size):
        return 1.0 - sum((c / size) ** 2 for c in counts) if size else 0.0

    parent = gini(total, n)
    best_gain, best_feature, best_threshold = 0.0, None, None
    for feature in range(n_features):
        order = np.argsort(X[:, feature], kind="stable").tolist()
        values = X[:, feature].tolist()
        labels = y.tolist()
        left = [0] * n_classes
        for i in range(n - 1):
            left[labels[order[i]]] += 1
            low, high = values[order[i]], values[order[i + 1]]
            if low == high:
                continue                                   # no cut between equal values
            right = [t - l for t, l in zip(total, left)]
            gain = parent - ((i + 1) * gini(left, i + 1) + (n - i - 1) * gini(right, n - i - 1)) / n
            if gain > best_gain + 1e-12:
                best_gain, best_feature, best_threshold = gain, feature, low / 2 + high / 2
    return best_feature, best_threshold, best_gain


def loop_predict(tree, X):
    # walk the tree one row at a time
    out = []
    for row in X:
        node = 0
        while tree.feature_[node] >= 0:
            go_left = row[tree.feature_[node]] <= tree.threshold_[node]
            node = tree.left_[node] if go_left else tree.right_[node]
        out.append(tree.classes_[tree.value_[node].argmax()])
    return np.array(out)


def bench_loop_vs_vectorised():
    rng = np.random.default_rng(0)
    rows = []

    for n in SPLIT_SIZES:
        X = rng.normal(size=(n, 5))
        y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)) > 0).astype(int)
        slow = best_time(lambda: loop_best_split(X, y, 2))
        fast = best_time(lambda: _best_split(X, y, 2, "gini"))
        a, b = loop_best_split(X, y, 2), _best_split(X, y, 2, "gini")
        same = a[0] == b[0] and np.isclose(a[1], b[1]) and np.isclose(a[2], b[2])
        rows.append(("best split", n, slow, fast, same))

    X = rng.normal(size=(2000, 5))
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=2000)) > 0).astype(int)
    tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
    for n in PREDICT_SIZES:
        X_new = rng.normal(size=(n, 5))
        slow = best_time(lambda: loop_predict(tree, X_new))
        fast = best_time(lambda: tree.predict(X_new))
        same = np.array_equal(loop_predict(tree, X_new), tree.predict(X_new))
        rows.append(("tree predict", n, slow, fast, same))
    return rows


# Table 2: different buffer_size values

def make_stream(seed=0):
    # 5 numeric columns. The label is (x0 + x1 > 0) before the drift and (x0 - x1 > 0) after,
    # with 5% of the labels flipped at random.
    rng = np.random.default_rng(seed)
    chunks = []
    for chunk in range(N_CHUNKS):
        X = rng.normal(size=(CHUNK_ROWS, 5))
        signal = X[:, 0] + X[:, 1] if chunk < DRIFT_CHUNK else X[:, 0] - X[:, 1]
        y = (signal > 0).astype(int)
        flip = rng.random(CHUNK_ROWS) < 0.05
        chunks.append((X, np.where(flip, 1 - y, y)))
    return chunks


def bench_buffer_sizes():
    chunks = make_stream()
    rows, rolling = [], {}
    for buffer_size in BUFFER_SIZES:
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size, random_state=0))])
        trainer = StreamTrainer(pipe, window=3)
        for X, y in chunks:                                # test-then-train
            trainer.score_chunk(X, y)
            trainer.fit_chunk(X, y)
        log = trainer.log
        accuracy = np.array([row["accuracy"] for row in log])
        rebuild_seconds = [row["fit_seconds"] for row in log if row["retrained"]]
        rows.append((buffer_size,
                     len(rebuild_seconds),
                     np.nanmean(accuracy[:DRIFT_CHUNK]),
                     np.nanmean(accuracy[DRIFT_CHUNK:]),
                     sum(row["fit_seconds"] for row in log),
                     np.mean(rebuild_seconds) if rebuild_seconds else float("nan"),
                     max(row["memory_bytes"] for row in log)))
        rolling[buffer_size] = [row["rolling_accuracy"] for row in log]
    return rows, rolling


# Printing

def print_table(title, headers, rows, markdown):
    cells = [[str(cell) for cell in row] for row in rows]
    print(f"\n{title}\n")
    if markdown:
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in cells:
            print("| " + " | ".join(row) + " |")
        return
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def milliseconds(seconds):
    return f"{seconds * 1000:.3f}" if seconds == seconds else "-"


def main():
    parser = argparse.ArgumentParser(description="NumCompute-Stream benchmarks")
    parser.add_argument("--markdown", action="store_true", help="print tables in markdown")
    parser.add_argument("--no-plot", action="store_true", help="do not save the buffer_size image")
    args = parser.parse_args()

    rows = bench_loop_vs_vectorised()
    print_table("Table 1: loop against vectorised (times in ms, best of 3)",
                ["operation", "rows", "loop ms", "vectorised ms", "speed-up", "same answer"],
                [(op, n, milliseconds(slow), milliseconds(fast), f"{slow / fast:.0f}x", "yes" if same else "NO")
                 for op, n, slow, fast, same in rows], args.markdown)

    rows, rolling = bench_buffer_sizes()
    print_table(f"Table 2: buffer_size on a stream of {N_CHUNKS} chunks of {CHUNK_ROWS} rows "
                f"(the rule changes at chunk {DRIFT_CHUNK})",
                ["buffer_size", "rebuilds", "accuracy before drift", "accuracy after drift",
                 "total fit ms", "ms per rebuild", "peak memory KB"],
                [(size, rebuilds, f"{before:.3f}", f"{after:.3f}", milliseconds(total),
                  milliseconds(per_rebuild), f"{memory / 1024:.1f}")
                 for size, rebuilds, before, after, total, per_rebuild, memory in rows], args.markdown)

    if not args.no_plot:
        target = Path(__file__).with_name("buffer_size_accuracy.png")
        smallest, largest = BUFFER_SIZES[0], BUFFER_SIZES[-1]
        compare_models(rolling[smallest], rolling[largest],
                       (f"buffer_size={smallest}", f"buffer_size={largest}"),
                       title=f"Rolling accuracy (last 3 chunks); the rule changes at chunk {DRIFT_CHUNK}",
                       ylabel="Accuracy", save_path=str(target), show=False)
        print(f"\nSaved {target}")


if __name__ == "__main__":
    main()
