import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth"]

_BLUE, _ORANGE = "#2a78d6", "#eb6834"      # first two colours of the palette
_INK, _GRID = "#52514e", "#d8d7d2"         # text and grid colours



def _as_values(metric_values, name):
    # a non-empty 1-D list of numbers, as floats. NaN is allowed (it leaves a gap in the line).
    values = np.asarray(metric_values)
    if values.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain numbers")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence, got shape {values.shape}")
    values = values.astype(float)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinity")
    return values


def _new_axes(title, xlabel, ylabel):
    # an empty figure with a light grid and no top or right border
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title, color=_INK)
    ax.set_xlabel(xlabel, color=_INK)
    ax.set_ylabel(ylabel, color=_INK)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b8b2")
    ax.tick_params(colors=_INK)
    return fig, ax


def _finish(fig, save_path, show):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig



def plot_metric_over_time(metric_values, title="", ylabel="", xlabel="Chunk",save_path=None, show=True):
    values = _as_values(metric_values, "metric_values")
    fig, ax = _new_axes(title, xlabel, ylabel)
    marker = "o" if values.size <= 40 else None          # dots only while they stay readable
    ax.plot(np.arange(values.size), values, color=_BLUE, linewidth=2, marker=marker, markersize=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels, title="Model comparison", ylabel="Metric",xlabel="Chunk", save_path=None, show=True):
    first = _as_values(metric1, "metric1")
    second = _as_values(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError(f"labels must have exactly two names, got {len(labels)}")

    fig, ax = _new_axes(title, xlabel, ylabel)
    ax.plot(np.arange(first.size), first, color=_BLUE, linewidth=2, label=str(labels[0]))
    ax.plot(np.arange(second.size), second, color=_ORANGE, linewidth=2, linestyle="--",label=str(labels[1]))                          # dashed too, so colour is not the only clue
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, labelcolor=_INK, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth (latest chunk)",xlabel="Row in the chunk", save_path=None, show=True):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        if labels.dtype.kind not in "biufU":
            raise TypeError(f"{name} must contain numbers or strings")
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError(f"{name} must be a non-empty 1-D array, got shape {labels.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same length, got {y_true.size} and {y_pred.size}")
    if (y_true.dtype.kind == "U") != (y_pred.dtype.kind == "U"):
        raise TypeError("y_true and y_pred must both be numbers or both strings")

    classes = np.unique(np.concatenate((y_true, y_pred)))
    height_true = np.searchsorted(classes, y_true)
    height_pred = np.searchsorted(classes, y_pred)
    rows = np.arange(y_true.size)
    wrong = height_true != height_pred

    fig, ax = _new_axes(f"{title}\n{wrong.sum()} wrong of {y_true.size}", xlabel, "Class")
    ax.scatter(rows, height_true, s=70, facecolors="none", edgecolors=_INK, linewidths=1.5,label="True label")
    ax.scatter(rows[~wrong], height_pred[~wrong], marker="x", s=30, color=_BLUE, linewidths=1.5,label="Correct prediction")
    ax.scatter(rows[wrong], height_pred[wrong], marker="X", s=80, color=_ORANGE,label="Wrong prediction")
    ax.set_yticks(np.arange(classes.size))
    ax.set_yticklabels([str(label) for label in classes])
    ax.set_ylim(-0.5, classes.size - 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, labelcolor=_INK, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    return _finish(fig, save_path, show)
