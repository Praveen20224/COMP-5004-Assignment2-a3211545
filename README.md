# NumCompute-Stream

## Setup

You need Python 3.10 or newer

```
pip install numpy matplotlib pytest
```

To use the package from any folder, install it from the repository folder (the one that
contains `numcompute_stream/` and `pyproject.toml`):

```
pip install -e .
```


## Example

```python
import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

# a scaler feeding a tree that rebuilds itself once 100 rows have been collected
pipe = Pipeline([("scale", StandardScaler()),
                 ("model", DecisionTreeClassifier(max_depth=5, buffer_size=100))])
trainer = StreamTrainer(pipe)

rng = np.random.default_rng(0)
for _ in range(10):                                  # ten chunks of 50 rows
    X = rng.normal(size=(50, 3))
    y = (X[:, 0] > 0).astype(int)
    trainer.score_chunk(X, y)                        # test first: predict before learning
    trainer.fit_chunk(X, y)                          # then train on the chunk

print(trainer.log[-1]["cumulative_accuracy"])        # accuracy over all scored chunks
```

`trainer.log` holds one row per chunk: its accuracy, the cumulative accuracy, the time the
training took, whether the tree was rebuilt, and the memory the pipeline holds.

To plot it:

```python
from numcompute_stream.visualise import plot_metric_over_time

accuracy = [row["accuracy"] for row in trainer.log]
plot_metric_over_time(accuracy, "Accuracy per chunk", "Accuracy")      # save_path="plot.png" saves it
```

## How to run things

Run the tests:

```
python -m pytest tests
```

Run the benchmarks (loop against vectorised, and different `buffer_size` values):

```
python numcompute_stream/stream_benchmarks.py
```

Add `--markdown` to print the tables in markdown, or `--no-plot` to skip saving the image.

Open the demo notebook, which runs a stream with drift and compares buffer sizes:

```
jupyter notebook demo/stream_demo.ipynb
```
