# CSEP 546 HW4 Code

This archive contains the starter code for the autograded k-means portion of
Homework 4. The CIFAR-10 and transformer parts of the assignment use the
notebooks linked from the homework PDF/course site; they are not packaged here.

## Contents

- `homeworks/k_means/k_means.py`: implement Lloyd's algorithm helper functions.
- `homeworks/k_means/main.py`: run k-means on MNIST and produce the cluster-center images for the PDF.
- `data/mnist/mnist.npz`: MNIST data used by `main.py`.
- `tests/public/k_means/test_k_means.py`: public tests for helper functions.

No other homework datasets or old assignment code are included.

## Setup

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from this directory:

```bash
uv sync
```

## Testing

Run all public tests:

```bash
uv run inv test
```

Run just the k-means public tests:

```bash
uv run inv test --problem k_means
```

## Submission

When you are done, run:

```bash
uv run inv submit
```

This creates a `submission_<timestamp>.zip` file. Upload that generated zip to
the HW4 coding submission on Gradescope.
