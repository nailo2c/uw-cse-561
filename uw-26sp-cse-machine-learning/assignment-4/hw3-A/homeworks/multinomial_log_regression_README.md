## Multinomial Logistic Regression

In this problem you will implement multiclass linear classification on MNIST.
Start by looking at `multinomial_log_regression.py`.

You will implement:

- `J_loss`, the joint negative log-likelihood loss.
- `L_loss`, the standard average cross-entropy loss.
- `accuracy`, which computes the fraction of correctly classified examples.
- `train`, which trains a weight matrix for one of the two losses.
- `main`, which trains both models, produces the requested plots, and reports train/test accuracy.

Run this problem from the root of the provided zip file:

```bash
python homeworks/multinomial_log_regression.py
```

The public tests are intentionally light. Passing them does not guarantee full
credit; use the written assignment as the specification for the plots and
reported accuracies.
