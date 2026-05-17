from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    residual = X @ weight + bias - y
    new_bias = bias - 2 * eta * np.sum(residual)
    
    z = weight - 2 * eta * (X.T @ residual)
    threshold = 2 * eta * _lambda
    new_weight = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)

    return (new_weight, new_bias)


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    residual = X @ weight + bias - y
    sse = np.sum(residual ** 2)
    reg = _lambda * np.sum(np.abs(weight))
    return sse + reg


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    else:
        start_weight = np.copy(start_weight)

    if start_bias is None:
        start_bias = 0

    old_w: Optional[np.ndarray] = None
    old_b: float = None
    
    while old_w is None or not convergence_criterion(
        start_weight, old_w, start_bias, old_b, convergence_delta
    ):
        old_w = np.copy(start_weight)
        old_b = start_bias

        start_weight, start_bias = step(X, y, start_weight, start_bias, _lambda, eta)

    return start_weight, start_bias


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    weight_change = np.max(np.abs(weight - old_w))
    bias_change = abs(bias - old_b)
    return weight_change <= convergence_delta and bias_change <= convergence_delta


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    rng = np.random.default_rng(0)

    n = 500
    d = 1000
    k = 100
    sigma = 1

    true_weight = np.zeros(d)
    true_weight[:k] = np.arange(1, k + 1) / k

    X = rng.normal(size=(n, d))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = X @ true_weight + rng.normal(0, sigma, size=n)

    lambda_max = np.max(2 * np.abs(X.T @ (y - y.mean())))
    augmented_X = np.column_stack([X, np.ones(n)])
    eta = 1 / (2 * np.linalg.norm(augmented_X, ord=2) ** 2)

    lambdas = []
    nonzeros = []
    fdrs = []
    tprs = []

    weight = None
    bias = None
    _lambda = lambda_max

    max_path_steps = 25
    for _ in range(max_path_steps):
        weight, bias = train(
            X,
            y,
            _lambda=_lambda,
            eta=eta,
            convergence_delta=1e-4,
            start_weight=weight,
            start_bias=bias,
        )

        selected = np.abs(weight) > 1e-6
        num_selected = np.sum(selected)
        false_selected = np.sum(selected[k:])
        true_selected = np.sum(selected[:k])

        fdr = false_selected / num_selected if num_selected > 0 else 0
        tpr = true_selected / k

        lambdas.append(_lambda)
        nonzeros.append(num_selected)
        fdrs.append(fdr)
        tprs.append(tpr)

        if num_selected >= 0.95 * d:
            break
        _lambda /= 2

    output_dir = Path(__file__).resolve().parent

    plt.figure()
    plt.plot(lambdas, nonzeros, marker="o")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("number of nonzero weights")
    plt.savefig(output_dir / "synthetic_nonzeros_path.png", bbox_inches="tight")
    plt.cla()
    plt.clf()

    plt.figure()
    plt.plot(fdrs, tprs, marker="o")
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.savefig(output_dir / "synthetic_fdr_tpr.png", bbox_inches="tight")
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    main()
