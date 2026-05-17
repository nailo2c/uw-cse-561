if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    target = "ViolentCrimesPerPop"
    y_train = df_train[target].values
    X_train_df = df_train.drop(target, axis=1)
    X_train = X_train_df.values

    y_test = df_test[target].values
    X_test = df_test.drop(target, axis=1).values

    feature_names = list(X_train_df.columns)
    path_features = [
        "agePct12t29",
        "pctWSocSec",
        "pctUrban",
        "agePct65up",
        "householdsize",
    ]
    path_feature_indices = [feature_names.index(name) for name in path_features]

    lambda_max = np.max(2 * np.abs(X_train.T @ (y_train - y_train.mean())))
    augmented_X = np.column_stack([X_train, np.ones(X_train.shape[0])])
    eta = 1 / (2 * np.linalg.norm(augmented_X, ord=2) ** 2)

    lambdas = []
    nonzeros = []
    path_weights = {name: [] for name in path_features}
    train_errors = []
    test_errors = []

    weight = None
    bias = None
    _lambda = lambda_max

    while _lambda >= 0.01:
        weight, bias = train(
            X_train,
            y_train,
            _lambda=_lambda,
            eta=eta,
            convergence_delta=1e-4,
            start_weight=weight,
            start_bias=bias,
        )

        train_residual = X_train @ weight + bias - y_train
        test_residual = X_test @ weight + bias - y_test

        lambdas.append(_lambda)
        nonzeros.append(np.sum(np.abs(weight) > 1e-6))
        train_errors.append(np.sum(train_residual ** 2))
        test_errors.append(np.sum(test_residual ** 2))

        for name, idx in zip(path_features, path_feature_indices):
            path_weights[name].append(weight[idx])

        _lambda /= 2

    output_dir = Path(__file__).resolve().parent

    plt.figure()
    plt.plot(lambdas, nonzeros, marker="o")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("number of nonzero weights")
    plt.savefig(output_dir / "crime_nonzeros_path.png", bbox_inches="tight")
    plt.cla()
    plt.clf()

    plt.figure()
    for name in path_features:
        plt.plot(lambdas, path_weights[name], marker="o", label=name)
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("coefficient value")
    plt.legend()
    plt.savefig(output_dir / "crime_regularization_paths.png", bbox_inches="tight")
    plt.cla()
    plt.clf()

    plt.figure()
    plt.plot(lambdas, train_errors, marker="o", label="train")
    plt.plot(lambdas, test_errors, marker="o", label="test")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("squared error")
    plt.legend()
    plt.savefig(output_dir / "crime_squared_error.png", bbox_inches="tight")
    plt.cla()
    plt.clf()

    weight_30, bias_30 = train(
        X_train,
        y_train,
        _lambda=30,
        eta=eta,
        convergence_delta=1e-4,
    )
    most_positive_idx = np.argmax(weight_30)
    most_negative_idx = np.argmin(weight_30)

    with open(output_dir / "crime_lambda_30_coefficients.txt", "w") as f:
        f.write(
            f"Most positive: {feature_names[most_positive_idx]} "
            f"({weight_30[most_positive_idx]})\n"
        )
        f.write(
            f"Most negative: {feature_names[most_negative_idx]} "
            f"({weight_30[most_negative_idx]})\n"
        )


if __name__ == "__main__":
    main()
