from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@problem.tag("hw3-A")
def J_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the joint negative log-likelihood loss.

    This is the sum of per-example negative log-likelihoods for the
    multinomial logistic regression model.

    Args:
        logits: FloatTensor of shape (n, k). Raw class scores.
        y: LongTensor of shape (n,). Class labels.

    Returns:
        A scalar tensor containing the loss.
    """
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -torch.sum(log_probs[torch.arange(len(y), device=logits.device), y])


@problem.tag("hw3-A")
def L_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard average cross-entropy loss.

    This is the mean per-example negative log-likelihood. You may use
    torch.nn.functional.cross_entropy or write the softmax/log expression
    directly.

    Args:
        logits: FloatTensor of shape (n, k). Raw class scores.
        y: LongTensor of shape (n,). Class labels.

    Returns:
        A scalar tensor containing the loss.
    """
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -torch.mean(log_probs[torch.arange(len(y), device=logits.device), y])


@problem.tag("hw3-A")
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        logits: FloatTensor of shape (n, k). Raw class scores.
        y: LongTensor of shape (n,). Class labels.

    Returns:
        Fraction of examples classified correctly.
    """
    predictions = torch.argmax(logits, dim=1)
    return torch.mean((predictions == y).float()).item()


@problem.tag("hw3-A")
def train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: LossFunction,
    *,
    learning_rate: float = 0.1,
    epochs: int = 50,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Train a multinomial logistic regression model on MNIST.

    Args:
        x_train: FloatTensor of shape (n, d).
        y_train: LongTensor of shape (n,).
        loss_fn: Either J_loss or L_loss.
        learning_rate: Step size for gradient descent.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.

    Returns:
        A tuple (W, losses), where W has shape (k, d) and losses stores one
        average training loss per epoch.
    """
    dataset = TensorDataset(x_train, y_train)
    loader_rng = torch.Generator()
    loader_rng.manual_seed(546)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=loader_rng
    )
    num_classes = int(torch.max(y_train).item()) + 1
    num_features = x_train.shape[1]
    rng = torch.Generator()
    rng.manual_seed(546)
    W = 0.01 * torch.randn(num_classes, num_features, generator=rng)
    W.requires_grad_()

    losses = []
    for _ in range(epochs):
        running_loss = 0.0

        for x_batch, y_batch in loader:
            logits = x_batch @ W.T
            loss = loss_fn(logits, y_batch)
            loss.backward()

            with torch.no_grad():
                W -= learning_rate * W.grad
                W.grad.zero_()

            if loss_fn is J_loss:
                running_loss += loss.item()
            else:
                running_loss += loss.item() * len(y_batch)

        losses.append(running_loss / len(dataset))

    return W.detach(), losses


def _evaluate(
    x: torch.Tensor, y: torch.Tensor, W: torch.Tensor, loss_fn: LossFunction
) -> Tuple[float, float]:
    with torch.no_grad():
        logits = x @ W.T
        raw_loss = loss_fn(logits, y).item()
        if loss_fn is J_loss:
            loss = raw_loss / len(y)
        else:
            loss = raw_loss
        return accuracy(logits, y), loss


@problem.tag("hw3-A", start_line=5)
def main() -> Dict[str, Dict[str, float]]:
    """
    Train multinomial logistic regression models with J_loss and L_loss.

    For each loss, this function should:
        1. Load MNIST.
        2. Train a multinomial logistic regression model.
        3. Plot training loss vs. epoch.
        4. Report training and test accuracy.

    Returns:
        A dictionary mapping loss names to accuracy summaries.
    """
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    W_J, losses_J = train(
        x_train,
        y_train,
        J_loss,
        learning_rate=1e-3,
        epochs=40,
        batch_size=512,
    )
    W_L, losses_L = train(
        x_train,
        y_train,
        L_loss,
        learning_rate=0.5,
        epochs=40,
        batch_size=512,
    )

    J_train_accuracy, J_train_loss = _evaluate(x_train, y_train, W_J, J_loss)
    J_test_accuracy, J_test_loss = _evaluate(x_test, y_test, W_J, J_loss)
    L_train_accuracy, L_train_loss = _evaluate(x_train, y_train, W_L, L_loss)
    L_test_accuracy, L_test_loss = _evaluate(x_test, y_test, W_L, L_loss)

    plt.figure()
    plt.plot(losses_J, label="J_loss")
    plt.plot(losses_L, label="L_loss")
    plt.xlabel("epoch")
    plt.ylabel("average cross-entropy loss")
    plt.title("Multinomial logistic regression")
    plt.legend()
    plt.savefig("multinomial_loss_comparison.png")
    plt.close()

    plt.figure()
    plt.plot(losses_J)
    plt.xlabel("epoch")
    plt.ylabel("average cross-entropy loss")
    plt.title("J_loss")
    plt.savefig("multinomial_J_loss.png")
    plt.close()

    plt.figure()
    plt.plot(losses_L)
    plt.xlabel("epoch")
    plt.ylabel("average cross-entropy loss")
    plt.title("L_loss")
    plt.savefig("multinomial_L_loss.png")
    plt.close()

    results = {
        "J_loss": {
            "train_accuracy": J_train_accuracy,
            "test_accuracy": J_test_accuracy,
            "train_loss": J_train_loss,
            "test_loss": J_test_loss,
        },
        "L_loss": {
            "train_accuracy": L_train_accuracy,
            "test_accuracy": L_test_accuracy,
            "train_loss": L_train_loss,
            "test_loss": L_test_loss,
        },
    }

    print("J_loss train accuracy:", J_train_accuracy)
    print("J_loss test accuracy:", J_test_accuracy)
    print("L_loss train accuracy:", L_train_accuracy)
    print("L_loss test accuracy:", L_test_accuracy)

    return results


if __name__ == "__main__":
    main()
