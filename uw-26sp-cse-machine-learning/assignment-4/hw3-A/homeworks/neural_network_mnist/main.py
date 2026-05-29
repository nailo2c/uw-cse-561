# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


def _uniform_parameter(*shape: int, input_dim: int) -> Parameter:
    alpha = 1 / math.sqrt(input_dim)
    values = Uniform(-alpha, alpha).sample(shape)
    return Parameter(values.float())


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.W0 = _uniform_parameter(d, h, input_dim=d)
        self.b0 = _uniform_parameter(h, input_dim=d)
        self.W1 = _uniform_parameter(h, k, input_dim=h)
        self.b1 = _uniform_parameter(k, input_dim=h)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden = relu(x @ self.W0 + self.b0)
        return hidden @ self.W1 + self.b1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.W0 = _uniform_parameter(d, h0, input_dim=d)
        self.b0 = _uniform_parameter(h0, input_dim=d)
        self.W1 = _uniform_parameter(h0, h1, input_dim=h0)
        self.b1 = _uniform_parameter(h1, input_dim=h0)
        self.W2 = _uniform_parameter(h1, k, input_dim=h1)
        self.b2 = _uniform_parameter(k, input_dim=h1)

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        hidden0 = relu(x @ self.W0 + self.b0)
        hidden1 = relu(hidden0 @ self.W1 + self.b1)
        return hidden1 @ self.W2 + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    losses = []
    train_accuracy = 0.0
    max_epochs = 100

    for _ in range(max_epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        losses.append(running_loss / len(train_loader.dataset))
        train_accuracy, _ = _evaluate(model, train_loader)

        if train_accuracy >= 0.99:
            break

    return losses


def _evaluate(model: Module, data_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            logits = model(x_batch)
            loss = cross_entropy(logits, y_batch)
            predictions = torch.argmax(logits, dim=1)
            correct += torch.sum(predictions == y_batch).item()
            total += len(y_batch)
            running_loss += loss.item() * len(y_batch)

    return correct / total, running_loss / total


def _num_parameters(model: Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _plot_losses(losses: List[float], title: str, filename: str) -> None:
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.savefig(filename)
    plt.close()


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=512)

    model_f1 = F1(h=64, d=784, k=10)
    optimizer_f1 = Adam(model_f1.parameters(), lr=1e-3)
    losses_f1 = train(model_f1, optimizer_f1, train_loader)
    accuracy_f1, loss_f1 = _evaluate(model_f1, test_loader)
    params_f1 = _num_parameters(model_f1)
    _plot_losses(losses_f1, "MNIST F1 loss", "mnist_f1_loss.png")

    model_f2 = F2(h0=32, h1=32, d=784, k=10)
    optimizer_f2 = Adam(model_f2.parameters(), lr=1e-3)
    losses_f2 = train(model_f2, optimizer_f2, train_loader)
    accuracy_f2, loss_f2 = _evaluate(model_f2, test_loader)
    params_f2 = _num_parameters(model_f2)
    _plot_losses(losses_f2, "MNIST F2 loss", "mnist_f2_loss.png")

    print("F1 test accuracy:", accuracy_f1)
    print("F1 test loss:", loss_f1)
    print("F1 parameters:", params_f1)
    print("F1 epochs:", len(losses_f1))
    print("F2 test accuracy:", accuracy_f2)
    print("F2 test loss:", loss_f2)
    print("F2 parameters:", params_f2)
    print("F2 epochs:", len(losses_f2))


if __name__ == "__main__":
    main()
