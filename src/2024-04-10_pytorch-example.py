"""
Basic example of training a DL model for regression with pytorch

Reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

"""

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

N_FEATURES = 10
SEED = 2024


class TabularData(Dataset):
    """
    This class creates objects that store the particular format of data required for our
    problem.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X.astype(np.float32))
        self.y = torch.tensor(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_FEATURES, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Calculate prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(-1), y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"Loss: {loss:>7f}, Current: {current}/{size}")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(-1), y).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f}")


if __name__ == "__main__":
    X, y = make_regression(n_samples=400, n_features=N_FEATURES, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    data_train = TabularData(X_train, y_train)
    data_test = TabularData(X_test, y_test)

    # DataLoader wraps an iterable around a Dataset
    dataloader_train = DataLoader(data_train, batch_size=8, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=8, shuffle=True)

    model = Model()
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before_training = model.forward(data_train[0][0])
    train(dataloader_train, model, loss_fn, optimizer)
    after_training = model.forward(data_train[0][0])

    test(dataloader_test, model, loss_fn)

    # Compare with LR:
    lrcv = LogisticRegressionCV()
    # todo: use nested CV to get best h-params, and get generalization error as well
