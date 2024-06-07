"""
Basic example of training a DL model for regression with pytorch

References:
- https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

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
torch.manual_seed(SEED)


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


def train_with_early_stopping(
    dataloader_train, dataloader_val, model, loss_fn, optimizer, num_epochs, patience
):
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print_epoch(epoch)
        loss = train(dataloader_train, model, loss_fn, optimizer)
        avg_loss_val = test(dataloader_val, model, loss_fn)

        if avg_loss_val < best_val_loss:
            best_val_loss = avg_loss_val
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch + 1}/{num_epochs}, Last batch training loss: {loss:.5f}")
        print(f"Avg val loss: {avg_loss_val}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            # TODO: how to revert to the best epoch?
            break


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    current = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Calculate prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(-1), y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        current = current + len(X)

        print_train_progress = False
        if print_train_progress:
            print(f"Train loss: {loss:.5f}, Current instance: {current}/{size}")
    return loss


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(-1), y).item()

    test_loss /= num_batches
    return test_loss


def print_epoch(epoch):
    print(f"{'-'*40}")
    print(f"Epoch: {epoch + 1}")
    print(f"{'-'*40}")


if __name__ == "__main__":
    X, y = make_regression(n_samples=400, n_features=N_FEATURES, random_state=SEED)
    X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_and_val, y_train_and_val, test_size=0.20, random_state=SEED
    )

    data_train = TabularData(X_train, y_train)
    data_val = TabularData(X_val, y_val)
    data_test = TabularData(X_test, y_test)

    # DataLoader wraps an iterable around a Dataset
    dataloader_train = DataLoader(data_train, batch_size=8, shuffle=True)
    dataloader_val = DataLoader(data_val, batch_size=8, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=8, shuffle=True)

    model = Model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    before_training = model.forward(data_train[0][0])

    num_epochs = 50
    patience = 3
    # for epoch in tqdm(range(num_epochs)):
    #     train(dataloader_train, model, loss_fn, optimizer)
    #     test(dataloader_test, model, loss_fn)

    train_with_early_stopping(
        dataloader_train,
        dataloader_val,
        model,
        loss_fn,
        optimizer,
        num_epochs,
        patience,
    )

    after_training = model.forward(data_train[0][0])

    # Compare with LR:
    lrcv = LogisticRegressionCV()
    # todo: use nested CV to get best h-params, and get generalization error as well

    print("done")
