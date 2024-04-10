import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

N_FEATURES = 10

X, y = make_regression(n_samples=400, n_features=N_FEATURES)
X_train, X_test, y_train, y_test = train_test_split(X, y)


class TabularData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


data_train = TabularData(X_train, y_train)
data_test = TabularData(X_test, y_test)

dataloader_train = DataLoader(data_train, batch_size=8, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=8, shuffle=True)


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


model = Model()
print(model)

try:
    m = Model()
    m(data_train[0][0])
except RuntimeError as e:
    print(f"{type(e).__name__} \nMessage: {e}")
