"""
# Creating API endpoint to return predictions from a model

Reference: https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/#what-is-fastapi  # noqa
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = FastAPI()

data_dir = Path(__file__).parents[1]

data = pd.read_csv(data_dir.joinpath("data/penguins.csv"))
data = data.dropna()

le = LabelEncoder()
X = data[["bill_length_mm", "flipper_length_mm"]]
le.fit(data["species"])
y = le.transform(data["species"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
clf = Pipeline(
    steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
)
clf.fit(X_train, y_train)


@app.get("/")
async def root():
    return {
        "name": "Penguins",
        "description": "Penguins prediction API",
    }


@app.get("/predict")
async def predict(bill_length_mm: float = 0.0, flipper_length_mm: float = 0.0):
    param = {
        "bill_length_mm": bill_length_mm,
        "flipper_length_mm": flipper_length_mm,
    }
    if bill_length_mm <= 0.0 or flipper_length_mm <= 0.0:
        return {"parameters": param, "error_msg": "Invalid input values"}
    else:
        result = clf.predict([[bill_length_mm, flipper_length_mm]])
        return {
            "parameters": param,
            "prediction": le.inverse_transform(result)[0],
        }


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}"}


@app.get("/hello_optional_name")
async def say_hello_optional_name(name: Optional[str] = None):
    if name:
        return {"message": f"Hello, {name}"}
    return {"message": "Hello, stranger"}


if __name__ == "__main__":
    uvicorn.run(app)
