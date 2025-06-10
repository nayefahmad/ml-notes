"""
# Creating API endpoint to return predictions from a model

Reference: https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/#what-is-fastapi  # noqa
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    set up our model with FastAPI lifespan events. The advantage of doing that is we
    can make sure no request will be accepted while the model is still being set up
    and the memory used will be cleaned up afterward.
    """
    df = pd.read_csv(Path(__file__).parents[1].joinpath("data", "penguins.csv"))
    df = df.dropna()

    le = LabelEncoder()
    le.fit(df["species"])
    y = le.transform(df["species"])

    X = df[["bill_length_mm", "flipper_length_mm"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )

    clf = Pipeline(
        steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
    )

    clf.fit(X_train, y_train)

    ml_models["clf"] = clf
    ml_models["le"] = le

    yield

    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


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
