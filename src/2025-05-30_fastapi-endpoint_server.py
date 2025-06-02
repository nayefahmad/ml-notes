"""
# Creating API endpoint to return predictions from a model

Reference: https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/#what-is-fastapi  # noqa
"""

from typing import Optional

import uvicorn
from fastapi import FastAPI

app = FastAPI()


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
