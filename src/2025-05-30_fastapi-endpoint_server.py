"""
# Creating API endpoint to return predictions from a model

Reference: https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/#what-is-fastapi  # noqa
"""

import uvicorn
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app)
