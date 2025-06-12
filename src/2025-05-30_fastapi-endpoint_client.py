"""
# Calling an API endpoint to return predictions from a model

Reference: https://blog.jetbrains.com/pycharm/2024/09/how-to-use-fastapi-for-machine-learning/#what-is-fastapi  # noqa
"""
from typing import Optional

import requests

base_url = "http://127.0.0.1:8000"


def call_hello_optional_name(name: Optional[str] = None):
    if name:
        response = requests.get(f"{base_url}/hello_optional_name?name={name}")
    else:
        response = requests.get(f"{base_url}/hello_optional_name")
    return response.json()


def predict_client(bill_length_mm: float = 0.0, flipper_length_mm: float = 0.0):
    response = requests.get(
        f"{base_url}/predict"
        f"?bill_length_mm={bill_length_mm}&flipper_length_mm={flipper_length_mm}"
    )
    return response.json()


if __name__ == "__main__":
    call_hello_optional_name()
    call_hello_optional_name("Nayef")
    predict_client(10.0, 11.2)
