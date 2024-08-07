"""
# Time series as supervised learning

References:
- https://machinelearningmastery.com/xgboost-for-time-series-forecasting/
"""
from pathlib import Path

import pandas as pd

# from xgboost import XGBRegressor


def read_data() -> pd.DataFrame:
    df = pd.read_csv(
        Path(__file__).parent.parent.joinpath(
            "data/2024-08-07_daily-total-female-births.csv"
        )
    )
    assert len(df) == 365
    return df


def time_series_to_supervised():
    pass


if __name__ == "__main__":
    df_raw = read_data()

    print("done")
