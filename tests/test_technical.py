"""Unit test untuk modul kalkulasi fitur teknikal."""

import numpy as np
import pandas as pd
from src.features.technical import calculate_technical_indicators


def test_calculate_technical_indicators():
    dates = pd.date_range("2023-01-01", periods=30)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.linspace(8000, 8500, 30),
            "High": np.linspace(8100, 8600, 30),
            "Low": np.linspace(7900, 8400, 30),
            "Close": np.linspace(8050, 8550, 30),
            "Volume": np.random.randint(1000, 5000, 30),
        }
    )

    res_df = calculate_technical_indicators(df)

    expected_cols = [
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "MA5",
        "MA20",
        "RSI",
        "Return",
    ]
    for col in expected_cols:
        assert col in res_df.columns
        assert not res_df[col].isna().any(), f"Kolom {col} mengandung NaN"
