"""Modul rekayasa fitur teknikal saham disesuaikan dengan notebook.

Fitur teknikal:
1. Close   - Harga penutupan
2. Open    - Harga pembukaan
3. High    - Harga tertinggi
4. Low     - Harga terendah
5. Volume  - Volume perdagangan
6. MA5     - Moving Average 5 Hari
7. MA20    - Moving Average 20 Hari
8. RSI     - Relative Strength Index 14 Hari
9. Return  - Daily Return (pct_change)
"""

import numpy as np
import pandas as pd


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Menghitung fitur teknikal sesuai skema notebook."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

    close = df["Close"]

    # MA5 & MA20
    df["MA5"] = close.rolling(window=5).mean()
    df["MA20"] = close.rolling(window=20).mean()

    # Return
    df["Return"] = close.pct_change()

    # RSI 14 Hari
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-8))))

    # Kolom teknikal utama
    tech_cols = ["Close", "Open", "High", "Low", "Volume", "MA5", "MA20", "RSI", "Return"]

    # Isi NaN pada window awal
    df[tech_cols] = df[tech_cols].bfill().fillna(0)

    return df
