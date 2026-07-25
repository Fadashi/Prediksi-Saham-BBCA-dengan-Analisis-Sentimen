"""Modul pengunduhan data harga historis BBCA via yfinance."""

import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.config import CONFIG, RAW_DATA_DIR, set_seed


def fetch_bbca_prices(
    ticker_symbol: str = None,
    start_date: str = None,
    end_date: str = None,
    output_path: Path = None,
) -> pd.DataFrame:
    """Mengunduh data harga historis BBCA dari yfinance dan menyimpannya ke CSV.

    Parameters:
    -----------
    ticker_symbol : str
        Ticker saham yfinance (misal 'BBCA.JK')
    start_date : str
        Tanggal mulai format YYYY-MM-DD
    end_date : str
        Tanggal selesai format YYYY-MM-DD
    output_path : Path
        Path file tujuan CSV

    Returns:
    --------
    pd.DataFrame
        Dataframe harga OHLCV BBCA
    """
    ticker_symbol = ticker_symbol or CONFIG["price_ticker"]
    start_date = start_date or CONFIG["period"]["price_start"]
    end_date = end_date or CONFIG["period"]["end"]
    output_path = output_path or (RAW_DATA_DIR / "prices_bbca.csv")

    print(f"Mengunduh harga {ticker_symbol} dari {start_date} hingga {end_date}...")
    df = yf.download(ticker_symbol, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(
            f"Gagal mengunduh data harga untuk {ticker_symbol}. Data kosong."
        )

    # Flatten multi-level columns if yfinance returns tuple column names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # Standardize column names
    cols_map = {
        "Date": "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    }
    df.rename(columns=cols_map, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.sort_values("Date", ascending=True, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"Data harga {ticker_symbol} berhasil disimpan: {output_path} ({len(df)} baris)"
    )

    return df


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    fetch_bbca_prices()
