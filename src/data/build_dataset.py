"""Modul penggabungan data harga dan sentimen menjadi dataset final.

Menghasilkan data/processed/dataset_final.csv dengan skema kolom yang sesuai notebook.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    CONFIG,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    set_seed,
)
from src.data.fetch_prices import fetch_bbca_prices
from src.features.technical import calculate_technical_indicators
from src.sentiment.aggregate_daily import aggregate_daily_sentiment


def build_final_dataset(
    price_file: Path = None,
    sentiment_file: Path = None,
    output_file: Path = None,
) -> pd.DataFrame:
    """Menggabungkan data harga + teknikal + sentimen harian."""
    price_file = price_file or (RAW_DATA_DIR / "prices_bbca.csv")
    sentiment_file = sentiment_file or (INTERIM_DATA_DIR / "daily_sentiment.csv")
    output_file = output_file or (PROCESSED_DATA_DIR / "dataset_final.csv")

    if not price_file.exists():
        fetch_bbca_prices(output_path=price_file)
    df_price = pd.read_csv(price_file)

    df_price = calculate_technical_indicators(df_price)

    if not sentiment_file.exists():
        if "indobert" in str(sentiment_file).lower():
            from src.sentiment.aggregate_indobert import aggregate_indobert_sentiment
            aggregate_indobert_sentiment(output_daily_file=sentiment_file)
        else:
            aggregate_daily_sentiment(output_file=sentiment_file)
            
    df_sentiment = pd.read_csv(sentiment_file)

    df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.strftime("%Y-%m-%d")
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"]).dt.strftime("%Y-%m-%d")

    sentiment_start = CONFIG["period"]["sentiment_start"]
    end_date = CONFIG["period"]["end"]

    df_price = df_price[
        (df_price["Date"] >= sentiment_start) & (df_price["Date"] <= end_date)
    ].copy()

    merged_df = pd.merge(df_price, df_sentiment, on="Date", how="left")

    # Imputasi Halus Missing Value (Forward-Fill -> Backward-Fill -> default 0.0)
    for sent_col in ["sentiment_score", "positive_ratio", "negative_ratio"]:
        merged_df[sent_col] = merged_df[sent_col].ffill().bfill().fillna(0.0)

    # Volume Diskusi Riil (Jumlah Postingan Riil per Hari)
    merged_df["discussion_volume"] = merged_df["discussion_volume"].fillna(0).astype(int)

    # Target: Harga Penutupan H+1
    merged_df["Target"] = merged_df["Close"].shift(-1)
    merged_df.dropna(subset=["Target"], inplace=True)
    merged_df.sort_values("Date", ascending=True, inplace=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
    print(
        f"[Dataset Final] Berhasil dibuat di {output_file} ({len(merged_df)} baris)"
    )

    return merged_df


def build_all_final_datasets():
    """Membuat dataset_final.csv (InSet Lexicon) dan dataset_final_indobert.csv (IndoBERT)."""
    print("[Build All Datasets] Creating dataset_final.csv (InSet Lexicon)...")
    build_final_dataset(
        sentiment_file=INTERIM_DATA_DIR / "daily_sentiment.csv",
        output_file=PROCESSED_DATA_DIR / "dataset_final.csv"
    )
    
    print("[Build All Datasets] Creating dataset_final_indobert.csv (IndoBERT)...")
    build_final_dataset(
        sentiment_file=INTERIM_DATA_DIR / "daily_sentiment_indobert.csv",
        output_file=PROCESSED_DATA_DIR / "dataset_final_indobert.csv"
    )


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    build_all_final_datasets()

