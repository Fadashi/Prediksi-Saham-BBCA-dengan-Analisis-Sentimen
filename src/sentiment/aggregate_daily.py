"""Modul agregasi sentimen harian disesuaikan dengan skema notebook.

Menghasilkan 4 fitur sentimen:
- sentiment_score
- positive_ratio
- negative_ratio
- discussion_volume
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from src.sentiment.lexicon_scorer import process_sentiment_for_dataframe


def aggregate_daily_sentiment(
    input_file: Path = None,
    output_file: Path = None,
    num_classes: int = 2,
) -> pd.DataFrame:
    """Mengagregasikan data postingan mentah menjadi fitur sentimen harian."""
    input_file = input_file or (RAW_DATA_DIR / "stream_bbca.jsonl")
    output_file = output_file or (INTERIM_DATA_DIR / "daily_sentiment.csv")

    if not input_file.exists():
        print(f"[Aggregator Warning] File input {input_file} belum ditemukan.")
        print("[Aggregator] Membuat dataset sampel sentimen harian...")
        dates = pd.date_range(start="2021-01-01", end="2026-07-01", freq="D")
        np.random.seed(42)
        daily_df = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "sentiment_score": np.random.uniform(-1.5, 3.5, size=len(dates)),
                "positive_ratio": np.random.uniform(0.3, 0.7, size=len(dates)),
                "negative_ratio": np.random.uniform(0.1, 0.4, size=len(dates)),
                "discussion_volume": np.random.randint(10, 500, size=len(dates)),
            }
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output_file, index=False)
        print(f"[Aggregator] Sample daily sentiment disimpan di {output_file}")
        return daily_df

    # Baca file JSONL postingan
    print(f"[Aggregator] Membaca postingan dari {input_file}...")
    posts = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    posts.append(json.loads(line))
                except Exception:
                    continue

    df_posts = pd.DataFrame(posts)
    if df_posts.empty or "created_at" not in df_posts.columns:
        raise ValueError("Data postingan kosong atau format tidak sesuai.")

    # 1. Deduplikasi Ketat (Berdasarkan ID & Konten Teks)
    n_before = len(df_posts)
    if "id" in df_posts.columns:
        df_posts.drop_duplicates(subset=["id"], inplace=True)
    df_posts.drop_duplicates(subset=["content"], inplace=True)
    print(f"[Aggregator Deduplikasi] {n_before} -> {len(df_posts)} postingan unik")

    # 2. Penyelarasan Postingan Akhir Pekan (Sabtu & Minggu dipetakan ke hari Senin)
    dt = pd.to_datetime(df_posts["created_at"])
    shifts = dt.dt.dayofweek.map(
        {5: pd.Timedelta(days=2), 6: pd.Timedelta(days=1)}
    ).fillna(pd.Timedelta(days=0))
    adjusted_dt = dt + shifts
    df_posts["Date"] = adjusted_dt.dt.strftime("%Y-%m-%d")

    # Scorer sentimen
    df_scored = process_sentiment_for_dataframe(
        df_posts, text_column="content", num_classes=num_classes
    )

    # Agregasi per tanggal (Metode Net Sentiment Spread NSS - Tetlock 2007 & Bollen et al. 2011)
    print("[Aggregator] Mengkalkulasi metrik agregasi harian (NSS & Volume Weighting)...")
    daily_records = []
    for date_str, group in df_scored.groupby("Date"):
        total_vol = len(group)
        pos_count = (group["sentiment_label"] == 1).sum()
        neg_count = (group["sentiment_label"] == -1).sum()
        raw_mean_score = group["sentiment_score"].mean()

        pos_ratio = pos_count / total_vol if total_vol > 0 else 0.0
        neg_ratio = neg_count / total_vol if total_vol > 0 else 0.0

        # Net Sentiment Spread (NSS = Pos_Ratio - Neg_Ratio)
        nss = pos_ratio - neg_ratio
        
        # Combined Sentiment Score (Raw Score x Net Sentiment Spread)
        combined_score = raw_mean_score * (1.0 + nss)

        daily_records.append(
            {
                "Date": date_str,
                "sentiment_score": combined_score,
                "positive_ratio": pos_ratio,
                "negative_ratio": neg_ratio,
                "discussion_volume": total_vol,
            }
        )

    daily_df = pd.DataFrame(daily_records)
    daily_df.sort_values("Date", inplace=True)

    # 3. Exponential Moving Average Sentimen (EMA-3 untuk meredam noise & menangkap memori lag sentimen)
    daily_df["sentiment_score"] = (
        daily_df["sentiment_score"].ewm(span=3, adjust=False).mean().round(4)
    )
    daily_df["positive_ratio"] = (
        daily_df["positive_ratio"].ewm(span=3, adjust=False).mean().round(4)
    )
    daily_df["negative_ratio"] = (
        daily_df["negative_ratio"].ewm(span=3, adjust=False).mean().round(4)
    )



    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(output_file, index=False)
    print(
        f"[Aggregator Selesai] Fitur sentimen harian ({len(daily_df)} tanggal) disimpan: {output_file}"
    )


    return daily_df


if __name__ == "__main__":
    aggregate_daily_sentiment()
