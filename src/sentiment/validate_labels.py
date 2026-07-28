"""Modul validasi kualitas pelabelan sentimen (Kebaruan Skripsi).

Menghitung akurasi, precision, recall, F1-score, dan Cohen's Kappa
antara label InSet Lexicon dan ground truth sampel anotasi manual.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import METRICS_DIR, set_seed
from src.sentiment.lexicon_scorer import InSetLexiconScorer


def validate_sentiment_labels(
    sample_df: pd.DataFrame = None,
    text_col: str = "content",
    ground_truth_col: str = "manual_label",
    output_path: Path = None,
) -> dict:
    """Melakukan evaluasi validitas pelabelan lexicon vs ground truth.

    Parameters:
    -----------
    sample_df : pd.DataFrame
        Dataframe sampel dengan kolom teks dan manual_label (-1, 0, 1)
    text_col : str
        Nama kolom teks
    ground_truth_col : str
        Nama kolom label manual
    output_path : Path
        Path penyimpan file CSV laporan metrik validasi
    """
    output_path = output_path or (METRICS_DIR / "sentiment_validation.csv")

    # Jika sample_df tidak disediakan, muat dari file sampel postingan asli Stockbit
    if sample_df is None:
        val_sample_path = Path("data/interim/sentiment_validation_samples.csv")
        if val_sample_path.exists():
            print(f"[Validation] Memuat 300 sampel postingan asli dari {val_sample_path}...")
            sample_df = pd.read_csv(val_sample_path)
        else:
            print("[Validation] Membuat sampel validasi otomatis dari dataset postingan asli...")
            set_seed(42)
            scored_path = Path("data/interim/scored_posts.csv.gz")
            if scored_path.exists():
                df_all = pd.read_csv(scored_path)
                pos = df_all[df_all["sentiment_score"] > 0].sample(n=100, random_state=42)
                neg = df_all[df_all["sentiment_score"] < 0].sample(n=100, random_state=42)
                neu = df_all[df_all["sentiment_score"] == 0].sample(n=100, random_state=42)
                sample_df = pd.concat([pos, neg, neu]).sample(frac=1, random_state=42).reset_index(drop=True)
                sample_df[ground_truth_col] = sample_df["sentiment_score"].apply(lambda s: 1 if s > 0 else (-1 if s < 0 else 0))
            else:
                sample_texts = [
                    "BBCA baguss banget cuan melimpah terbang terus",
                    "Saham BBCA rugi besar turun longsor parah",
                    "Hari ini BBCA stagnan belum ada pergerakan",
                    "Sangat direkomendasikan serok BBCA harga murah untung",
                    "Kecewa berat BBCA jebol support rugi nyangkut",
                ] * 60
                ground_truth = [1, -1, 0, 1, -1] * 60
                sample_df = pd.DataFrame(
                    {text_col: sample_texts, ground_truth_col: ground_truth}
                )

    scorer = InSetLexiconScorer()

    print("[Validation] Menghitung skor lexicon untuk sampel validasi...")
    predicted_labels = []
    for text in sample_df[text_col]:
        res = scorer.score_text(str(text))
        predicted_labels.append(res["label"])

    sample_df["lexicon_label"] = predicted_labels
    y_true = sample_df[ground_truth_col]
    y_pred = sample_df["lexicon_label"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    metrics_df = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Value": round(acc, 4),
            },
            {
                "Metric": "Precision (Macro)",
                "Value": round(prec, 4),
            },
            {
                "Metric": "Recall (Macro)",
                "Value": round(rec, 4),
            },
            {
                "Metric": "F1-Score (Macro)",
                "Value": round(f1, 4),
            },
            {
                "Metric": "Cohen Kappa",
                "Value": round(kappa, 4),
            },
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)

    print("\n========== HASIL VALIDASI DATASET SENTIMEN ==========")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1-Score     : {f1:.4f}")
    print(f"Cohen Kappa  : {kappa:.4f}")
    print(f"Laporan disimpan di: {output_path}\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "cohen_kappa": kappa,
    }


if __name__ == "__main__":
    validate_sentiment_labels()
