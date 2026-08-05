"""Skrip untuk mengestrak 10.000 sampel postingan terlabeli InSet Lexicon
untuk keperluan pengujian dan validasi oleh Validator Ahli.

Kolom Output:
- content       : Teks asli postingan Stockbit
- clean_text    : Teks hasil preprocessing (folding, cleaning, stopwords, stemming)
- lexicon_score : Skor sentimen terbobot InSet Lexicon
- label         : Label sentimen (Positif (+1), Negatif (-1), Netral (0))
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(".").resolve()
INPUT_FILE = BASE_DIR / "data" / "interim" / "scored_posts.csv.gz"
OUTPUT_FILE_DATA = BASE_DIR / "data" / "interim" / "sampel_10000_inset_validator.csv"
OUTPUT_FILE_REPORTS = BASE_DIR / "reports" / "dokumentasi_skripsi" / "sampel_10000_inset_validator.csv"

print(f"[INFO] Membaca data berlabel InSet Lexicon dari: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"[INFO] Total data mentah terlabeli: {len(df):,} baris.")

# Sampling 10.000 baris secara terstrata berdasarkan label sentimen
sample_size = 10000
train_size = sample_size / len(df)

df_sampled, _ = train_test_split(
    df,
    train_size=sample_size,
    stratify=df["sentiment_label"],
    random_state=42
)

# Urutkan berdasarkan index agar rapi
df_sampled = df_sampled.sort_index().reset_index(drop=True)

# Pilih & ganti nama kolom sesuai spesifikasi persis user: content, clean_text, lexicon_score, label
df_out = df_sampled[["content", "cleaned_content", "sentiment_score", "sentiment_label"]].copy()
df_out.columns = ["content", "clean_text", "lexicon_score", "label"]

# Simpan ke CSV dengan encoding utf-8-sig
OUTPUT_FILE_DATA.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE_REPORTS.parent.mkdir(parents=True, exist_ok=True)

df_out.to_csv(OUTPUT_FILE_DATA, index=False, encoding="utf-8-sig")
df_out.to_csv(OUTPUT_FILE_REPORTS, index=False, encoding="utf-8-sig")

print(f"\n[SUCCESS] 10.000 sampel data berhasil diekstrak dan disimpan ke:")
print(f"  1. {OUTPUT_FILE_DATA}")
print(f"  2. {OUTPUT_FILE_REPORTS}")

print("\n--- DISTRIBUSI LABEL SENTIMEN PADA 10.000 SAMPEL VALIDATOR ---")
dist = df_out["label"].value_counts()
dist_pct = df_out["label"].value_counts(normalize=True) * 100
for lbl, count in dist.items():
    print(f"* {lbl:<20}: {count:>6,} baris ({dist_pct[lbl]:.2f}%)")

print("\n--- INFORMASI FILE RESULT ---")
print(f"Total baris   : {len(df_out):,}")
print(f"Kolom         : {list(df_out.columns)}")
