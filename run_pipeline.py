"""Skrip otomasi pipeline end-to-end Sistem Prediksi Saham BBCA.

Menjalankan seluruh alur:
1. Fetch Harga Saham BBCA
2. InSet Lexicon Download & Sentiment Validation
3. Agregasi Sentimen Harian
4. Build Dataset Final (Gabungan Harga + Sentimen)
5. Skenario Feature Assembly (S1 vs S2)
6. Pelatihan & Evaluasi 6 Konfigurasi (LSTM, GRU, CNN 1D x S1, S2)
7. Pemilihan & Pembuatan Artefak Model Terbaik (best_model.pt)
"""

import sys
import time
from pathlib import Path

# Registrasi root project
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import CONFIG, set_seed
from src.data.build_dataset import build_final_dataset
from src.data.fetch_prices import fetch_bbca_prices
from src.features.assemble import prepare_scenario_data
from src.models.evaluate import evaluate_all_configurations
from src.sentiment.aggregate_daily import aggregate_daily_sentiment
from src.sentiment.validate_labels import validate_sentiment_labels


def main():
    start_time = time.time()
    set_seed(CONFIG["seed"])

    print("==========================================================")
    print("  MEMULAI EKSREKUSI PIPELINE END-TO-END PREDIKSI BBCA")
    print("==========================================================")

    # 1. Fetch Harga BBCA
    print("\n--- STEP 1: Mengunduh Data Harga Saham BBCA ---")
    fetch_bbca_prices()

    # 2. Validasi Sentiment Labeling (InSet Lexicon)
    print("\n--- STEP 2: Melakukan Validasi Dataset Sentimen (InSet Lexicon) ---")
    validate_sentiment_labels()

    # 3. Agregasi Sentimen Harian
    print("\n--- STEP 3: Mengagregasi Sentimen Harian ---")
    aggregate_daily_sentiment(num_classes=CONFIG["sentiment"]["classes"])

    # 4. Build Dataset Final
    print("\n--- STEP 4: Membangun Dataset Final (Harga + Sentimen) ---")
    build_final_dataset()

    # 5. Feature Assembly Check
    print("\n--- STEP 5: Memverifikasi Perakitan Fitur S1 & S2 ---")
    prepare_scenario_data(scenario="S1")
    prepare_scenario_data(scenario="S2")

    # 6. Pemodelan & Evaluasi 6 Konfigurasi
    print(
        "\n--- STEP 6: Melatih & Mengevaluasi 6 Konfigurasi Model (StockLSTM, StockGRU, StockCNN) ---"
    )
    evaluate_all_configurations(selected_cfg_id="K3", runs=3)


    elapsed = time.time() - start_time
    print("==========================================================")
    print(f" PIPELINE SELESAI DENGAN SUKSES DALAM {elapsed:.2f} DETIK!")
    print(" Jalankan GUI dengan perintah: streamlit run app/streamlit_app.py")
    print("==========================================================")


if __name__ == "__main__":
    main()
