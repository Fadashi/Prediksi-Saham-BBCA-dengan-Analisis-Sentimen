# Sistem Prediksi Pergerakan Harga Saham BBCA Berbasis Deep Learning & Analisis Sentimen Stockbit

Repositori ini berisi implementasi lengkap **Sistem Prediksi Harga Saham BBCA** berbasis Deep Learning (LSTM, GRU, 1D CNN) yang memanfaatkan analisis sentimen diskusi investor dari Stockbit (menggunakan InSet Lexicon) serta 9 indikator teknikal.

---

## 🛠️ Tech Stack & Requirements
- **Bahasa:** Python 3.10+
- **Data & Analisis:** Pandas, NumPy, yfinance, Scikit-learn
- **NLP & Lexicon:** InSet Lexicon (diunduh dari GitHub), Sastrawi, NLTK, WordCloud
- **Deep Learning:** PyTorch (`torch`)
- **GUI Interaktif:** Streamlit, Plotly
- **Testing & Quality:** Pytest, Black, Ruff

---

## 🚀 Panduan Setup & Eksekusi

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline End-to-End
Jalankan alur pengunduhan data harga, pemrosesan sentimen, perakitan fitur S1 (baseline) vs S2 (+sentimen), pelatihan 6 konfigurasi model, dan seleksi model terbaik:
```bash
python run_pipeline.py
```

### 3. Menjalankan Unit Tests
```bash
pytest -q
```

### 4. Menjalankan Aplikasi Web GUI (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

---

## 📁 Struktur Repositori
```
Code Akhir/Sistem/
├── README.md                           # Panduan penggunaan
├── BLUEPRINT_Sistem_Prediksi_Saham_BBCA.md # Blueprint spesifikasi teknis
├── AGENTS.md                           # Panduan konvensi kode
├── requirements.txt                    # Daftar dependensi Python
├── config.yaml                         # Konfigurasi parameter & rentang tanggal
├── run_pipeline.py                     # Skrip eksekusi pipeline end-to-end
├── data/                               # Direktori data (raw, interim, processed)
├── models/                             # Artefak model (.pt) & scaler (.pkl)
├── reports/                            # Laporan metrik CSV & grafik PNG
├── lexicon/                            # Kamus InSet Lexicon (positive.tsv, negative.tsv)
├── src/                                # Source code modul utama
│   ├── config.py                       # Loader konfigurasi & global seed
│   ├── data/                           # Module fetch harga, crawler & build dataset
│   ├── sentiment/                      # Preprocessing, lexicon scorer & agregator
│   ├── features/                       # Fitur teknikal 9 & perakitan S1/S2
│   ├── models/                         # Arsitektur PyTorch, train, tune, evaluate
│   └── utils/                          # Utilitas visualisasi & I/O
├── app/
│   └── streamlit_app.py                # Aplikasi GUI Streamlit & Plotly
└── tests/                              # Unit test suite
```

---

## 📊 Hasil Eksperimen (S1 vs S2)
Eksperimen mengevaluasi 6 konfigurasi (3 Arsitektur Deep Learning × 2 Skenario):
- **S1 (Baseline):** 9 Fitur Teknikal (`Close`, `MA5`, `MA10`, `MA20`, `RSI`, `MACD`, `Bollinger%`, `Volume`, `Return`).
- **S2 (Teknikal + Sentimen Stockbit):** 9 Fitur Teknikal + 4 Fitur Sentimen (`sent_pos_ratio`, `sent_neg_ratio`, `sent_mean_score`, `sent_volume`).
- Model terbaik dengan MAPE terendah secara otomatis disimpan ke `models/best_model.pt` dan dimuat oleh aplikasi Streamlit.
