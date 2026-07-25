# Blueprint & Implementation Plan
## Sistem Prediksi Pergerakan Harga Saham BBCA Berbasis Deep Learning dengan Pemanfaatan Analisis Sentimen Diskusi Investor (Stockbit)

> **Dokumen ini adalah spesifikasi kerja untuk dieksekusi bersama Claude Code.**
> Baca bagian [Cara Memakai Dokumen Ini dengan Claude Code](#0-cara-memakai-dokumen-ini-dengan-claude-code) lebih dulu.
> Semua kode ditulis dalam **Python**; komentar & dokumen boleh Bahasa Indonesia.

---

## Daftar Isi
0. [Cara Memakai Dokumen Ini dengan Claude Code](#0-cara-memakai-dokumen-ini-dengan-claude-code)
1. [Konteks & Tujuan](#1-konteks--tujuan)
2. [Ruang Lingkup & Keputusan Desain](#2-ruang-lingkup--keputusan-desain)
3. [Arsitektur Sistem (7 Lapisan)](#3-arsitektur-sistem-7-lapisan)
4. [Tech Stack](#4-tech-stack)
5. [Struktur Repositori](#5-struktur-repositori)
6. [Spesifikasi Pipeline Data](#6-spesifikasi-pipeline-data)
7. [Metodologi Analisis Sentimen](#7-metodologi-analisis-sentimen)
8. [Rekayasa Fitur & Skenario Eksperimen](#8-rekayasa-fitur--skenario-eksperimen)
9. [Pemodelan Deep Learning](#9-pemodelan-deep-learning)
10. [Protokol Evaluasi](#10-protokol-evaluasi)
11. [Aplikasi / GUI (Streamlit)](#11-aplikasi--gui-streamlit)
12. [Implementation Plan (Fase & Checklist)](#12-implementation-plan-fase--checklist)
13. [Konvensi Kode & Kualitas](#13-konvensi-kode--kualitas)
14. [Traceability ke Catatan Revisi Penguji](#14-traceability-ke-catatan-revisi-penguji)
15. [Risiko & Mitigasi](#15-risiko--mitigasi)
16. [Lampiran: Perintah & Dependensi](#16-lampiran-perintah--dependensi)

---

## 0. Cara Memakai Dokumen Ini dengan Claude Code

1. Letakkan file ini di root project sebagai `BLUEPRINT.md`.
2. Buat file `AGENTS.md` (atau `CLAUDE.md`) berisi ringkasan aturan main dari bagian [13](#13-konvensi-kode--kualitas) agar Claude Code selalu membacanya.
3. Kerjakan **per fase** di bagian [12](#12-implementation-plan-fase--checklist). Untuk tiap fase, gunakan prompt seperti:
   > "Baca `BLUEPRINT.md` bagian Fase N. Implementasikan semua task di fase itu, ikuti struktur repositori & konvensi kode. Buatkan test bila disebutkan, lalu jalankan dan tunjukkan hasilnya."
4. Jangan lompati urutan fase — tiap fase punya **Definition of Done (DoD)**. Minta Claude Code memenuhi DoD sebelum lanjut.
5. Commit di akhir setiap fase (`git commit`) dengan pesan yang jelas.

---

## 1. Konteks & Tujuan

**Latar:** Penelitian skripsi — memprediksi pergerakan harga saham **BBCA** dengan memanfaatkan sinyal **sentimen diskusi investor** dari aplikasi **Stockbit**, dipadukan dengan **indikator teknikal** harga historis. Tiga arsitektur deep learning (**LSTM, GRU, CNN 1D**) dibandingkan untuk memilih mesin prediksi terbaik, lalu dibungkus menjadi **sistem prediksi interaktif ber-GUI**.

**Tujuan sistem:**
1. Mengumpulkan & memproses data harga historis + sentimen Stockbit.
2. Membuktikan **kontribusi fitur sentimen** terhadap akurasi prediksi (skenario tanpa vs dengan sentimen).
3. Memilih model terbaik dan menyajikannya sebagai **aplikasi prediksi** yang bisa dipakai (grafik prediksi, metrik, indikator sentimen, word cloud).

**Metrik utama:** MAPE (utama), didukung RMSE, MAE.

---

## 2. Ruang Lingkup & Keputusan Desain

| Aspek | Keputusan | Catatan / perlu konfirmasi pembimbing |
|---|---|---|
| Saham | BBCA | Justifikasi: blue-chip, likuiditas tinggi, volume diskusi besar |
| Sumber sentimen | Stockbit Stream (ticker BBCA) | Tanpa API resmi → endpoint internal + token |
| Periode harga | ≈ 10 tahun | Mudah via `yfinance` (`BBCA.JK`) |
| Periode sentimen | **Window padat hasil pilot** (mis. ~2021–2026) | Ditentukan dari `volume_summary.csv`; butuh justifikasi |
| Metode sentimen | Lexicon-based **InSet Lexicon** | Kebaruan diperkuat di sisi **validasi dataset** |
| Kelas sentimen | **Positif / Negatif** (netral dipertimbangkan dibuang) | Keputusan final via eksperimen kualitas label |
| Model | LSTM, GRU, CNN 1D | Perbandingan = tahap pemilihan mesin sistem |
| Target prediksi | Harga penutupan (regresi) + arah (opsional) | Tetapkan 1: regresi harga (default) |
| GUI | Streamlit | Prototipe fungsional, bukan aplikasi produksi |

---

## 3. Arsitektur Sistem (7 Lapisan)

```
[1] SUMBER DATA        Stockbit Stream (teks)        Harga historis BBCA (OHLCV)
                              |                                |
[2] AKUISISI           Crawler stream  ------------  Downloader harga (yfinance)
                              |                                |
[3] PEMROSESAN     Jalur A (teks/sentimen)        Jalur B (harga/teknikal)
                   - preprocessing teks            - preprocessing harga
                   - InSet Lexicon scoring         - indikator teknikal (9 fitur)
                   - agregasi harian + validasi
                              \                              /
[4] REKAYASA FITUR         Gabung fitur -> dataset final (skenario 9 vs 13 fitur)
                              |
[5] PEMODELAN          LSTM | GRU | CNN 1D -> tuning -> evaluasi -> PILIH TERBAIK
                              |
[6] APLIKASI (GUI)     Streamlit: grafik aktual vs prediksi, metrik,
                       indikator sentimen, word cloud
                              |
[7] PENGGUNA           Investor / analis (pendukung keputusan)
```

---

## 4. Tech Stack

| Kategori | Pilihan |
|---|---|
| Bahasa | Python 3.10+ |
| Data | pandas, numpy |
| Crawling | requests, (fallback: playwright / selenium) |
| Harga | yfinance |
| NLP / sentimen | Sastrawi (stemming), nltk, kamus **InSet Lexicon**, wordcloud |
| Deep Learning | TensorFlow / Keras (atau PyTorch — pilih satu, default Keras) |
| Tuning | KerasTuner atau grid manual + seed control |
| Evaluasi | scikit-learn (metrics), matplotlib/plotly |
| GUI | Streamlit, plotly |
| Konfigurasi | YAML (`config.yaml`) + python-dotenv (`.env` untuk token) |
| Kualitas | black, ruff, pytest |
| Reproduksibilitas | seed global, `requirements.txt`, `README.md` |

---

## 5. Struktur Repositori

```
stockbit-bbca-forecast/
├─ README.md
├─ BLUEPRINT.md                # dokumen ini
├─ AGENTS.md                   # aturan singkat untuk Claude Code
├─ requirements.txt
├─ config.yaml                 # semua parameter (window, hyperparam, path)
├─ .env.example                # STOCKBIT_TOKEN, STREAM_ENDPOINT
├─ .gitignore
├─ data/
│  ├─ raw/                     # data mentah (jsonl, csv) — tidak di-commit
│  ├─ interim/                 # hasil setengah proses
│  └─ processed/               # dataset final siap model
├─ models/                     # artefak model terlatih (.h5/.keras) + scaler
├─ reports/
│  ├─ figures/                 # grafik hasil (loss, prediksi, wordcloud)
│  └─ metrics/                 # tabel metrik per model & skenario (csv)
├─ lexicon/                    # kamus InSet (positive.tsv, negative.tsv)
├─ src/
│  ├─ config.py                # loader config.yaml + seed
│  ├─ data/
│  │  ├─ crawl_stream.py       # crawler Stockbit (produksi, dari pilot)
│  │  ├─ fetch_prices.py       # unduh harga via yfinance
│  │  └─ build_dataset.py      # gabung -> data/processed
│  ├─ sentiment/
│  │  ├─ preprocess_text.py    # cleaning, normalisasi, stopword, stemming
│  │  ├─ lexicon_scorer.py     # InSet Lexicon -> skor & label
│  │  ├─ aggregate_daily.py    # agregasi sentimen harian
│  │  └─ validate_labels.py    # validasi kualitas label (sampling, metrik)
│  ├─ features/
│  │  ├─ technical.py          # 9 indikator teknikal
│  │  └─ assemble.py           # skenario 9 vs 13 fitur, scaling, windowing
│  ├─ models/
│  │  ├─ architectures.py      # build_lstm(), build_gru(), build_cnn()
│  │  ├─ train.py              # training loop + callbacks
│  │  ├─ tune.py               # hyperparameter search
│  │  └─ evaluate.py           # MAPE/RMSE/MAE + plot, pilih terbaik
│  └─ utils/
│     ├─ io.py
│     └─ plotting.py
├─ app/
│  └─ streamlit_app.py         # GUI sistem prediksi
└─ tests/
   ├─ test_preprocess.py
   ├─ test_technical.py
   └─ test_features.py
```

---

## 6. Spesifikasi Pipeline Data

### 6.1 Crawling sentimen (`crawl_stream.py`)
- Basis dari skrip pilot yang sudah teruji. Tambahkan: checkpoint/resume (simpan cursor terakhir), dedup by post id, penyimpanan `data/raw/stream_bbca.jsonl`.
- Field minimal yang disimpan: `id`, `created_at`, `content`, (opsional) `like/agree count`.
- **Anonimkan**: jangan simpan username/PII pada dataset olahan.
- Rate limit: jeda + retry pada HTTP 429.

### 6.2 Harga (`fetch_prices.py`)
- `yfinance.download("BBCA.JK", start=..., end=...)` → `data/raw/prices_bbca.csv`.
- Kolom: Date, Open, High, Low, Close, Volume.

### 6.3 Pembentukan dataset (`build_dataset.py`)
- Gabung sentimen harian + harga + fitur teknikal pada indeks tanggal.
- Tangani hari non-bursa & hari tanpa post (isi 0 / forward-fill sesuai aturan yang didokumentasikan).
- Output: `data/processed/dataset_final.csv`.

**DoD data layer:** dataset final punya kolom lengkap, tidak ada kebocoran masa depan (no look-ahead), rentang tanggal sesuai window terpilih.

---

## 7. Metodologi Analisis Sentimen

### 7.1 Preprocessing teks (`preprocess_text.py`)
Urutan: case folding → hapus URL/mention/emoji/cashtag → normalisasi slang (kamus) → tokenizing → stopword removal (ID) → stemming (Sastrawi).

### 7.2 Pelabelan lexicon (`lexicon_scorer.py`)
- Skor = Σ bobot kata positif − Σ bobot kata negatif (InSet Lexicon).
- Aturan label: skor > 0 → positif; skor < 0 → negatif; skor = 0 → netral.
- **Keputusan kelas netral** (jawab Bu Dian): jalankan dua varian —
  (a) 3 kelas (pos/net/neg), (b) 2 kelas (buang netral). Bandingkan dampaknya ke performa & kualitas; pilih yang paling baik + dokumentasikan.

### 7.3 Validasi dataset (`validate_labels.py`) — KEBARUAN
- Ambil sampel acak (mis. 300–400 post), lakukan **anotasi manual/expert** sebagai ground truth.
- Hitung **akurasi/precision/recall/F1** label lexicon vs ground truth + **Cohen's Kappa** (kesepakatan).
- Laporkan di `reports/metrics/sentiment_validation.csv`. Ini bukti validitas dataset yang diminta penguji.

### 7.4 Agregasi harian (`aggregate_daily.py`)
- Per tanggal: proporsi positif, proporsi negatif, skor sentimen rata-rata, jumlah post (volume).
- Fitur sentimen final (contoh 4): `sent_pos_ratio`, `sent_neg_ratio`, `sent_mean_score`, `sent_volume`.

---

## 8. Rekayasa Fitur & Skenario Eksperimen

### 8.1 Indikator teknikal (`technical.py`) — 9 fitur
Contoh set (sesuaikan dg skripsi): `Close`, `MA5`, `MA10`, `MA20`, `RSI`, `MACD`, `Bollinger%`, `Volume`, `Return`.

### 8.2 Skenario (jawaban "urgensi" Bu Dian)
| Skenario | Fitur | Tujuan |
|---|---|---|
| **S1 – Baseline** | 9 fitur teknikal | Kontrol tanpa sentimen |
| **S2 – Sentimen** | 9 teknikal + 4 sentimen = **13 fitur** | Uji kontribusi sentimen |

Dua skenario × tiga model = **6 konfigurasi** dievaluasi dengan setup identik.

### 8.3 Windowing & scaling (`assemble.py`)
- Normalisasi (MinMaxScaler) — **fit di train saja**, transform ke val/test.
- Sliding window (mis. lookback = 30 hari) → supervised sequences.
- Split **kronologis** (bukan acak): train / val / test (mis. 70/15/15) untuk mencegah kebocoran waktu.

---

## 9. Pemodelan Deep Learning

### 9.1 Arsitektur (`architectures.py`)
- `build_lstm(...)`, `build_gru(...)`, `build_cnn1d(...)` — antarmuka seragam (input_shape, params) agar mudah dibandingkan adil.
- Output: 1 neuron (regresi harga penutupan berikutnya).

### 9.2 Training (`train.py`)
- Loss MSE, optimizer Adam. Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
- **Seed global** + simpan history loss ke `reports/figures/`.

### 9.3 Tuning (`tune.py`)
- Grid/random search: units, layers, dropout, learning_rate, batch_size, lookback.
- Simpan konfigurasi terbaik per model ke `config.yaml`/`models/`.

---

## 10. Protokol Evaluasi

- Metrik: **MAPE (utama)**, RMSE, MAE (jelaskan makna tiap metrik — jawab Pak Radit).
- Inverse-transform prediksi ke skala harga asli sebelum menghitung metrik.
- Tabel hasil: `reports/metrics/results.csv` (kolom: model, skenario, MAPE, RMSE, MAE).
- Grafik: aktual vs prediksi tiap model; bar-chart perbandingan MAPE.
- **Pemilihan model terbaik**: MAPE test terendah pada skenario S2 → jadi mesin GUI.
- Uji konsistensi: multi-run (mis. 3× seed) lalu laporkan rata-rata ± std.

**DoD modeling:** tabel 6 konfigurasi lengkap + model terbaik tersimpan di `models/best_model.keras` beserta scaler.

---

## 11. Aplikasi / GUI (Streamlit)

`app/streamlit_app.py` — fitur:
1. **Sidebar**: pilih rentang tanggal / tombol refresh data; pilih model (default: terbaik).
2. **Panel utama**:
   - Grafik **harga aktual vs prediksi** (plotly, interaktif).
   - Kartu **nilai prediksi** + metrik (MAPE/RMSE) periode uji.
   - **Indikator sentimen**: ringkasan pos/neg + tren harian.
   - **Word cloud** dari diskusi terbaru (jawab Pak Hendri & Penguji 3).
3. Model & scaler di-load **sekali** saat start (cache `@st.cache_resource`) — tidak melatih ulang.
4. Sisipkan disclaimer: bukan saran finansial.

**DoD GUI:** `streamlit run app/streamlit_app.py` jalan tanpa error dan menampilkan keempat komponen.

---

## 12. Implementation Plan (Fase & Checklist)

> Kerjakan berurutan. Centang tiap task; penuhi **DoD** sebelum lanjut fase berikut.

### Fase 0 — Setup Proyek
- [ ] Inisialisasi repo, `venv`, `requirements.txt`, `.gitignore`.
- [ ] Buat `config.yaml`, `.env.example`, `src/config.py` (loader + seed global).
- [ ] Buat `AGENTS.md` dari bagian [13](#13-konvensi-kode--kualitas).
- **DoD:** `python -c "import src.config"` sukses; struktur folder sesuai bagian [5](#5-struktur-repositori).

### Fase 1 — Akuisisi Data
- [ ] `fetch_prices.py`: unduh harga BBCA → `data/raw/prices_bbca.csv`.
- [ ] `crawl_stream.py`: crawler produksi (checkpoint, dedup, resume) berdasarkan pilot.
- [ ] Tentukan **window periode** dari `volume_summary.csv` → tulis ke `config.yaml`.
- **DoD:** data mentah harga & stream tersimpan untuk window terpilih.

### Fase 2 — Analisis Sentimen
- [ ] `preprocess_text.py` + test.
- [ ] `lexicon_scorer.py` (varian 2-kelas & 3-kelas).
- [ ] `validate_labels.py` → metrik validasi + Kappa.
- [ ] `aggregate_daily.py` → fitur sentimen harian.
- **DoD:** `reports/metrics/sentiment_validation.csv` ada; keputusan netral terdokumentasi.

### Fase 3 — Rekayasa Fitur
- [ ] `technical.py` (9 indikator) + test.
- [ ] `assemble.py`: skenario S1 & S2, scaling (fit-train-only), windowing, split kronologis.
- [ ] `build_dataset.py` → `data/processed/dataset_final.csv`.
- **DoD:** array train/val/test untuk S1 & S2 tersimpan; tidak ada look-ahead.

### Fase 4 — Pemodelan & Evaluasi
- [ ] `architectures.py` (LSTM/GRU/CNN antarmuka seragam).
- [ ] `train.py` + callbacks + simpan history.
- [ ] `tune.py` hyperparameter search.
- [ ] `evaluate.py`: metrik + grafik + pilih terbaik.
- [ ] Jalankan **6 konfigurasi** (3 model × 2 skenario), multi-seed.
- **DoD:** `reports/metrics/results.csv` lengkap; `models/best_model.keras` + scaler tersimpan.

### Fase 5 — Aplikasi GUI
- [ ] `streamlit_app.py`: grafik prediksi, metrik, indikator sentimen, word cloud.
- [ ] Caching model/scaler; disclaimer.
- **DoD:** app jalan lokal & menampilkan semua komponen.

### Fase 6 — Finalisasi & Reproduksibilitas
- [ ] `README.md`: cara setup, jalankan pipeline end-to-end, jalankan app.
- [ ] Skrip orkestrasi `make all` / `run_pipeline.py` (crawl→sentimen→fitur→train→evaluate).
- [ ] Rapikan `reports/figures` untuk dilampirkan ke skripsi (diagram alur, wordcloud, grafik prediksi).
- **DoD:** pipeline bisa dijalankan ulang dari nol mengikuti README.

---

## 13. Konvensi Kode & Kualitas

- **Struktur:** semua logika di `src/`, fungsi kecil & dapat diuji; tidak ada hardcode path (pakai `config.yaml`).
- **Reproduksibilitas:** set seed (numpy, random, TF) di `src/config.py`; scaler & model disimpan sebagai artefak.
- **No data leakage:** scaler fit hanya di train; split kronologis; tidak memakai info masa depan.
- **Gaya:** `black` + `ruff`; docstring singkat Bahasa Indonesia; type hints bila memungkinkan.
- **Testing:** `pytest` untuk preprocessing, indikator teknikal, dan windowing.
- **Rahasia:** token Stockbit hanya di `.env` (jangan commit). `.env.example` sebagai contoh.
- **Git:** commit per fase, pesan jelas (mis. `feat(sentiment): tambah validasi label`).
- **Data besar:** `data/raw/` & `models/` masuk `.gitignore`.

---

## 14. Traceability ke Catatan Revisi Penguji

| Catatan penguji | Ditangani oleh |
|---|---|
| Fokus tak jelas (sentimen vs prediksi vs banding) | Reframe ke **sistem**; perbandingan = tahap pemilihan (Fase 4) |
| Judul–isi tak korelasi | Skenario S1/S2 + output sistem menjawab judul |
| Urgensi pengujian model (Bu Dian 3) | Skenario S1 vs S2 membuktikan nilai sentimen |
| Data terlalu pendek (semua penguji) | Harga ~10th; window sentimen padat berbasis pilot |
| Validasi dataset sentimen (Bu Dian 11) | `validate_labels.py` + Kappa (Fase 2) |
| Kualitas kelas netral (Bu Dian 12–13) | Varian 2 vs 3 kelas, pilih terbaik |
| Justifikasi "hanya sentimen" (Bu Dian 9) | Ditulis di laporan + jurnal pendukung |
| Definisi & makna metrik (Pak Radit 5) | Protokol evaluasi (MAPE/RMSE/MAE) bagian [10](#10-protokol-evaluasi) |
| Word cloud (Pak Hendri, Penguji 3) | Komponen GUI + `reports/figures` |
| Diagram alur metode (Penguji 3) | Arsitektur 7 lapisan bagian [3](#3-arsitektur-sistem-7-lapisan) |

---

## 15. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Token Stockbit kedaluwarsa saat crawl panjang | Crawl putus | Checkpoint/resume + refresh token |
| Volume post sangat besar | Proses berat | Batasi window padat; simpan interim; batch processing |
| Kualitas label netral rendah | Bias hasil | Varian 2-kelas + validasi manual |
| Data tahun lama tipis | Sinyal lemah | Justifikasi window; kecualikan tahun keropos |
| Overfitting model | Metrik test buruk | EarlyStopping, dropout, multi-seed |
| ToS / etika data | Masalah kepatuhan | Akademik saja, anonimisasi, rate limit sopan |

---

## 16. Lampiran: Perintah & Dependensi

### 16.1 `requirements.txt` (awal)
```
pandas
numpy
yfinance
requests
Sastrawi
nltk
wordcloud
scikit-learn
tensorflow
keras-tuner
matplotlib
plotly
streamlit
pyyaml
python-dotenv
black
ruff
pytest
```

### 16.2 Perintah umum
```bash
# setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# pipeline end-to-end (contoh, sesuaikan setelah dibuat)
python -m src.data.fetch_prices
python -m src.data.crawl_stream
python -m src.sentiment.lexicon_scorer
python -m src.sentiment.validate_labels
python -m src.sentiment.aggregate_daily
python -m src.features.assemble
python -m src.models.train
python -m src.models.evaluate

# jalankan aplikasi
streamlit run app/streamlit_app.py

# kualitas
black src app && ruff check src app && pytest -q
```

### 16.3 Contoh `config.yaml`
```yaml
symbol: BBCA
price_ticker: BBCA.JK
period:
  price_start: "2016-01-01"
  sentiment_start: "2021-01-01"   # dari hasil pilot; sesuaikan
  end: "2026-07-01"
sentiment:
  classes: 2          # 2 (buang netral) atau 3
features:
  lookback: 30
  technical: [close, ma5, ma10, ma20, rsi, macd, boll_pct, volume, return]
split:
  train: 0.70
  val: 0.15
  test: 0.15
seed: 42
```

---

*Dokumen ini dirancang sebagai spesifikasi eksekusi. Jalankan fase demi fase bersama Claude Code, penuhi setiap Definition of Done, dan commit di akhir tiap fase.*
