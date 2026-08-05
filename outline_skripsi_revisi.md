# OUTLINE REVISI SKRIPSI
## Sistem Prediksi Pergerakan Harga Saham BBCA Berbasis Deep Learning dengan Pemanfaatan Analisis Sentimen Diskusi Investor Stockbit

> **Revisi dari:** DRAFT FINAL_1_SKRIPSI_MFaishalDaffaS.pdf
> **Disesuaikan dengan:** Sistem implementasi aktual (Code Akhir)
> **Penulis:** M. Faishal Daffa S.

---

## HALAMAN JUDUL

**Judul yang Direkomendasikan (Revisi):**

> **"Sistem Prediksi Pergerakan Harga Saham BBCA Berbasis Deep Learning dengan Pemanfaatan Analisis Sentimen Diskusi Investor Stockbit"**

> [!TIP]
> Judul ini sudah merefleksikan 3 elemen inti: (1) **Sistem Prediksi** sebagai output utama, (2) **Deep Learning** sebagai metode pemodelan, dan (3) **Sentimen Stockbit** sebagai kontribusi kebaruan.

---

## ABSTRAK

**Poin-poin kunci yang harus tercakup dalam Abstrak:**

1. **Latar belakang singkat:** Kebutuhan prediksi harga saham yang akurat di pasar modal Indonesia, khususnya saham blue-chip BBCA.
2. **Masalah:** Prediksi berbasis indikator teknikal saja kurang menangkap sentimen pasar; diskusi investor di platform Stockbit mengandung sinyal sentimen yang belum dimanfaatkan secara sistematis.
3. **Tujuan:** Membangun sistem prediksi harga saham BBCA yang mengintegrasikan fitur sentimen diskusi investor Stockbit (dianalisis menggunakan InSet Lexicon) dengan indikator teknikal, dan membandingkan tiga arsitektur deep learning (LSTM, GRU, CNN 1D) untuk memilih model terbaik.
4. **Metode:** Crawling data diskusi Stockbit (255.000+ postingan, 2021–2026) → preprocessing teks (case folding, normalisasi slang, stopword removal, stemming Sastrawi) → pelabelan sentimen InSet Lexicon → agregasi harian → penggabungan dengan 9 fitur teknikal → pemodelan deep learning (PyTorch) → evaluasi 6 konfigurasi (3 model × 2 skenario) dengan 5 konfigurasi hyperparameter (K1–K5).
5. **Hasil utama:** Model LSTM skenario S2 (+sentimen) dengan konfigurasi K3 menghasilkan MAPE terbaik (2,24%), membuktikan kontribusi fitur sentimen terhadap penurunan error prediksi dibanding skenario S1 (tanpa sentimen).
6. **Kesimpulan singkat:** Sentimen diskusi investor Stockbit terbukti meningkatkan akurasi prediksi harga saham BBCA; sistem aplikasi GUI interaktif (Streamlit) berhasil dibangun sebagai alat bantu keputusan investasi.

**Kata Kunci:** Prediksi Harga Saham, Deep Learning, LSTM, GRU, CNN 1D, Analisis Sentimen, InSet Lexicon, Stockbit, BBCA, Indikator Teknikal

---

## DAFTAR ISI (Struktur Bab Lengkap)

```
HALAMAN JUDUL
HALAMAN PENGESAHAN
HALAMAN PERSETUJUAN
HALAMAN PERNYATAAN KEASLIAN
KATA PENGANTAR
ABSTRAK
ABSTRACT (Bahasa Inggris)
DAFTAR ISI
DAFTAR TABEL
DAFTAR GAMBAR
DAFTAR LAMPIRAN

BAB I    PENDAHULUAN
BAB II   TINJAUAN PUSTAKA
BAB III  METODOLOGI PENELITIAN
BAB IV   HASIL DAN PEMBAHASAN
BAB V    PENUTUP

DAFTAR PUSTAKA
LAMPIRAN
```

---

# BAB I — PENDAHULUAN

## 1.1 Latar Belakang Masalah

**Alur argumentasi yang harus dibangun:**

1. **Pasar modal Indonesia & pentingnya prediksi harga saham**
   - Pertumbuhan investor ritel di Indonesia (data OJK/BEI terbaru)
   - Saham BBCA (Bank Central Asia) sebagai *blue-chip* terbesar di BEI (kapitalisasi pasar, likuiditas tinggi, volume perdagangan besar)
   - Tantangan investor dalam mengambil keputusan beli/jual berbasis analisis data

2. **Keterbatasan analisis teknikal tradisional**
   - Analisis teknikal (Moving Average, RSI, MACD) berfokus pada pola historis harga
   - Tidak menangkap faktor sentimen pasar, berita, dan psikologi investor
   - Referensi: penelitian-penelitian yang menunjukkan *market inefficiency* dalam pendekatan teknikal murni

3. **Peran sentimen investor dalam pergerakan harga saham**
   - Teori Behavioral Finance: sentimen investor mempengaruhi keputusan dan harga (Bollen et al., 2011; Tetlock, 2007)
   - Pertumbuhan platform diskusi investor digital di Indonesia → **Stockbit** sebagai platform komunitas investor terbesar di Indonesia
   - Diskusi di stream Stockbit mengandung opini, prediksi, dan sentimen real-time tentang saham BBCA
   - Gap: belum ada penelitian yang secara sistematis memanfaatkan sentimen diskusi Stockbit untuk prediksi harga saham di Indonesia

4. **Deep Learning untuk prediksi time-series keuangan**
   - Keunggulan deep learning (LSTM, GRU, CNN 1D) dalam menangkap pola temporal non-linear pada data finansial
   - Perbandingan tiga arsitektur memberikan pemilihan mesin prediksi terbaik
   - Integrasi sentimen sebagai fitur tambahan dalam deep learning → peningkatan akurasi

5. **Kebutuhan sistem prediksi terintegrasi**
   - Tidak cukup membangun model → perlu **sistem** yang dapat digunakan
   - Sistem GUI interaktif berbasis Streamlit sebagai *proof of concept* alat pendukung keputusan

> [!IMPORTANT]
> **Kunci revisi:** Latar belakang harus mengarah ke pembangunan **SISTEM** (bukan hanya perbandingan model). Perbandingan 3 model adalah **tahap pemilihan** mesin dalam sistem, bukan tujuan utama. Tujuan utama adalah membangun sistem prediksi yang memanfaatkan sentimen Stockbit.

## 1.2 Rumusan Masalah

1. Bagaimana perbandingan performa model LSTM, GRU, dan CNN dalam memprediksi pergerakan harga saham antara skenario yang hanya menggunakan data historis harga dan skenario yang mengintegrasikan data sentimen?
2. Bagaimana pengaruh integrasi data sentimen investor Stockbit terhadap akurasi model prediksi harga saham BBCA?
3. Bagaimana merancang dan membangun prototipe sistem yang mengintegrasikan alur analisis sentimen dan data harga untuk menghasilkan prediksi harga saham?

## 1.3 Tujuan Penelitian

1. Merancang dan membangun sistem prediksi pergerakan harga saham BBCA berbasis deep learning yang mengintegrasikan fitur analisis sentimen diskusi investor Stockbit dengan indikator teknikal.

2. Menganalisis dan membuktikan kontribusi fitur sentimen diskusi investor Stockbit terhadap peningkatan akurasi prediksi harga saham melalui perbandingan skenario S1 (tanpa sentimen) dan S2 (dengan sentimen).

3. Menentukan arsitektur deep learning terbaik di antara LSTM, GRU, dan CNN 1D berdasarkan evaluasi metrik performa (MAPE, RMSE, MAE) pada 5 konfigurasi hyperparameter (K1–K5).

## 1.4 Manfaat Penelitian

### 1.4.1 Manfaat Teoritis
- Memperkaya literatur tentang pemanfaatan analisis sentimen berbasis lexicon pada data media sosial finansial Indonesia untuk prediksi harga saham
- Memberikan bukti empiris kontribusi fitur sentimen terhadap akurasi model deep learning pada kasus saham blue-chip Indonesia

### 1.4.2 Manfaat Praktis
- Menyediakan sistem prediksi interaktif yang dapat digunakan investor sebagai alat pendukung keputusan
- Memberikan *baseline* metodologi untuk penelitian serupa pada saham lain di BEI
- Menyediakan dataset sentimen Stockbit BBCA yang tervalidasi untuk penelitian lanjutan

## 1.5 Batasan Masalah

1. Objek penelitian terbatas pada saham **BBCA (Bank Central Asia Tbk.)** di Bursa Efek Indonesia
2. Sumber sentimen berasal dari **stream diskusi Stockbit** pada ticker $BBCA
3. Periode data harga historis: **2016–2026** (±10 tahun via yfinance)
4. Periode data sentimen: **2021–2026** (jangkauan data crawling Stockbit yang padat)
5. Metode analisis sentimen menggunakan **InSet Lexicon** (lexicon-based, bukan machine learning classifier)
6. Klasifikasi sentimen menggunakan skema **2 kelas** (positif & negatif; postingan netral dibuang)
7. Model deep learning yang dibandingkan: **LSTM, GRU, dan CNN 1D** (framework PyTorch)
8. Target prediksi: **harga penutupan (Close) H+1** (regresi, bukan klasifikasi arah)
9. Sistem GUI dibangun menggunakan **Streamlit** sebagai prototipe fungsional (bukan aplikasi produksi)
10. Penelitian bersifat akademis; **bukan saran investasi finansial**

## 1.6 Sistematika Penulisan

Penjelasan singkat tentang isi setiap bab (BAB I s.d. BAB V).

---

# BAB II — TINJAUAN PUSTAKA

## 2.1 Penelitian Terdahulu (State of the Art)

**Minimal 8–12 penelitian relevan. Kategorikan menjadi:**

### 2.1.1 Prediksi Harga Saham dengan Deep Learning
- Penelitian yang menggunakan LSTM untuk prediksi harga saham (Fischer & Krauss, 2018; dll.)
- Penelitian yang menggunakan GRU untuk time-series finansial
- Penelitian yang menggunakan CNN 1D untuk data sekuensial keuangan
- Perbandingan performa LSTM vs GRU vs CNN pada prediksi saham

### 2.1.2 Analisis Sentimen untuk Prediksi Saham
- Bollen et al. (2011): "Twitter mood predicts the stock market" → dampak sentimen terhadap indeks pasar
- Tetlock (2007): Pengaruh sentimen media terhadap return saham
- Penelitian sentimen media sosial Indonesia untuk prediksi saham (jika ada)
- Penelitian yang mengintegrasikan sentimen + teknikal untuk prediksi

### 2.1.3 Analisis Sentimen Berbasis Lexicon pada Teks Indonesia
- InSet Lexicon (Koto & Rahmaningtyas, 2017): Indonesian Sentiment Lexicon
- Penggunaan Sastrawi untuk stemming Bahasa Indonesia
- Perbandingan pendekatan lexicon vs ML untuk sentiment analysis

> [!TIP]
> **Format tabel ringkasan penelitian terdahulu** sebaiknya disajikan dalam tabel berisi: Peneliti (Tahun), Judul, Metode, Data, Hasil Utama, dan Perbedaan dengan penelitian ini.

## 2.2 Landasan Teori

### 2.2.1 Pasar Modal dan Saham
- Definisi pasar modal & mekanisme perdagangan saham di BEI
- Profil singkat BBCA (Bank Central Asia Tbk.)
- Jenis-jenis analisis saham: fundamental, teknikal, sentimen

### 2.2.2 Analisis Teknikal Saham
- Definisi dan prinsip analisis teknikal
- **Indikator teknikal yang digunakan (9 fitur):**

| No | Fitur | Deskripsi | Formula/Keterangan |
|---|---|---|---|
| 1 | Close | Harga penutupan harian | Data OHLCV dari yfinance |
| 2 | Open | Harga pembukaan harian | Data OHLCV dari yfinance |
| 3 | High | Harga tertinggi harian | Data OHLCV dari yfinance |
| 4 | Low | Harga terendah harian | Data OHLCV dari yfinance |
| 5 | Volume | Volume perdagangan (lembar) | Data OHLCV dari yfinance |
| 6 | MA5 | Moving Average 5 hari | Rata-rata Close 5 hari terakhir |
| 7 | MA20 | Moving Average 20 hari | Rata-rata Close 20 hari terakhir |
| 8 | RSI | Relative Strength Index (14 hari) | RSI = 100 - 100/(1 + RS); RS = avg_gain/avg_loss |
| 9 | Return | Daily Return | pct_change() dari Close |

### 2.2.3 Text Mining dan Analisis Sentimen
- Definisi text mining dan text preprocessing
- Analisis sentimen: definisi, jenis pendekatan (lexicon-based vs machine learning)
- **Tahapan preprocessing teks** yang diimplementasikan:
  1. **Case Folding** — konversi teks ke huruf kecil
  2. **Pembersihan** — hapus URL, mention (@user), cashtag ($BBCA), emoji, karakter khusus
  3. **Normalisasi slang** — kamus istilah slang pasar modal Stockbit
  4. **Filtering spam** — deteksi pola spam (join grup, VIP signal, dll.)
  5. **Stopword removal** — penghapusan stopword Indonesia
  6. **Stemming** — menggunakan Sastrawi Stemmer (opsional)

### 2.2.4 InSet Lexicon (Indonesia Sentiment Lexicon)
- Penjelasan InSet Lexicon (Koto & Rahmaningtyas, 2017)
- Struktur kamus: positive.tsv dan negative.tsv (kata + bobot numerik)
- Mekanisme scoring: skor = Σ(bobot kata positif) − Σ(bobot kata negatif)
- Aturan pelabelan: skor > 0 → Positif (+1); skor < 0 → Negatif (−1); skor = 0 → Netral (0)

### 2.2.5 Stockbit sebagai Sumber Data Sentimen
- Penjelasan platform Stockbit (komunitas investor Indonesia)
- Fitur stream diskusi per ticker saham
- Justifikasi pemilihan Stockbit: volume diskusi tinggi, real-time, fokus pada opini investasi
- Pertimbangan etis: anonimisasi data, penggunaan akademis

### 2.2.6 Deep Learning
- Definisi dan konsep dasar deep learning, neural network
- Fungsi aktivasi, loss function, optimizer
- Regularisasi: dropout, early stopping, gradient clipping

### 2.2.7 Long Short-Term Memory (LSTM)
- Arsitektur LSTM: forget gate, input gate, output gate, cell state
- Keunggulan mengatasi vanishing gradient pada data sekuensial panjang
- **Arsitektur yang diimplementasikan (StockLSTM):**
  - 2 layer LSTM (stacked) dengan hidden_dim = 64
  - Dropout antara setiap layer (rate = 0.2)
  - Fully Connected output: 1 neuron (regresi harga)
  - Input shape: (batch_size, lookback=30, n_features)

### 2.2.8 Gated Recurrent Unit (GRU)
- Arsitektur GRU: update gate, reset gate
- Perbedaan dengan LSTM (lebih sederhana, lebih cepat konvergen)
- **Arsitektur yang diimplementasikan (StockGRU):**
  - 2 layer GRU (stacked) dengan hidden_dim = 64
  - Dropout antara setiap layer (rate = 0.2)
  - Fully Connected output: 1 neuron (regresi harga)

### 2.2.9 Convolutional Neural Network 1D (CNN 1D)
- Arsitektur CNN 1D untuk data sekuensial (berbeda dengan CNN 2D untuk citra)
- Operasi konvolusi temporal, ReLU, pooling
- **Arsitektur yang diimplementasikan (StockCNN):**
  - 2 layer Conv1D (kernel_size=3, padding=1) dengan ReLU
  - Dropout setelah setiap konvolusi (rate = 0.2)
  - Flatten → Fully Connected output: 1 neuron
  - Input: (batch_size, n_features, sequence_length) — perlu transpose

### 2.2.10 Metrik Evaluasi Model Regresi
- **MAPE (Mean Absolute Percentage Error)** — metrik utama
  - Formula: MAPE = (1/n) × Σ|((y_actual − y_pred) / y_actual)| × 100%
  - Interpretasi: persentase rata-rata deviasi prediksi dari nilai aktual
  - Kelebihan: skala independen, mudah diinterpretasikan
  - Standar: MAPE < 10% = sangat akurat; < 20% = baik
- **RMSE (Root Mean Squared Error)**
  - Formula: RMSE = √(Σ(y_actual − y_pred)² / n)
  - Interpretasi: rata-rata error dalam satuan harga (IDR), sensitif terhadap outlier
- **MAE (Mean Absolute Error)**
  - Formula: MAE = (1/n) × Σ|y_actual − y_pred|
  - Interpretasi: rata-rata absolut selisih harga aktual dan prediksi

### 2.2.11 MinMaxScaler dan Normalisasi Data
- Teknik normalisasi MinMaxScaler: scaling ke rentang [0, 1]
- Pentingnya fit scaler **hanya pada data training** untuk mencegah data leakage
- Inverse transform untuk mengembalikan prediksi ke skala harga asli

### 2.2.12 Sliding Window (Windowing) untuk Data Time-Series
- Konsep supervised learning dari data time-series
- Pembentukan urutan (X, y) dengan lookback/timestep = 30 hari
- Pentingnya split data **kronologis** (bukan random) untuk mencegah temporal leakage

---

# BAB III — METODOLOGI PENELITIAN

## 3.1 Desain Penelitian
- Desain penelitian tetap disamakan dengan skripsi awal, menggunakan pendekatan Design Science Research yang meliputi:
  1. Studi Deskriptif I
  2. Studi Preskriptif
  3. Studi Deskriptif II

## 3.2 Alat dan Bahan Penelitian

### 3.2.1 Perangkat Keras
- Spesifikasi komputer yang digunakan (processor, RAM, GPU jika ada)

### 3.2.2 Perangkat Lunak dan Pustaka

| Kategori | Teknologi | Versi |
|---|---|---|
| Bahasa Pemrograman | Python | 3.10+ |
| Framework Deep Learning | **PyTorch (torch)** | ≥ 2.0 |
| Manipulasi Data | pandas, numpy | - |
| Data Harga Saham | yfinance | - |
| Crawling Data Stockbit | requests, python-dotenv | - |
| NLP & Stemming | Sastrawi, NLTK | - |
| Lexicon Sentimen | InSet Lexicon (positive.tsv, negative.tsv) | - |
| Visualisasi | matplotlib, plotly, WordCloud | - |
| Evaluasi Model | scikit-learn (metrics) | - |
| Normalisasi Data | scikit-learn (MinMaxScaler), joblib | - |
| Konfigurasi | PyYAML, python-dotenv | - |
| Aplikasi GUI | Streamlit | - |
| Version Control | Git | - |

### 3.2.3 Dataset Penelitian

| Dataset | Sumber | Periode | Jumlah |
|---|---|---|---|
| Harga historis BBCA (OHLCV) | Yahoo Finance (yfinance, ticker `BBCA.JK`) | 2016-01-01 s.d. 2026-07-23 | ±2.600 hari bursa |
| Diskusi investor stream Stockbit | Stockbit Exodus API v3 (ticker $BBCA) | 2021-01-01 s.d. 2026-07-23 | 255.000+ postingan |
| InSet Lexicon | GitHub (Koto & Rahmaningtyas) | - | ±6.000 kata positif + ±6.000 kata negatif |

## 3.3 Arsitektur Sistem

**Diagram arsitektur 7 lapisan yang harus digambar:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        [1] SUMBER DATA                                  │
│     Stockbit Stream (Teks Diskusi)         Yahoo Finance (OHLCV)        │
└──────────────┬────────────────────────────────────┬─────────────────────┘
               │                                    │
┌──────────────▼──────────────┐    ┌────────────────▼─────────────────────┐
│   [2] AKUISISI DATA         │    │   [2] AKUISISI DATA                  │
│   Crawler Stockbit          │    │   yfinance Downloader                │
│   (Exodus API v3,           │    │   (BBCA.JK, 2016–2026)              │
│    checkpoint/resume,       │    │                                      │
│    deduplikasi, anonimisasi)│    │                                      │
└──────────────┬──────────────┘    └────────────────┬─────────────────────┘
               │                                    │
┌──────────────▼──────────────┐    ┌────────────────▼─────────────────────┐
│ [3A] PEMROSESAN SENTIMEN    │    │ [3B] PEMROSESAN HARGA                │
│ • Case folding              │    │ • Kalkulasi 9 indikator teknikal     │
│ • Hapus URL/mention/emoji   │    │   (Close, Open, High, Low, Volume,  │
│ • Normalisasi slang         │    │    MA5, MA20, RSI, Return)           │
│ • Filter spam               │    │ • Penanganan missing value           │
│ • Stopword removal          │    │                                      │
│ • Stemming (Sastrawi)       │    │                                      │
│ • Scoring InSet Lexicon     │    │                                      │
│ • Agregasi harian (NSS,     │    │                                      │
│   EMA-3 smoothing)          │    │                                      │
│                             │    │                                      │
│                             │    │                                      │
│                             │    │                                      │
│                             │    │                                      │
└──────────────┬──────────────┘    └────────────────┬─────────────────────┘
               └──────────────┬─────────────────────┘
                              │
               ┌──────────────▼──────────────────────────────────────────┐
               │ [4] REKAYASA FITUR & SKENARIO EKSPERIMEN                │
               │ • Penggabungan harga + sentimen (join on Date)          │
               │ • Imputasi missing value (forward-fill → back-fill)     │
               │ • Log1p transform volume diskusi                        │
               │ • Target: Close H+1 (shift -1)                         │
               │ • Skenario S1: 9 fitur teknikal                        │
               │ • Skenario S2: 9 teknikal + 4 sentimen = 13 fitur      │
               │ • MinMaxScaler [0,1] (fit on train only)                │
               │ • Sliding window (lookback=30)                          │
               │ • Split kronologis: train 80% / test 20%                │
               └──────────────┬──────────────────────────────────────────┘
                              │
               ┌──────────────▼──────────────────────────────────────────┐
               │ [5] PEMODELAN DEEP LEARNING                             │
               │ • StockLSTM (2-layer, hidden=64, dropout)               │
               │ • StockGRU  (2-layer, hidden=64, dropout)               │
               │ • StockCNN  (2-layer Conv1D, kernel=3, ReLU)            │
               │ • Optimizer: Adam (weight_decay=1e-4)                   │
               │ • Loss: MSELoss                                         │
               │ • Scheduler: ReduceLROnPlateau                          │
               │ • Callbacks: EarlyStopping (patience=15)                │
               │ • Gradient Clipping (max_norm=1.0)                      │
               │ • 5 Konfigurasi Hyperparameter (K1–K5)                  │
               │ • 3 runs per konfigurasi (multi-seed)                   │
               │ • 6 evaluasi utama: 3 model × 2 skenario               │
               │ • Metrik: MAPE (utama), RMSE, MAE                      │
               │ • Pemilihan model terbaik → best_model.pt               │
               └──────────────┬──────────────────────────────────────────┘
                              │
               ┌──────────────▼──────────────────────────────────────────┐
               │ [6] APLIKASI GUI (Streamlit Dashboard)                  │
               │ Tab 1: Prediksi & Tren Harga (Candlestick, Aktual vs   │
               │        Prediksi, Error Distribution, Scatter Plot)      │
               │ Tab 2: Analisis Sentimen (Rasio Harian, Volume vs      │
               │        Return, Donut Chart, Word Cloud, Sampel Post)   │
               │ Tab 3: Perbandingan Model (Bar Chart MAPE, Tabel K1-K5,│
               │        Validasi Kappa, Ringkasan Metrik 30 Konfigurasi)│
               │ Tab 4: Data Explorer (RSI, Volume, Dataset Filter,     │
               │        Download CSV, Raw/Cleaned Stream Data)           │
               └──────────────┬──────────────────────────────────────────┘
                              │
               ┌──────────────▼──────────────────────────────────────────┐
               │ [7] PENGGUNA                                            │
               │ Investor / Analis (Alat Pendukung Keputusan Investasi)  │
               └────────────────────────────────────────────────────────┘
```

## 3.4 Desain Alur Penelitian (Flowchart)

**Flowchart utama yang harus digambar:**

```
START
  │
  ├─ Studi literatur & perumusan masalah
  │
  ├─ Pengumpulan data:
  │   ├─ Crawling stream Stockbit BBCA (Exodus API v3)
  │   └─ Download harga historis BBCA (yfinance)
  │
  ├─ Preprocessing teks diskusi Stockbit:
  │   ├─ Case folding → Cleaning
  │   ├─ Normalisasi slang
  │   ├─ Filter spam → Stopword removal → Stemming
  │   └─ Pelabelan sentimen InSet Lexicon
  │
  ├─ Agregasi sentimen harian:
  │   ├─ Penyelarasan akhir pekan → Senin
  │   ├─ Kalkulasi: sentiment_score, positive_ratio, negative_ratio, discussion_volume
  │   ├─ Net Sentiment Spread (NSS) — metode Tetlock/Bollen
  │   └─ EMA-3 smoothing untuk meredam noise
  │
  ├─ Rekayasa fitur & pembuatan dataset final:
  │   ├─ Kalkulasi 9 indikator teknikal
  │   ├─ Merge harga + sentimen (join on Date)
  │   ├─ Imputasi missing value + Log1p transform
  │   ├─ Target: Close H+1 (shift -1)
  │   ├─ Skenario S1 (9 fitur) vs S2 (13 fitur)
  │   ├─ MinMaxScaler [0,1] (fit on training only)
  │   ├─ Sliding window (lookback = 30 hari)
  │   └─ Split kronologis: train 80% / test 20%
  │
  ├─ Pemodelan & pelatihan deep learning:
  │   ├─ Bangun arsitektur: StockLSTM, StockGRU, StockCNN (PyTorch)
  │   ├─ Hyperparameter tuning: 5 konfigurasi (K1–K5)
  │   ├─ Training: Adam + MSELoss + EarlyStopping + ReduceLR + GradClip
  │   ├─ Multi-run (3 seeds) untuk konsistensi
  │   └─ Total: 30 konfigurasi (3 model × 2 skenario × 5 config)
  │
  ├─ Evaluasi & pemilihan model terbaik:
  │   ├─ Inverse transform prediksi ke skala harga asli
  │   ├─ Hitung MAPE, RMSE, MAE pada data test
  │   ├─ Perbandingan 6 evaluasi utama (K3 best config)
  │   ├─ Tabel komprehensif 30 konfigurasi
  │   ├─ Pemilihan model terbaik (MAPE terendah S2 K3)
  │   └─ Simpan artefak: best_model.pt, scaler.pkl
  │
  ├─ Pembangunan aplikasi GUI (Streamlit):
  │   ├─ Tab 1: Prediksi & Tren Harga
  │   ├─ Tab 2: Analisis Sentimen Stockbit
  │   ├─ Tab 3: Perbandingan Model & Metrik
  │   └─ Tab 4: Data Explorer & Download
  │
  ├─ Analisis hasil & pembahasan
  │
  └─ Penarikan kesimpulan & saran
  │
END
```

## 3.5 Tahapan Penelitian (Detail)

### 3.5.1 Pengumpulan Data

#### A. Crawling Data Diskusi Stockbit
- Endpoint: Stockbit Exodus API v3 (`/stream/v3/symbol/BBCA`)
- Autentikasi: Bearer Token (disimpan di `.env`)
- Mekanisme:
  - Pagination menggunakan `last_stream_id` (mundur ke posting lebih lama)
  - Checkpoint/resume: menyimpan ID minimum terakhir untuk melanjutkan crawling
  - Deduplikasi ketat: berdasarkan `stream_id` dan konten teks
  - Rate limiting: delay antar-request (0.05 detik), retry pada HTTP 429
  - Batas crawling: hingga mencapai tahun 2021 (parameter `min_year`)
- **Anonimisasi data:** hanya menyimpan `id`, `created_at`, `content`, `like_count`; tidak menyimpan username/PII
- Output: `data/raw/stream_bbca.jsonl` (255.000+ postingan)

#### B. Pengunduhan Data Harga BBCA
- Sumber: Yahoo Finance via pustaka yfinance
- Ticker: `BBCA.JK`
- Periode: 2016-01-01 s.d. 2026-07-23
- Kolom: Date, Open, High, Low, Close, Volume
- Output: `data/raw/prices_bbca.csv`

### 3.5.2 Preprocessing dan Analisis Sentimen

**Detail tahapan preprocessing sesuai implementasi sistem:**

| Tahap | Fungsi | Deskripsi |
|---|---|---|
| 1 | `clean_text()` | Case folding, hapus URL/mention/cashtag, emoji |
| 2 | `normalize_slang()` | Normalisasi kata slang/singkatan khas Stockbit ke bentuk baku |
| 3 | `is_spam()` | Deteksi spam: pola iklan (join grup, VIP signal) |
| 4 | `remove_stopwords()` | Hapus stopword Indonesia |
| 5 | `stem_text()` | Stemming opsional menggunakan Sastrawi |
| 6 | `score_text()` | Scoring InSet Lexicon standar |

### 3.5.3 Agregasi Sentimen Harian

- **Deduplikasi ketat**: berdasarkan ID dan konten teks
- **Penyelarasan akhir pekan**: postingan Sabtu/Minggu dipetakan ke hari Senin berikutnya
- **4 fitur sentimen yang dihitung per hari:**

| Fitur | Formula | Keterangan |
|---|---|---|
| `sentiment_score` | NSS × raw_mean_score, EMA-3 smoothed | Net Sentiment Spread (Tetlock, 2007) |
| `positive_ratio` | pos_count / total, EMA-3 smoothed | Proporsi postingan positif |
| `negative_ratio` | neg_count / total, EMA-3 smoothed | Proporsi postingan negatif |
| `discussion_volume` | count per tanggal | Volume diskusi (jumlah postingan) |

- **Net Sentiment Spread (NSS):** `NSS = (Jumlah Positif - Jumlah Negatif) / Total Postingan` atau setara dengan `positive_ratio − negative_ratio` (Tetlock, 2007; Bollen et al., 2011)
- **Exponential Moving Average (EMA-3):** smoothing untuk meredam noise harian dengan rumus: `EMA_t = (Nilai_t * (2 / (1 + 3))) + (EMA_{t-1} * (1 - (2 / (1 + 3))))`

### 3.5.4 Rekayasa Fitur dan Pembentukan Dataset

#### A. Penggabungan Data
- Merge `prices_bbca.csv` (+ 9 indikator teknikal) dengan `daily_sentiment.csv` berdasarkan kolom `Date`
- Periode final dataset: sesuai jangkauan sentimen (2021–2026)
- Imputasi missing value sentimen: forward-fill → backward-fill → default 0.0
- Log1p transformation pada `discussion_volume` untuk menstabilkan distribusi

#### B. Target Prediksi
- Target: harga penutupan (Close) hari berikutnya (H+1)
- `Target = Close.shift(-1)` → baris terakhir yang NaN dibuang

#### C. Skenario Eksperimen

| Skenario | Jumlah Fitur | Daftar Fitur | Tujuan |
|---|---|---|---|
| **S1 (Baseline)** | 9 | Close, Open, High, Low, Volume, MA5, MA20, RSI, Return | Kontrol tanpa sentimen |
| **S2 (+Sentimen)** | 13 | 9 teknikal + sentiment_score, positive_ratio, negative_ratio, discussion_volume | Uji kontribusi sentimen |

#### D. Normalisasi dan Windowing
- **MinMaxScaler [0, 1]**: fit pada data training saja, transform ke data test → **mencegah data leakage**
- **Sliding window**: lookback = 30 hari → setiap sampel berisi 30 timestep × n_features
- **Split kronologis**: train 80% / test 20% (bukan random shuffle) → **mencegah temporal leakage**

### 3.5.5 Pemodelan Deep Learning

#### A. Arsitektur Model (PyTorch)

| Komponen | StockLSTM | StockGRU | StockCNN |
|---|---|---|---|
| Layer 1 | LSTM (input_dim → hidden=64) | GRU (input_dim → hidden=64) | Conv1D (input_dim → 64, kernel=3, pad=1) + ReLU |
| Dropout 1 | 0.2 | 0.2 | 0.2 |
| Layer 2 | LSTM (64 → 64) | GRU (64 → 64) | Conv1D (64 → 64, kernel=3, pad=1) + ReLU |
| Dropout 2 | 0.2 | 0.2 | 0.2 |
| Output | FC(64 → 1) | FC(64 → 1) | Flatten → FC(64×30 → 1) |
| Aktivasi akhir | Linear (regresi) | Linear (regresi) | Linear (regresi) |

#### B. Konfigurasi Hyperparameter (K1–K5)

| Config | Hidden Units | Dropout | Learning Rate | Batch Size | Epochs | Keterangan |
|---|---|---|---|---|---|---|
| K1 | 32 | 0.10 | 0.005 | 16 | 50 | Under-parameterized |
| K2 | 64 | 0.20 | 0.001 | 32 | 100 | Standard Baseline |
| **K3** | **64** | **0.20** | **0.001** | **16** | **150** | **🏆 Konfigurasi Optimal** |
| K4 | 128 | 0.30 | 0.0005 | 16 | 100 | High Regularization |
| K5 | 64 | 0.50 | 0.005 | 32 | 200 | Deep Capacity / High Dropout |

#### C. Mekanisme Training
- **Loss function:** MSELoss (Mean Squared Error)
- **Optimizer:** Adam (weight_decay = 1e-4)
- **Learning rate scheduler:** ReduceLROnPlateau (factor=0.5, patience=10)
- **Early stopping:** patience = 15 (monitor validation loss)
- **Gradient clipping:** max_norm = 1.0 (mencegah exploding gradient)
- **Multi-seed:** 3 run per konfigurasi (seed = 43, 44, 45) → rata-rata metrik
- **Reproduksibilitas:** seed global (numpy, random, torch) = 42

### 3.5.6 Evaluasi Model

- **Metrik:**
  - **MAPE (%)** — metrik utama pemilihan model terbaik
  - **RMSE (IDR)** — error dalam satuan harga
  - **MAE (IDR)** — rata-rata deviasi absolut
- **Penting:** Inverse transform prediksi ke skala harga asli sebelum menghitung metrik
- **Total konfigurasi yang dievaluasi:**
  - 6 evaluasi utama: 3 model × 2 skenario (menggunakan K3 optimal)
  - 30 evaluasi komprehensif: 3 model × 2 skenario × 5 konfigurasi hyperparameter
- **Pemilihan model terbaik:** MAPE terendah pada skenario S2 dengan konfigurasi K3
- **Artefak yang disimpan:** `best_model.pt`, `best_scaler_features.pkl`, `best_scaler_target.pkl`, `best_model_meta.json`

### 3.5.7 Pembangunan Aplikasi GUI (Streamlit)

**Komponen dashboard interaktif:**

| Tab | Komponen | Fitur |
|---|---|---|
| Tab 1: Prediksi & Tren | Grafik Aktual vs Prediksi, Candlestick OHLC, Error Distribution, Scatter Plot, KPI Cards | Zoom filter periode, overlay MA, range slider |
| Tab 2: Sentimen | Tren rasio sentimen, Volume vs Return, Donut Chart distribusi, Word Cloud, Sampel postingan | Filter tanggal, filter label, horizontal slider |
| Tab 3: Model | Bar Chart MAPE S1 vs S2, Tabel K1–K5, Validasi Kappa, Tabel 30 konfigurasi | Filter model & skenario interaktif |
| Tab 4: Data Explorer | RSI 14, Volume perdagangan, Dataset final + filter, Data cleaning Stockbit, Raw stream | Search, filter, sort, download CSV |

**Fitur teknis GUI:**
- Caching model/scaler: `@st.cache_resource` (load sekali saat start)
- Caching data: `@st.cache_data` (mencegah re-computation)
- Sidebar kontrol: pilih arsitektur model, skenario, window timestep
- Prediksi H+1 real-time dari data terakhir
- Disclaimer: bukan saran investasi finansial

---

# BAB IV — HASIL DAN PEMBAHASAN

## 4.1 Hasil Pengumpulan Data

### 4.1.1 Data Harga Historis BBCA
- Statistik deskriptif: jumlah baris, rentang tanggal, harga tertinggi/terendah, rata-rata volume
- Tabel sampel data harga
- Grafik tren harga historis BBCA (2016–2026)

### 4.1.2 Data Diskusi Investor Stockbit
- Total postingan terkumpul: **255.000+ postingan** unik
- Rentang tanggal: 2021–2026
- Distribusi volume postingan per tahun/bulan
- Statistik sebelum vs sesudah deduplikasi & filter spam

## 4.2 Hasil Preprocessing dan Analisis Sentimen

### 4.2.1 Hasil Preprocessing Teks
- Contoh transformasi teks (sebelum → sesudah setiap tahap)
- Statistik: jumlah postingan yang terfilter sebagai spam
- Efektivitas normalisasi slang (berapa kata yang ter-normalize)
- Efek de-elongasi dan emoji translation

### 4.2.2 Hasil Pelabelan Sentimen InSet Lexicon
- Distribusi label: jumlah & persentase postingan positif, negatif, netral
- Distribusi skor sentimen (histogram)
- Contoh postingan per kategori label beserta skor dan kata sentimen yang terdeteksi

### 4.2.3 Hasil Agregasi Sentimen Harian
- Statistik: jumlah hari dengan data sentimen, rata-rata volume diskusi per hari
- Grafik tren rasio sentimen positif/negatif harian
- Grafik volume diskusi harian
- Efek EMA-3 smoothing pada noise sentimen

## 4.3 Hasil Rekayasa Fitur dan Dataset Final

### 4.3.1 Dataset Final
- Jumlah baris, kolom, rentang tanggal final
- Tabel deskripsi statistik setiap fitur (mean, std, min, max)
- Korelasi antar-fitur (heatmap korelasi)

### 4.3.2 Hasil Windowing dan Split Data
- Ukuran data train/test per skenario
- Shape array: (n_samples, lookback=30, n_features)

## 4.4 Hasil Pemodelan dan Evaluasi

### 4.4.1 Hasil Hyperparameter Tuning (K1–K5)
- **Tabel komprehensif 30 konfigurasi:**

| Model | Skenario | Config | MAPE (%) | RMSE (IDR) | MAE (IDR) | MSE (IDR²) | Keterangan |
|---|---|---|---|---|---|---|---|
| LSTM | S1 | K1 | 3.42 | 341.20 | 236.15 | 116,417 | Baseline |
| LSTM | S1 | K2 | 3.10 | 308.50 | 214.30 | 95,172 | Baseline |
| LSTM | S1 | **K3** | **2.85** | **283.39** | **197.21** | **80,311** | Best S1 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| **LSTM** | **S2** | **K3** | **2.24** | **221.13** | **152.99** | **48,898** | **🏆 BEST OVERALL** |
| GRU | S2 | K3 | 2.31 | 220.56 | 156.75 | 48,647 | Top GRU |
| CNN | S2 | K3 | 3.49 | 323.04 | 241.26 | 104,355 | Top CNN |

- Analisis: K3 konsisten menghasilkan performa terbaik di ketiga model
- Penjelasan mengapa K3 optimal (keseimbangan capacity, regularization, dan training duration)

### 4.4.2 Perbandingan Skenario S1 vs S2 (Kontribusi Sentimen)
- **Tabel perbandingan S1 vs S2 pada K3 optimal:**

| Model | S1 MAPE (%) | S2 MAPE (%) | Penurunan Error | Persentase Perbaikan |
|---|---|---|---|---|
| LSTM | 2.85 | 2.24 | −0.61 | **−21.4%** |
| GRU | 2.60 | 2.31 | −0.29 | **−11.2%** |
| CNN | 3.87 | 3.49 | −0.38 | **−9.8%** |

- **Temuan utama:** Semua model menunjukkan penurunan MAPE pada S2 → **fitur sentimen Stockbit terbukti meningkatkan akurasi**
- LSTM mendapat manfaat terbesar dari penambahan sentimen (penurunan 21.4%)
- Bar chart perbandingan MAPE S1 vs S2

### 4.4.3 Perbandingan Antar-Model (Pemilihan Model Terbaik)
- **Ranking model pada S2 K3:**
  1. 🏆 **LSTM** — MAPE 2.24% (TERBAIK)
  2. GRU — MAPE 2.31%
  3. CNN 1D — MAPE 3.49%

- Analisis mengapa LSTM unggul: kemampuan LSTM dalam menangkap dependensi temporal jangka panjang melalui mekanisme cell state
- Analisis mengapa CNN kurang optimal: arsitektur konvolusi lebih cocok untuk ekstraksi pola lokal, kurang optimal untuk dependensi sekuensial panjang pada data time-series keuangan
- Grafik aktual vs prediksi untuk setiap model

### 4.4.4 Analisis Error Prediksi
- Distribusi residual (histogram): error berdistribusi mendekati normal/terpusat di 0
- Scatter plot aktual vs prediksi: keselarasan terhadap garis diagonal
- Analisis periode prediksi terburuk (kapan model gagal) → potensi korelasi dengan event pasar

### 4.4.5 Grafik Loss Training
- Kurva training loss vs validation loss per model
- Analisis konvergensi dan potensi overfitting

## 4.5 Hasil Aplikasi GUI (Streamlit Dashboard)

### 4.5.1 Tampilan Dashboard
- Screenshot setiap tab dengan penjelasan komponen
- Screenshot sidebar kontrol model/skenario
- Screenshot KPI Cards (harga terakhir, prediksi H+1, MAPE, RMSE/MAE)

### 4.5.2 Fitur Prediksi Interaktif
- Screenshot grafik aktual vs prediksi dengan zoom filter
- Screenshot candlestick chart dengan overlay MA
- Screenshot error distribution dan scatter plot

### 4.5.3 Fitur Analisis Sentimen
- Screenshot tren rasio sentimen harian
- Screenshot korelasi volume diskusi vs daily return
- Screenshot donut chart distribusi sentimen
- Screenshot word cloud (all, positif, negatif)
- Screenshot sampel postingan sentimen (slider horizontal)

### 4.5.4 Fitur Data Explorer
- Screenshot tabel data interaktif dengan filter tanggal
- Screenshot fitur download CSV
- Screenshot data cleaning Stockbit & raw stream data

## 4.6 Pembahasan

### 4.6.1 Pembahasan Kontribusi Sentimen (Menjawab Rumusan Masalah 2)
- Bukti kuantitatif: penurunan MAPE di semua model ketika sentimen ditambahkan
- Penjelasan mekanisme: sentimen menangkap faktor psikologi pasar yang tidak ada di teknikal
- Hubungan antara volume diskusi tinggi dan volatilitas harga
- Perbandingan dengan temuan penelitian terdahulu (Bollen et al., 2011)

### 4.6.2 Pembahasan Pemilihan Model Terbaik (Menjawab Rumusan Masalah 3)
- Justifikasi keunggulan LSTM: mekanisme cell state untuk long-term dependency
- Analisis trade-off: LSTM paling akurat tetapi paling lambat; GRU trade-off terbaik; CNN paling cepat tetapi kurang akurat
- Perbandingan dengan temuan penelitian sejenis

### 4.6.3 Pembahasan Sistem Keseluruhan (Menjawab Rumusan Masalah 1)
- Keterintegrasian pipeline end-to-end: dari data mentah hingga prediksi GUI
- Keunggulan arsitektur modular 7 lapisan
- Reproduksibilitas: seed global, konfigurasi YAML, artefak model tersimpan
- Keterbatasan sistem dan potensi pengembangan

---

# BAB V — PENUTUP

## 5.1 Kesimpulan

**Kesimpulan yang harus menjawab setiap rumusan masalah:**

1. **Sistem prediksi** berhasil dirancang dan dibangun dengan arsitektur 7 lapisan yang mengintegrasikan data harga historis (9 indikator teknikal) dengan analisis sentimen diskusi investor Stockbit (4 fitur sentimen, InSet Lexicon), menggunakan tiga arsitektur deep learning (LSTM, GRU, CNN 1D) berbasis PyTorch, dan disajikan melalui aplikasi GUI interaktif Streamlit.

2. **Fitur sentimen** terbukti meningkatkan akurasi prediksi pada semua model. Pada konfigurasi optimal K3, LSTM mengalami penurunan MAPE sebesar 21.4% (dari 2.85% menjadi 2.24%), GRU turun 11.2% (dari 2.60% menjadi 2.31%), dan CNN 1D turun 9.8% (dari 3.87% menjadi 3.49%) ketika fitur sentimen Stockbit ditambahkan.

3. **Model LSTM skenario S2 dengan konfigurasi K3** menghasilkan performa terbaik dengan MAPE 2.24%, RMSE Rp 221.13, dan MAE Rp 152.99, sehingga dipilih sebagai mesin prediksi utama dalam sistem.


## 5.2 Saran

1. **Peningkatan metode sentimen:** Eksplorasi metode deep learning (BERT, IndoBERT) untuk analisis sentimen yang lebih akurat, terutama dalam menangani sarkasme dan konteks implisit.

2. **Perluasan sumber data:** Integrasi sentimen dari sumber lain (Twitter/X, berita keuangan) untuk memperkaya sinyal sentimen.

3. **Perluasan objek saham:** Menguji sistem pada saham lain di BEI (BMRI, TLKM, ASII) untuk menguji generalisasi.

4. **Arsitektur model lanjutan:** Eksplorasi Transformer, Attention-based LSTM, atau hybrid CNN-LSTM untuk potensi peningkatan akurasi.

5. **Deployment produksi:** Pengembangan sistem ke deployment cloud dengan pembaruan data real-time dan model retraining otomatis.

6. **Validasi sentimen skala lebih besar:** Melakukan anotasi manual pada sampel lebih besar (500–1000 postingan) dengan multiple annotator untuk meningkatkan reliabilitas validasi.

---

# DAFTAR PUSTAKA (Referensi Kunci yang Harus Dicantumkan)

**Referensi utama yang wajib ada:**

| Kategori | Referensi |
|---|---|
| Sentimen & Saham | Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science*. |
| Sentimen & Saham | Tetlock, P. C. (2007). Giving content to investor sentiment. *The Journal of Finance*. |
| InSet Lexicon | Koto, F., & Rahmaningtyas, G. Y. (2017). InSet Lexicon: Evaluation of a word list for Indonesian sentiment analysis. *PACLIC 31*. |
| LSTM | Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*. |
| GRU | Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. *EMNLP*. |
| CNN untuk Time-Series | Gamboa, J. C. B. (2017). Deep learning for time-series analysis. *arXiv preprint*. |
| LSTM Saham | Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*. |
| Sastrawi | Sastrawi Stemmer — Indonesian Stemming Library |
| PyTorch | Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS*. |
| Behavioral Finance | Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *The Journal of Finance*. |

---

# LAMPIRAN

## Lampiran yang Harus Disertakan

1. **Kode sumber utama** (potongan kode kunci dari setiap modul)
2. **Tabel komprehensif 30 konfigurasi evaluasi** (K1–K5 × 3 model × 2 skenario)
3. **Grafik loss training** setiap model
4. **Screenshot lengkap aplikasi GUI** Streamlit (setiap tab & fitur)
5. **Struktur repositori & file** sistem
6. **Contoh data mentah** Stockbit (10–15 postingan, ter-anonimisasi)
7. **Contoh data cleaned** setelah preprocessing (menunjukkan transformasi)

---

# CATATAN PENTING REVISI (DRAFT AWAL → SISTEM AKTUAL)

> [!WARNING]
> ## Perbandingan Detail: Skripsi Draft Awal vs Sistem Aktual

Berikut adalah analisis perbandingan mendalam antara isi skripsi **DRAFT FINAL_1** (yang telah lulus sidang awal) dengan **sistem yang sekarang telah diimplementasikan**. Outline revisi di atas telah disesuaikan sepenuhnya dengan sistem aktual.

---

### ✅ ELEMEN YANG DIPERTAHANKAN DARI DRAFT AWAL

| Elemen | Detail di Draft Awal | Status |
|---|---|---|
| **Judul Inti** | "Pemanfaatan Analisis Sentimen Diskusi Investor pada Aplikasi Stockbit untuk Memprediksi Pergerakan Harga Saham" | ✅ Dipertahankan, diperkuat menjadi "Sistem Prediksi" |
| **Objek Saham** | BBCA (Bank Central Asia Tbk.) | ✅ Tetap |
| **Sumber Sentimen** | Stockbit Stream forum diskusi investor | ✅ Tetap |
| **Metode Sentimen** | InSet Lexicon (Lexicon-Based) | ✅ Tetap |
| **3 Arsitektur DL** | LSTM, GRU, CNN 1D | ✅ Tetap |
| **2 Skenario** | S1 (teknikal saja) vs S2 (teknikal + sentimen) | ✅ Tetap |
| **4 Fitur Sentimen** | sentiment_score, positive_ratio, negative_ratio, discussion_volume | ✅ Tetap, nama fitur identik |
| **Metrik Evaluasi** | MAE, MSE, RMSE, MAPE (utama) | ✅ Tetap (MAPE tetap metrik utama) |
| **5 Config K1–K5** | Grid search 5 konfigurasi hyperparameter | ✅ Tetap |
| **Multi-Run (3×)** | 3 seed berbeda per konfigurasi | ✅ Tetap |
| **Framework** | PyTorch (2.9.0 + CUDA) | ✅ Tetap |
| **Split Data** | 80% train / 20% test (kronologis) | ✅ Tetap |
| **Lookback** | 30 timestep | ✅ Tetap |
| **Normalisasi** | MinMaxScaler [0,1] (fit on train only) | ✅ Tetap |
| **Pendekatan Metodologi** | Design Science Research (Studi Deskriptif I → Preskriptif → Deskriptif II) | ✅ Tetap |
| **GQM Framework** | Goal-Question-Metric untuk instrumen penelitian | ✅ Tetap |

---

### 🔄 ELEMEN YANG DIREVISI / DIPERKUAT

| Aspek | Draft Awal | Sistem Aktual (Revisi) | Catatan |
|---|---|---|---|
| **Periode Data Sentimen** | "sepanjang tahun perdagangan 2025" (1 tahun) | **2021–2026** (±5 tahun, 255.000+ postingan) | 📌 **Revisi kritis**: Data diperluas jauh lebih lama atas masukan penguji bahwa data terlalu pendek |
| **Periode Data Harga** | 2025 (1 tahun) | **2016–2026** (±10 tahun) | 📌 Periode harga diperluas signifikan |
| **Jumlah Data Sentimen** | Tidak disebutkan secara eksplisit | **255.000+ postingan** unik terkumpul | 📌 Skala data jauh lebih besar |
| **Fitur Teknikal (9)** | Close, Open, High, Low, Volume, MA5, MA20, RSI14, Daily Return | Close, Open, High, Low, Volume, MA5, MA20, RSI14, Return | ✅ Identik, hanya penamaan "Daily Return" → "Return" |
| **Fitur Teknikal Hilang** | Draft menyebut MACD, MA10, Bollinger Band di beberapa bagian | Tidak diimplementasikan di sistem aktual | 📌 Sesuaikan: pastikan konsisten 9 fitur seperti di kode |
| **Preprocessing Teks** | Case folding, Remove punctuation, Tokenization, Stopwords removal, Normalisasi, Stemming | Sama dengan draft awal | ✅ Tetap |
| **Pelabelan Sentimen** | Scoring InSet Lexicon standar (positif/negatif/netral) | Scoring InSet Lexicon standar | ✅ Tetap |
| **Kelas Sentimen** | 3 kelas (positif, negatif, netral) | **2 kelas** (positif, negatif; netral dibuang) — config `classes: 2` | 📌 Keputusan desain: kelas netral dibuang untuk mengurangi noise |
| **Agregasi Sentimen** | Rata-rata skor + proporsi + volume | + **Net Sentiment Spread (NSS)** formula (Tetlock 2007), + **EMA-3 smoothing**, + Penyelarasan akhir pekan (Sabtu/Minggu → Senin) | 📌 Agregasi lebih sophisticated di sistem aktual |
| **Arsitektur Model** | Stacked 2-layer (LSTM/GRU), CNN 1D | Identik, tetapi **detail implementasi PyTorch** lebih jelas: gradient clipping, weight decay, ReduceLROnPlateau | 📌 Mekanisme training lebih robust |
| **Konfigurasi K1–K5** | K1(32,0.1,0.005,16,50), K2(64,0.2,0.001,32,50), K3(128,0.2,0.001,32,50), K4(128,0.3,0.0005,64,50), K5(256,0.4,0.0001,64,50) | **Sistem aktual** menggunakan K3(64,0.2,0.001,16,150) sebagai optimal + **kalibrasi per model** (dropout & lr disesuaikan per arsitektur/skenario) | 📌 K3 direvisi + adaptive tuning per model |
| **Hasil MAPE (Draft)** | LSTM terbaik S1=1.02%, GRU=1.03%, CNN=1.63% | **Sistem aktual**: LSTM S2=2.24%, GRU S2=2.31%, CNN S2=3.49% | 📌 Angka berbeda karena periode data dan konfigurasi berbeda |
| **Temuan Sentimen (Draft)** | CNN mendapat benefit terbesar (−0.23pp), LSTM/GRU benefit minimal | **Sistem aktual**: LSTM mendapat benefit terbesar (−21.4%), GRU (−11.2%), CNN (−9.8%) | 📌 Temuan berubah: semua model benefit signifikan |
| **GUI Streamlit** | Tidak dibahas di draft (hanya di blueprint) | **4 tab dashboard** interaktif dengan fitur premium (candlestick, donut chart, word cloud, data explorer, post slider, KPI cards) | 📌 **Revisi mayor**: GUI sekarang menjadi komponen utama "prototipe sistem" |

---

### ➕ ELEMEN BARU YANG DITAMBAHKAN (Tidak Ada di Draft Awal)

| Elemen Baru | Deskripsi | Relevansi Bab |
|---|---|---|
| **Net Sentiment Spread (NSS)** | Formula: NSS = pos_ratio − neg_ratio; Combined Score = raw_score × (1 + NSS) | BAB III §3.5.4 |
| **EMA-3 Smoothing** | Exponential Moving Average span=3 pada fitur sentimen harian untuk meredam noise | BAB III §3.5.4 |
| **Penyelarasan Akhir Pekan** | Postingan Sabtu → Senin (+2 hari), Minggu → Senin (+1 hari) | BAB III §3.5.4 |
| **Log1p Transformation** | `np.log1p()` pada volume diskusi untuk menstabilkan distribusi | BAB III §3.5.5 |
| **Gradient Clipping** | `torch.nn.utils.clip_grad_norm_` max_norm=1.0 mencegah exploding gradient | BAB III §3.5.6C |
| **Weight Decay** | Adam optimizer weight_decay=1e-4 untuk regularisasi | BAB III §3.5.6C |
| **ReduceLROnPlateau** | Learning rate scheduler factor=0.5, patience=10 | BAB III §3.5.6C |
| **Adaptive Dropout/LR per Model** | Kalibrasi dropout dan learning rate berbeda per arsitektur (LSTM, GRU, CNN) dan skenario (S1, S2) | BAB III §3.5.6C |
| **Prediksi H+1 Real-Time** | Sistem GUI menghitung prediksi harga esok hari dari data terakhir | BAB III §3.5.8, BAB IV §4.5.2 |
| **Data Explorer** | Tab 4 GUI: pencarian, filter tanggal, sort, download CSV dataset | BAB III §3.5.8 |
| **Sampel Postingan Live** | Horizontal slider card menampilkan postingan terbaru dengan label dan skor sentimen | BAB III §3.5.8 |
| **Crawler Exodus API v3** | Modul crawler produksi dengan checkpoint/resume, deduplikasi, rate limiting, anonimisasi PII | BAB III §3.5.1A |
| **Pipeline End-to-End** | `run_pipeline.py` — skrip otomasi dari crawling hingga evaluasi model terbaik | BAB III §3.3 |

---

### ⚠️ POIN KRITIS YANG HARUS DIPERHATIKAN SAAT REVISI

> [!CAUTION]
> 1. **Konsistensi Angka Hasil**: Draft awal melaporkan MAPE sangat rendah (1.02%–1.63%) pada data 2025 saja. Sistem aktual dengan periode lebih panjang (2021–2026) menghasilkan MAPE 2.24%–3.49%. **Pastikan angka di revisi merujuk ke hasil run terbaru dari sistem aktual**, bukan dari draft lama.
>
> 2. **Temuan Sentimen Berubah**: Di draft awal, CNN mendapat benefit terbesar dan LSTM/GRU benefit minimal. Di sistem aktual, **LSTM mendapat benefit terbesar** (−21.4%). Ini **mengubah narasi kesimpulan** secara signifikan.
>
> 3. **Konfigurasi K1–K5 Berbeda**: Spesifikasi K1–K5 di draft awal berbeda dari yang ada di sistem aktual. **Gunakan spesifikasi dari kode `evaluate.py`** yang sebenarnya.
>
> 4. **GUI Baru**: Draft awal tidak membahas GUI sama sekali di bab Hasil. Sistem aktual memiliki **1.573 baris kode Streamlit** dengan 4 tab interaktif premium. Ini harus menjadi **sub-bab tersendiri di BAB IV**.
>
> 5. **Framing "Sistem"**: Draft awal lebih berfokus pada "perbandingan model". Revisi harus me-reframe menjadi **"pembangunan prototipe sistem"** di mana perbandingan model adalah tahap pemilihan mesin dalam sistem.
>
> 6. **Struktur Bab Berubah**: Draft awal menggunakan 3.1 Metode → 3.2 Klarifikasi → 3.3 Studi Deskriptif I → 3.4 Studi Preskriptif → 3.5 Studi Deskriptif II. **Pertahankan kerangka Design Science Research ini** tetapi perkaya kontennya dengan detail implementasi aktual.
>
> 7. **Periode Data**: Jelaskan justifikasi perpanjangan periode dari "2025 saja" menjadi "2021–2026" sebagai respons terhadap masukan penguji tentang data yang terlalu pendek.

---

## Mapping File Kode ↔ Bab Skripsi

| File Kode Sistem | Relevansi Bab |
|---|---|
| [config.yaml](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/config.yaml) | BAB III §3.2 (parameter penelitian) |
| [src/config.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/config.py) | BAB III §3.5.6 (reproduksibilitas, seed) |
| [src/data/crawl_stream.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/data/crawl_stream.py) | BAB III §3.5.1A (crawling Stockbit) |
| [src/data/fetch_prices.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/data/fetch_prices.py) | BAB III §3.5.1B (unduh harga) |
| [src/sentiment/preprocess_text.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/sentiment/preprocess_text.py) | BAB III §3.5.2 (preprocessing teks) |
| [src/sentiment/lexicon_scorer.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/sentiment/lexicon_scorer.py) | BAB III §3.5.2 & BAB II §2.2.4 (InSet Lexicon) |
| [src/sentiment/validate_labels.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/sentiment/validate_labels.py) | BAB III §3.5.3 (validasi label) |
| [src/sentiment/aggregate_daily.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/sentiment/aggregate_daily.py) | BAB III §3.5.4 (agregasi harian) |
| [src/features/technical.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/features/technical.py) | BAB II §2.2.2 & BAB III §3.5.5A |
| [src/features/assemble.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/features/assemble.py) | BAB III §3.5.5 (rekayasa fitur) |
| [src/data/build_dataset.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/data/build_dataset.py) | BAB III §3.5.5A (penggabungan data) |
| [src/models/architectures.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/models/architectures.py) | BAB III §3.5.6A (arsitektur model) |
| [src/models/train.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/models/train.py) | BAB III §3.5.6C (training) |
| [src/models/evaluate.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/src/models/evaluate.py) | BAB III §3.5.7 & BAB IV §4.4 |
| [app/streamlit_app.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/app/streamlit_app.py) | BAB III §3.5.8 & BAB IV §4.5 |
| [run_pipeline.py](file:///d:/Skripsi/Proses%20SKRIPSI/Dokumen%20Skripsi/Bismillah%20Sidang%20akhir/Code%20Akhir/Sistem/run_pipeline.py) | BAB III §3.3 (arsitektur pipeline) |
