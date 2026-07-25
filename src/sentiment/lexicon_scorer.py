"""Modul pelabelan dan kalkulasi skor sentimen berbasis InSet Lexicon.

Secara otomatis mengunduh InSet Lexicon dari GitHub ke folder lexicon/ jika belum ada.
"""

from pathlib import Path
import pandas as pd
import requests

from src.config import CONFIG, LEXICON_DIR
from src.sentiment.preprocess_text import preprocess_text_full


class InSetLexiconScorer:
    def __init__(self, lexicon_dir: Path = LEXICON_DIR):
        self.lexicon_dir = Path(lexicon_dir)
        self.pos_file = self.lexicon_dir / "positive.tsv"
        self.neg_file = self.lexicon_dir / "negative.tsv"

        self.pos_url = CONFIG["sentiment"]["lexicon_urls"]["positive"]
        self.neg_url = CONFIG["sentiment"]["lexicon_urls"]["negative"]

        self.lexicon = {}
        self.ensure_lexicon_available()
        self.load_lexicon()

    def download_file(self, url: str, target_path: Path):
        """Mengunduh file lexicon dari GitHub."""
        print(f"[Lexicon] Mengunduh {target_path.name} dari {url}...")
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"[Lexicon] Berhasil disimpan di {target_path}")

    def ensure_lexicon_available(self):
        """Memastikan file positive.tsv dan negative.tsv dari InSet Lexicon tersedia dan lengkap dari GitHub."""
        if not self.pos_file.exists() or self.pos_file.stat().st_size < 500:
            try:
                self.download_file(self.pos_url, self.pos_file)
            except Exception as e:
                print(f"[Lexicon Error] Gagal mengunduh positive.tsv: {e}")
                if not self.pos_file.exists():
                    self._create_fallback_lexicon(self.pos_file, is_positive=True)

        if not self.neg_file.exists() or self.neg_file.stat().st_size < 500:
            try:
                self.download_file(self.neg_url, self.neg_file)
            except Exception as e:
                print(f"[Lexicon Error] Gagal mengunduh negative.tsv: {e}")
                if not self.neg_file.exists():
                    self._create_fallback_lexicon(self.neg_file, is_positive=False)

    def _create_fallback_lexicon(self, path: Path, is_positive: bool):
        """Menyediakan lexicon dasar jika unduhan dari GitHub terkendala."""
        path.parent.mkdir(parents=True, exist_ok=True)
        sample_data = (
            "word\tweight\nuntung\t4\nnaik\t3\nbagus\t3\ncuan\t5\nbeli\t2\n"
            if is_positive
            else "word\tweight\nrugi\t-4\nturun\t-3\njelek\t-3\nloss\t-5\njual\t-2\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(sample_data)

    def load_lexicon(self):
        """Memuat kata dan bobot dari TSV InSet Lexicon ke dalam dictionary."""
        self.lexicon = {}
        for is_pos, file_path in [(True, self.pos_file), (False, self.neg_file)]:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, sep="\t")
                    # Standarisasi nama kolom
                    df.columns = [c.strip().lower() for c in df.columns]
                    word_col = df.columns[0]
                    weight_col = df.columns[1]

                    for _, row in df.iterrows():
                        word = str(row[word_col]).strip().lower()
                        try:
                            weight = float(row[weight_col])
                            if is_pos:
                                weight = abs(weight)
                                self.lexicon[word] = weight
                            else:
                                weight = -abs(weight)
                                # Jangan menimpa kata positif yang sudah terdaftar di positive.tsv
                                if word not in self.lexicon:
                                    self.lexicon[word] = weight
                        except ValueError:
                            continue
                except Exception as e:
                    print(f"[Lexicon Error] Gagal membaca {file_path.name}: {e}")


        # Injeksi kata & slang pasar modal Stockbit khusus dengan bobot sentimen presisi
        # Injeksi kata & slang pasar modal Stockbit khusus dengan bobot sentimen presisi
        STOCK_SLANG_WEIGHTS = {
            # Positif (Bullish, Profit, Rebound, Accumulation, Dividends)
            "cuan": 4.5, "untung": 4.0, "profit": 4.0, "terbang": 3.5, "haka": 3.5, "serok": 3.0,
            "bullish": 3.5, "bulish": 3.5, "uptrend": 3.5, "uptren": 3.5, "bagus": 3.0, "dividen": 3.0,
            "naik": 3.0, "arah": 4.0, "ara": 4.0, "hijau": 2.5, "ijo": 2.5, "joss": 3.0, "jos": 3.0,
            "gass": 2.5, "gas": 2.5, "gasken": 2.5, "moon": 3.5, "to-the-moon": 4.0, "masuk": 1.5,
            "buy": 2.0, "pantau": 1.0, "rebound": 3.5, "mantul": 3.5, "breakout": 3.5, "cumdate": 3.0,
            "stocksplit": 3.0, "buyback": 3.5, "bagger": 4.5, "multibagger": 4.5,

            # Negatif (Bearish, Panic, Drops, Traps, Heavy Sell)
            "nyangkut": -4.5, "rugi": -4.0, "loss": -4.0, "longsor": -4.0, "anjlok": -4.0,
            "haki": -3.5, "bearish": -3.5, "bearis": -3.5, "downtrend": -3.5, "downtren": -3.5,
            "melorot": -3.5, "tenggelam": -3.5, "meluncur": -3.5, "terperosok": -3.5, "kebanting": -3.5,
            "kejatuhan": -4.0, "jatuh": -3.5, "bahaya": -3.5, "gagal": -3.0, "ancaman": -3.0,
            "jelek": -3.0, "turun": -3.0, "merah": -2.5, "arb": -4.0, "kebakaran": -3.5, "cutloss": -4.0,
            "ambles": -3.5, "sangkur": -4.5, "buntung": -4.0, "hancur": -4.0, "ancur": -4.0, "parah": -3.0,
            "sell": -2.0, "keringat dingin": -4.0, "panic": -3.5, "panik": -3.5, "guyur": -4.0,
            "diguyur": -4.0, "banting": -4.0, "dibanting": -4.0, "terjun": -4.5, "nyungsep": -4.0,
            "trap": -4.0, "exdate": -2.5, "delisting": -5.0, "pompom": -3.0, "pompon": -3.0
        }

        # Netralkan kata-kata formal bursa & partisipan pasar yang tidak memiliki emosi di InSet bawaan
        FORMAL_STOCK_NEUTRAL_WORDS = {
            "indonesia", "perusahaan", "pasar", "saham", "pusat", "bursa", "gabungan", "direksi",
            "laporan", "harga", "sahamnya", "jakarta", "emiten", "indeks", "sahamku", "rupiah",
            "dolar", "investasi", "investor", "keuangan", "perbankan", "bank", "central", "asia",
            "transaksi", "perdagangan", "penutupan", "sesi", "analisis", "rekomendasi", "keluar",
            "bandar", "bandarmologi", "asing", "foreign", "retail", "ritel", "ihsg", "lot", "lembar"
        }
        for nw in FORMAL_STOCK_NEUTRAL_WORDS:
            self.lexicon[nw] = 0.0



        for slang_w, slang_val in STOCK_SLANG_WEIGHTS.items():
            self.lexicon[slang_w] = slang_val

        print(
            f"[Lexicon] Total {len(self.lexicon)} entri kata lexicon (termasuk slang bursa & penetralan formal) berhasil dimuat."
        )



    def score_text(self, text: str) -> dict:
        """Menghitung total skor sentimen dari teks terpilih dengan negation handling."""
        from src.sentiment.preprocess_text import NEGATION_WORDS

        tokens = text.split()
        score = 0.0
        pos_words = []
        neg_words = []

        for i, token in enumerate(tokens):
            if token in NEGATION_WORDS:
                continue

            if token in self.lexicon:
                weight = self.lexicon[token]

                # Periksa apakah ada kata negasi 1 atau 2 token sebelum kata ini
                is_negated = False
                for prev_offset in [1, 2]:
                    if i - prev_offset >= 0 and tokens[i - prev_offset] in NEGATION_WORDS:
                        is_negated = True
                        break

                if is_negated:
                    weight = -weight

                score += weight
                if weight > 0:
                    pos_words.append(token)
                elif weight < 0:
                    neg_words.append(token)

        if score > 0:
            label = 1  # Positif
        elif score < 0:
            label = -1  # Negatif
        else:
            label = 0  # Netral

        return {
            "score": score,
            "label": label,
            "pos_words_count": len(pos_words),
            "neg_words_count": len(neg_words),
        }


def process_sentiment_for_dataframe(
    df: pd.DataFrame, text_column: str = "content", num_classes: int = 2
) -> pd.DataFrame:
    """Memproses dataframe postingan, menambahkan kolom skor sentimen & label."""
    scorer = InSetLexiconScorer()

    df = df.copy()
    print("[Sentiment Scorer] Memproses preprocessing dan scoring teks...")

    df["cleaned_content"] = df[text_column].apply(
        lambda t: preprocess_text_full(str(t))
    )
    scores = df["cleaned_content"].apply(scorer.score_text)

    df["sentiment_score"] = [s["score"] for s in scores]
    df["sentiment_label"] = [s["label"] for s in scores]

    if num_classes == 2:
        print("[Sentiment Scorer] Membuang kelas netral (label == 0)...")
        df = df[df["sentiment_label"] != 0].copy()

    return df
