"""Modul preprocessing teks diskusi saham.

Tahapan preprocessing:
1. Case folding (lowercase)
2. Pembersihan URL, mention (@user), cashtag ($BBCA), emoji, dan karakter non-alfabet
3. Normalisasi kata gaul/slang finansial
4. Filtering stopword Bahasa Indonesia (mempertahankan kata negasi)
5. Stemming Bahasa Indonesia (Sastrawi)
"""

import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# Kamus slang umum & istilah pasar modal Stockbit lengkap
SLANG_DICTIONARY = {
    # Stockbit positive sentiment slang & typos
    "cuan": "untung",
    "cuanz": "untung",
    "cuann": "untung",
    "cuannn": "untung",
    "cuannnn": "untung",
    "profit": "untung",
    "tp": "untung",
    "takeprofit": "untung",
    "haka": "beli",
    "hakaa": "beli",
    "hakaaa": "beli",
    "serok": "beli",
    "serokk": "beli",
    "serokkk": "beli",
    "buyback": "beli",
    "terbang": "naik",
    "terbanggg": "naik",
    "to-the-moon": "naik",
    "moon": "naik",
    "bullish": "naik",
    "bulish": "naik",
    "bulis": "naik",
    "bull": "naik",
    "uptrend": "naik",
    "uptren": "naik",
    "ijo": "naik",
    "hijau": "naik",
    "jos": "bagus",
    "joss": "bagus",
    "josss": "bagus",
    "gass": "naik",
    "gasken": "naik",
    "arah": "naik",
    "ara": "naik",
    "dividen": "dividen untung",

    # Stockbit negative sentiment slang & typos
    "nyangkut": "rugi",
    "nyangkuttt": "rugi",
    "sangkur": "rugi",
    "buntung": "rugi",
    "loss": "rugi",
    "cl": "rugi",
    "cutloss": "rugi",
    "cut": "rugi",
    "haki": "jual",
    "hakii": "jual",
    "hakiii": "jual",
    "longsor": "turun",
    "longsoor": "turun",
    "longsorrr": "turun",
    "anjlok": "turun",
    "anjlokkk": "turun",
    "bearish": "turun",
    "bearis": "turun",
    "beris": "turun",
    "bear": "turun",
    "downtrend": "turun",
    "downtren": "turun",
    "melorot": "turun",
    "tenggelam": "turun",
    "meluncur": "turun",
    "terperosok": "turun",
    "kebanting": "turun",
    "guyur": "jual turun",
    "diguyur": "jual turun",
    "dibanting": "turun rugi",
    "terjun": "turun rugi",
    "nyungsep": "turun rugi",
    "pompom": "jelek bohong",
    "pompon": "jelek bohong",
    "mantul": "naik untung",
    "rebound": "naik untung",
    "breakout": "naik untung",
    "bahaya": "jelek rugi",
    "gagal": "jelek rugi",
    "ancaman": "jelek rugi",
    "sideway": "tahan",
    "sideways": "tahan",
    "merah": "turun",
    "berdarah": "turun",
    "kebakaran": "turun",
    "arb": "turun",
    "ambles": "turun",
    "gugur": "turun",
    "drop": "turun",
    "parah": "jelek",
    "ancur": "jelek",
    "hancur": "jelek",



    # General informal & negation slang
    "sgt": "sangat",
    "bgt": "banget",
    "gak": "tidak",
    "ga": "tidak",
    "g": "tidak",
    "tdk": "tidak",
    "blm": "belum",
    "belom": "belum",
    "dr": "dari",
    "dgn": "dengan",
    "klo": "kalau",
    "kl": "kalau",
    "utk": "untuk",
    "sdh": "sudah",
    "dah": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "hold": "tahan",
    "keknya": "sepertinya",
    "kayaknya": "sepertinya",
    "kayanya": "sepertinya",
}


# Download NLTK stopwords jika belum ada
try:
    INDONESIAN_STOPWORDS = set(stopwords.words("indonesian"))
except Exception:
    nltk.download("stopwords", quiet=True)
    INDONESIAN_STOPWORDS = set(stopwords.words("indonesian"))

# Pertahankan kata negasi kritis untuk analisis sentimen
NEGATION_WORDS = {
    "tidak",
    "bukan",
    "belum",
    "tanpa",
    "tidaklah",
    "jangan",
    "kurang",
    "gak",
    "ga",
    "tdk",
}
INDONESIAN_STOPWORDS = INDONESIAN_STOPWORDS - NEGATION_WORDS

# Inisialisasi Sastrawi Stemmer secara lazy/cached
_STEMMER_INSTANCE = None


def get_stemmer():
    global _STEMMER_INSTANCE
    if _STEMMER_INSTANCE is None:
        factory = StemmerFactory()
        _STEMMER_INSTANCE = factory.create_stemmer()
    return _STEMMER_INSTANCE


def clean_text(text: str) -> str:
    """Membersihkan teks dari URL, mention, emoji (diterjemahkan ke sentimen), cashtag, karakter khusus, dan de-elongasi huruf."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()

    # 1. Konversi HTML entities
    text = text.replace("&amp;", " dan ").replace("&gt;", " ").replace("&lt;", " ").replace("&quot;", " ")

    # 2. Terjemahkan emoji sentimen bursa kritis ke kata sentimen sebelum penghapusan simbol
    text = re.sub(r"[\U0001F680\U0001F4C8\U0001F7E2\U0001F48E\U0001F525\U0001F402\U0001F911\U0001F4B0]", " untung naik ", text)
    text = re.sub(r"[\U0001F4C9\U0001F7E1\U0001F7E0\U0001F534\U0001FA78\U0001F43B\U0001F62D\U0001F92E\U0001F4B8]", " rugi turun ", text)

    # 3. Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # 4. Hapus cashtag ($BBCA) dan mention (@user)
    text = re.sub(r"[\$@]\w+", "", text)
    # 5. Hapus karakter khusus dan angka, hanya sisakan huruf & spasi
    text = re.sub(r"[^a-z\s]", " ", text)
    # 6. De-elongasi kata berulang (misal: "naikkkkk" -> "naik", "cuannnn" -> "cuan", "longsooor" -> "longsor")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    # 7. Normalkan spasi ganda
    text = re.sub(r"\s+", " ", text).strip()

    return text




def normalize_slang(text: str, slang_map: dict = None) -> str:
    """Mengganti kata slang/singkatan dengan bentuk baku."""
    slang_map = slang_map or SLANG_DICTIONARY
    tokens = text.split()
    normalized_tokens = [slang_map.get(w, w) for w in tokens]
    return " ".join(normalized_tokens)


def remove_stopwords(text: str, stop_words: set = None) -> str:
    """Menghapus stopword Bahasa Indonesia (mempertahankan kata negasi)."""
    stop_words = stop_words or INDONESIAN_STOPWORDS
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(filtered_tokens)


def stem_text(text: str) -> str:
    """Melakukan stemming teks menggunakan Sastrawi Stemmer."""
    if not text.strip():
        return ""
    stemmer = get_stemmer()
    return stemmer.stem(text)


SPAM_PATTERNS = [
    r"join\s+(grup|telegram|channel|wa|whatsapp|link)",
    r"vip\s+(group|signal|member)",
    r"titip\s+dana",
    r"hubungi\s+admin",
    r"promo\s+diskon",
    r"pendaftaran\s+gratis",
    r"link\s+di\s+bio",
    r"signal\s+gratis",
    r"random\s+tag",
    r"spam\s+tag",
    r"tagging",
]


def is_spam(text: str) -> bool:
    """Mendeteksi apakah teks postingan merupakan spam, iklan promosi, atau random tag multi-ticker."""
    if not isinstance(text, str) or len(text.strip()) < 5:
        return True
    
    text_lower = text.lower()

    # 1. Filter pola kata spam & random tag
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text_lower):
            return True

    # 2. Filter Spam Multi-Ticker Tagging (> 4 cashtag saham berbeda)
    cashtags = re.findall(r"\$\w+", text)
    if len(cashtags) > 4:
        return True

    return False



def preprocess_text_full(
    text: str, use_stemming: bool = False
) -> str:
    """Pipeline preprocessing teks lengkap: clean -> normalize -> stopword -> (optional) stem."""
    if is_spam(text):
        return ""
    text = clean_text(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    if use_stemming:
        text = stem_text(text)
    return text

