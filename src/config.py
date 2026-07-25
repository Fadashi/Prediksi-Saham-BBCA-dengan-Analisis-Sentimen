import os
import random
from pathlib import Path
import numpy as np
import yaml
from dotenv import load_dotenv

# Base Directory Path (Root Project)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Membaca file konfigurasi config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan di {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42) -> None:
    """Mengatur seed global untuk reproduksibilitas eksperimen."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


# Instance konfigurasi global & direktori utama
CONFIG = load_config()

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
LEXICON_DIR = BASE_DIR / "lexicon"

# Pastikan folder esensial tersedia
for folder in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LEXICON_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
