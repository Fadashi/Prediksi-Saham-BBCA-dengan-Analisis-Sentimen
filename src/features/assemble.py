"""Modul perakitan fitur (S1 vs S2), MinMaxScaler, dan Windowing disesuaikan persis dengan notebook.

S1: ["Close", "Open", "High", "Low", "Volume", "MA5", "MA20", "RSI", "Return"]
S2: ["Close", "Open", "High", "Low", "Volume", "MA5", "MA20", "RSI", "Return", "sentiment_score", "positive_ratio", "negative_ratio", "discussion_volume"]
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import CONFIG, MODELS_DIR, PROCESSED_DATA_DIR
from src.data.build_dataset import build_final_dataset

TECHNICAL_FEATURES = [
    "Close",
    "Open",
    "High",
    "Low",
    "Volume",
    "MA5",
    "MA20",
    "RSI",
    "Return",
]

SENTIMENT_FEATURES = [
    "sentiment_score",
    "positive_ratio",
    "negative_ratio",
    "discussion_volume",
]


def create_sliding_window_sequences(
    feature_array: np.ndarray, target_array: np.ndarray, lookback: int = 30
):
    """Membentuk urutan data sliding window (Samples, Lookback, Features)."""
    X, y = [], []
    for i in range(lookback, len(feature_array)):
        X.append(feature_array[i - lookback : i])
        y.append(target_array[i])
    return np.array(X), np.array(y)


def prepare_scenario_data(
    scenario: str = "S2",
    lookback: int = 30,
    dataset_path: str = None,
    train_ratio: float = None,
):
    """Menyiapkan data (X_train, y_train, X_test, y_test) sesuai fungsi SplitData notebook."""
    train_ratio = train_ratio or CONFIG.get("split", {}).get("train", 0.80)
    
    if dataset_path is None:
        if scenario.upper() == "S3":
            dataset_path = PROCESSED_DATA_DIR / "dataset_final_indobert.csv"
        else:
            dataset_path = PROCESSED_DATA_DIR / "dataset_final.csv"
            
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        if scenario.upper() == "S3":
            build_final_dataset(
                sentiment_file=INTERIM_DATA_DIR / "daily_sentiment_indobert.csv",
                output_file=dataset_path
            )
        else:
            build_final_dataset(output_file=dataset_path)

    df = pd.read_csv(dataset_path)

    df_features = df.copy()
    if "discussion_volume" in df_features.columns:
        df_features["discussion_volume"] = np.log1p(df_features["discussion_volume"].fillna(0))

    if scenario.upper() == "S1":
        feature_cols = TECHNICAL_FEATURES
    elif scenario.upper() in ["S2", "S3"]:
        feature_cols = TECHNICAL_FEATURES + SENTIMENT_FEATURES
    else:
        raise ValueError(f"Skenario {scenario} tidak dikenal. Gunakan 'S1', 'S2', atau 'S3'.")

    features = df_features[feature_cols].values
    target = df[["Target"]].values

    n_total = len(df)
    n_train = int(n_total * train_ratio)

    # Split train & test (sesuai notebook logic)
    train_features = features[:n_train]
    train_target = target[:n_train]

    test_features = features[n_train:]
    test_target = target[n_train:]

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    train_features_scaled = feature_scaler.fit_transform(train_features)
    train_target_scaled = target_scaler.fit_transform(train_target).flatten()

    test_features_scaled = feature_scaler.transform(test_features)
    test_target_scaled = target_scaler.transform(test_target).flatten()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_scaler, MODELS_DIR / f"scaler_features_{scenario.lower()}.pkl")
    joblib.dump(target_scaler, MODELS_DIR / f"scaler_target_{scenario.lower()}.pkl")

    X_train, y_train = create_sliding_window_sequences(
        train_features_scaled, train_target_scaled, lookback=lookback
    )

    # Windowing test (dengan menggabungkan ujung train_features_scaled sesuai notebook)
    test_data_combined = np.vstack([train_features_scaled[-lookback:], test_features_scaled])
    test_target_combined = np.concatenate([train_target_scaled[-lookback:], test_target_scaled])

    X_test, y_test = create_sliding_window_sequences(
        test_data_combined, test_target_combined, lookback=lookback
    )

    print(
        f"[Assemble {scenario}] Train shape: {X_train.shape}, Test shape: {X_test.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_test[: int(len(X_test) * 0.5)], # Untuk validation callback jika dibutuhkan
        "y_val": y_test[: int(len(y_test) * 0.5)],
        "X_test": X_test,
        "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "test_dates": df["Date"].values[n_train:],
    }


if __name__ == "__main__":
    prepare_scenario_data(scenario="S1")
    prepare_scenario_data(scenario="S2")
