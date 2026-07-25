"""Modul pelatihan model deep learning (LSTM, GRU, CNN 1D)."""

from pathlib import Path
import matplotlib.pyplot as plt

from src.config import CONFIG, FIGURES_DIR, MODELS_DIR, set_seed
from src.features.assemble import prepare_scenario_data
from src.models.architectures import PyTorchSeqRegressor


def train_model(
    model_name: str = "LSTM",
    scenario: str = "S2",
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    seed: int = None,
):
    """Melatih satu model deep learning pada skenario tertentu."""
    seed = seed or CONFIG["seed"]
    set_seed(seed)

    print(
        f"\n[Train] Memulai pelatihan Model={model_name.upper()} | Skenario={scenario.upper()} | Seed={seed}"
    )

    data = prepare_scenario_data(scenario=scenario, lookback=CONFIG["features"]["lookback"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    regressor = PyTorchSeqRegressor(
        model_name=model_name, input_dim=input_dim, seq_len=seq_len, seed=seed
    )

    history = regressor.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )

    # Simpan plot loss curve
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = FIGURES_DIR / f"loss_curve_{model_name.lower()}_{scenario.lower()}.png"

    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss (MSE)")
    plt.plot(history["val_loss"], label="Validation Loss (MSE)")
    plt.title(f"Loss Curve - {model_name.upper()} ({scenario.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Simpan artefak model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name.lower()}_{scenario.lower()}.pt"
    regressor.save(model_path)

    print(f"[Train Selesai] Model disimpan: {model_path}")
    print(f"[Train Selesai] Loss plot disimpan: {plot_path}")

    return regressor, history, data


if __name__ == "__main__":
    train_model(model_name="LSTM", scenario="S1")
    train_model(model_name="LSTM", scenario="S2")
