"""Modul hyperparameter search / tuning sederhana untuk model deep learning."""

from src.config import CONFIG, set_seed
from src.features.assemble import prepare_scenario_data
from src.models.architectures import PyTorchSeqRegressor


def tune_hyperparameters(model_name: str = "LSTM", scenario: str = "S2"):
    """Melakukan pencarian kombinasi hyperparameter terbaik (learning rate)."""
    set_seed(CONFIG["seed"])
    print(f"[Tune] Tuning hyperparameter untuk {model_name} ({scenario})...")

    data = prepare_scenario_data(scenario=scenario, lookback=CONFIG["features"]["lookback"])
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    learning_rates = [0.005, 0.001, 0.0005]
    best_lr = 0.001
    best_val_loss = float("inf")

    for lr in learning_rates:
        regressor = PyTorchSeqRegressor(
            model_name=model_name,
            input_dim=X_train.shape[2],
            seq_len=X_train.shape[1],
            lr=lr,
        )
        history = regressor.fit(
            X_train, y_train, X_val, y_val, epochs=20, batch_size=32, patience=5
        )
        final_val_loss = min(history["val_loss"])
        print(f"LR: {lr} -> Val Loss Terendah: {final_val_loss:.6f}")

        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_lr = lr

    print(f"[Tune Selesai] Learning Rate Terbaik untuk {model_name}: {best_lr}")
    return {"best_lr": best_lr, "best_val_loss": best_val_loss}


if __name__ == "__main__":
    tune_hyperparameters("LSTM", "S2")
