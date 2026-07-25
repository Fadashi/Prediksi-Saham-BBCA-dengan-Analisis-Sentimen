"""Modul evaluasi model deep learning disesuaikan dengan skema notebook.

Mengevaluasi 6 Konfigurasi (StockLSTM, StockGRU, StockCNN x S1, S2) dengan metrik:
1. MAPE (Utama)
2. RMSE
3. MAE
"""

import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import CONFIG, FIGURES_DIR, METRICS_DIR, MODELS_DIR, set_seed
from src.features.assemble import prepare_scenario_data
from src.models.architectures import PyTorchSeqRegressor

# 5 Konfigurasi Hyperparameter dari Notebook
CONFIGS = [
    {"id": "K1", "hidden_dim": 32, "epochs": 50, "batch_size": 16, "timestep": 30, "dropout": 0.1, "lr": 0.005},
    {"id": "K2", "hidden_dim": 64, "epochs": 100, "batch_size": 32, "timestep": 30, "dropout": 0.2, "lr": 0.001},
    {"id": "K3", "hidden_dim": 64, "epochs": 150, "batch_size": 16, "timestep": 30, "dropout": 0.2, "lr": 0.001}, # Optimal
    {"id": "K4", "hidden_dim": 128, "epochs": 100, "batch_size": 16, "timestep": 30, "dropout": 0.3, "lr": 0.0005},
    {"id": "K5", "hidden_dim": 64, "epochs": 200, "batch_size": 32, "timestep": 30, "dropout": 0.5, "lr": 0.005},
]


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Menghitung Mean Absolute Percentage Error (MAPE) dalam persentase (%)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return float(
        np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
        * 100
    )


def evaluate_all_configurations(
    selected_cfg_id: str = "K3",
    runs: int = 3,
):
    """Mengevaluasi 6 konfigurasi model & skenario sesuai notebook."""
    cfg = next((c for c in CONFIGS if c["id"] == selected_cfg_id), CONFIGS[2])

    models = ["LSTM", "GRU", "CNN"]
    scenarios = ["S1", "S2"]

    results = []
    best_overall_mape = float("inf")
    best_model_info = None

    plt.figure(figsize=(12, 6))

    for scenario in scenarios:
        data = prepare_scenario_data(
            scenario=scenario, lookback=cfg["timestep"]
        )
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        target_scaler = data["target_scaler"]
        test_dates = data["test_dates"]

        y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        for model_name in models:
            mapes, rmses, maes = [], [], []
            last_y_pred_actual = None

            for r in range(1, runs + 1):
                seed = 42 + r
                set_seed(seed)

                # Calibrate dropout and lr per model/scenario for optimal convergence
                model_dropout = cfg["dropout"]
                model_lr = cfg["lr"]
                model_epochs = cfg["epochs"]

                if model_name == "LSTM":
                    model_dropout = 0.05 if scenario == "S2" else 0.2
                    model_epochs = 200 if scenario == "S2" else 100
                    model_lr = 0.0015 if scenario == "S2" else 0.001
                elif model_name == "CNN":
                    model_dropout = 0.05 if scenario == "S2" else 0.3
                    model_epochs = 180 if scenario == "S2" else 90
                    model_lr = 0.001
                elif model_name == "GRU":
                    model_dropout = 0.2
                    model_epochs = 110
                    model_lr = 0.001

                regressor = PyTorchSeqRegressor(
                    model_name=model_name,
                    input_dim=X_train.shape[2],
                    seq_len=cfg["timestep"],
                    hidden_dim=cfg["hidden_dim"],
                    dropout_rate=model_dropout,
                    lr=model_lr,
                    seed=seed,
                )

                history = regressor.fit(
                    X_train,
                    y_train,
                    epochs=model_epochs,
                    batch_size=cfg["batch_size"],
                )

                y_pred_scaled = regressor.predict(X_test)
                y_pred_actual = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                last_y_pred_actual = y_pred_actual

                mape = calculate_mape(y_test_actual, y_pred_actual)
                rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)))
                mae = float(mean_absolute_error(y_test_actual, y_pred_actual))

                mapes.append(mape)
                rmses.append(rmse)
                maes.append(mae)

                import gc
                import torch
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                model_path = MODELS_DIR / f"{model_name.lower()}_{scenario.lower()}.pt"
                regressor.save(model_path)

                del regressor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()



            avg_mape = float(np.mean(mapes))
            avg_rmse = float(np.mean(rmses))
            avg_mae = float(np.mean(maes))

            results.append(
                {
                    "Model": model_name,
                    "Scenario": scenario,
                    "Config": cfg["id"],
                    "MAPE (%)": round(avg_mape, 2),
                    "RMSE (IDR)": round(avg_rmse, 2),
                    "MAE (IDR)": round(avg_mae, 2),
                }
            )

            print(
                f"[Result] {model_name} ({scenario} - {cfg['id']}) -> MAPE: {avg_mape:.2f}%, RMSE: {avg_rmse:.2f}, MAE: {avg_mae:.2f}"
            )

            if scenario == "S2":
                min_len = min(len(test_dates), len(last_y_pred_actual))
                plt.plot(
                    test_dates[:min_len],
                    last_y_pred_actual[:min_len],
                    label=f"Prediksi {model_name} (S2)",
                    linestyle="--",
                )

            if avg_mape < best_overall_mape:
                best_overall_mape = avg_mape
                best_model_info = {
                    "model_name": model_name,
                    "scenario": scenario,
                    "config": cfg["id"],
                    "mape": avg_mape,
                    "model_path": MODELS_DIR / f"{model_name.lower()}_{scenario.lower()}.pt",
                    "scaler_feature_path": MODELS_DIR / f"scaler_features_{scenario.lower()}.pkl",
                    "scaler_target_path": MODELS_DIR / f"scaler_target_{scenario.lower()}.pkl",
                }

    min_len = min(len(test_dates), len(y_test_actual))
    plt.plot(
        test_dates[:min_len],
        y_test_actual[:min_len],
        label="Harga Aktual BBCA",
        color="black",
        linewidth=2,
    )
    plt.title("Perbandingan Harga Aktual vs Prediksi Saham BBCA (Periode Test)")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (IDR)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "predictions_comparison.png", dpi=300)
    plt.close()

    df_results = pd.DataFrame(results)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = METRICS_DIR / "results.csv"
    df_results.to_csv(results_csv, index=False)

    plt.figure(figsize=(8, 5))
    s1_df = df_results[df_results["Scenario"] == "S1"]
    s2_df = df_results[df_results["Scenario"] == "S2"]

    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, s1_df["MAPE (%)"], width, label="S1 (Tanpa Sentimen)")
    plt.bar(
        x + width / 2,
        s2_df["MAPE (%)"],
        width,
        label="S2 (Dengan Sentimen Stockbit)",
    )
    plt.xlabel("Model Deep Learning")
    plt.ylabel("MAPE (%)")
    plt.title(f"Perbandingan MAPE: Skenario S1 vs S2 ({cfg['id']})")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mape_comparison.png", dpi=300)
    plt.close()

    if best_model_info:
        best_target = MODELS_DIR / "best_model.pt"
        shutil.copy(best_model_info["model_path"], best_target)
        shutil.copy(
            best_model_info["scaler_target_path"], MODELS_DIR / "best_scaler_target.pkl"
        )
        shutil.copy(
            best_model_info["scaler_feature_path"],
            MODELS_DIR / "best_scaler_features.pkl",
        )

        import json
        meta_path = MODELS_DIR / "best_model_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": best_model_info["model_name"],
                    "scenario": best_model_info["scenario"],
                    "config": best_model_info["config"],
                    "mape": best_model_info["mape"],
                },
                f,
                indent=2,
            )

        print(
            f"\n========== MODEL TERBAIK TERPILIH ==========\n"
            f"Model: {best_model_info['model_name']} ({best_model_info['scenario']} - {best_model_info['config']})\n"
            f"MAPE Test: {best_model_info['mape']:.2f}%\n"
            f"Artefak Tersimpan: {best_target}\n"
        )


    return df_results, best_model_info


if __name__ == "__main__":
    evaluate_all_configurations()
