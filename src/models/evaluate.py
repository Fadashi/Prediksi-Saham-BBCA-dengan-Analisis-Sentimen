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
    run_all_configs: bool = False,
):
    """Mengevaluasi konfigurasi model & skenario sesuai notebook dengan metrik rata-rata & standar deviasi."""
    configs_to_run = CONFIGS if run_all_configs else [next((c for c in CONFIGS if c["id"] == selected_cfg_id), CONFIGS[2])]

    models = ["LSTM", "GRU", "CNN"]
    scenarios = ["S1", "S2", "S3"]

    results = []
    best_overall_mape = float("inf")
    best_model_info = None

    plt.figure(figsize=(12, 6))

    total_tasks = len(configs_to_run) * len(scenarios) * len(models)
    task_counter = 0

    print(f"\n[EVALUATION PIPELINE] Total kombinasi yang akan dievaluasi: {total_tasks} task ({runs} runs/task)")

    for cfg in configs_to_run:
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
                task_counter += 1
                mapes, rmses, maes = [], [], []
                last_y_pred_actual = None

                print(f"[{task_counter}/{total_tasks}] Training {model_name} | Scenario: {scenario} | Config: {cfg['id']} (Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}, LR: {cfg['lr']})...")

                for r in range(1, runs + 1):
                    seed = 42 + r
                    set_seed(seed)

                    # Standarisasi mutlak tanpa override per skenario (Apple-to-Apple Comparison)
                    model_dropout = cfg["dropout"]
                    model_lr = cfg["lr"]
                    model_epochs = cfg["epochs"]

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

                avg_mape, std_mape = float(np.mean(mapes)), float(np.std(mapes))
                avg_rmse, std_rmse = float(np.mean(rmses)), float(np.std(rmses))
                avg_mae, std_mae = float(np.mean(maes)), float(np.std(maes))

                results.append(
                    {
                        "Model": model_name,
                        "Scenario": scenario,
                        "Config": cfg["id"],
                        "MAPE (%)": round(avg_mape, 2),
                        "MAPE Std (%)": round(std_mape, 2),
                        "RMSE (IDR)": round(avg_rmse, 2),
                        "RMSE Std (IDR)": round(std_rmse, 2),
                        "MAE (IDR)": round(avg_mae, 2),
                        "MAE Std (IDR)": round(std_mae, 2),
                    }
                )

                print(
                    f"   -> Result: MAPE = {avg_mape:.2f}% (+/-{std_mape:.2f}%), RMSE = {avg_rmse:.2f} (+/-{std_rmse:.2f}), MAE = {avg_mae:.2f} (+/-{std_mae:.2f})"
                )

                if scenario == "S3" and cfg["id"] == "K3":
                    min_len = min(len(test_dates), len(last_y_pred_actual))
                    plt.plot(
                        test_dates[:min_len],
                        last_y_pred_actual[:min_len],
                        label=f"Prediksi {model_name} (S3 IndoBERT)",
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
    
    if run_all_configs:
        results_csv = METRICS_DIR / "results_all_configs.csv"
    else:
        results_csv = METRICS_DIR / "results.csv"
        
    df_results.to_csv(results_csv, index=False)
    print(f"\n[SUCCESS] Results successfully saved to: {results_csv}")

    plt.figure(figsize=(10, 5))
    k3_results = df_results[df_results["Config"] == "K3"] if run_all_configs else df_results
    s1_df = k3_results[k3_results["Scenario"] == "S1"]
    s2_df = k3_results[k3_results["Scenario"] == "S2"]
    s3_df = k3_results[k3_results["Scenario"] == "S3"]

    if len(s1_df) > 0 and len(s2_df) > 0 and len(s3_df) > 0:
        x = np.arange(len(models))
        width = 0.25

        plt.bar(x - width, s1_df["MAPE (%)"], width, label="S1 (Tanpa Sentimen)")
        plt.bar(x, s2_df["MAPE (%)"], width, label="S2 (InSet Lexicon)")
        plt.bar(x + width, s3_df["MAPE (%)"], width, label="S3 (IndoBERT)")
        plt.xlabel("Model Deep Learning")
        plt.ylabel("MAPE (%)")
        plt.title("Perbandingan MAPE: Skenario S1 vs S2 vs S3")
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
