"""Modul arsitektur Deep Learning (StockLSTM, StockGRU, StockCNN) disesuaikan persis dengan notebook.

Menggunakan PyTorch untuk kompatibilitas penuh dan performa optimal.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import set_seed


class StockLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super(StockLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc(out)
        return out.squeeze(-1)


class StockGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super(StockGRU, self).__init__()
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.gru2 = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        out, _ = self.gru2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc(out)
        return out.squeeze(-1)


class StockCNN(nn.Module):
    def __init__(self, input_dim: int, sequence_length: int = 30, hidden_dim: int = 64, dropout_rate: float = 0.2):
        super(StockCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim * sequence_length, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.squeeze(-1)


class PyTorchSeqRegressor:
    """Wrapper kelas untuk melatih dan mengevaluasi model PyTorch sesuai setup notebook."""

    def __init__(
        self,
        model_name: str,
        input_dim: int,
        seq_len: int = 30,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
        lr: float = 0.001,
        seed: int = 42,
    ):
        set_seed(seed)
        self.model_name = model_name.upper()
        self.lr = lr
        self.device = torch.device("cpu")




        if self.model_name == "LSTM":
            self.net = StockLSTM(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(self.device)
        elif self.model_name == "GRU":
            self.net = StockGRU(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(self.device)
        elif self.model_name in ["CNN", "CNN1D", "CNN 1D"]:
            self.net = StockCNN(
                input_dim=input_dim, sequence_length=seq_len, hidden_dim=hidden_dim, dropout_rate=dropout_rate
            ).to(self.device)
        else:
            raise ValueError(f"Model {model_name} tidak dikenali.")

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 150,
        batch_size: int = 16,
        patience: int = 15,
    ):
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.net.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                preds = self.net(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(X_train)
            history["train_loss"].append(epoch_loss)

            if X_val is not None and y_val is not None:
                self.net.eval()
                X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                with torch.no_grad():
                    val_pred = self.net(X_val_t)
                    val_loss = self.criterion(val_pred, y_val_t).item()
                history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = self.net.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            else:
                self.scheduler.step(epoch_loss)

        if best_weights is not None:
            self.net.load_state_dict(best_weights)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return history


    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.net(X_t).cpu().numpy()
        return preds

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), filepath)

    def load(self, filepath: Path):
        self.net.load_state_dict(torch.load(filepath, map_location=self.device))

    @classmethod
    def from_file(
        cls,
        filepath: Path,
        input_dim: int,
        seq_len: int = 30,
        hidden_dim: int = 64,
        seed: int = 42,
    ):
        """Memuat model secara otomatis mendeteksi arsitektur (StockLSTM, StockGRU, StockCNN) dari state_dict."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(filepath, map_location=device)

        if any(k.startswith("gru") for k in state_dict.keys()):
            model_name = "GRU"
        elif any(k.startswith("conv") for k in state_dict.keys()):
            model_name = "CNN"
        else:
            model_name = "LSTM"

        regressor = cls(
            model_name=model_name,
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            seed=seed,
        )
        try:
            regressor.net.load_state_dict(state_dict)
        except RuntimeError:
            regressor.net.load_state_dict(state_dict, strict=False)

        return regressor

