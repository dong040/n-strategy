"""PyTorch K-line sequence model.

This module is intentionally self-contained so we can iterate on deep sequence
filters without disturbing the rule-based scanner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SequenceConfig:
    window: int = 30
    hidden_dim: int = 64
    conv_channels: int = 32
    dropout: float = 0.15
    lr: float = 1e-3
    epochs: int = 35
    batch_size: int = 64


def build_kline_tensor(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    end_idx: int | None = None,
    window: int = 30,
) -> np.ndarray | None:
    """Convert a recent OHLCV window into a normalized deep-learning tensor.

    Output shape: [window, 8]
    Channels:
      0 close-to-close return
      1 open gap
      2 candle body / prev close
      3 full range / prev close
      4 upper shadow / prev close
      5 lower shadow / prev close
      6 volume ratio vs 5-bar mean
      7 close position in daily range
    """
    if end_idx is None:
        end_idx = len(closes) - 1
    start = end_idx - window + 1
    if start < 1:
        return None

    feats = []
    for i in range(start, end_idx + 1):
        prev_close = max(float(closes[i - 1]), 1e-9)
        day_range = max(float(highs[i] - lows[i]), 1e-9)
        body_top = max(float(opens[i]), float(closes[i]))
        body_bottom = min(float(opens[i]), float(closes[i]))
        upper_shadow = max(float(highs[i]) - body_top, 0.0)
        lower_shadow = max(body_bottom - float(lows[i]), 0.0)
        vol5 = float(np.mean(vols[max(0, i - 4):i + 1]))
        close_pos = (float(closes[i]) - float(lows[i])) / day_range
        feats.append(
            [
                (float(closes[i]) - prev_close) / prev_close,
                (float(opens[i]) - prev_close) / prev_close,
                (float(closes[i]) - float(opens[i])) / prev_close,
                day_range / prev_close,
                upper_shadow / prev_close,
                lower_shadow / prev_close,
                float(vols[i]) / max(vol5, 1e-9),
                close_pos,
            ]
        )

    arr = np.asarray(feats, dtype=np.float32)
    arr[:, :7] = np.clip(arr[:, :7], -3.0, 3.0)
    return arr


class KLineSequenceNet(nn.Module):
    def __init__(self, input_dim: int = 8, conv_channels: int = 32, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = x.transpose(1, 2)       # [B, C, T]
        x = self.conv(x)
        x = x.transpose(1, 2)       # [B, T, C]
        seq, _ = self.gru(x)
        last = seq[:, -1, :]
        return self.head(last).squeeze(-1)


def train_sequence_model(
    X_seq: np.ndarray,
    y: np.ndarray,
    config: SequenceConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = SequenceConfig(window=X_seq.shape[1])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X_seq) < 40:
        raise ValueError("Sequence dataset too small; need at least 40 samples")

    split = max(int(len(X_seq) * 0.75), len(X_seq) - 20)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y[:split], y[split:]

    model = KLineSequenceNet(
        input_dim=X_seq.shape[2],
        conv_channels=config.conv_channels,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_state = None
    best_val_loss = float("inf")
    history = []

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(X_val))
            val_loss = float(loss_fn(val_logits, torch.from_numpy(y_val)).item())
            val_prob = torch.sigmoid(val_logits).numpy()
            val_pred = (val_prob >= 0.5).astype(np.float32)
            val_acc = float((val_pred == y_val).mean()) if len(y_val) else 0.0

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(float(np.mean(train_losses)), 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "version": 1,
        "model_type": "torch_kline_sequence",
        "config": config.__dict__,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "history": history,
        "input_dim": X_seq.shape[2],
    }


def predict_sequence_prob(artifact: dict[str, Any], seq_tensor: np.ndarray) -> float:
    cfg = artifact.get("config", {})
    model = KLineSequenceNet(
        input_dim=int(artifact.get("input_dim", seq_tensor.shape[-1])),
        conv_channels=int(cfg.get("conv_channels", 32)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        dropout=float(cfg.get("dropout", 0.15)),
    )
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    x = torch.from_numpy(np.asarray(seq_tensor, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return float(prob)


def save_sequence_model(artifact: dict[str, Any], path: str | Path) -> None:
    torch.save(artifact, str(path))


def load_sequence_model(path: str | Path) -> dict[str, Any]:
    return torch.load(str(path), map_location="cpu")
