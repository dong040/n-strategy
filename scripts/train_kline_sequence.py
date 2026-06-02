"""Train the PyTorch K-line sequence model from a prepared NPZ dataset.

Expected NPZ keys:
  - X_seq: [N, T, C]
  - y: [N]
"""

from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from strategy.kline_sequence import SequenceConfig, save_sequence_model, train_sequence_model


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python3 scripts/train_kline_sequence.py data/sequences.npz [output_path]")

    npz_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJECT_ROOT, "data", "kline_sequence_model.pt")

    data = np.load(npz_path)
    X_seq = data["X_seq"]
    y = data["y"]

    artifact = train_sequence_model(
        X_seq=X_seq,
        y=y,
        config=SequenceConfig(window=X_seq.shape[1]),
    )
    save_sequence_model(artifact, output_path)

    print(f"样本数: {len(X_seq)}")
    print(f"窗口: {X_seq.shape[1]} x 通道: {X_seq.shape[2]}")
    print(f"模型已保存 -> {output_path}")
    print("最近 5 轮:")
    for row in artifact["history"][-5:]:
        print(
            f"epoch={row['epoch']:>2} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
