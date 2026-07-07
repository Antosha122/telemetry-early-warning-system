"""Инференс (прогноз аварий) загруженной моделью."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .model import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_batch(cfg_path: str, input_csv: str | None = None) -> pd.DataFrame:
    """Делает предсказание по батчу данных."""
    cfg = load_config(cfg_path)

    n_features = 3600

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.model.save_path, n_features, cfg.model).to(device)

    # Загружаем scaler
    scaler_path = Path(cfg.model.save_path).with_suffix(".scaler.pkl")
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None

    # Загружаем входные данные
    input_path = Path(input_csv) if input_csv else Path(cfg.data.source_dir) / cfg.data.stpa_file
    logger.info("Loading input: %s", input_path)

    df = pd.read_csv(input_path)
    feat_cols = sorted(
        [c for c in df.columns if c.startswith("v_")],
        key=lambda c: int(c.split("_")[1]),
    )
    X = df[feat_cols].values.astype(np.float32)

    if scaler:
        X = scaler.transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= cfg.prediction.threshold).astype(int)

    result = pd.DataFrame(
        {
            "probability": probs,
            "prediction": preds,
            "threshold": cfg.prediction.threshold,
        }
    )

    output_path = Path(cfg.prediction.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", output_path, len(result))

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict emergencies")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Input CSV for prediction")
    args = parser.parse_args()
    predict_batch(args.config, args.input)


if __name__ == "__main__":
    main()
