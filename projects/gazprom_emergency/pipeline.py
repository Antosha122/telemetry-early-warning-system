"""sklearn Pipeline для защиты от data leakage (замечание senior review #16).

Явный Pipeline из sklearn гарантирует правильный порядок операций:
1. split (train/val) — ДО любых преобразований,
2. fit scaler/feature_engineer ТОЛЬКО на train,
3. transform val (без fit),
4. SMOTE ТОЛЬКО на train (после scaler).

Для MLP (PyTorch) используется обёртка-адаптер, совместимая с sklearn Pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import Config
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


class TorchMLPWrapper:
    """Адаптер PyTorch MLP для sklearn Pipeline."""

    def __init__(
        self,
        cfg: Config,
        input_dim: int | None = None,
        device: str = "auto",
    ) -> None:
        self.cfg = cfg
        self.input_dim = input_dim
        self.device = device
        self._device: Any = None
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPWrapper":
        """Обучает MLP."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from .model import build_model
        from .optimizers import build_optimizer
        from .utils import set_seed

        set_seed(self.cfg.training.seed)
        self.input_dim = X.shape[1]
        if self.device == "auto":
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(self.device)

        self._model = build_model(self.input_dim, self.cfg.model).to(self._device)

        pos_weight = self.cfg.training.pos_weight
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], device=self._device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = build_optimizer(self._model, self.cfg)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        for epoch in range(self.cfg.training.epochs):
            self._model.train()
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятность положительного класса."""
        import torch

        if self._model is None:
            raise RuntimeError("Model not fitted")
        self._model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model(X_tensor)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Возвращает бинарные предсказания."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0, 1])


def build_preprocessing_pipeline(
    cfg: Config,
    batch_times: np.ndarray | None = None,
) -> ImbPipeline:
    """Собирает preprocessing pipeline (без модели)."""
    steps: list[tuple[str, Any]] = []

    fe_cfg = cfg.feature_engineering
    if (
        fe_cfg.add_time_features
        or fe_cfg.lag_sizes
        or fe_cfg.rolling_windows
        or fe_cfg.pca_components is not None
    ):
        feature_engineer = FeatureEngineer(cfg=fe_cfg, batch_times=batch_times)
        steps.append(("feature_engineer", feature_engineer))

    scaler_type = getattr(cfg.training, "scaler_type", "standard")
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    elif scaler_type == "none":
        scaler = None
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type!r}")
    if scaler is not None:
        steps.append(("scaler", scaler))

    if cfg.training.use_smote:
        smote = SMOTE(
            sampling_strategy=cfg.training.smote_sampling_strategy,
            random_state=cfg.training.random_state,
        )
        steps.append(("smote", smote))

    pipeline = ImbPipeline(steps)
    logger.info(
        "Preprocessing pipeline built: %d steps (leakage-safe)", len(steps)
    )
    return pipeline


def build_full_pipeline(
    cfg: Config,
    model_name: str = "mlp",
    batch_times: np.ndarray | None = None,
) -> ImbPipeline:
    """Собирает полный pipeline: preprocessing + модель."""
    pipeline = build_preprocessing_pipeline(cfg, batch_times)

    if model_name == "mlp":
        final_model = TorchMLPWrapper(cfg)
    else:
        from .baselines import build_baseline

        baseline = build_baseline(model_name, cfg)
        final_model = baseline.model

    pipeline.steps.append((f"model_{model_name}", final_model))
    logger.info("Full pipeline built: %s", model_name)
    return pipeline


def transform_with_pipeline(
    pipeline: ImbPipeline,
    X_memmap: np.memmap,
    indices: np.ndarray,
    chunk_size: int = 50000,
) -> np.ndarray:
    """Применяет pipeline.transform к memmap-данным по индексам чанками."""
    sorted_idx = np.sort(indices)
    n = len(sorted_idx)
    result_chunks: list[np.ndarray] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_idx = sorted_idx[start:end]
        chunk = np.asarray(X_memmap[chunk_idx], dtype=np.float32)
        transformed = pipeline.transform(chunk)
        result_chunks.append(transformed)

    return np.vstack(result_chunks)