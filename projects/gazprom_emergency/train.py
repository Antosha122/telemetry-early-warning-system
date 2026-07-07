"""Обучение модели прогноза аварий."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .data import get_processed_paths, load_memmap, merge_to_memmap
from .model import build_model, save_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train(cfg_path: str) -> None:
    cfg = load_config(cfg_path)

    # --- Подготовка данных ---
    x_path, y_path = get_processed_paths(cfg.data)
    if not x_path.exists():
        logger.info("Processed data not found, running merge_to_memmap...")
        x_path, y_path, n_features = merge_to_memmap(cfg.data)
    else:
        n_features = 3600

    X, y = load_memmap(x_path, y_path, n_features)
    logger.info("Loaded memmap: X=%s, y=%s", X.shape, y.shape)

    # --- Разделение ---
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X),
        np.array(y),
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=y,
    )

    # --- Масштабирование ---
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- SMOTE ---
    if cfg.training.use_smote:
        logger.info("Applying SMOTE (sampling_strategy=%.2f)...", cfg.training.smote_sampling_strategy)
        smote = SMOTE(
            sampling_strategy=cfg.training.smote_sampling_strategy,
            random_state=cfg.training.random_state,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info("After SMOTE: X_train=%s", X_train.shape)

    # --- DataLoader ---
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    # --- Модель ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(n_features, cfg.model).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer_name = cfg.training.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.training.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    # --- Цикл обучения ---
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Валидация
        model.eval()
        val_loss = 0.0
        all_preds, all_probs, all_true = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= cfg.prediction.threshold).float()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())
        val_loss /= len(test_loader.dataset)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f acc=%.4f f1=%.4f auc=%.4f",
            epoch + 1,
            cfg.training.epochs,
            train_loss,
            val_loss,
            accuracy_score(all_true, all_preds),
            f1_score(all_true, all_preds),
            roc_auc_score(all_true, all_probs),
        )

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_model(model, cfg.model.save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # --- Финальные метрики ---
    logger.info("\n=== Final Metrics ===")
    logger.info("Accuracy:  %.4f", accuracy_score(all_true, all_preds))
    logger.info("Precision: %.4f", precision_score(all_true, all_preds))
    logger.info("Recall:    %.4f", recall_score(all_true, all_preds))
    logger.info("F1-score:  %.4f", f1_score(all_true, all_preds))
    logger.info("ROC-AUC:   %.4f", roc_auc_score(all_true, all_probs))
    logger.info("LogLoss:   %.4f", log_loss(all_true, all_probs))
    logger.info("MCC:       %.4f", matthews_corrcoef(all_true, all_preds))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(all_true, all_preds))
    logger.info("\n%s", classification_report(all_true, all_preds))

    # Сохраняем scaler
    scaler_path = Path(cfg.model.save_path).with_suffix(".scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", scaler_path)


def main():
    parser = argparse.ArgumentParser(description="Train emergency predictor")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
