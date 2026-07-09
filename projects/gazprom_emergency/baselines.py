"""Бейзлайн-модели и сравнение (замечание senior review #13).

Добавляет классические модели для табличных данных:
- Logistic Regression (простейший линейный бейзлайн),
- Random Forest,
- XGBoost / LightGBM (градиентный бустинг).

Все модели регистрируются в реестре и реализуют единый интерфейс:
``fit(X, y)``, ``predict_proba(X)`` (sklearn-style).

Для табличных данных бустинги часто превосходят MLP при меньших затратах,
поэтому сравнение обязательно для выбора лучшей архитектуры.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .config import Config, CostMatrixConfig

logger = logging.getLogger(__name__)


# ============================================================
# Единый интерфейс моделей-классификаторов
# ============================================================


class BaselineClassifier:
    """Обёртка над sklearn-классификатором с единым интерфейсом.

    Унифицирует API: ``fit``, ``predict_proba``, ``save``, ``load``.
    Поддерживает cost-sensitive learning через ``class_weight``.
    """

    def __init__(self, name: str, model: Any) -> None:
        self.name = name
        self.model = model
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weight: dict[int, float] | None = None,
    ) -> "BaselineClassifier":
        """Обучает модель. Если class_weight передан, применяет его."""
        if class_weight is not None and hasattr(self.model, "class_weight"):
            self.model.set_params(class_weight=class_weight)
        self.model.fit(X, y)
        self._fitted = True
        logger.info("Baseline '%s' fitted on %d samples", self.name, len(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Возвращает вероятность положительного класса."""
        if not self._fitted:
            raise RuntimeError(f"Model {self.name!r} is not fitted")
        proba = self.model.predict_proba(X)
        # Возвращаем вероятность класса 1
        return proba[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Возвращает бинарные предсказания по порогу."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ============================================================
# Реестр фабрик бейзлайн-моделей
# ============================================================


_BASELINE_REGISTRY: dict[str, Any] = {}


def register_baseline(name: str) -> Any:
    """Декоратор для регистрации фабрики бейзлайн-модели."""

    def decorator(factory: Any) -> Any:
        _BASELINE_REGISTRY[name] = factory
        return factory

    return decorator


@register_baseline("logreg")
def _build_logreg(cfg: Config) -> BaselineClassifier:
    """Логистическая регрессия — простейший линейный бейзлайн."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=cfg.training.random_state,
        n_jobs=-1,
        solver="lbfgs",
    )
    return BaselineClassifier("logreg", model)


@register_baseline("random_forest")
def _build_random_forest(cfg: Config) -> BaselineClassifier:
    """Random Forest — ансамбль деревьев решений."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        random_state=cfg.training.random_state,
        n_jobs=-1,
    )
    return BaselineClassifier("random_forest", model)


@register_baseline("xgboost")
def _build_xgboost(cfg: Config) -> BaselineClassifier:
    """XGBoost — градиентный бустинг (если установлен)."""
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError(
            "XGBoost is not installed. Install with: pip install xgboost"
        ) from e
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=cfg.training.random_state,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    return BaselineClassifier("xgboost", model)


@register_baseline("lightgbm")
def _build_lightgbm(cfg: Config) -> BaselineClassifier:
    """LightGBM — градиентный бустинг (если установлен)."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError(
            "LightGBM is not installed. Install with: pip install lightgbm"
        ) from e
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=cfg.training.random_state,
        n_jobs=-1,
        verbose=-1,
    )
    return BaselineClassifier("lightgbm", model)


def available_baselines() -> list[str]:
    """Возвращает список зарегистрированных бейзлайн-моделей."""
    return sorted(_BASELINE_REGISTRY.keys())


def build_baseline(name: str, cfg: Config) -> BaselineClassifier:
    """Создаёт бейзлайн-модель по имени из реестра."""
    if name not in _BASELINE_REGISTRY:
        available = ", ".join(available_baselines())
        raise ValueError(
            f"Unknown baseline model: {name!r}. Available: [{available}]."
        )
    return _BASELINE_REGISTRY[name](cfg)


# ============================================================
# Сравнение моделей с K-Fold CV
# ============================================================


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_names: list[str],
    cfg: Config,
    selection_metric: str = "pr_auc",
) -> dict[str, dict[str, float]]:
    """Сравнивает модели на валидационной выборке.

    Args:
        X_train, y_train: обучающая выборка.
        X_val, y_val: валидационная выборка.
        model_names: список имён моделей (из реестра).
        cfg: полная конфигурация.
        selection_metric: метрика выбора ("pr_auc", "roc_auc", "f1").

    Returns:
        Словарь {model_name: {metric: value, ...}}.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    results: dict[str, dict[str, float]] = {}
    cost_matrix = cfg.cost_matrix

    for name in model_names:
        logger.info("Training baseline: %s", name)
        try:
            model = build_baseline(name, cfg)
            # Cost-sensitive learning
            class_weight = cost_matrix.class_weight if name != "mlp" else None
            model.fit(X_train, y_train, class_weight=class_weight)

            proba = model.predict_proba(X_val)
            preds = (proba >= 0.5).astype(int)

            tp = int(((preds == 1) & (y_val == 1)).sum())
            fp = int(((preds == 1) & (y_val == 0)).sum())
            fn = int(((preds == 0) & (y_val == 1)).sum())
            tn = int(((preds == 0) & (y_val == 0)).sum())

            metrics = {
                "pr_auc": float(average_precision_score(y_val, proba)),
                "roc_auc": float(roc_auc_score(y_val, proba)),
                "f1": float(f1_score(y_val, preds, zero_division=0)),
                "precision": float(precision_score(y_val, preds, zero_division=0)),
                "recall": float(recall_score(y_val, preds, zero_division=0)),
                "cost": float(cost_matrix.total_cost(tp, fp, fn, tn)),
            }
            results[name] = metrics
            logger.info(
                "  %s: pr_auc=%.4f roc_auc=%.4f f1=%.4f cost=%.2f",
                name,
                metrics["pr_auc"],
                metrics["roc_auc"],
                metrics["f1"],
                metrics["cost"],
            )
        except ImportError as e:
            logger.warning("Skipping %s: %s", name, e)
            results[name] = {"error": str(e)}
        except Exception as e:
            logger.error("Failed to train %s: %s", name, e)
            results[name] = {"error": str(e)}

    return results


def select_best_model(
    results: dict[str, dict[str, float]],
    selection_metric: str = "pr_auc",
) -> str:
    """Выбирает лучшую модель по заданной метрике.

    Для метрик "pr_auc", "roc_auc", "f1", "recall", "precision" — больше лучше.
    Для "cost" — больше лучше (положительный = выгода).
    """
    best_name = ""
    best_score = float("-inf")
    for name, metrics in results.items():
        if "error" in metrics:
            continue
        score = metrics.get(selection_metric, float("-inf"))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


# ============================================================
# K-Fold Cross-Validation
# ============================================================


def cross_validate_baseline(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    cfg: Config,
    n_folds: int = 5,
) -> dict[str, list[float]]:
    """K-Fold CV для бейзлайн-модели.

    Возвращает список метрик по фолдам для оценки устойчивости.

    Args:
        X, y: полная выборка (train+val).
        model_name: имя модели из реестра.
        cfg: конфигурация.
        n_folds: число фолдов.

    Returns:
        Словарь {metric: [fold1, fold2, ...]}.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.training.random_state)
    fold_metrics: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "f1": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model = build_baseline(model_name, cfg)
        model.fit(
            X[train_idx],
            y[train_idx],
            class_weight=cfg.cost_matrix.class_weight,
        )

        proba = model.predict_proba(X[val_idx])
        preds = (proba >= 0.5).astype(int)

        fold_metrics["pr_auc"].append(
            float(average_precision_score(y[val_idx], proba))
        )
        fold_metrics["roc_auc"].append(
            float(roc_auc_score(y[val_idx], proba))
        )
        fold_metrics["f1"].append(
            float(f1_score(y[val_idx], preds, zero_division=0))
        )
        logger.info(
            "CV Fold %d/%d (%s): pr_auc=%.4f roc_auc=%.4f f1=%.4f",
            fold,
            n_folds,
            model_name,
            fold_metrics["pr_auc"][-1],
            fold_metrics["roc_auc"][-1],
            fold_metrics["f1"][-1],
        )

    # Средние значения
    for metric, values in fold_metrics.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        logger.info(
            "CV %s: %s = %.4f ± %.4f", model_name, metric, mean_val, std_val
        )

    return fold_metrics