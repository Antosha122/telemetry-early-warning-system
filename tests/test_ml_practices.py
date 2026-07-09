"""Тесты для новых модулей: validation, features, baselines, threshold, pipeline."""

from __future__ import annotations

import numpy as np
import pytest
import polars as pl

from projects.gazprom_emergency.config import (
    ValidationConfig,
    FeatureEngineeringConfig,
    CostMatrixConfig,
    ThresholdConfig,
)
from projects.gazprom_emergency.validation import (
    validate_opers,
    validate_stpa,
    validate_inputs,
    ValidationReport,
    validate_numpy_features,
)


# ============================================================
# Validation tests (#11)
# ============================================================

class TestValidationConfig:
    def test_defaults(self):
        cfg = ValidationConfig()
        assert cfg.enabled is True
        assert cfg.nan_strategy == "fill_median"
        assert cfg.duplicate_keys_strategy == "fail"

    def test_custom(self):
        cfg = ValidationConfig(nan_strategy="fill_zero", enabled=False)
        assert cfg.nan_strategy == "fill_zero"
        assert cfg.enabled is False


class TestValidateOpers:
    def test_no_duplicates_passes(self):
        opers = pl.DataFrame({
            "batch_time": ["2023-01-01", "2023-01-02"],
            "is_emergency": [0, 1],
        })
        cfg = ValidationConfig()
        report = ValidationReport()
        result = validate_opers(opers, cfg, report)
        assert not report.has_errors
        assert len(result) == 2

    def test_duplicates_fail_strategy(self):
        opers = pl.DataFrame({
            "batch_time": ["2023-01-01", "2023-01-01"],
            "is_emergency": [0, 1],
        })
        cfg = ValidationConfig(duplicate_keys_strategy="fail")
        report = ValidationReport()
        validate_opers(opers, cfg, report)
        assert report.has_errors

    def test_duplicates_drop_strategy(self):
        opers = pl.DataFrame({
            "batch_time": ["2023-01-01", "2023-01-01"],
            "is_emergency": [0, 1],
        })
        cfg = ValidationConfig(duplicate_keys_strategy="drop")
        report = ValidationReport()
        result = validate_opers(opers, cfg, report)
        assert len(result) == 1

    def test_invalid_target_values(self):
        opers = pl.DataFrame({
            "batch_time": ["2023-01-01"],
            "is_emergency": [5],
        })
        cfg = ValidationConfig(strict_binary_target=True)
        report = ValidationReport()
        validate_opers(opers, cfg, report)
        assert report.has_errors


class TestValidateNumpyFeatures:
    def test_nan_fill_median(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        cfg = ValidationConfig(nan_strategy="fill_median")
        X_clean, report = validate_numpy_features(X, cfg)
        assert not np.isnan(X_clean).any()

    def test_nan_fail(self):
        X = np.array([[1.0, np.nan]])
        cfg = ValidationConfig(nan_strategy="fail")
        _, report = validate_numpy_features(X, cfg)
        assert report.has_errors

    def test_inf_replace(self):
        X = np.array([[1.0, np.inf]])
        cfg = ValidationConfig(
            inf_strategy="replace_with_nan", nan_strategy="fill_zero"
        )
        X_clean, report = validate_numpy_features(X, cfg)
        assert np.isfinite(X_clean).all()


class TestCostMatrix:
    def test_total_cost(self):
        cm = CostMatrixConfig(cost_fn=100, cost_fp=10, benefit_tp=90, benefit_tn=0)
        cost = cm.total_cost(tp=10, fp=5, fn=2, tn=100)
        # 10*90 + 100*0 - 2*100 - 5*10 = 900 - 200 - 50 = 650
        assert cost == 650

    def test_class_weight(self):
        cm = CostMatrixConfig(cost_fn=100, cost_fp=10)
        cw = cm.class_weight
        assert cw[0] == 1.0
        assert cw[1] == 10.0


# ============================================================
# Feature engineering tests (#12)
# ============================================================

class TestFeatureEngineering:
    def test_time_features_shape(self):
        from projects.gazprom_emergency.features import extract_time_features
        times = np.array(["2023-01-01", "2023-06-15"], dtype="datetime64[ns]")
        feats = extract_time_features(times)
        assert feats.shape == (2, 10)

    def test_lag_features(self):
        from projects.gazprom_emergency.features import add_lag_features
        X = np.arange(10).reshape(5, 2).astype(np.float32)
        lag_X, names = add_lag_features(X, [1], n_features=2)
        assert lag_X.shape == (5, 2)
        assert len(names) == 2
        # First row should be NaN (no history)
        assert np.isnan(lag_X[0]).all()
        # Second row should have first row values
        np.testing.assert_array_equal(lag_X[1], X[0])

    def test_rolling_features(self):
        from projects.gazprom_emergency.features import add_rolling_features
        X = np.arange(12).reshape(6, 2).astype(np.float32)
        roll_X, names = add_rolling_features(X, [3], n_features=2)
        assert roll_X.shape == (6, 4)  # 2 features * 2 stats (mean, std)

    def test_interpolate_nan_rows(self):
        from projects.gazprom_emergency.features import interpolate_nan_rows
        X = np.array([
            [1.0, 2.0],
            [np.nan, np.nan],
            [5.0, 6.0],
        ])
        result = interpolate_nan_rows(X)
        # Middle row should be average of neighbors
        np.testing.assert_array_almost_equal(result[1], [3.0, 4.0])

    def test_feature_engineer_fit_transform(self):
        from projects.gazprom_emergency.features import FeatureEngineer
        cfg = FeatureEngineeringConfig(
            add_time_features=False,
            lag_sizes=[1],
            rolling_windows=[3],
            rolling_n_features=2,
        )
        X = np.random.rand(10, 5).astype(np.float32)
        fe = FeatureEngineer(cfg=cfg)
        transformed = fe.fit(X).transform(X)
        assert transformed.shape[0] == 10
        # Original 5 + lag 2 + rolling 4 = 11
        assert transformed.shape[1] == 11


# ============================================================
# Threshold optimization tests (#14)
# ============================================================

class TestThresholdOptimization:
    def test_find_threshold_f1(self):
        from projects.gazprom_emergency.threshold import find_optimal_threshold
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.45, 0.55, 0.8, 0.9])
        cfg = ThresholdConfig(metric="f1")
        threshold = find_optimal_threshold(y_true, y_proba, cfg)
        assert 0.0 < threshold < 1.0

    def test_find_threshold_youden(self):
        from projects.gazprom_emergency.threshold import find_optimal_threshold
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        cfg = ThresholdConfig(metric="youden")
        threshold = find_optimal_threshold(y_true, y_proba, cfg)
        assert 0.0 < threshold < 1.0

    def test_find_threshold_cost(self):
        from projects.gazprom_emergency.threshold import find_optimal_threshold
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])
        cfg = ThresholdConfig(metric="cost")
        cm = CostMatrixConfig(cost_fn=100, cost_fp=10)
        threshold = find_optimal_threshold(y_true, y_proba, cfg, cm)
        assert 0.0 < threshold < 1.0

    def test_optimize_threshold_pipeline(self):
        from projects.gazprom_emergency.threshold import optimize_threshold_pipeline
        np.random.seed(42)
        y_true = np.array([0]*90 + [1]*10)
        y_proba = np.concatenate([
            np.random.uniform(0, 0.4, 90),
            np.random.uniform(0.5, 1.0, 10),
        ])
        result = optimize_threshold_pipeline(
            y_true, y_proba, cfg=ThresholdConfig(optimize=True, metric="f1")
        )
        assert "optimal_threshold" in result
        assert "metrics" in result
        assert "f1" in result["metrics"]


# ============================================================
# Baselines tests (#13)
# ============================================================

class TestBaselines:
    def test_available_baselines(self):
        from projects.gazprom_emergency.baselines import available_baselines
        baselines = available_baselines()
        assert "logreg" in baselines
        assert "random_forest" in baselines

    def test_build_logreg(self):
        from projects.gazprom_emergency.baselines import build_baseline
        from projects.gazprom_emergency.config import Config
        cfg = Config()
        model = build_baseline("logreg", cfg)
        assert model.name == "logreg"

    def test_unknown_baseline_raises(self):
        from projects.gazprom_emergency.baselines import build_baseline
        from projects.gazprom_emergency.config import Config
        with pytest.raises(ValueError, match="Unknown baseline"):
            build_baseline("nonexistent", Config())

    def test_baseline_fit_predict(self):
        from projects.gazprom_emergency.baselines import build_baseline
        from projects.gazprom_emergency.config import Config
        np.random.seed(42)
        X = np.random.rand(100, 5)
        y = (X[:, 0] > 0.5).astype(int)
        cfg = Config()
        model = build_baseline("logreg", cfg)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100,)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_select_best_model(self):
        from projects.gazprom_emergency.baselines import select_best_model
        results = {
            "logreg": {"pr_auc": 0.7, "roc_auc": 0.8},
            "rf": {"pr_auc": 0.85, "roc_auc": 0.9},
        }
        best = select_best_model(results, "pr_auc")
        assert best == "rf"


# ============================================================
# Pipeline tests (#16)
# ============================================================

class TestPipeline:
    def test_build_preprocessing_pipeline(self):
        from projects.gazprom_emergency.pipeline import build_preprocessing_pipeline
        from projects.gazprom_emergency.config import Config
        cfg = Config()
        cfg.training.use_smote = False
        cfg.feature_engineering.add_time_features = False
        cfg.feature_engineering.lag_sizes = []
        cfg.feature_engineering.rolling_windows = []
        pipeline = build_preprocessing_pipeline(cfg)
        assert pipeline is not None

    def test_torch_mlp_wrapper_predict(self):
        from projects.gazprom_emergency.pipeline import TorchMLPWrapper
        from projects.gazprom_emergency.config import Config
        cfg = Config()
        cfg.training.epochs = 1
        cfg.model.hidden_dims = [8]
        np.random.seed(42)
        X = np.random.rand(20, 4).astype(np.float32)
        y = np.array([0]*10 + [1]*10)
        model = TorchMLPWrapper(cfg, device="cpu")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (20,)
        assert (proba >= 0).all() and (proba <= 1).all()