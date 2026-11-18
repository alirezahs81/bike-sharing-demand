"""Unit tests for preprocessing, training pipeline and API."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import TARGET, engineer_features, get_feature_names, load_raw  # noqa: E402
from src.train import get_models, regression_metrics  # noqa: E402

DATA_AVAILABLE = os.path.exists(os.path.join("data", "raw", "hour.csv"))


@pytest.fixture(scope="module")
def raw_df():
    return engineer_features(load_raw())


@pytest.mark.skipif(not DATA_AVAILABLE, reason="raw dataset not downloaded")
def test_dataset_loads(raw_df):
    assert len(raw_df) == 17379
    assert TARGET in raw_df.columns


@pytest.mark.skipif(not DATA_AVAILABLE, reason="raw dataset not downloaded")
def test_no_leakage_columns_in_features():
    features = get_feature_names(engineer_features(load_raw()))
    assert "casual" not in features
    assert "registered" not in features
    assert "cnt" not in features


def test_engineered_features_range(raw_df):
    # Cyclical features must be within [-1, 1].
    for col in ["hr_sin", "hr_cos", "dow_sin", "mnth_cos"]:
        assert raw_df[col].between(-1.0, 1.0).all()
    # Rush hour only for known peak hours.
    assert raw_df.loc[raw_df["hr"] == 8, "rush_hour"].eq(1).all()
    assert raw_df.loc[raw_df["hr"] == 3, "rush_hour"].eq(0).all()


def test_get_models_returns_fitted_able_models():
    models = get_models()
    assert set(models) >= {"linear_regression", "random_forest", "gradient_boosting"}


def test_regression_metrics_perfect():
    y = [1.0, 2.0, 3.0]
    m = regression_metrics(y, y)
    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)


def test_regression_metrics_imperfect():
    m = regression_metrics([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert m["mae"] == pytest.approx(1.0)
    assert m["r2"] < 1.0  # worse than perfect; may even be negative vs the mean


def test_api_feature_pipeline():
    from src.api import FEATURES, PredictionInput, build_features

    payload = PredictionInput(
        season=1, yr=1, mnth=1, hr=8, holiday=0, weekday=3, workingday=1,
        weathersit=1, temp=0.24, atemp=0.2879, hum=0.81, windspeed=0.0,
    )
    X = build_features(payload)
    assert list(X.columns) == FEATURES
    assert X["rush_hour"].iloc[0] == 1
    assert -1 <= X["hr_sin"].iloc[0] <= 1
