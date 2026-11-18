"""Train regression models on the UCI Bike Sharing dataset (hourly demand).

Compares Linear Regression, Random Forest and Gradient Boosting on a
chronological split, selects the best by validation R2, and saves the
artifacts to models/ and a metrics report to reports/.
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
TARGET = "cnt"
RANDOM_STATE = 42


def load_splits() -> dict[str, pd.DataFrame]:
    """Load the processed train/valid/test splits."""
    parts = {}
    for name in ["train", "valid", "test"]:
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run preprocess.py first.")
        parts[name] = pd.read_csv(path)
    return parts


def get_models() -> dict:
    """Candidate regressors with light hyperparameters."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.08, max_depth=5, random_state=RANDOM_STATE
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2, MAE and RMSE for one set of predictions."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    parts = load_splits()
    NON_FEATURES = [TARGET, "instant", "dteday", "casual", "registered"]
    feature_cols = [c for c in parts["train"].columns if c not in NON_FEATURES]
    print(f"Training on {len(feature_cols)} features, {len(parts['train'])} rows")

    results: dict[str, dict] = {}
    best_name, best_model, best_r2 = None, None, -np.inf
    for name, model in get_models().items():
        model.fit(parts["train"][feature_cols], parts["train"][TARGET])
        val_pred = model.predict(parts["valid"][feature_cols])
        m = regression_metrics(parts["valid"][TARGET].values, val_pred)
        results[name] = m
        print(f"{name}: valid R2={m['r2']:.4f} MAE={m['mae']:.1f} RMSE={m['rmse']:.1f}")
        if m["r2"] > best_r2:
            best_name, best_model, best_r2 = name, model, m["r2"]

    # Final report on the untouched test set.
    test_pred = best_model.predict(parts["test"][feature_cols])
    test_m = regression_metrics(parts["test"][TARGET].values, test_pred)
    print(f"\nBest model: {best_name} (valid R2={best_r2:.4f})")
    print(f"Test metrics: R2={test_m['r2']:.4f} MAE={test_m['mae']:.1f} RMSE={test_m['rmse']:.1f}")

    # Feature importance (tree models only).
    if hasattr(best_model, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": feature_cols, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(MODELS_DIR, "feature_importance.csv"), index=False)
        print("\nTop 10 features:")
        print(imp.head(10).to_string(index=False))

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    comparison = pd.DataFrame(results).T
    comparison.loc["best_on_test"] = test_m
    comparison.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    print(f"\nSaved artifacts to {MODELS_DIR}/ and {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
