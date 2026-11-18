"""Data preprocessing for the UCI Bike Sharing dataset (hourly records).

Loads the raw dataset, engineers time/season features, and produces a
leakage-free train/valid/test split ordered by time (no shuffling), so the
model is evaluated on future demand.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW_PATH = os.path.join("data", "raw", "hour.csv")
PROCESSED_DIR = os.path.join("data", "processed")

TARGET = "cnt"
# `casual` + `registered` sum to `cnt`, so keeping them would leak the target.
LEAKAGE_COLS = ["casual", "registered"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw hourly CSV and raise a clear error if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, parse_dates=["dteday"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create cyclical and interaction features from time columns."""
    df = df.copy()
    # Cyclical encoding for hour-of-day so 23:00 and 00:00 are close.
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    # Rush-hour flag and temp x humidity interaction.
    df["rush_hour"] = df["hr"].isin([7, 8, 17, 18, 19]).astype(int)
    df["temp_hum"] = df["temp"] * df["hum"]
    return df


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (target, leakage and id columns removed)."""
    drop = [TARGET, "instant", "dteday", *LEAKAGE_COLS]
    return [c for c in df.columns if c not in drop]


def preprocess(path: str = RAW_PATH) -> dict:
    """Full preprocessing pipeline: features, split, save to data/processed."""
    df = engineer_features(load_raw(path))
    features = get_feature_names(df)

    # Chronological split: first 2 years train, next 2 months valid, last 2 test.
    df = df.sort_values("dteday").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.80)
    valid_end = int(n * 0.90)

    splits = {
        "train": df.iloc[:train_end],
        "valid": df.iloc[train_end:valid_end],
        "test": df.iloc[valid_end:],
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)

    print(f"Features ({len(features)}): {features}")
    for name, part in splits.items():
        print(f"{name}: {len(part)} rows | target mean {part[TARGET].mean():.1f}")
    print(f"Saved splits to {PROCESSED_DIR}/")
    return splits


if __name__ == "__main__":
    preprocess()
