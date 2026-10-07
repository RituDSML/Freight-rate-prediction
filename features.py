"""Data loading, cleaning and feature engineering for the freight-rate task."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "posted_rate"
EQUIP = {"Dry Van": 0, "Flatbed": 1, "Reefer": 2}
T0 = pd.Timestamp("2025-01-01")


def haversine_miles(lat1, lon1, lat2, lon2):
    r, p = 3958.8, np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_frames(train_path: str, valid_path: str) -> pd.DataFrame:
    """Read both files into one frame (flag `is_valid`). Features only are shared across files, so
    date-level aggregates of *unlabeled* features (market_index) can be computed for every date."""
    tr = pd.read_csv(train_path, parse_dates=["date"]).assign(is_valid=False)
    va = pd.read_csv(valid_path, parse_dates=["date"]).assign(is_valid=True)
    return pd.concat([tr, va], ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # --- data-quality fixes -------------------------------------------------
    d["weight_was_negative"] = (d["weight"] < 0).astype(int)   # sign-flipped weights
    d["weight"] = d["weight"].abs()
    d["weight_missing"] = d["weight"].isna().astype(int)         # kept as NaN; LightGBM handles it
    d["mi_missing"] = d["market_index"].isna().astype(int)
    # market_index = date-level signal + per-load noise -> date mean is a clean estimate
    d["mi_date"] = d.groupby("date")["market_index"].transform("mean")
    d["mi"] = d["market_index"].fillna(d["mi_date"])
    # --- features -------------------------------------------------------------
    d["equip"] = d["equipment"].map(EQUIP)
    d["ldist"] = np.log(d["distance"])
    d["gc_miles"] = haversine_miles(d.pickup_lat, d.pickup_lon, d.delivery_lat, d.delivery_lon)
    d["circuity"] = d["distance"] / d["gc_miles"].clip(lower=1)
    d["dlat"] = d.delivery_lat - d.pickup_lat
    d["dlon"] = d.delivery_lon - d.pickup_lon
    d["dow"] = d["date"].dt.dayofweek
    d["t"] = (d["date"] - T0).dt.days
    if TARGET in d:
        d["lrpm"] = np.log(d[TARGET] / d["distance"])      # log rate-per-mile target
    return d


BASE_FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "ldist", "gc_miles", "circuity", "equip", "weight", "weight_missing",
    "mi", "mi_date", "quote_signal", "dow",
]
