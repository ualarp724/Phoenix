from typing import List

import pandas as pd
import phoenix_config as config


def validate_features(df: pd.DataFrame, features: List[str] | None = None) -> None:
    features = features or config.FEATURES
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")


def validate_no_nulls(df: pd.DataFrame, features: List[str] | None = None) -> None:
    features = features or config.FEATURES
    nulls = df[features].isnull().sum().sum()
    if nulls:
        raise ValueError(f"Null values found in features: {nulls}")


def validate_target(df: pd.DataFrame) -> None:
    if "Target" not in df.columns:
        raise ValueError("Target column missing")
