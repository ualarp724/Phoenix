#!/usr/bin/env python3
"""Train EURUSD Elite model/scaler for multi-asset master."""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")


TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"

ELITE_PARAMS_PATH = "eurusd_elite_v1.json"


def _load_and_resample_m15(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep="\t")
        if len(df.columns) < 2:
            df = pd.read_csv(csv_path, sep=",")
    except Exception as exc:
        raise SystemExit(f"Error leyendo CSV: {exc}")

    col_map = {}
    for col in df.columns:
        c = col.upper().replace("<", "").replace(">", "")
        if "DATE" in c:
            col_map[col] = "Date"
        elif "TIME" in c:
            col_map[col] = "Time"
        elif "OPEN" in c:
            col_map[col] = "Open"
        elif "HIGH" in c:
            col_map[col] = "High"
        elif "LOW" in c:
            col_map[col] = "Low"
        elif "CLOSE" in c:
            col_map[col] = "Close"
        elif "VOL" in c:
            col_map[col] = "Volume"

    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated()]
    if "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    else:
        df["Datetime"] = pd.to_datetime(df["Date"])
    df = df.set_index("Datetime").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    resampled = df.resample("15min").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()
    resampled = resampled.reset_index()
    resampled["Date"] = resampled["Datetime"].dt.date.astype(str)
    resampled["Time"] = resampled["Datetime"].dt.time.astype(str)
    return resampled[["Date", "Time", "Open", "High", "Low", "Close", "Volume"]]


def _prepare_dataset(csv_path: str) -> pd.DataFrame:
    df_m15 = _load_and_resample_m15(csv_path)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        df_m15.to_csv(tmp.name, index=False)
        temp_path = tmp.name

    processor = PhoenixDataProcessor(temp_path)
    df = processor.clean_and_prepare(apply_train_filter=False, relax_factor=0.0)
    df = add_mtf_features_multi(df, config.MTF_CONFIGS)

    for col in config.FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    df = df.dropna()
    return df


def main() -> None:
    config.apply_asset("EURUSD")
    config.TIMEFRAME = "M15"

    elite_params = {}
    if Path(ELITE_PARAMS_PATH).exists():
        elite_params = json.loads(Path(ELITE_PARAMS_PATH).read_text())

    df_all = _prepare_dataset("vantage_eurusd.csv")
    df_all = df_all.sort_index()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    if train_df.empty:
        raise SystemExit("Train vacío para EURUSD.")

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)

    params = dict(config.LGBM_PARAMS)
    params.update(
        {
            "max_depth": 7,
            "n_estimators": 600,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.25,
            "reg_lambda": 2.0,
            "num_leaves": 55,
            "min_data_in_leaf": 150,
            "min_gain_to_split": 0.05,
            "max_bin": 127,
            "verbose": -1,
        }
    )

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    joblib.dump(model, config.LGBM_MODEL_PATH)
    joblib.dump(scaler, config.SCALER_SAVE_PATH)

    print("✅ EURUSD modelo y scaler guardados")
    print(f"Modelo: {config.LGBM_MODEL_PATH}")
    print(f"Scaler: {config.SCALER_SAVE_PATH}")
    if elite_params:
        print(f"EURUSD_ELITE_V1 cargado: {elite_params}")


if __name__ == "__main__":
    main()
