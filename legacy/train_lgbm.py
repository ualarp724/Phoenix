#!/usr/bin/env python3
"""
Train and save LightGBM model using Phoenix features and lookback.
"""

from pathlib import Path
import joblib

import phoenix_config as config
from phoenix_processor import PhoenixDataProcessor
from phoenix_brain import preparar_secuencias_flat_con_scaler
from core.mtf import add_mtf_features_multi

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"LightGBM no disponible: {exc}")


def main():
    processor = PhoenixDataProcessor(config.DATA_RAW)
    df_all = add_mtf_features_multi(
        processor.clean_and_prepare(apply_train_filter=False),
        config.MTF_CONFIGS,
    )

    train_df = df_all[(df_all.index >= config.TRAIN_START_DATE) & (df_all.index <= config.TRAIN_END_DATE)]
    if len(train_df) < config.LOOKBACK_WINDOW + 100:
        raise SystemExit("Datos de entrenamiento insuficientes")

    features = config.FEATURES
    scaler = joblib.load(config.SCALER_SAVE_PATH) if Path(config.SCALER_SAVE_PATH).exists() else None
    if scaler is None or getattr(scaler, "n_features_in_", None) != len(features):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_df[features].values)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)

    X_train, y_train = preparar_secuencias_flat_con_scaler(train_df, scaler)
    model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    model.fit(X_train, y_train)

    joblib.dump(model, config.LGBM_MODEL_PATH)
    print(f"✅ LightGBM guardado en {config.LGBM_MODEL_PATH}")


if __name__ == "__main__":
    main()
