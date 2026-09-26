import numpy as np
from sklearn.preprocessing import StandardScaler

import phoenix_config as config


def preparar_secuencias_flat(df, test_size=0.3):
    """
    Prepara datos para XGBoost: escala y aplana lookback*features.
    Retorna X_train, y_train, X_test, y_test (numpy) y scaler.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")

    features = config.FEATURES
    if "Target" not in df.columns:
        raise ValueError("Falta Target")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")

    data = df[features].values
    target = df["Target"].values

    split_idx = int(len(data) * (1 - test_size))
    data_train = data[:split_idx]
    data_test = data[split_idx:]
    target_train = target[:split_idx]
    target_test = target[split_idx:]

    scaler = StandardScaler()
    scaler.fit(data_train)
    data_train_scaled = scaler.transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    lookback = config.LOOKBACK_WINDOW

    def crear_secuencias_flat(data_scaled, target_data):
        X_seq, y_seq = [], []
        for i in range(lookback, len(data_scaled)):
            X_seq.append(data_scaled[i - lookback:i].reshape(-1))
            y_seq.append(target_data[i])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train = crear_secuencias_flat(data_train_scaled, target_train)
    X_test, y_test = crear_secuencias_flat(data_test_scaled, target_test)

    return X_train, y_train, X_test, y_test, scaler


def preparar_secuencias_flat_con_scaler(df, scaler):
    """Construye X/y para inferencia XGBoost usando scaler existente."""
    features = config.FEATURES
    if "Target" not in df.columns:
        raise ValueError("Falta Target")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")

    data = df[features].values
    target = df["Target"].values
    data_scaled = scaler.transform(data)
    lookback = config.LOOKBACK_WINDOW

    X_seq, y_seq = [], []
    for i in range(lookback, len(data_scaled)):
        X_seq.append(data_scaled[i - lookback:i].reshape(-1))
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)