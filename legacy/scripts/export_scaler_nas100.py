import sys
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from optuna_nas100_vantage_m15 import _prepare_dataset, TRAIN_START, TRAIN_END
from sklearn.preprocessing import StandardScaler
import phoenix_config as config

def main():
    config.apply_asset("NAS100")
    config.TIMEFRAME = "M15"
    df_all = _prepare_dataset("vantage_nas100.csv").sort_index()
    df_all = df_all.loc[TRAIN_START:TRAIN_END].copy()
    train_df = df_all.copy()
    scaler = StandardScaler()
    scaler.fit(train_df[config.FEATURES].values)
    joblib.dump(scaler, "phoenix_nas100_scaler.pkl")
    print("✅ Scaler NAS100 exportado: phoenix_nas100_scaler.pkl")

if __name__ == "__main__":
    main()
