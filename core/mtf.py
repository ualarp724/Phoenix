import pandas as pd

import phoenix_config as config


def _add_single_mtf(df: pd.DataFrame, prefix: str, timeframe: str) -> pd.DataFrame:
    if f"{prefix}Trend_Score" in df.columns:
        return df

    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in base_cols if c in df.columns]
    if "Close" not in available or "High" not in available or "Low" not in available:
        return df

    ohlc = df[available].copy()
    if "Open" not in ohlc.columns:
        ohlc["Open"] = ohlc["Close"]
    if "Volume" not in ohlc.columns:
        ohlc["Volume"] = 0.0

    freq = timeframe.lower() if isinstance(timeframe, str) else timeframe
    resampled = ohlc.resample(freq).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()

    resampled["EMA_50"] = resampled["Close"].ewm(span=50).mean()
    resampled["Trend_Score"] = (resampled["Close"] - resampled["EMA_50"]) / resampled["Close"] * 1000

    delta = resampled["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    resampled["RSI"] = 100 - (100 / (1 + rs))

    sma = resampled["Close"].rolling(20).mean()
    std = resampled["Close"].rolling(20).std()
    resampled["BB_Pos"] = (resampled["Close"] - (sma - std * 2)) / (std * 4)

    resampled = resampled[["Trend_Score", "RSI", "BB_Pos"]].rename(
        columns={
            "Trend_Score": f"{prefix}Trend_Score",
            "RSI": f"{prefix}RSI",
            "BB_Pos": f"{prefix}BB_Pos",
        }
    )

    aligned = resampled.reindex(df.index, method="ffill")
    return df.join(aligned)


def add_mtf_features_multi(df: pd.DataFrame, mtf_configs=None) -> pd.DataFrame:
    if not config.USE_MTF_CONFIRM:
        return df

    mtf_configs = mtf_configs or [("M15_", "15min"), ("H1_", "1h")]
    for prefix, timeframe in mtf_configs:
        df = _add_single_mtf(df, prefix, timeframe)
    return df


def add_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_MTF_CONFIRM:
        return df

    return _add_single_mtf(df, config.MTF_PREFIX, config.MTF_TIMEFRAME)


def mtf_confirm(pred: int, row, prefix: str | None = None) -> bool:
    if not config.USE_MTF_CONFIRM:
        return True

    prefix = prefix or config.MTF_PREFIX
    trend = float(row.get(f"{prefix}Trend_Score", 0.0))
    rsi = float(row.get(f"{prefix}RSI", 50.0))
    bb_pos = float(row.get(f"{prefix}BB_Pos", 0.5))

    if pd.isna(trend) or pd.isna(rsi) or pd.isna(bb_pos):
        return True

    if pred == 1:
        trend_ok = True if not config.MTF_ENFORCE_TREND else trend >= config.MTF_TREND_MIN
        return (
            trend_ok
            and rsi <= config.MTF_RSI_MAX_BUY
            and bb_pos <= config.MTF_BB_POS_MAX_BUY
        )
    if pred == 2:
        trend_ok = True if not config.MTF_ENFORCE_TREND else trend <= -config.MTF_TREND_MIN
        return (
            trend_ok
            and rsi >= config.MTF_RSI_MIN_SELL
            and bb_pos >= config.MTF_BB_POS_MIN_SELL
        )
    return False
