import pandas as pd
import numpy as np


def _load_m15(csv_path: str) -> pd.DataFrame:
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
    if "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    else:
        df["Datetime"] = pd.to_datetime(df["Date"])

    df = df.set_index("Datetime").sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    m15 = (
        df.resample("15min")
        .agg(
            Open=pd.NamedAgg(column="Open", aggfunc="first"),
            High=pd.NamedAgg(column="High", aggfunc="max"),
            Low=pd.NamedAgg(column="Low", aggfunc="min"),
            Close=pd.NamedAgg(column="Close", aggfunc="last"),
            Volume=pd.NamedAgg(column="Volume", aggfunc="sum"),
        )
        .dropna()
    )
    return m15


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(n).mean()
    avg_loss = loss.rolling(n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def main() -> None:
    m15 = _load_m15("vantage_gold.csv")
    m15 = m15.loc["2022-01-01":"2024-12-31"].copy()

    # Candle anatomy
    m15["Range"] = (m15["High"] - m15["Low"]).clip(lower=1e-9)
    m15["Body"] = (m15["Close"] - m15["Open"]).abs()
    m15["UpperWick"] = m15["High"] - m15[["Open", "Close"]].max(axis=1)
    m15["LowerWick"] = m15[["Open", "Close"]].min(axis=1) - m15["Low"]
    m15["BodyRatio"] = (m15["Body"] / m15["Range"]).clip(0, 1)
    m15["WickRatio"] = ((m15["UpperWick"] + m15["LowerWick"]) / m15["Range"]).clip(0, 1)

    # Indicators
    m15["EMA200"] = m15["Close"].ewm(span=200, adjust=False).mean()
    m15["ATR"] = _atr(m15)
    m15["ATR_PCT"] = (m15["ATR"] / m15["Close"]) * 100
    m15["RSI"] = _rsi(m15["Close"]).clip(0, 100)

    # Future returns for pattern validation
    horizon = 8  # 2h
    m15["FWD_RET"] = (m15["Close"].shift(-horizon) - m15["Close"]) / m15["Close"] * 100

    # Session buckets (UTC)
    hour = m15.index.hour
    session = pd.cut(
        hour,
        bins=[-1, 6, 12, 16, 21, 23],
        labels=["Asia", "London", "NY_Open", "NY_Late", "After"],
    )
    m15["Session"] = session.astype(str)

    # Summary stats
    summary = {
        "rows": int(len(m15)),
        "body_ratio_p50": float(m15["BodyRatio"].median()),
        "body_ratio_p75": float(m15["BodyRatio"].quantile(0.75)),
        "wick_ratio_p50": float(m15["WickRatio"].median()),
        "wick_ratio_p75": float(m15["WickRatio"].quantile(0.75)),
        "atr_pct_p50": float(m15["ATR_PCT"].median()),
        "atr_pct_p75": float(m15["ATR_PCT"].quantile(0.75)),
        "atr_pct_p90": float(m15["ATR_PCT"].quantile(0.90)),
        "trend_up_pct": float((m15["Close"] > m15["EMA200"]).mean() * 100),
        "trend_down_pct": float((m15["Close"] < m15["EMA200"]).mean() * 100),
    }

    # Session stats
    sess_stats = (
        m15.groupby("Session")
        .agg(
            bars=("Close", "size"),
            atr_pct_p50=("ATR_PCT", "median"),
            atr_pct_p75=("ATR_PCT", lambda s: s.quantile(0.75)),
            body_p50=("BodyRatio", "median"),
            wick_p50=("WickRatio", "median"),
            fwd_ret_p50=("FWD_RET", "median"),
        )
        .sort_values("atr_pct_p50", ascending=False)
    )

    # Pattern probes
    strong_body = m15["BodyRatio"] >= m15["BodyRatio"].quantile(0.75)
    long_wick = m15["WickRatio"] >= m15["WickRatio"].quantile(0.75)
    rsi_oversold = m15["RSI"] <= 30
    rsi_overbought = m15["RSI"] >= 70

    pattern_table = {
        "strong_body_fwd_ret_p50": float(m15.loc[strong_body, "FWD_RET"].median()),
        "long_wick_fwd_ret_p50": float(m15.loc[long_wick, "FWD_RET"].median()),
        "oversold_fwd_ret_p50": float(m15.loc[rsi_oversold, "FWD_RET"].median()),
        "overbought_fwd_ret_p50": float(m15.loc[rsi_overbought, "FWD_RET"].median()),
    }

    print("SUMMARY", summary)
    print("SESSION_STATS")
    print(sess_stats.to_string())
    print("PATTERN_STATS", pattern_table)


if __name__ == "__main__":
    main()
