import pandas as pd
import numpy as np


def main() -> None:
    csv_path = "vantage_gold.csv"
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
    m15 = m15.loc["2022-01-01":"2024-12-31"].copy()

    m15["Range"] = (m15["High"] - m15["Low"]).clip(lower=1e-9)
    m15["Body"] = (m15["Close"] - m15["Open"]).abs()
    m15["UpperWick"] = m15["High"] - m15[["Open", "Close"]].max(axis=1)
    m15["LowerWick"] = m15[["Open", "Close"]].min(axis=1) - m15["Low"]
    m15["BodyRatio"] = (m15["Body"] / m15["Range"]).clip(0, 1)
    m15["WickRatio"] = ((m15["UpperWick"] + m15["LowerWick"]) / m15["Range"]).clip(0, 1)

    ema200 = m15["Close"].ewm(span=200, adjust=False).mean()
    trend_up = (m15["Close"] > ema200).mean()
    trend_down = (m15["Close"] < ema200).mean()

    tr = pd.concat(
        [
            m15["High"] - m15["Low"],
            (m15["High"] - m15["Close"].shift()).abs(),
            (m15["Low"] - m15["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = (atr / m15["Close"]).dropna()

    summary = {
        "rows": len(m15),
        "body_ratio_p50": float(m15["BodyRatio"].median()),
        "body_ratio_p75": float(m15["BodyRatio"].quantile(0.75)),
        "wick_ratio_p50": float(m15["WickRatio"].median()),
        "wick_ratio_p75": float(m15["WickRatio"].quantile(0.75)),
        "trend_up_pct": float(trend_up * 100),
        "trend_down_pct": float(trend_down * 100),
        "atr_pct_p50": float(atr_pct.median() * 100),
        "atr_pct_p75": float(atr_pct.quantile(0.75) * 100),
        "atr_pct_p90": float(atr_pct.quantile(0.90) * 100),
    }

    print(summary)


if __name__ == "__main__":
    main()
