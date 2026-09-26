import phoenix_config as config


def detect_regime(row) -> str:
    trend = float(row.get("Trend_Score", 0.0))
    bb_width = float(row.get("BB_Width", 0.0))
    if abs(trend) >= config.REGIME_TREND_THRESHOLD and bb_width >= config.REGIME_BB_WIDTH_THRESHOLD:
        return "trend"
    return "range"


def entry_allowed(pred: int, row) -> bool:
    rsi = float(row.get("RSI", 50.0))
    bb_pos = float(row.get("BB_Pos", 0.5))
    trend = float(row.get("Trend_Score", 0.0))
    regime = detect_regime(row)

    if pred == 1:  # BUY
        if regime == "trend":
            return trend > 0 and rsi < config.RSI_PULLBACK_MAX and bb_pos < config.BB_PULLBACK_MAX
        return rsi < config.RSI_RANGE_LOW and bb_pos < config.BB_RANGE_LOW
    if pred == 2:  # SELL
        if regime == "trend":
            return trend < 0 and rsi > (100 - config.RSI_PULLBACK_MAX) and bb_pos > (1 - config.BB_PULLBACK_MAX)
        return rsi > config.RSI_RANGE_HIGH and bb_pos > config.BB_RANGE_HIGH
    return False
