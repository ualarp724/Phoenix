import phoenix_config as config
from phoenix_backtester_pro import ProfessionalBacktester


def test_backtester_runs_minimal():
    bt = ProfessionalBacktester(config.MODEL_SAVE_PATH, config.SCALER_SAVE_PATH)
    # Backtest sobre un segmento corto
    from phoenix_processor import PhoenixDataProcessor

    processor = PhoenixDataProcessor(config.DATA_RAW)
    df = processor.clean_and_prepare()
    df_slice = df.iloc[-(config.LOOKBACK_WINDOW + 200):].copy()

    result = bt.backtest_period(df_slice)
    # Puede no haber trades, pero no debe fallar
    assert result is None or "capital_final" in result
