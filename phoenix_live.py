import torch
import pandas as pd
import numpy as np
import joblib
import time
import os
import argparse
from phoenix_processor import PhoenixDataProcessor
import phoenix_config as config
from core.risk_manager import RiskManager
from core.calibration import apply_temperature, load_temperature
from core.regime import entry_allowed
from core.mtf import add_mtf_features_multi, mtf_confirm

# --- CONFIGURACIÓN DE OPERACIÓN REAL ---
CONFIANZA_MINIMA = config.UMBRAL_CONFIANZA  # Alineado con el umbral optimizado
ARCHIVO_MT5 = config.LIVE_FILE  # Archivo que MT5 debe actualizar

def parse_args():
    parser = argparse.ArgumentParser(description="Phoenix Live multi-asset")
    parser.add_argument("--symbol", type=str, default=None, help="Símbolo a operar (ej: XAUUSD, BTCUSD)")
    parser.add_argument("--mt5-file", type=str, default=None, help="CSV live generado por MT5")
    return parser.parse_args()

def cargar_activos():
    print("--- [SISTEMA] Cargando Cerebro Phoenix... ---")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    scaler = joblib.load(config.SCALER_SAVE_PATH)
    if config.MODEL_TYPE in {"xgboost", "ensemble"}:
        model_xgb = joblib.load(config.XGB_MODEL_PATH)
        model_lgbm = None
        if config.MODEL_TYPE == "ensemble":
            model_lgbm = joblib.load(config.LGBM_MODEL_PATH)
        return model_xgb, model_lgbm, scaler, device
    raise RuntimeError("Modelo LSTM no soportado. Usa MODEL_TYPE='xgboost' o 'ensemble'.")

def analizar_mercado_actual(model_xgb, model_lgbm, scaler, device):
    try:
        # 1. Leer el archivo que viene de MT5
        processor = PhoenixDataProcessor(ARCHIVO_MT5)
        df = add_mtf_features_multi(processor.clean_and_prepare(), config.MTF_CONFIGS)
        
        # 2. Tomar las últimas 60 velas (Lookback)
        data_scaled = scaler.transform(df[config.FEATURES].tail(config.LOOKBACK_WINDOW).values)

        if config.MODEL_TYPE in {"xgboost", "ensemble"}:
            flat = data_scaled.reshape(1, -1)
            probs_xgb = model_xgb.predict_proba(flat)
            pred_xgb = int(np.argmax(probs_xgb, axis=1)[0])
            conf_xgb = float(np.max(probs_xgb, axis=1)[0])

            if config.MODEL_TYPE == "ensemble" and model_lgbm is not None:
                probs_lgb = model_lgbm.predict_proba(flat)
                pred_lgb = int(np.argmax(probs_lgb, axis=1)[0])
                conf_lgb = float(np.max(probs_lgb, axis=1)[0])
                weights = config.ENSEMBLE_WEIGHTS

                if config.ENSEMBLE_REQUIRE_CONSENSUS:
                    if pred_xgb != pred_lgb or pred_xgb == 0:
                        return 0, 0.0, df["Close"].iloc[-1]
                    conf = (weights["xgboost"] * conf_xgb) + (weights["lightgbm"] * conf_lgb)
                    return pred_xgb, conf, df["Close"].iloc[-1]

                combined = (weights["xgboost"] * probs_xgb) + (weights["lightgbm"] * probs_lgb)
                buy_score = float(combined[0][1])
                sell_score = float(combined[0][2])
                if buy_score >= sell_score:
                    pred = 1 if buy_score >= config.UMBRAL_BUY else 0
                    conf = buy_score if pred == 1 else 0.0
                else:
                    pred = 2 if sell_score >= config.UMBRAL_SELL else 0
                    conf = sell_score if pred == 2 else 0.0
                return pred, conf, df["Close"].iloc[-1]

            return pred_xgb, conf_xgb, df["Close"].iloc[-1]

        tensor_ventana = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0).to(device)

        # 3. Predicción de IA
        temperature = load_temperature()
        with torch.no_grad():
            output = apply_temperature(model(tensor_ventana), temperature)
            probs = torch.nn.functional.softmax(output, dim=1)
            confianza, prediccion = torch.max(probs, dim=1)

        return prediccion.item(), confianza.item(), df['Close'].iloc[-1]
    except Exception as e:
        print(f"[ERROR] Esperando datos válidos de MT5... {e}")
        return 0, 0, 0

def iniciar_bot():
    global ARCHIVO_MT5
    args = parse_args()
    if args.symbol:
        config.apply_asset(args.symbol)
    if args.mt5_file:
        ARCHIVO_MT5 = args.mt5_file
    else:
        ARCHIVO_MT5 = config.LIVE_FILE
    print(f"[LIVE] Símbolo: {config.SYMBOL} | Archivo MT5: {ARCHIVO_MT5}")

    model_xgb, model_lgbm, scaler, device = cargar_activos()
    risk_manager = RiskManager(account_balance=config.CAPITAL_INICIAL)
    print("--- [PHOENIX LIVE] Monitor de Mercado Iniciado ---")
    
    ultima_vela = None
    
    while True:
        # Analizamos cada 30 segundos para no saturar la CPU
        pred, conf, precio = analizar_mercado_actual(model_xgb, model_lgbm, scaler, device)
        
        if pred != 0 and conf > CONFIANZA_MINIMA:
            # Filtros de calidad (estrategia híbrida IA + tendencia/volatilidad)
            last_row = None
            try:
                processor = PhoenixDataProcessor(ARCHIVO_MT5)
                df_now = add_mtf_features_multi(processor.clean_and_prepare(), config.MTF_CONFIGS)
                last_row = df_now.iloc[-1]
            except Exception:
                last_row = None

            if last_row is not None:
                natr = last_row.get('NATR', 0.0)
                if natr < config.MIN_ATR_THRESHOLD:
                    time.sleep(30)
                    continue

                # Filtro mechas (wick) con cuerpo vs cierre previo
                try:
                    prev_close = df_now['Close'].iloc[-2]
                    body = abs(last_row.get('Close', 0.0) - prev_close)
                    wick = max(last_row.get('High', 0.0) - last_row.get('Low', 0.0) - body, 0.0)
                    if body == 0 or wick > (0.5 * body):
                        time.sleep(30)
                        continue
                except Exception:
                    pass

                # Flash crash: ATR 3 velas > 200% del promedio diario
                try:
                    atr_series = (df_now['NATR'] * df_now['Close'] / 100.0)
                    atr_last3 = atr_series.tail(3).mean()
                    atr_daily_avg = atr_series.tail(288).mean()
                    if risk_manager.block_flash_crash(atr_last3, atr_daily_avg):
                        time.sleep(30)
                        continue
                except Exception:
                    pass

                if last_row.get('Vol_Rel', 0.0) < config.MIN_VOL_REL:
                    time.sleep(30)
                    continue

                if config.USE_REGIME_FILTER and not entry_allowed(pred, last_row):
                    time.sleep(30)
                    continue

                if config.USE_MTF_CONFIRM and not mtf_confirm(pred, last_row):
                    time.sleep(30)
                    continue

            stop_loss_pips = config.DEFAULT_STOP_LOSS_PIPS
            take_profit_pips = max(
                config.MIN_TAKE_PROFIT_PIPS,
                int(stop_loss_pips * config.RISK_REWARD_RATIO_TARGET),
            )
            target_mult = (config.TARGET_DAILY_USD / config.EXPECTED_DAILY_USD) if config.EXPECTED_DAILY_USD else 1.0
            riesgo_pct = config.RIESGO_POR_OPERACION * target_mult * config.RISK_MULTIPLIER
            riesgo_pct = min(config.MAX_RISK_PER_TRADE_PCT, riesgo_pct)
            lot_size = risk_manager.calculate_position_size(
                stop_loss_pips,
                config.SYMBOL,
                risk_pct=riesgo_pct,
            )
            can_trade, reason = risk_manager.can_open_trade(
                symbol=config.SYMBOL,
                lot_size=lot_size,
                stop_loss_pips=stop_loss_pips,
                take_profit_pips=take_profit_pips,
            )
            if not can_trade:
                print(f"⚠️ [RIESGO] Señal bloqueada: {reason}")
                time.sleep(30)
                continue

            tipo = "COMPRA" if pred == 1 else "VENTA"
            print(f"🔥 [ALERTA] SEÑAL DETECTADA: {tipo} | Precio: {precio} | Confianza: {conf:.4f}")
            print(
                f"💰 [ORDEN] Abrir {lot_size:.2f} lotes en Vantage. "
                f"TP: {take_profit_pips} pips | SL: {stop_loss_pips} pips."
            )
            risk_manager.record_trade_opened(config.SYMBOL, lot_size)
            
            # Aquí esperaríamos 5 minutos para la siguiente vela
            time.sleep(300) 
        
        time.sleep(30)

if __name__ == "__main__":
    iniciar_bot()