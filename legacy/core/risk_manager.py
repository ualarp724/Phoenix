from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple

import phoenix_config as config
from core.news_checker import NewsFilter


@dataclass
class PositionSnapshot:
    symbol: str
    lot_size: float
    exposure_usd: float
    opened_at: datetime = field(default_factory=lambda: datetime.utcnow())


class RiskManager:
    """
    Risk Manager con validaciones CFO obligatorias.
    Mantiene estado en memoria para el runtime del bot.
    """

    def __init__(self, account_balance: float):
        self.account_balance = account_balance
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.open_positions: List[PositionSnapshot] = []
        self._last_day = datetime.utcnow().date()
        self._last_week_start = self._week_start(datetime.utcnow())
        self._news_filter = NewsFilter()

    def _week_start(self, dt: datetime) -> datetime:
        return dt - timedelta(days=dt.weekday())

    def _reset_if_new_period(self, now: datetime | None = None) -> None:
        now = now or datetime.utcnow()
        if now.date() != self._last_day:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self._last_day = now.date()

        week_start = self._week_start(now)
        if week_start.date() != self._last_week_start.date():
            self.weekly_pnl = 0.0
            self._last_week_start = week_start

    def _current_exposure(self) -> float:
        return sum(p.exposure_usd for p in self.open_positions)

    def _calculate_exposure(self, lot_size: float) -> float:
        return lot_size * 100000.0

    def is_trading_hours(self, now: datetime | None = None) -> bool:
        now = now or datetime.utcnow()
        if config.AVOID_SUNDAY_OPEN and now.weekday() == 6:
            return False
        if config.AVOID_FRIDAY_AFTER_16 and now.weekday() == 4 and now.hour >= 16:
            return False
        return config.TRADING_HOURS_UTC["start"] <= now.hour < config.TRADING_HOURS_UTC["end"]

    def high_impact_news_soon(self, now: datetime | None = None) -> bool:
        if now is None:
            return self._news_filter.high_impact_soon()
        return self._news_filter.high_impact_at(now)

    def calculate_position_size(self, stop_loss_pips: int, symbol: str, risk_pct: float | None = None) -> float:
        risk_pct = config.RIESGO_POR_OPERACION if risk_pct is None else risk_pct
        risk_pct = risk_pct * config.RISK_MULTIPLIER
        if symbol == "BTCUSD":
            risk_pct = risk_pct * getattr(config, "BTC_RISK_MULTIPLIER", 0.25)
        risk_pct = min(risk_pct, config.MAX_RISK_PER_TRADE_PCT)
        risk_usd = self.account_balance * risk_pct
        pip_value_map = {
            "EURUSD": 0.10,
            "GBPUSD": 0.10,
            "AUDUSD": 0.10,
            "USDJPY": 0.09,
            "USDCHF": 0.10,
            "XAUUSD": 0.10,
            "BTCUSD": 0.10,
        }
        pip_value_micro = pip_value_map.get(symbol)
        if pip_value_micro is None:
            return config.MIN_LOT_SIZE

        lots = risk_usd / (stop_loss_pips * pip_value_micro)
        lots = round(lots / 0.01) * 0.01
        return max(config.MIN_LOT_SIZE, min(lots, config.MAX_LOT_SIZE))

    def block_flash_crash(self, atr_last3: float, atr_daily_avg: float) -> bool:
        if atr_daily_avg <= 0:
            return False
        return atr_last3 > (2.0 * atr_daily_avg)

    def can_open_trade(
        self,
        symbol: str,
        lot_size: float,
        stop_loss_pips: int,
        take_profit_pips: int,
        conservative_mode: bool = True,
        current_time: datetime | None = None,
    ) -> Tuple[bool, str]:
        self._reset_if_new_period(current_time)

        if self.daily_pnl <= -config.MAX_DAILY_LOSS_USD:
            return False, f"❌ LÍMITE DIARIO ALCANZADO: ${self.daily_pnl:.2f}"

        if self.weekly_pnl <= -config.MAX_WEEKLY_LOSS_USD:
            return False, f"❌ LÍMITE SEMANAL ALCANZADO: ${self.weekly_pnl:.2f}"

        if self.daily_pnl >= config.DAILY_PROFIT_TARGET_USD and conservative_mode:
            return False, f"✅ OBJETIVO DIARIO CUMPLIDO: ${self.daily_pnl:.2f}"

        if lot_size > config.MAX_LOT_SIZE or lot_size < config.MIN_LOT_SIZE:
            return False, f"❌ LOTE {lot_size:.2f} FUERA DE RANGO"

        new_exposure = self._calculate_exposure(lot_size)
        total_exposure = self._current_exposure() + new_exposure
        effective_leverage = total_exposure / max(self.account_balance, 1.0)

        if total_exposure > config.MAX_EXPOSURE_USD:
            return False, f"❌ EXPOSICIÓN ${total_exposure:.2f} EXCEDE ${config.MAX_EXPOSURE_USD}"

        if effective_leverage > config.MAX_EFFECTIVE_LEVERAGE:
            return False, f"❌ APALANCAMIENTO {effective_leverage:.1f}x EXCEDE {config.MAX_EFFECTIVE_LEVERAGE}x"

        if not (config.MIN_STOP_LOSS_PIPS <= stop_loss_pips <= config.MAX_STOP_LOSS_PIPS):
            return False, f"❌ STOP LOSS {stop_loss_pips} FUERA DE RANGO"

        rr_ratio = take_profit_pips / max(stop_loss_pips, 1)
        if rr_ratio < config.RISK_REWARD_RATIO_MIN:
            return False, f"❌ R:R {rr_ratio:.2f} < {config.RISK_REWARD_RATIO_MIN}"

        if symbol not in config.ALLOWED_PAIRS:
            return False, f"❌ PAR {symbol} NO PERMITIDO"

        if not (config.BACKTEST_IGNORE_TIME_NEWS and current_time is not None):
            if not self.is_trading_hours(current_time):
                return False, "⚠️ FUERA DE HORARIO"

        if self.daily_trades >= config.MAX_TRADES_PER_DAY:
            return False, f"⚠️ MÁXIMO DIARIO: {self.daily_trades}/{config.MAX_TRADES_PER_DAY}"

        if not (config.BACKTEST_IGNORE_TIME_NEWS and current_time is not None):
            if self.high_impact_news_soon(current_time):
                return False, "📰 NOTICIA DE ALTO IMPACTO <30min"

        if self.consecutive_losses >= 3:
            return False, "🔴 3 PÉRDIDAS CONSECUTIVAS"

        if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, f"⚠️ MÁXIMO POSICIONES: {len(self.open_positions)}/{config.MAX_OPEN_POSITIONS}"

        return True, "✅ APROBADO"

    def record_trade_opened(self, symbol: str, lot_size: float) -> None:
        exposure = self._calculate_exposure(lot_size)
        self.open_positions.append(PositionSnapshot(symbol=symbol, lot_size=lot_size, exposure_usd=exposure))
        self.daily_trades += 1

    def record_trade_closed(self, pnl_usd: float) -> None:
        self.daily_pnl += pnl_usd
        self.weekly_pnl += pnl_usd
        if pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        if self.open_positions:
            self.open_positions.pop(0)
