"""
Phoenix Metrics - Análisis completo de performance de trading
Incluye: Sharpe, Calmar, Profit Factor, Recovery Factor, Maximum Drawdown, etc.
"""
import numpy as np
import pandas as pd
from datetime import datetime

class TradingMetrics:
    """Calcula métricas de trading profesionales"""
    
    def __init__(self, capital_inicial, trades_log=None):
        self.capital_inicial = capital_inicial
        self.trades = trades_log or []
    
    def add_trade(self, entry_price, exit_price, direction, size, entry_time, exit_time, pnl):
        """Registra una operación completada"""
        self.trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'direction': direction,  # 'BUY' o 'SELL'
            'size': size,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'pnl': pnl,
            'return_pct': (pnl / (entry_price * size * 100)) * 100 if size > 0 else 0
        })
    
    def calcular_metricas_basicas(self):
        """Retorna diccionario con métricas básicas"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] < 0]
        
        total_profit = trades_df['pnl'].sum()
        total_wins = winning['pnl'].sum() if len(winning) > 0 else 0
        total_losses = abs(losing['pnl'].sum()) if len(losing) > 0 else 0
        
        win_rate = (len(winning) / len(trades_df) * 100) if len(trades_df) > 0 else 0
        avg_win = total_wins / len(winning) if len(winning) > 0 else 0
        avg_loss = total_losses / len(losing) if len(losing) > 0 else 0
        
        profit_factor = total_wins / total_losses if total_losses != 0 else 0
        
        return {
            'total_trades': len(trades_df),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def calcular_sharpe_ratio(self, capital_series, risk_free_rate=0.02):
        """
        Sharpe Ratio = (Retorno Promedio - Tasa Libre Riesgo) / Desv Estándar Retornos
        Valores > 1.0 son buenos, > 2.0 son excelentes
        """
        if len(capital_series) < 2:
            return 0
        
        returns = np.diff(capital_series) / capital_series[:-1]
        excess_returns = np.mean(returns) - (risk_free_rate / 252)  # 252 períodos trading/año
        std_returns = np.std(returns)
        
        if std_returns == 0:
            return 0
        return excess_returns / std_returns * np.sqrt(252)
    
    def calcular_calmar_ratio(self, capital_series):
        """
        Calmar Ratio = CAGR / Maximum Drawdown
        Mide retorno ajustado por riesgo de caída
        Valores > 1.0 son buenos
        """
        if len(capital_series) < 2:
            return 0
        
        # CAGR (Compound Annual Growth Rate)
        total_return = (capital_series[-1] / capital_series[0]) - 1
        periods = len(capital_series) / 252  # Asumimos 252 períodos/año
        cagr = (1 + total_return) ** (1 / periods) - 1 if periods > 0 else 0
        
        # Maximum Drawdown
        running_max = np.maximum.accumulate(capital_series)
        drawdown = (capital_series - running_max) / running_max
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        if max_dd == 0:
            return 0
        return cagr / max_dd if max_dd > 0 else 0
    
    def calcular_recovery_factor(self, capital_series):
        """
        Recovery Factor = Total Profit / Maximum Drawdown (en $)
        Mide cuánto ganaste vs qué perdiste en el peor momento
        Mayor es mejor
        """
        if len(capital_series) < 2:
            return 0
        
        total_profit = capital_series[-1] - capital_series[0]
        
        running_max = np.maximum.accumulate(capital_series)
        drawdown_usd = capital_series - running_max
        max_dd_usd = abs(np.min(drawdown_usd)) if len(drawdown_usd) > 0 else 1
        
        return total_profit / max_dd_usd if max_dd_usd > 0 else 0

    def calcular_max_drawdown_pct(self, capital_series):
        """Retorna el máximo drawdown en %"""
        if len(capital_series) < 2:
            return 0
        running_max = np.maximum.accumulate(capital_series)
        drawdown = (capital_series - running_max) / running_max
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        return max_dd * 100

    def calcular_metricas_diarias(self):
        """Calcula métricas diarias usando timestamps en trades_log"""
        if not self.trades:
            return {
                'avg_daily_profit': 0,
                'daily_profit_std': 0,
                'days_traded': 0
            }

        trades_df = pd.DataFrame(self.trades)
        ts_col = None
        for candidate in ('timestamp', 'exit_time', 'entry_time'):
            if candidate in trades_df.columns:
                ts_col = candidate
                break
        if ts_col is None:
            return {
                'avg_daily_profit': 0,
                'daily_profit_std': 0,
                'days_traded': 0
            }

        trades_df[ts_col] = pd.to_datetime(trades_df[ts_col], errors='coerce')
        trades_df = trades_df.dropna(subset=[ts_col])
        if trades_df.empty:
            return {
                'avg_daily_profit': 0,
                'daily_profit_std': 0,
                'days_traded': 0
            }

        daily = trades_df.groupby(trades_df[ts_col].dt.date)['pnl'].sum()
        return {
            'avg_daily_profit': float(daily.mean()) if len(daily) > 0 else 0,
            'daily_profit_std': float(daily.std()) if len(daily) > 1 else 0,
            'days_traded': int(len(daily))
        }
    
    def calcular_consecutive_wins_losses(self):
        """Retorna las rachas máximas de ganancias y pérdidas"""
        if not self.trades:
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}
        
        trades_df = pd.DataFrame(self.trades)
        is_win = trades_df['pnl'] > 0
        
        # Calcular cambios
        changes = is_win.astype(int).diff().fillna(0)
        groups = (changes != 0).cumsum()
        
        consecutive = is_win.groupby(groups).size()
        wins = consecutive[is_win.groupby(groups).first()]
        losses = consecutive[~is_win.groupby(groups).first()]
        
        max_wins = wins.max() if len(wins) > 0 else 0
        max_losses = losses.max() if len(losses) > 0 else 0
        
        return {
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses
        }
    
    def generar_reporte_completo(self, capital_series):
        """Genera un reporte completo de todas las métricas"""
        basicas = self.calcular_metricas_basicas()
        
        reporte = {
            **basicas,
            'sharpe_ratio': self.calcular_sharpe_ratio(capital_series),
            'calmar_ratio': self.calcular_calmar_ratio(capital_series),
            'recovery_factor': self.calcular_recovery_factor(capital_series),
            'max_drawdown_pct': self.calcular_max_drawdown_pct(capital_series),
            **self.calcular_consecutive_wins_losses(),
            **self.calcular_metricas_diarias()
        }
        
        return reporte
    
    def imprimir_reporte(self, capital_series, titulo="REPORTE DE TRADING"):
        """Imprime un reporte formateado"""
        reporte = self.generar_reporte_completo(capital_series)
        
        print("\n" + "="*60)
        print(f" {titulo}")
        print("="*60)
        print(f"Total Operaciones:        {reporte['total_trades']}")
        print(f"Ganancias:                {reporte['winning_trades']}")
        print(f"Pérdidas:                 {reporte['losing_trades']}")
        print(f"Win Rate:                 {reporte['win_rate']:.2f}%")
        print(f"Ganancia Total:           ${reporte['total_profit']:.2f}")
        print(f"Promedio Ganancia:        ${reporte['avg_win']:.2f}")
        print(f"Promedio Pérdida:         ${reporte['avg_loss']:.2f}")
        print(f"Profit Factor:            {reporte['profit_factor']:.2f}")
        print(f"\nSharpe Ratio:             {reporte['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio:             {reporte['calmar_ratio']:.2f}")
        print(f"Recovery Factor:          {reporte['recovery_factor']:.2f}")
        print(f"Max Wins Consecutivos:    {reporte['max_consecutive_wins']}")
        print(f"Max Losses Consecutivos:  {reporte['max_consecutive_losses']}")
        print("="*60 + "\n")
        
        return reporte
