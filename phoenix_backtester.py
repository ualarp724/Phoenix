import phoenix_config as config

def ejecutar_auditoria():
    """Wrapper de alto nivel: usa el backtester profesional."""
    from phoenix_backtester_pro import ejecutar_backtest_profesional

    print("⚠️  phoenix_backtester.py está deprecado. Usando backtester profesional.")
    ejecutar_backtest_profesional()

if __name__ == "__main__":
    ejecutar_auditoria()