import json
import re
from pathlib import Path

LOG_PATH = Path("reports/optuna_logs/nas100_m15_50_dd20.log")
OUTPUT_PATH = Path("nas100_best_dd20.json")

text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")

best_match = re.findall(r"Best is trial (\d+) with value: ([0-9.]+)", text)
best_value = float(best_match[-1][1].rstrip(".")) if best_match else None

trial_match = re.findall(
    r"Trial (\d+) finished with value: ([0-9eE+\-\.]+) and parameters: (\{.*?\})",
    text,
)

best_trial = None
best_params = None
best_score = None

for tid, val, params in trial_match:
    try:
        score = float(val)
    except Exception:
        continue
    if best_value is not None and abs(score - best_value) < 1e-6:
        best_trial = int(tid)
        best_score = score
        best_params = params
        break

if best_params is None and trial_match:
    tid, val, params = max(trial_match, key=lambda x: float(x[1]))
    best_trial = int(tid)
    best_score = float(val)
    best_params = params

metrics = None
for m in re.finditer(
    r"Profit: ([0-9.\-]+) \| Winrate: ([0-9.]+)% \| Sharpe: ([0-9.\-]+) \| DD%: ([0-9.\-]+) \| Trades/día: ([0-9.]+) \| Score: ([0-9.\-]+)",
    text,
):
    profit, winrate, sharpe, dd, trades_day, score = m.groups()
    if best_score is not None and abs(float(score) - best_score) < 1e-3:
        metrics = {
            "profit": float(profit),
            "winrate": float(winrate),
            "sharpe": float(sharpe),
            "dd": float(dd),
            "trades_per_day": float(trades_day),
            "score": float(score),
        }
        break

if best_params is None:
    raise SystemExit("No best params found in log")

params_dict = eval(best_params)

out = {
    "name": "NAS100_M15_DD20_BEST",
    "source_log": str(LOG_PATH),
    "best_trial": best_trial,
    "score": best_score,
    "params": params_dict,
    "metrics": metrics,
}

OUTPUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"Wrote {OUTPUT_PATH}")
