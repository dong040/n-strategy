# n-strategy

N-strategy is a rule-based stock strategy project with ML filters, backtesting, and scan tooling.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
python3 scripts/run_scan.py
python3 scripts/quick_backtest.py
python3 scripts/train_ml_walkforward.py
python3 scripts/auto_iterate_strategy.py
python3 scripts/train_kline_sequence.py data/sequences.npz
```

## Project Layout

- `src/strategy/` core strategy logic, backtest, ML filters, and sequence model code
- `src/screener/` scanning and data fetching helpers
- `scripts/` runnable entry points for scans, backtests, and model training
- `data/` local datasets and model artifacts

## Notes

- Keep secrets in a local `.env` file.
- Generated files such as `.venv/`, `logs/`, `eval_results/`, and local cache/data artifacts are ignored by Git.
- The main ML model artifact is stored at `data/xgboost_n_pattern.pkl` when trained.
