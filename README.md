## Running a file:
uv run --env-file .env.run path-to-file


## Config-driven pipeline

Set values once in `pipeline.toml`, then run scripts with default config:

uv run --env-file .env.run scripts/run_weekly_update.py
uv run --env-file .env.run scripts/run_daily_update.py
uv run --env-file .env.run scripts/run_prepare.py
uv run --env-file .env.run scripts/run_train.py
uv run --env-file .env.run scripts/run_benchmark.py
uv run --env-file .env.run scripts/run_predict.py
uv run --env-file .env.run scripts/run_plot.py

To use a different config file:

uv run --env-file .env.run scripts/run_train.py --config pipeline.toml

Benchmark outputs are written to:

prepared/<experiment-id>/benchmark/<benchmark-id>/

The benchmark suite compares random trading, a one-layer CNN baseline, the
configured single model with meta layer, and the council/ensemble with meta
layer. It writes per-strategy evaluation CSVs, `summary.csv`,
`random_runs.csv`, `equity_comparison.csv`, and benchmark figures.


`run_daily_update.py` requires `ALPHAVANTAGE_API_KEY` in your env file.
