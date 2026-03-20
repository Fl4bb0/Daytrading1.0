## Running a file:
uv run --env-file .env.run path-to-file


## Config-driven pipeline

Set values once in `pipeline.toml`, then run scripts with default config:

uv run --env-file .env.run scripts/run_weekly_update.py
uv run --env-file .env.run scripts/run_daily_update.py
uv run --env-file .env.run scripts/run_prepare.py
uv run --env-file .env.run scripts/run_train.py
uv run --env-file .env.run scripts/run_predict.py
uv run --env-file .env.run scripts/run_plot.py

To use a different config file:

uv run --env-file .env.run scripts/run_train.py --config pipeline.toml


`run_daily_update.py` requires `ALPHAVANTAGE_API_KEY` in your env file.
