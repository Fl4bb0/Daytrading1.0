# daytrading

A config-driven pipeline for training and evaluating intraday (1-minute bar)
trading models. Data is fetched/cached locally, prepared into labeled
windows, trained with PyTorch, scored with an optional meta/ranking layer,
and evaluated with benchmarks, walk-forward analysis, and plots.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
uv sync
cp .env.template .env.run   # then fill in your API key(s)
```

`.env.run` is passed to every script via `--env-file` and must contain:

```
PYTHONPATH="src"
ALPHAVANTAGE_API_KEY=<your key>   # used by data retrieval helpers
```

All scripts are run from the project root with:

```bash
uv run --env-file .env.run <path-to-script>
```

## Configuration

All pipeline behavior (symbols, date ranges, model choice, training
hyperparameters, meta layer, benchmarking, walk-forward) is controlled by
`pipeline.toml`. Edit it once, then every script below reads from it by
default. To use a different config file, pass `--config <path>` to any
script.

Key sections in `pipeline.toml`:

- `[paths]` — local data store, prepared/checkpoint output roots
- `[data]` — interval, ticker universe, pre/post market inclusion
- `[prepare]` — train/val/test splits, window size, labeling/barrier params
- `[walk_forward]` — enable rolling/expanding walk-forward evaluation instead
  of a single train/test split
- `[train]` — model architecture, epochs, batch size, learning rate
- `[predict]` — inference thresholds, position sizing/limits
- `[meta]` — optional meta-ranking layer trained on validation predictions
- `[benchmark]` — benchmark suite settings (random baseline seeds, etc.)
- `[ensemble]` — models included in the council/ensemble
- `[data_sources]` / `[hf_config]` — HuggingFace historical backfill vs.
  Yahoo Finance incremental updates

## Updating local data

```bash
# One-time backfill of historical 1-minute bars from HuggingFace
uv run --env-file .env.run scripts/run_hf_backfill.py

# Incremental Mon-Fri update from Yahoo Finance (safe to re-run)
uv run --env-file .env.run scripts/run_weekly_update.py
```

`run_weekly_update.py` is idempotent — it tracks which days are already
fetched and only retrieves what's missing. Run it weekly (e.g. after Friday
close) or mid-week to pick up completed days so far.

## Running the pipeline

### Quick start (single train/test split)

```bash
bash run_pipeline.sh [pipeline.toml]
```

This runs, in order: prepare → train → (train_meta, if `[meta].enabled`) →
predict → plot. If `[walk_forward].enabled = true` in the config, it instead
runs only the walk-forward step.

### Individual steps

```bash
uv run --env-file .env.run scripts/run_prepare.py     # build labeled windows from raw bars
uv run --env-file .env.run scripts/run_train.py       # train the configured model
uv run --env-file .env.run scripts/run_train_meta.py  # train the meta/ranking layer (if enabled)
uv run --env-file .env.run scripts/run_predict.py     # run inference + evaluation CSVs
uv run --env-file .env.run scripts/run_plot.py        # generate figures from evaluation CSVs
```

Each accepts `--config <path>` to override `pipeline.toml`.

### Walk-forward evaluation

```bash
uv run --env-file .env.run scripts/run_walk_forward.py
uv run --env-file .env.run scripts/run_walk_forward_compare.py
```

`run_walk_forward.py` prepares, trains, and evaluates successive
expanding/rolling folds as configured under `[walk_forward]`. Results are
written to `prepared/walkforward/comparisons/latest/`, including
`run_comparison_summary.csv`, `fold_comparison.csv`, and plots for net
return, accuracy, stability, and fold distributions.

`run_walk_forward_compare.py` compares multiple prior walk-forward runs
against each other (`--run-ids` to pick specific runs, `--max-runs` to limit
how many).

### Benchmarking

```bash
uv run --env-file .env.run scripts/run_benchmark.py
uv run --env-file .env.run scripts/run_benchmark.py --benchmark-id final_test
```

Compares four strategies: the council/ensemble + meta layer, the configured
single model + meta layer, a one-layer CNN baseline, and random trading over
multiple seeds. Output is written to
`prepared/<experiment-id>/benchmark/<benchmark-id>/`, including per-strategy
evaluation CSVs, `summary.csv`, `random_runs.csv`, `equity_comparison.csv`,
and comparison figures.

## Output layout

- `data/1m/` — local cache of raw 1-minute OHLCV bars (HuggingFace + Yahoo)
- `prepared/<experiment-id>/` — prepared datasets, evaluation CSVs, figures
- `checkpoints/` — trained model weights

## Tests

```bash
uv run --env-file .env.run pytest
```