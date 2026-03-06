## Running a file:
uv run --env-file .env.run path-to-file


## Full pipeline (Yahoo data)

### Step 1 — Prepare the experiment (feature engineering, labelling, sampling):
uv run --env-file .env.run .\src\kvant\ml_prepare_data\prepare_experiment.py yahoo --symbols AAPL MSFT --period 7d --interval 1m

### Step 2 — Run the labeller parameter sweep:
uv run --env-file .env.run .\src\kvant\ml_prepare_data\plot_labelling\vary_labeller_runs.py yahoo --symbols AAPL MSFT --period 7d --interval 1m

### Step 3 — Train the model on the prepared experiment:
uv run --env-file .env.run .\src\kvant\ml_framework\scripts\train_experiment.py