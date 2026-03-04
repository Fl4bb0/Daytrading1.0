# pip install huggingface_hub pyarrow pandas vectorbt

import pandas as pd
import vectorbt as vbt



REPO_ID = "mito0o852/OHLCV-1m"

# Pick a month shard that exists in the dataset repo under /data
# Example shown in the dataset: data/ohlcv_1992-01.parquet
MONTH_FILE = "data/ohlcv_2025-01.parquet"   # change if needed
TICKER = "VUZI"

# MA windows in *bars* (here bars are minutes if you regularize to 1-minute)
FAST = 20    # 20 minutes
SLOW = 60    # 60 minutes


def run_ma_crossover(close: pd.Series) -> vbt.Portfolio:
    fast_ma = vbt.MA.run(close, FAST, short_name="fast")  # ([vectorbt.dev](https://vectorbt.dev/api/indicators/basic/?utm_source=openai))
    slow_ma = vbt.MA.run(close, SLOW, short_name="slow")
    entries = fast_ma.ma_crossed_above(slow_ma)           # ([vectorbt.dev](https://vectorbt.dev/?utm_source=openai))
    exits = fast_ma.ma_crossed_below(slow_ma)

    # Basic backtest assumptions (tune!)
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        freq="1T",          # 1-minute bars
        fees=0.0005,        # 5 bps per trade side (example)
        slippage=0.0002     # 2 bps per trade side (example)
    )  # ([vectorbt.dev](https://vectorbt.dev/api/portfolio/base/?utm_source=openai))
    return pf


def main():
    from src.kvant.kdata import load_one_month
    df_month = load_one_month(MONTH_FILE)

    # Print tickers in this month shard
    tickers = sorted(df_month["ticker"].dropna().unique().tolist())
    print(f"\nTickers in {MONTH_FILE}: {len(tickers)}")
    print(tickers[:50], "..." if len(tickers) > 50 else "")

    # Filter one ticker
    df = df_month.loc[df_month["ticker"] == TICKER].copy()
    if df.empty:
        raise ValueError(f"No rows for ticker={TICKER} in {MONTH_FILE}")

    df = df.set_index("timestamp").sort_index()

    # --- Optional but recommended: regularize to a complete 1-minute grid ---
    # This makes MA windows behave like “minutes” rather than “activity bars”.
    full_idx = pd.date_range(df.index.min().floor("T"), df.index.max().ceil("T"), freq="1T", tz="UTC")
    df = df.reindex(full_idx)

    # Fill missing OHLC with previous close; volume=0 (simple time-bar imputation)
    df["close"] = df["close"].ffill()
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"]  = df["low"].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    close = df["close"].dropna()

    # Run strategy
    pf = run_ma_crossover(close)

    print("\n=== Portfolio stats ===")
    print(pf.stats())

    print("\n=== First few orders ===")
    # records_readable exists for orders; exact field names can vary by vectorbt version
    print(pf.orders.records_readable.head(10))

    # Optional: plot (requires plotly installed)
    # pf.plot().show()

if __name__ == "__main__":
    main()