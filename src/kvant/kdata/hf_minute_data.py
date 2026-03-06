
# from historical_stuff.data_vectorbt_example import MONTH_FILE
from kvant.kdata.hf_download_utils import load_one_month
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import os
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from collections import defaultdict
import tqdm

REPO_ID = "mito0o852/OHLCV-1m"

# Pick a month shard that exists in the dataset repo under /data
# Example shown in the dataset: data/ohlcv_1992-01.parquet
MONTH_FILE = "data/ohlcv_2025-01.parquet"   # change if needed


def get_dataset_file(repo_id: str, filename: str) -> str:
    try:
        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_files_only=True,   # <- guarantees no network
        )
    except LocalEntryNotFoundError:
        print("> Cache unavailable, downloading remotely from hugging face", filename)
        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )


def get_raw_monthly_data(year, month_zero_indexed):
    month_file = f"data/ohlcv_{year}-{(month_zero_indexed+1):02}.parquet"
    return get_dataset_file(repo_id=REPO_ID, filename=month_file)





def prepare_single_ticker(month_file : str, ticker : str, impute_missing_minutes=True) -> dict[str, pd.DataFrame]:
    df_month = load_one_month(month_file)

    tickers = sorted(df_month["ticker"].dropna().unique().tolist())
    print(f"\nTickers in {MONTH_FILE}: {len(tickers)}")
    print(tickers[:50], "..." if len(tickers) > 50 else "")
    TICKER = ticker
    df = df_month.loc[df_month["ticker"] == TICKER].copy()
    if df.empty:
        raise ValueError(f"No rows for ticker={TICKER} in {MONTH_FILE}")

    df = df.set_index("timestamp").sort_index()
    # --- Optional but recommended: regularize to a complete 1-minute grid ---
    # This makes MA windows behave like “minutes” rather than “activity bars”.
    full_idx = pd.date_range(df.index.min().floor("min"), df.index.max().ceil("min"), freq="1min", tz="UTC")
    df = df.reindex(full_idx)

    # Fill missing OHLC with previous close; volume=0 (simple time-bar imputation)
    df["close"] = df["close"].ffill()
    df["open"] = df["open"].fillna(df["close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"] = df["low"].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df['ticker'] = ticker
    return {ticker: df}
    return df

def google_1_month():
    # return prepare_single_ticker()
    ticker = 'GOOG'
    tickers = prepare_single_ticker(month_file=MONTH_FILE, ticker=ticker, impute_missing_minutes=True)
    return tickers

from dataclasses import dataclass

@dataclass
class DatasetConfiguration:
    year_quarter_train : list[tuple]
    year_quarter_test : list[tuple]
    year_quarter_val : list[tuple]

    year_month_train : list[tuple]
    year_month_test : list[tuple]
    year_month_val : list[tuple]


def available_datasets(first_year=2015, warmup_quarters=16):
    # Which months go with which quarters.
    q2m = {0 : [0, 1, 2], 1 : [3, 4, 5], 2 : [6, 7, 8], 3 : [9, 10, 11]}

    # first_year = 2015
    first_quarter = 0
    last_year = 2025
    last_quarter = 1
    # warmup_quarters = 16

    avail_quarters = (last_year - first_year)  * 4 + last_quarter - first_quarter

    # Zero-indexed quarters to [(year, month), ...] format.
    def q2qy(quarter):
        dy = (first_quarter  + quarter ) // 4
        dq = (first_quarter + quarter ) % 4
        return (first_year + dy, dq)

    # Convert a list of [(year, quarter), ...] to a longer list of [(year, month), ...]
    def yq2ym(year_quarter_list):
        ym_list = []
        for y, q in year_quarter_list:
            for m in q2m[q]:
                ym_list.append(  (y, m) )
        return ym_list

    steps = avail_quarters - warmup_quarters
    datasets_configurations = []

    for k in range(steps):
        q_train = [k + i for i in range(warmup_quarters) ]
        q_val = [k + 0 + warmup_quarters]
        q_test = [k + 1 + warmup_quarters]

        year_quarter_train = list( map(q2qy, q_train) )
        year_quarter_val = list(map(q2qy, q_val))
        year_quarter_test = list(map(q2qy, q_test))

        ds = DatasetConfiguration(year_quarter_train=year_quarter_train,
                             year_quarter_test=year_quarter_test,
                             year_quarter_val=year_quarter_val,
                             year_month_train=yq2ym(year_quarter_train),
                             year_month_test=yq2ym(year_quarter_test),
                             year_month_val=yq2ym(year_quarter_val))
        datasets_configurations.append(ds)
    return datasets_configurations


@dataclass
class DownloadedDatasetSplit:
    split: DatasetConfiguration
    tickers_train : list[str]
    monthly_pq_train : list[str]
    monthly_pq_test: list[str]
    monthly_pq_val: list[str]



def download_and_create_dataset(dataset_configurations : list[DatasetConfiguration], use_top_n_tickers : int, cache_dir : str, blacklisted_tickers=None):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    if blacklisted_tickers is None:
        blacklisted_tickers = tuple()

    import pickle
    # out_root = f"{cache_dir}/monthly_raw"  # choose your destination
    if os.path.isfile(pkl_cache := f"{cache_dir}/download_data_cached.pkl"):
        with open(pkl_cache, "rb") as f:
            return pickle.load(f)
    else:
        # Check if already downloaded and prepared. If so, return the prepared version.
        ds_top_tickers = {}
        # This code will load all datasets, determine top_k stocks across the entire dataset.
        for kk, dc in enumerate(dataset_configurations):
            total_vol = defaultdict(float)
            for year, month in tqdm.tqdm(dc.year_month_train):
                local_path = get_raw_monthly_data(year, month)

                # Load only what we need
                table = pq.read_table(local_path, columns=["ticker", "volume", "close"])

                # dollar = volume * close
                dollar = pc.multiply(table["volume"], table["close"])
                table2 = table.append_column("dollar", dollar)

                # Group by ticker and sum dollars
                agg = table2.group_by("ticker").aggregate([("dollar", "sum")])  # ticker, dollar_sum

                for ticker, dv in zip(agg["ticker"].to_pylist(), agg["dollar_sum"].to_pylist()):
                    if dv is not None:
                        total_vol[ticker] += float(dv)

            total_vol = {ticker : v for ticker, v in total_vol.items() if ticker not in blacklisted_tickers}

            top100_tickers_sorted = sorted([t for t, _ in sorted(total_vol.items(), key=lambda kv: kv[1], reverse=True)[:use_top_n_tickers]])
            ds_top_tickers[kk] = top100_tickers_sorted

        all_active_tickers = []
        for kk, tickers in ds_top_tickers.items():
            all_active_tickers  += tickers

        ym_out = {}

        # Now get all year_month configurations:
        for dc in dataset_configurations:
            for year, month in dc.year_month_train + dc.year_month_test + dc.year_month_val:
                if (year, month) in ym_out:
                    pass
                else:
                    local_path = get_raw_monthly_data(year, month)
                    table = pq.read_table(local_path)  # optionally pass columns=[...] to reduce IO
                    value_set = pa.array(all_active_tickers)  # or pa.array(..., type=pa.string())
                    mask = pc.is_in(table["ticker"], value_set=value_set)
                    filtered = table.filter(mask)
                    # write one file per month (fast to load later)
                    out_dir = os.path.join(cache_dir, "monthly_raw", f"year={year}", f"month={month:02d}")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, "data.parquet")
                    ym_out[(year, month)] = out_path
                    pq.write_table(filtered, out_path, compression="zstd")

        downloaded_datasets = []

        for kk, dc in enumerate(dataset_configurations):
            f_train = [ym_out[year, month] for year, month in dc.year_month_train]
            f_test = [ym_out[year, month] for year, month in dc.year_month_test]
            f_val = [ym_out[year, month] for year, month in dc.year_month_val]

            ddset = DownloadedDatasetSplit(split=dc, tickers_train=ds_top_tickers[kk], monthly_pq_train=f_train,
                                   monthly_pq_test=f_test, monthly_pq_val=f_val)
            downloaded_datasets.append(ddset)

        with open(pkl_cache, "wb") as f:
            pickle.dump(downloaded_datasets, f)


    return downloaded_datasets



def get_huggingface_top_200_splits():
    from src import kvant
    cache_dir = os.path.abspath(kvant.__path__[0] + "/../../cache/top_200")
    downloaded_datasets = download_and_create_dataset(available_datasets(), use_top_n_tickers=200, cache_dir=cache_dir)
    print("-"*80)
    print("""
Congratulations! You have downloaded minute-by-minute data split into train/test/validation folds.
The data has been cached and for each split, limited to the top-200 stocks.
Note: You have to make sure that the datasets only contain the listed tickers. Since the monthly data overlaps, they may contain
too many tickers!    
    """.strip() )
    print("-"*80)
    return downloaded_datasets



def get_huggingface_top_n_tiny_splits(n=4, warmup_quarters=1, blacklisted_tickers=None):
    from src import kvant
    if blacklisted_tickers is None:
        blacklisted_tickers = ("SPY", "QQQ")

    cache_dir = os.path.abspath(f"{kvant.__path__[0]}/../../cache/top_{n}_warm_{warmup_quarters}_tiny")
    downloaded_datasets = download_and_create_dataset(available_datasets(first_year=2020, warmup_quarters=warmup_quarters),
                                                      use_top_n_tickers=n,
                                                      cache_dir=cache_dir,
                                                      blacklisted_tickers=blacklisted_tickers)
    print("-" * 80)
    print("""
    Congratulations! You have downloaded minute-by-minute data split into train/test/validation folds.
    The data has been cached and for each split, limited to the top-200 stocks.
    Note: You have to make sure that the datasets only contain the listed tickers. Since the monthly data overlaps, they may contain
    too many tickers!    
        """.strip())
    print("-" * 80)
    return downloaded_datasets


# def get_huggingface_top_n_normal_splits(n=4, warmup_quarters=4):
#     from src import kvant
#     cache_dir = os.path.abspath(f"{kvant.__path__[0]}/../../cache/top_{n}_warm_{warmup_quarters}_tiny")
#     downloaded_datasets = download_and_create_dataset(available_datasets(first_year=2020, warmup_quarters=warmup_quarters), use_top_n_tickers=n, cache_dir=cache_dir)
#     print("-" * 80)
#     print("""
#     Congratulations! You have downloaded minute-by-minute data split into train/test/validation folds.
#     The data has been cached and for each split, limited to the top-200 stocks.
#     Note: You have to make sure that the datasets only contain the listed tickers. Since the monthly data overlaps, they may contain
#     too many tickers!
#         """.strip())
#     print("-" * 80)
#     return downloaded_datasets

def get_huggingface_top_4_tiny_splits():
    return get_huggingface_top_n_tiny_splits(n=4)

def get_huggingface_top_5_small_splits():
    return get_huggingface_top_n_tiny_splits(n=5, warmup_quarters=16)


def get_huggingface_top_10_tiny_splits():
    return get_huggingface_top_n_tiny_splits(n=10, warmup_quarters=2)


from typing import Optional, Iterable


def get_ticker_data(downloaded_dataset : DownloadedDatasetSplit):
    def load_concat_split_by_ticker(pq_files: list[str], only_use_tickers: Optional[Iterable[str]] = None):
        # 1) Load + concatenate
        dfs = [pd.read_parquet(p) for p in pq_files]
        df = pd.concat(dfs, ignore_index=True)

        # 2) Ensure timestamp is tz-aware UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Optional early filter for speed
        if only_use_tickers is not None:
            only_use_tickers = set(only_use_tickers)
            df = df[df["ticker"].isin(only_use_tickers)]
        # 3) Sort so "keep first" is well-defined
        #    We sort by (ticker, timestamp) so duplicates for a ticker are adjacent.
        df = df.sort_values(["ticker", "timestamp"], kind="mergesort")
        # 4) Drop duplicate rows for the same (ticker, timestamp), keeping the first
        df = df.drop_duplicates(subset=["ticker", "timestamp"], keep="first")
        # 5) Set index to timestamp (now safe for time-series ops)
        df = df.set_index("timestamp")
        # 6) Split into dict of per-ticker dataframes (time-indexed)
        out = {t: g.copy() for t, g in df.groupby("ticker", sort=True)}
        return out

    ds = downloaded_dataset

    ticker_data_train = load_concat_split_by_ticker(ds.monthly_pq_train, only_use_tickers=ds.tickers_train)
    ticker_data_val = load_concat_split_by_ticker(ds.monthly_pq_val, only_use_tickers=ds.tickers_train)
    ticker_data_test = load_concat_split_by_ticker(ds.monthly_pq_test, only_use_tickers=ds.tickers_train)
    return ticker_data_train, ticker_data_val, ticker_data_test

if __name__ == "__main__":
    # load_one_month(MONTH_FILE)
    # Download and get top-200 stock splits (train, test, val).
    # downloaded_splits = get_huggingface_top_200_splits()
    downloaded_splits = get_huggingface_top_4_tiny_splits()
    # Now the dataset is downloaded and prepared.
    # Try to train on a single split?
    pq_file = downloaded_splits[-1].monthly_pq_train[-1]
    table = pq.read_table(pq_file)
    df = table.to_pandas()

    # Normalize
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df2 = df[df['ticker'] == 'PLTR']

    for k in range(60*10):
        break
        t = df2['timestamp'][k]
        vol = df2['volume'][k]
        is_nyse = is_nyse_available(t, minutes_after_open=0, minutes_before_close=0)
        print(">", t, is_nyse, vol)

    ticker_data_train, ticker_data_val, ticker_data_test = get_ticker_data( downloaded_splits[-1])
    ticker = list(ticker_data_train.keys())[2]

    df = ticker_data_train[ticker]
    # df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    from kvant.labelling import tripple_bar_label
    # for kk, t in enumerate(df.index):
    #     a, b = tripple_bar_label(df,  time_start=t, width=32, height=0.025)
    #     print(t, a,b)
    #     if a is not None:
    #         break
    import numpy as np
    import matplotlib.pyplot as plt
    # df is indexed by timestamp already in your example.
    # If not, ensure UTC datetime index:
    # df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # df = df.sort_values("timestamp").set_index("timestamp")

    # Now compute oracle profits:

    df = df.sort_index()
    idx = df.index
    width = 60
    height = 0.02

    labels = np.empty(len(idx), dtype=object)  # store 0/1/2 or None

    trades = []
    print(len(idx))
    t0, tn = None, None
    for i, t in tqdm.tqdm(enumerate(idx)):
        if i == 0:
            t0 = t
        bar = tripple_bar_label(df, time_start=t, width=width, height=height)
        if bar is None:
            continue
        if bar.label == 1:
            continue
        elif bar.label == 0:

            trades.append( -bar.pnl_absolute )
        elif bar.label == 2:
            trades.append( bar.pnl_absolute )

        # if i > 30000:
        tn = t
        # break
        # labels[i] = label

    x = idx
    y = df.loc[idx, "close"].to_numpy()
    print("Total profit was", sum(trades))

    def plot_triple_bar_first_n(df, n=600, width=60, height=0.005):
        df = df.sort_index()
        idx = df.index[:n]

        labels = np.empty(len(idx), dtype=object)  # store 0/1/2 or None
        for i, t in enumerate(idx):
            bar = tripple_bar_label(df, time_start=t, width=width, height=height)
            if bar is None:
                label = None
            else:
                label = bar.label
            labels[i] = label

        x = idx
        y = df.loc[idx, "close"].to_numpy()

        # color map for labels; None -> black
        color_map = {0: "red", 1: "blue", 2: "green", None: "black"}
        colors = [color_map.get(l, "black") for l in labels]

        plt.figure(figsize=(14, 5))
        plt.scatter(x, y, c=colors, s=9, linewidths=0)

        plt.xlabel("time (UTC)")
        plt.ylabel("close")
        plt.title(f"Triple-bar labels (n={n}, width={width}m, height={height}); (red: down, green: up)")

    plot_triple_bar_first_n(df, n=3000, width=120, height=0.02)
    plt.show()

