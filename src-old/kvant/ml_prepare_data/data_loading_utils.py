from typing import Any, Dict
import numpy as np


def summary(self, display: bool = True) -> Dict[str, Any]:
    """
    Summarize this dataset split (as defined by self.index).

    Computes per-ticker:
      - n: number of samples in this dataset for that ticker
      - y_counts: counts of classes {0,1,2}
      - first_ts / last_ts: timestamps at min/max tpos for that ticker

    Also computes an 'overall' summary aggregated across tickers.

    If display=True, prints a tabulated report.

    Returns a dict:
      {
        "overall": {...},
        "per_ticker": {
           "AAPL": {...},
           ...
        }
      }
    """
    # Defensive: empty dataset
    if self.index is None or int(self.index.shape[0]) == 0:
        out = {
            "overall": {
                "n": 0,
                "y_counts": {0: 0, 1: 0, 2: 0},
                "first_ts": None,
                "last_ts": None,
            },
            "per_ticker": {},
        }
        if display:
            try:
                from tabulate import tabulate
                print(tabulate([], headers=["ticker", "n", "y=0", "y=1", "y=2", "first_ts", "last_ts"]))
            except Exception:
                print("(empty dataset)")
        return out

    # Pull ids/positions
    tids = self.index[:, 0].astype(np.int64, copy=False)
    tposs = self.index[:, 1].astype(np.int64, copy=False)
    uniq_tids = np.unique(tids)

    per_ticker: Dict[str, Any] = {}

    overall_n = 0
    overall_counts = np.zeros(3, dtype=np.int64)
    overall_first_ts = None  # np.datetime64
    overall_last_ts = None  # np.datetime64

    for tid in uniq_tids:
        tid_i = int(tid)
        ticker = self.store.tickers_all[tid_i]

        mask = (tids == tid)
        pos = tposs[mask]
        n = int(pos.shape[0])
        overall_n += n

        # Labels for those positions (fast: direct array indexing, no window extraction)
        y_arr = np.asarray(self.store._labels[tid_i][pos], dtype=np.int64)
        # In case y contains unexpected values, we only count 0/1/2
        y_arr = y_arr[(y_arr >= 0) & (y_arr <= 2)]
        counts = np.bincount(y_arr, minlength=3).astype(np.int64)
        overall_counts += counts

        # Timestamps: take min/max by position (position order corresponds to time)
        ts_arr = self.store._timestamps[tid_i]
        pmin = int(pos.min())
        pmax = int(pos.max())
        first_ts = ts_arr[pmin]
        last_ts = ts_arr[pmax]

        if overall_first_ts is None or first_ts < overall_first_ts:
            overall_first_ts = first_ts
        if overall_last_ts is None or last_ts > overall_last_ts:
            overall_last_ts = last_ts

        per_ticker[ticker] = {
            "tid": tid_i,
            "n": n,
            "y_counts": {0: int(counts[0]), 1: int(counts[1]), 2: int(counts[2])},
            "first_ts": None if first_ts is None else str(np.datetime_as_string(first_ts, unit="s")),
            "last_ts": None if last_ts is None else str(np.datetime_as_string(last_ts, unit="s")),
        }

    out = {
        "overall": {
            "n": int(overall_n),
            "y_counts": {0: int(overall_counts[0]), 1: int(overall_counts[1]), 2: int(overall_counts[2])},
            "first_ts": None if overall_first_ts is None else str(np.datetime_as_string(overall_first_ts, unit="s")),
            "last_ts": None if overall_last_ts is None else str(np.datetime_as_string(overall_last_ts, unit="s")),
        },
        "per_ticker": per_ticker,
    }

    if display:
        try:
            from tabulate import tabulate

            # rows sorted by ticker symbol for stable display
            rows = []
            for ticker in sorted(per_ticker.keys()):
                d = per_ticker[ticker]
                rows.append([
                    ticker,
                    d["n"],
                    d["y_counts"][0],
                    d["y_counts"][1],
                    d["y_counts"][2],
                    d["first_ts"],
                    d["last_ts"],
                ])

            print(tabulate(
                rows,
                headers=["ticker", "n", "y=0", "y=1", "y=2", "first_ts", "last_ts"],
                tablefmt="github",
            ))

            # overall line
            o = out["overall"]
            print("\nOverall:")
            print(tabulate([[
                o["n"], o["y_counts"][0], o["y_counts"][1], o["y_counts"][2], o["first_ts"], o["last_ts"]
            ]],
                headers=["n", "y=0", "y=1", "y=2", "first_ts", "last_ts"],
                tablefmt="github",
            ))
        except ImportError:
            print("tabulate is not installed. Run: pip install tabulate")
        except Exception as e:
            print("Failed to display summary:", e)

    return out