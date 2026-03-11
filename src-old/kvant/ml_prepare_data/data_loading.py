from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _load_jsonl(path: Path) -> List[Optional[dict]]:
    """
    Loads a JSONL file where each line is either:
      - "null"  -> None
      - JSON object -> dict
    """
    out: List[Optional[dict]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class PreparedStore:
    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.tickers_all = json.loads((exp_dir / "tickers_all.json").read_text())
        self.ticker_to_id = {t: i for i, t in enumerate(self.tickers_all)}

        self._features: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._timestamps: List[np.ndarray] = []
        self._label_metadata: List[List[Optional[dict]]] = []

        for t in self.tickers_all:
            tdir = exp_dir / "tickers" / t
            X = np.load(tdir / "features.npy", mmap_mode="r")
            y = np.load(tdir / "labels.npy", mmap_mode="r")
            ts = np.load(tdir / "timestamps.npy", mmap_mode="r")

            meta_path = tdir / "label_metadata.jsonl"
            if meta_path.exists():
                md = _load_jsonl(meta_path)
                if len(md) != len(y):
                    raise RuntimeError(
                        f"{t}: label_metadata length {len(md)} != labels length {len(y)}"
                    )
            else:
                md = [None] * int(len(y))

            self._features.append(X)
            self._labels.append(y)
            self._timestamps.append(ts)
            self._label_metadata.append(md)

        self.n_features = int(self._features[0].shape[1])

    def window_and_label(self, tid: int, tpos: int, L: int) -> Tuple[np.ndarray, int]:
        X = self._features[tid]
        y = self._labels[tid]
        x_win = X[tpos - L : tpos, :]  # (L,F)
        label = int(y[tpos])
        return x_win, label

    def metadata(self, tid: int, tpos: int) -> Optional[dict]:
        return self._label_metadata[tid][tpos]

    def metadata_for_index(self, index: np.ndarray) -> List[Optional[dict]]:
        """
        index: (N,2) array of (tid, tpos). Returns list[Optional[dict]] aligned to rows.
        """
        out: List[Optional[dict]] = []
        for i in range(int(index.shape[0])):
            tid = int(index[i, 0])
            tpos = int(index[i, 1])
            out.append(self.metadata(tid, tpos))
        return out

class IndexWindowDataset(Dataset):
    def __init__(self, store: PreparedStore, index: np.ndarray, lookback_L: int):
        self.store = store
        self.index = index
        self.L = int(lookback_L)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int):
        tid, tpos = int(self.index[i, 0]), int(self.index[i, 1])
        x_win, y = self.store.window_and_label(tid, tpos, self.L)

        x_np = np.array(x_win, dtype=np.float32, copy=True)  # (L, F)
        x_t = torch.from_numpy(x_np.T).contiguous()  # (F, L)

        y_t = torch.as_tensor(y, dtype=torch.long)

        tid_t = torch.tensor(tid, dtype=torch.int32)
        tpos_t = torch.tensor(tpos, dtype=torch.int32)
        return x_t, y_t, tid_t, tpos_t

    def get_id(self, i: int) -> tuple[int, int]:
        """i is the global index of a sample, i.e., dataset[i]."""
        tid, tpos = int(self.index[i, 0]), int(self.index[i, 1])
        return tid, tpos

    def get_info(self, i: int) -> dict:
        tid, tpos = self.get_id(i)
        return {
            "tid": tid,
            "tpos": tpos,
            "ticker": self.store.ticker(tid),
            "timestamp": self.store.timestamp(tid, tpos),
            "label_metadata": self.store.metadata(tid, tpos),
        }

    def summary(self, display : bool = True):
        from kvant.ml_prepare_data.data_loading_utils import summary
        return summary(self, display=display)


import os

class PreparedExperiment:
    """
    Owns: config + store + split indices.
    Provides: get_datasets() and get_loaders() returning train/val/test.
    """
    @classmethod
    def does_experiment_exist(cls, exp_dir: Path) -> bool:
        return os.path.isfile(exp_dir / "config.json")

    def __init__(self, exp_dir: Path):
        self.exp_dir = exp_dir
        self.cfg = json.loads((exp_dir / "config.json").read_text())
        self.L = int(self.cfg["lookback_L"])

        self.store = PreparedStore(exp_dir)

        self.index_train = np.asarray(np.load(exp_dir / "index_train.npy", mmap_mode="r"))
        self.index_val = np.asarray(np.load(exp_dir / "index_val.npy", mmap_mode="r"))
        self.index_test = np.asarray(np.load(exp_dir / "index_test.npy", mmap_mode="r"))

    def get_datasets(self) -> Tuple[IndexWindowDataset, IndexWindowDataset, IndexWindowDataset]:
        ds_train = IndexWindowDataset(self.store, self.index_train, self.L)
        ds_val = IndexWindowDataset(self.store, self.index_val, self.L)
        ds_test = IndexWindowDataset(self.store, self.index_test, self.L)
        return ds_train, ds_val, ds_test

    def get_loaders(
            self,
            train_batch_size: int = 256,
            eval_batch_size: int = 512,
            num_workers: int = 0,
            pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        ds_train, ds_val, ds_test = self.get_datasets()

        dl_train = DataLoader(
            ds_train,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=eval_batch_size,
            shuffle=False,  # keep this for alignment / reproducibility
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return dl_train, dl_val, dl_test

    def get_split_metadata(self) -> Tuple[List[Optional[dict]], List[Optional[dict]], List[Optional[dict]]]:
        metas_train = self.store.metadata_for_index(self.index_train)
        metas_val = self.store.metadata_for_index(self.index_val)
        metas_test = self.store.metadata_for_index(self.index_test)
        return metas_train, metas_val, metas_test
