from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def class_weights_from_dataset(ds: Dataset, n_classes: int = 3) -> np.ndarray:
    ys = []
    for batch in DataLoader(ds, batch_size=4096, shuffle=False):
        # batch can be (x,y) or (x,y,tid,tpos)
        y = batch[1]
        ys.append(y.detach().cpu().numpy())

    y_all = np.concatenate(ys) if ys else np.asarray([], dtype=np.int64)
    if len(y_all) == 0:
        return np.ones(n_classes, dtype=np.float32)

    counts = np.bincount(y_all, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / counts
    return (w / w.mean()).astype(np.float32)