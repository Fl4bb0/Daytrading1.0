"""
training.predict — Batch inference helpers for PyTorch models.

predict_loader(net, loader, device)  → Dict[str, np.ndarray]
    Keys: "y_true", "y_pred", and optionally "tid", "tpos"
    if the DataLoader yields 4-element batches (x, y, tid, tpos).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict_loader(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run inference over *loader* and collect predictions.

    Supports both 2-element batches ``(x, y)`` and 4-element batches
    ``(x, y, tid, tpos)`` as produced by the prepared-dataset loaders.

    Returns
    -------
    dict with keys:
      ``y_true`` : int64 array, shape (n,)
      ``y_pred`` : int64 array, shape (n,)
      ``tid``    : int64 array, shape (n,)  — only if batch has ≥ 3 elements
      ``tpos``   : int64 array, shape (n,)  — only if batch has ≥ 4 elements
    """
    net.eval()
    yt, yp, tids, tposs = [], [], [], []
    has_tid = False

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]

        pred = net(x).argmax(dim=1).detach().cpu().numpy()
        yt.append(y.detach().cpu().numpy())
        yp.append(pred)

        if len(batch) >= 3:
            has_tid = True
            tids.append(batch[2].detach().cpu().numpy())
        if len(batch) >= 4:
            tposs.append(batch[3].detach().cpu().numpy())

    out: Dict[str, Any] = {
        "y_true": np.concatenate(yt)  if yt  else np.asarray([], dtype=np.int64),
        "y_pred": np.concatenate(yp)  if yp  else np.asarray([], dtype=np.int64),
    }
    if has_tid:
        out["tid"]  = np.concatenate(tids).astype(np.int64)
    if tposs:
        out["tpos"] = np.concatenate(tposs).astype(np.int64)
    return out
