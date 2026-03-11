from __future__ import annotations

from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    yt, yp, tids, tposs = [], [], [], []

    for batch in loader:
        x, y, tid, tpos = batch

        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

        yt.append(y.detach().cpu().numpy())
        yp.append(pred)

        # if tid is not None:
        tids.append(tid.detach().cpu().numpy())
        tposs.append(tpos.detach().cpu().numpy())

    out: Dict[str, Any] = {
        "y_true": np.concatenate(yt) if yt else np.asarray([], dtype=np.int64),
        "y_pred": np.concatenate(yp) if yp else np.asarray([], dtype=np.int64),
    }
    # if tids:
    out["tid"] = np.concatenate(tids).astype(np.int64, copy=False)
    out["tpos"] = np.concatenate(tposs).astype(np.int64, copy=False)
    return out