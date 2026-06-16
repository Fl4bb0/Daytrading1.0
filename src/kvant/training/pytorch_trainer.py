"""
training.pytorch_trainer — Concrete Trainer for all PyTorch KvantModel subclasses.

Wraps the raw nn.Module inside a KvantModel and owns:
  - the train/eval loop with early stopping
  - per-epoch logging (stdout + optional W&B)
  - best-checkpoint saving / restoring
  - class-weight-balanced CrossEntropyLoss

Usage
-----
    from kvant.models import Conv1DModel, MODEL_REGISTRY
    from kvant.training.pytorch_trainer import PytorchTrainer
    from kvant.training.trainer import TrainConfig

    model  = Conv1DModel(n_features=10, n_classes=3)
    cfg    = TrainConfig(epochs=100, learning_rate=1e-3, batch_size=256)
    trainer = PytorchTrainer(model, cfg)

    history = trainer.fit(X_train, y_train, X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from typing import runtime_checkable, Protocol

from kvant.models.base import KvantModel
from kvant.training.trainer import Trainer, TrainConfig
from kvant.training.metrics import classification_metrics
from kvant.training.predict import predict_loader


# ---------------------------------------------------------------------------
# Protocol — any KvantModel that owns a raw nn.Module
# ---------------------------------------------------------------------------

@runtime_checkable
class _HasNet(Protocol):
    net: nn.Module
    device: torch.device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    return DataLoader(
        TensorDataset(tx, ty),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
    )


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    # sqrt-dampened inverse frequency: protects rare classes without letting
    # extreme rarity (e.g. HOLD at <1% of labels) dominate the loss and flood
    # predictions toward that class.
    w = np.sqrt(counts.sum() / counts)
    w = (w / w.mean()).astype(np.float32)
    return torch.tensor(w)


# ---------------------------------------------------------------------------
# PytorchTrainer
# ---------------------------------------------------------------------------

class PytorchTrainer(Trainer):
    """
    Full training loop for any KvantModel whose ``.net`` is a ``torch.nn.Module``.

    Parameters
    ----------
    model  : KvantModel  — must expose a ``.net`` attribute (nn.Module) and a ``.device``.
    cfg    : TrainConfig — hyperparameters.
    logger : optional    — any object with a ``.log(dict, step=int)`` method (e.g. wandb.run).
    """

    def __init__(
        self,
        model: KvantModel,
        cfg: TrainConfig,
        logger: Optional[Any] = None,
    ) -> None:
        super().__init__(model, cfg)
        if not isinstance(model, _HasNet):
            raise TypeError(
                f"PytorchTrainer requires model to have a .net (nn.Module) attribute, "
                f"got {type(model).__name__}"
            )
        self._pytorch_model: _HasNet = model
        self.logger = logger
        self._device: torch.device = model.device

    # ------------------------------------------------------------------
    # Trainer interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model and return a history dict with keys:
          ``train_loss``, ``val_accuracy`` (if val provided),
          ``best_val_accuracy``, ``best_epoch``.
        """
        net: nn.Module = self._pytorch_model.net
        cfg = self.cfg

        pin_memory = self._device.type == "cuda"
        train_loader = _to_loader(
            X_train,
            y_train,
            cfg.batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )
        val_loader = (
            _to_loader(X_val, y_val, cfg.batch_size, pin_memory=pin_memory)
            if X_val is not None
            else None
        )

        n_classes = int(y_train.max()) + 1
        weights   = _class_weights(y_train, n_classes).to(self._device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=cfg.learning_rate, weight_decay=1e-5
        )
        scheduler = None
        if getattr(cfg, "lr_schedule", "none") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.01
            )

        history: Dict[str, List[float]] = defaultdict(list)
        best_metric   = -float("inf")
        best_state    = None
        best_epoch    = 0
        patience_left = cfg.early_stopping_patience

        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(net, train_loader, optimizer, criterion)
            history["train_loss"].append(train_loss)

            val_f1 = 0.0
            if val_loader is not None:
                val_f1 = self._macro_f1(net, val_loader)
                history["val_macro_f1"].append(val_f1)

            elapsed = time.time() - t0

            # Checkpoint on best macro-F1 (or train_loss if no val).
            # Accuracy is misleading with imbalanced HOLD/BUY/SHORT — F1 is not.
            metric = val_f1 if val_loader is not None else -train_loss
            if metric > best_metric:
                best_metric = metric
                best_epoch  = ep
                best_state  = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                patience_left = cfg.early_stopping_patience
            else:
                patience_left -= 1

            print(
                f"epoch={ep:04d}  loss={train_loss:.4f}  "
                f"val_f1={val_f1:.4f}  best={best_metric:.4f}  "
                f"[{elapsed:.1f}s]"
            )

            if self.logger is not None:
                self.logger.log(
                    {"train/loss": train_loss, "val/macro_f1": val_f1},
                    step=ep,
                )

            if scheduler is not None:
                scheduler.step()

            if patience_left <= 0:
                print(f"Early stopping at epoch {ep} (best epoch {best_epoch})")
                break

        # Restore best weights
        if best_state is not None:
            net.load_state_dict(best_state)
            if cfg.checkpoint_dir is not None:
                self.model.save(Path(cfg.checkpoint_dir))

        return {
            "train_loss":        history["train_loss"],
            "val_macro_f1":      history.get("val_macro_f1", []),
            "best_val_macro_f1": best_metric if val_loader is not None else None,
            "best_epoch":        best_epoch,
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Return classification metrics for the given split."""
        loader = _to_loader(X, y, self.cfg.batch_size)
        out = predict_loader(self._pytorch_model.net, loader, self._device)
        metrics = classification_metrics(out["y_true"], out["y_pred"])
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        net: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        net.train()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            x, y = batch[0].to(self._device), batch[1].to(self._device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _macro_f1(self, net: nn.Module, loader: DataLoader) -> float:
        net.eval()
        all_pred, all_true = [], []
        for batch in loader:
            x, y = batch[0].to(self._device), batch[1].to(self._device)
            all_pred.append(net(x).argmax(dim=1).cpu().numpy())
            all_true.append(y.cpu().numpy())
        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
