from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .evaluator import ExperimentEvaluator


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_batch_size: int = 256
    eval_batch_size: int = 512
    checkpoint_metric: str = "val/accuracy"

    # NEW: run full evaluator only every N epochs (but still log loss+accuracy each epoch)
    full_eval_every: int = 10


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        evaluator: Optional[ExperimentEvaluator] = None,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.evaluator = evaluator
        self.logger = logger

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x, y = batch[0], batch[1]  # ignore tid/tpos

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def accuracy_only(self, loader: DataLoader) -> float:
        """
        Fast accuracy computation only (no storing all preds, no metadata, no per-ticker).
        Batch is (x, y, tid, tpos) but we ignore ids.
        """
        self.model.eval()
        n_correct = 0
        n_total = 0
        counts = {l : 0 for l in range(3)}
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            pred = torch.argmax(self.model(x), dim=1)
            n_correct += int((pred == y).sum().item())
            n_total += int(y.numel())
            for l in counts:
                counts[l] += sum(pred == l)

        return float(n_correct / max(n_total, 1))

    def fit(
        self,
        *,
        train_loader: DataLoader,
        train_eval_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        cfg: TrainConfig,
    ) -> Dict[str, Any]:
        best_state = None
        best_metric = -float("inf")

        import time
        import numpy as np
        from collections import defaultdict

        tspend = defaultdict(list)

        def do_full_eval(epoch: int) -> bool:
            every = max(1, int(cfg.full_eval_every))
            # common pattern: always do epoch 1 + last, plus every N
            return (epoch % every == 0) or (epoch == 1) or (epoch == cfg.epochs)

        for ep in range(1, cfg.epochs + 1):
            # ---- train
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader)
            tspend["train"].append(time.time() - t0)

            if self.logger is not None:
                self.logger.log({"train/loss": train_loss}, step=ep)

            # ---- evaluation
            metrics: Dict[str, Any] = {}

            full_eval = (self.evaluator is not None) and do_full_eval(ep)
            t0 = time.time()

            if full_eval:
                # Heavy eval (per-ticker, profit, confusion matrices, etc.)
                loaders = {}
                if train_eval_loader is not None:
                    loaders["train"] = train_eval_loader
                if val_loader is not None:
                    loaders["val"] = val_loader
                if test_loader is not None:
                    loaders["test"] = test_loader

                # evaluator will log a lot of stuff; that's intended only on full eval steps
                metrics = self.evaluator.evaluate_all(self.model, loaders, step=ep)
                # (metrics likely contains val/accuracy etc., which is fine)

            else:
                if val_loader is not None:
                    metrics["val/accuracy"] = self.accuracy_only(val_loader)

            tspend["eval"].append(time.time() - t0)

            # ---- checkpointing uses the always-available accuracy metric
            metric_val = float(metrics.get(cfg.checkpoint_metric, -float("inf")))
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            # Log time spend on the train vs. eval.
            totals = {k: sum(v) for k, v in tspend.items()}
            totals = {k + "(pct)" : v / sum(totals.values()) for k, v in totals.items()}
            metrics = {'train/epoch' : np.mean( tspend['train'] )} | totals
            metrics = [f"{k}: {np.mean(v):.2f}" for k, v in metrics.items() ]

            print(
                f"epoch={ep:04d} train_loss={train_loss:.4f} "
                f"{cfg.checkpoint_metric}={metric_val:.4f} best={best_metric:.4f} "
                f"[{' '.join(metrics)}]"
            )

        return {"best_state": best_state, "best_metric": best_metric}