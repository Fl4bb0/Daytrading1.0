from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kvant.ml_prepare_data.data_loading import PreparedExperiment
from kvant.ml_framework.models import Conv1DClassifier
from kvant.ml_framework.train import Trainer, TrainConfig, ExperimentEvaluator, EvalConfig
from kvant.ml_framework.train.utils import class_weights_from_dataset
from kvant.ml_framework.logging import WandbLogger

import os
from dotenv import load_dotenv

load_dotenv()
project = os.environ.get("WANDB_PROJECT", "kvant-stocks")
entity = os.environ.get("WANDB_ENTITY", None)


def parse_args() -> argparse.Namespace:
    exp_id = "11bc7f8b735e5936"
    exp_id = "sb_L_400_w120_h2pct"
    exp_id = "sb_L_40_w120_h1.5_TBPD20"
    from kvant.ml_prepare_data import prepared_data_root
    with open(prepared_data_root / "last_experiment.txt", 'r') as f:
        exp_id = f.read().strip()

    from kvant.ml_prepare_data import prepared_data_root
    default_exp_dir = prepared_data_root / exp_id
    # default_exp_dir = Path("../src/kvant/ml_framework/prepared") / exp_id

    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=False, default=default_exp_dir)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--wandb-project", type=str, default="kvant-stocks")
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--no-return-stats", action="store_true")
    p.add_argument("--topk-ticker-plots", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    exp = PreparedExperiment(args.exp_dir)
    dl_train, dl_val, dl_test = exp.get_loaders(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # Train eval loader (no shuffle) so "train" metrics are stable and comparable
    dl_train_eval = DataLoader(
        dl_train.dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Optional local sanity check
    ds_train, ds_val, ds_test = exp.get_datasets()

    for ds, l in [(ds_train, "train"), (ds_val, "val"), (ds_test, "test")]:
        print(f"Dataset {l}")
        ds.summary(display=True)
        print("-"*10,"\n")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv1DClassifier(n_features=exp.store.n_features, n_classes=3).to(device)

    w = class_weights_from_dataset(ds_train, n_classes=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger = WandbLogger(
        project=project,
        entity=entity,
        name=args.wandb_name or "stocks-run",
        config={
            "exp_dir": str(args.exp_dir),
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "L": exp.L,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "class_weights": w.tolist(),
        },
    )

    # Log static dataset stats + configure ticker charts
    logger.setup(
        exp=exp,
        loaders={"train": dl_train_eval, "val": dl_val, "test": dl_test},
    )

    evaluator = ExperimentEvaluator(
        store=exp.store,
        device=device,
        logger=logger,
        cfg=EvalConfig(
            compute_per_ticker_accuracy=True,
        ),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        evaluator=evaluator,
        logger=logger,
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        checkpoint_metric="val/accuracy",
    )

    out = trainer.fit(
        train_loader=dl_train,
        train_eval_loader=dl_train_eval,
        val_loader=dl_val,
        test_loader=dl_test,
        cfg=cfg,
    )

    # Restore best checkpoint
    if out["best_state"] is not None:
        model.load_state_dict(out["best_state"])

    # Final evaluation (log at step=epochs+1)
    evaluator.evaluate_all(
        model,
        {"train": dl_train_eval, "val": dl_val, "test": dl_test},
        step=args.epochs + 1,
    )
    logger.stop()

if __name__ == "__main__":
    main()