"""
training.trainer — Abstract base for the training loop.

Concrete implementations:
  - PytorchTrainer   (pytorch_trainer.py)
  - SklearnTrainer   (sklearn_trainer.py)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

from kvant.models.base import KvantModel


@dataclass
class TrainConfig:
    """Hyperparameters that every trainer understands."""
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    checkpoint_dir: Optional[Path] = None
    lr_schedule: str = "cosine"  # "none" | "cosine"
    extra: Dict[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    """
    Owns the fit loop, checkpointing, and W&B logging.
    Decoupled from any specific model or framework.
    """

    def __init__(self, model: KvantModel, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run the full training loop and return a history dict."""
        ...

    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Return a dict of metric_name → value for the given split."""
        ...
