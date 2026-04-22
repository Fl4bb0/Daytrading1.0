"""
models.base — Abstract base (KvantModel) that every model must implement.

This interface decouples training, evaluation, and inference from any
particular architecture. Adding a new model means creating a new file
(e.g. models/lstm.py) and subclassing KvantModel — nothing else changes.

Concrete implementations:
  - Conv1DModel       (conv1d.py)       — 1-D convolutional price model
  - TransformerModel  (transformer.py)  — attention-based sequence model
  - ResNLSModel       (resnls.py)       — residual NLS architecture
  - PriceTextFusion   (fusion.py)       — combines price features + LLM signal (future)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch.nn as nn


class KvantModel(ABC):
    """
    Unified interface for all trainable models in the kvant framework.

    Every method receives/returns plain numpy arrays so the interface
    stays framework-agnostic (PyTorch, sklearn, etc. all conform).

    PyTorch subclasses should expose a ``net: nn.Module`` attribute so
    :class:`~kvant.training.PytorchTrainer` can access the raw module.
    """

    # Declared here so type-checkers understand PytorchTrainer's model.net accesses.
    # Concrete PyTorch subclasses must set self.net in __init__.
    net: nn.Module

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in artifact paths and W&B run names."""
        ...

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Returns a dict of training history / metrics (e.g. {'val_loss': [...]}).
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions (int array, shape (n,)).
        """
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities (float array, shape (n, n_classes)).
        Override if the model supports soft predictions.

        Array format: rows are samples, columns are classes, values are probabilities.
        """
        raise NotImplementedError(f"{self.name} does not support predict_proba.")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights/state to `path` (directory or file)."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "KvantModel":
        """Restore a model from `path` and return the instance."""
        ...
