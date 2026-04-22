# models — model definitions (price-only, text-only, fusion)
from kvant.models.base import KvantModel
from kvant.models.conv1d import Conv1DModel
from kvant.models.conv3d import Conv3DModel
from kvant.models.ensemble import AveragingEnsembleModel
from kvant.models.resnls import ResNLSModel
from kvant.models.shallow_cnn import ShallowCNNModel
from kvant.models.tsb import TSBModel

# Registry: map name string → class, used by run_train.py and sweeps
MODEL_REGISTRY: dict[str, type[KvantModel]] = {
    "conv1d":  Conv1DModel,
    "conv3d":  Conv3DModel,
    "resnls":  ResNLSModel,
    "shallow_cnn": ShallowCNNModel,
    "tsb":     TSBModel,
}

__all__ = [
    "KvantModel",
    "Conv1DModel",
    "Conv3DModel",
    "AveragingEnsembleModel",
    "ResNLSModel",
    "ShallowCNNModel",
    "TSBModel",
    "MODEL_REGISTRY",
]
