from flax import nnx

from earshot.models.dscnn import DSCNN
from earshot.models.gru import GRUKWS
from earshot.models.kwt import KWT

MODELS = {"dscnn": DSCNN, "gru": GRUKWS, "kwt": KWT}

# Per-architecture optimizer defaults (train.py flags override). Same epochs,
# batch size, schedule and augmentation for every model.
TRAIN_DEFAULTS = {
    "dscnn": {"lr": 3e-3, "weight_decay": 1e-4, "grad_clip": None},
    "gru": {"lr": 3e-3, "weight_decay": 1e-4, "grad_clip": 1.0},
    "kwt": {"lr": 1e-3, "weight_decay": 1e-2, "grad_clip": 1.0},
}


def build_model(name: str, num_classes: int, seed: int = 0, **kwargs) -> nnx.Module:
    return MODELS[name](num_classes=num_classes, **kwargs, rngs=nnx.Rngs(seed))


def count_params(model: nnx.Module) -> int:
    return sum(p[...].size for _, p in nnx.to_flat_state(nnx.state(model, nnx.Param)))
