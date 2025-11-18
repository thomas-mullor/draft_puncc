import os
import warnings
from deel.puncc.config import _freeze_backend_flag, set_backend
from typing import Callable

# Requires keras >= 3.3 for numpy backend support
_MIN_KERAS = (3, 3, 0)

def _parse_version(v: str) -> tuple[int, int, int]:
    parts = v.split(".")
    try:
        return tuple(int(x) for x in parts[:3])
    except Exception:
        return (0, 0, 0)

_defaulted_to_numpy = False
if not os.environ.get("KERAS_BACKEND"):
    set_backend("numpy")
    _defaulted_to_numpy = True

# Keras import for the whole module
import keras

_kver = getattr(keras, "__version__", "0.0.0")
if _parse_version(_kver) < _MIN_KERAS:
    raise RuntimeError(
        f"Keras {_kver} detected. This library requires Keras >= {'.'.join(map(str, _MIN_KERAS))} "
        "to use the 'numpy' backend. Upgrade with: pip install -U keras"
    )

# Prefer float32 where supported. Ignore if not available.
try:
    keras.backend.set_floatx("float32")
except Exception:
    pass

# Warn once if we silently defaulted to numpy
if _defaulted_to_numpy and keras.backend.backend() == "numpy":
    warnings.warn(
        "Puncc: no backend configured; defaulting to 'numpy'. "
        "Set explicitly with: deel.puncc.config.set_backend('torch'|'jax'|'tensorflow'|'numpy')",
        RuntimeWarning,
        stacklevel=2,
    )

# Freeze backend choice for the remainder of the process
_freeze_backend_flag()

# Public handles reused everywhere else
ops = keras.ops
backend = keras.backend
random = keras.random
BACKEND_NAME = backend.backend()  # "numpy" | "torch" | "jax" | "tensorflow"

# Other definitions dynamically added to ops :
# TODO : maybe define a deel.puncc.ops for such utilities ?

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ops.inf = ops.convert_to_tensor(1.0) / ops.convert_to_tensor(0.0)
    ops.ninf = -ops.inf

def flatten(x):
    """Backend-agnostic equivalent of np.flatten(x)."""
    return ops.reshape(x, (-1,))
ops.flatten = flatten

def setdiff1d(a, b):
    """Backend-agnostic equivalent of np.setdiff1d(a, b)."""
    # Ensure both are 1D tensors
    a = ops.reshape(a, (-1,))
    b = ops.reshape(b, (-1,))

    # For each element in a, check if it exists in b
    isin = ops.any(ops.expand_dims(a, -1) == b, axis=-1)
    mask = ~isin

    # Gather the elements where mask == True
    # There’s no boolean_mask, but we can simulate it with ops.where and indexing
    
    indices = ops.flatten(ops.where(mask))
    return ops.take(a, indices)
ops.setdiff1d = setdiff1d

def weighted_quantile(x, q, weights=None, axis=None, keepdims=False):
    if weights is None:
        weights = ops.ones_like(x)
    if axis is None:
        x = ops.flatten(x)
        weights = ops.flatten(weights)
        axis = 0
    weights = weights / ops.sum(weights, axis=axis, keepdims=True)
    sorted_indices = ops.argsort(x, axis=axis)
    sorted_cumsum_weights = ops.cumsum(ops.take_along_axis(weights, sorted_indices, axis=axis), axis=axis)
    idx = ops.sum(sorted_cumsum_weights < q, axis=axis, keepdims=keepdims)
    sorted_a = ops.take_along_axis(x, sorted_indices, axis=axis)
    res = ops.take_along_axis(sorted_a, ops.expand_dims(idx, axis=axis), axis=axis)
    return ops.squeeze(res, axis=axis)


ops.weighted_quantile = weighted_quantile