# Keep this import to ensure all default ops are available
from deel.puncc._keras import keras
from keras.ops import *
import warnings

ops = keras.ops

# Other definitions dynamically added to ops :
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    inf = ops.convert_to_tensor(1.0) / ops.convert_to_tensor(0.0)
    ninf = -inf

def flatten(x):
    """Backend-agnostic equivalent of np.flatten(x)."""
    return ops.reshape(x, (-1,))

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
    
    indices = flatten(ops.where(mask))
    return ops.take(a, indices)

def weighted_quantile(x, q, weights=None, axis=None, keepdims=False):
    if weights is None:
        weights = ops.ones_like(x)
    if axis is None:
        x = flatten(x)
        weights = flatten(weights)
        axis = 0
    weights = weights / ops.sum(weights, axis=axis, keepdims=True)
    sorted_indices = ops.argsort(x, axis=axis)
    sorted_cumsum_weights = ops.cumsum(ops.take_along_axis(weights, sorted_indices, axis=axis), axis=axis)
    idx = ops.sum(sorted_cumsum_weights < q, axis=axis, keepdims=keepdims)
    sorted_a = ops.take_along_axis(x, sorted_indices, axis=axis)
    res = ops.take_along_axis(sorted_a, ops.expand_dims(idx, axis=axis), axis=axis)
    return ops.squeeze(res, axis=axis)

patchs = ["inf", "ninf", "flatten", "setdiff1d", "weighted_quantile"]
__all__ = [*dir(ops), *patchs]
