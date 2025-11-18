from typing import Iterable
import warnings
from deel.puncc.typing import TensorLike, NCScoreFunction
from deel.puncc._keras import ops, random

# TODO : rename stuff in this file to name differently NCS function and NCS function generators.

def absolute_difference(y_pred:TensorLike, y_true:TensorLike) -> Iterable[float]:
    return ops.abs(y_pred - y_true)

def scaled_ad(eps:float=1e-12)-> NCScoreFunction:
    def _scaled_ad(y_pred:TensorLike, y_true:TensorLike) -> Iterable[float]:
        mean_pred = y_pred[:, 0]
        var_pred = y_pred[:, 1]
        mean_abs_dev = ops.abs(mean_pred - y_true)
        if ops.any(var_pred + eps <= 0):
            warnings.warn("Warning: calibration points with MAD predictions below -eps won't be used for calibration.",
                RuntimeWarning,
                stacklevel=2,
            )
        nonneg = var_pred + eps > 0
        return mean_abs_dev[nonneg] / (var_pred[nonneg] + eps)
    return _scaled_ad

def cqr_score(y_pred:TensorLike, y_true:TensorLike) -> Iterable[float]:
    lower_pred = y_pred[:, 0]
    upper_pred = y_pred[:, 1]
    return ops.maximum(lower_pred - y_true, y_true - upper_pred)

def scaled_bbox_difference(y_pred:TensorLike, y_true:TensorLike) -> Iterable[float]:
    x_min, y_min, x_max, y_max = ops.unstack(y_pred, axis=1)
    dx = ops.abs(x_max - x_min)
    dy = ops.abs(y_max - y_min)
    return (y_pred - y_true) / ops.hstack([dx, dy, dx, dy])

def lac_score(y_pred:TensorLike, y_true:TensorLike) -> Iterable[float]:
    return 1 - y_pred[ops.arange(ops.shape(y_true)[0]), y_true]

def raps_score(lambd:float=0, k_reg:int=1, rand:bool=True)->NCScoreFunction:
    def _raps_score(y_pred:TensorLike, y_true:TensorLike) -> Iterable[float]:
        condition = y_pred>=ops.take_along_axis(y_pred, y_true[..., None], axis=-1)
        s = ops.sum(ops.where(condition, y_pred, 0), axis=-1)
        nb_cum_elems = ops.sum(condition, axis=-1)
        regul = lambd * ops.maximum(nb_cum_elems - k_reg, ops.zeros_like(nb_cum_elems))
        rand_correction = 0
        if rand:
            u = random.uniform(ops.shape(s))
            rand_correction = u * ops.take_along_axis(y_pred, y_true[..., None], axis=-1).squeeze(-1)
        return s + regul - rand_correction
    return _raps_score

aps_score = raps_score(lambd=0, k_reg=0)