from typing import Any
from deel.puncc.typing import TensorLike, PredSetFunction
from deel.puncc import ops
from deel.puncc._keras import random


def _constant_interval(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    lower_bounds = y_pred - quantile
    upper_bounds = y_pred + quantile
    return ops.stack([lower_bounds, upper_bounds], axis=0)

def constant_interval()->PredSetFunction:
    return _constant_interval

def scaled_interval(eps:float=1e-12)->PredSetFunction:
    def _scaled_interval(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
        mean_pred = ops.take(y_pred, 0, axis=-1)
        var_pred = ops.take(y_pred, 1, axis=-1)
        
        y_low = ops.zeros_like(mean_pred)
        y_high = ops.zeros_like(mean_pred)

        nonneg = var_pred + eps > 0

        y_low[nonneg] = mean_pred[nonneg] - quantile * (var_pred[nonneg] + eps)
        y_high[nonneg] = mean_pred[nonneg] + quantile * (var_pred[nonneg] + eps)
        y_low[~nonneg] = ops.ninf
        y_high[~nonneg] = ops.inf
        return ops.stack([y_low, y_high], axis=0)
    return _scaled_interval

def _cqr_interval(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    lower_pred = y_pred[:, 0]
    upper_pred = y_pred[:, 1]
    y_low = lower_pred - quantile
    y_high = upper_pred + quantile
    return ops.stack([y_low, y_high], axis=0)

def cqr_interval()->PredSetFunction:
    return _cqr_interval

def _lac_set(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    # FIXME : see the [-1] indexation for cases where y_pred.ndim > 2, change the use of .where
    return [ops.where(pred >= 1 - quantile)[-1] for pred in y_pred]

def lac_set()->PredSetFunction:
    return _lac_set

def raps_set(lambd:float=0, k_reg:int=1, rand:bool=False)->PredSetFunction:
    # TODO : I think this implementation is clearly suboptimal, see if it can be improved
    def _raps_set(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
        sorted_index = ops.argsort(-y_pred, axis=-1)
        sorted_p = ops.take_along_axis(y_pred, sorted_index, axis=-1)
        cs = ops.cumsum(sorted_p, axis=-1)
        # FIXME : lol, i completely forgot to add the regularization term here
        
        index_limit = ops.sum(cs < quantile, axis=-1) + 1
        if rand:
            sm = ops.max(ops.where(cs < quantile, cs, ops.zeros_like(cs)), axis=-1)
            sp = ops.min(ops.where(cs >= quantile, cs, ops.full_like(cs, ops.inf)), axis=-1)
            threshold = (quantile - sm) / (sp - sm)
            exclude_last = random.uniform(ops.shape(sp)) > threshold
            index_limit = index_limit - exclude_last
        return [p[:lim] for p, lim in zip(sorted_index, index_limit)]
    return _raps_set

def aps_set(rand:bool=False)->PredSetFunction:
    return raps_set(lambd=1, k_reg=1, rand=rand)

def _constant_bbox(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    # Coordinates of covering bbox (upperbounds)
    x_min_lo, y_min_lo = x_min - quantile[0], y_min - quantile[1]
    x_max_hi, y_max_hi = x_max + quantile[2], y_max + quantile[3]
    Y_pred_hi = ops.hstack([x_min_lo, y_min_lo, x_max_hi, y_max_hi])

    # Coordinates of included bbox (lowerbounds)
    x_min_hi, y_min_hi = x_min + quantile[0], y_min + quantile[1]
    x_max_lo, y_max_lo = x_max - quantile[2], y_max - quantile[3]
    Y_pred_lo = ops.hstack([x_min_hi, y_min_hi, x_max_lo, y_max_lo])
    return ops.stack([Y_pred_lo, Y_pred_hi], axis=0)

def constant_bbox()->PredSetFunction:
    return _constant_bbox

def _scaled_bbox(y_pred:TensorLike, quantile:float|TensorLike) -> Any:
    print("quantile : ", quantile)
    x_min, y_min, x_max, y_max = ops.split(y_pred, 4, axis=1)
    dx = ops.abs(x_max - x_min)
    dy = ops.abs(y_max - y_min)
    qd = [quantile[0] * dx, quantile[1] * dy, quantile[2] * dx, quantile[3] * dy]
    # Coordinates of covering bbox (upperbounds)
    x_min_lo = x_min - qd[0]
    y_min_lo = y_min - qd[1]
    x_max_hi = x_max + qd[2]
    y_max_hi = y_max + qd[3]

    Y_pred_outer = ops.hstack([x_min_lo, y_min_lo, x_max_hi, y_max_hi])

    # Coordinates of included bbox (lowerbounds)
    x_min_hi, y_min_hi = (
        x_min + qd[0],
        y_min + qd[1],
    )
    x_max_lo, y_max_lo = (
        x_max - qd[2],
        y_max - qd[3],
    )
    Y_pred_inner = ops.hstack([x_min_hi, y_min_hi, x_max_lo, y_max_lo])

    return ops.stack([Y_pred_inner, Y_pred_outer], axis=0)

def scaled_bbox():
    return _scaled_bbox