from typing import Any
from deel.puncc.typing import TensorLike, PredSetFunction
from deel.puncc._keras import ops, random

def constant_interval(y_pred:TensorLike, quantile:float) -> Any:
    lower_bounds = y_pred - quantile
    upper_bounds = y_pred + quantile
    return ops.stack([lower_bounds, upper_bounds], axis=-1)

def scaled_interval(eps:float=1e-12)->PredSetFunction:
    def _scaled_interval(y_pred:TensorLike, quantile:float) -> Any:
        mean_pred = y_pred[:, 0]
        var_pred = y_pred[:, 1]
        
        y_low = ops.zeros_like(mean_pred)
        y_high = ops.zeros_like(mean_pred)

        nonneg = var_pred + eps > 0

        y_low[nonneg] = mean_pred[nonneg] - quantile * (var_pred[nonneg] + eps)
        y_high[nonneg] = mean_pred[nonneg] + quantile * (var_pred[nonneg] + eps)
        y_low[~nonneg] = ops.ninf
        y_high[~nonneg] = ops.inf
        return ops.stack([y_low, y_high], axis=-1)
    return _scaled_interval

def cqr_interval(y_pred:TensorLike, quantile:float) -> Any:
    lower_pred = y_pred[:, 0]
    upper_pred = y_pred[:, 1]
    y_low = lower_pred - quantile
    y_high = upper_pred + quantile
    return ops.stack([y_low, y_high], axis=-1)

def lac_set(y_pred:TensorLike, quantile:float) -> Any:
    # FIXME : see the [-1] indexation for cases where y_pred.ndim > 2, change the use of .where
    return [ops.where(pred >= 1 - quantile)[-1] for pred in y_pred]

def raps_set(lambd:float=0, k_reg:int=1, rand:bool=False)->PredSetFunction:
    # TODO : I think this implementation is clearly suboptimal, see if it can be improved
    def _raps_set(y_pred:TensorLike, quantile:float) -> Any:
        sorted_index = ops.argsort(-y_pred, axis=-1)
        sorted_p = ops.take_along_axis(y_pred, sorted_index, axis=-1)
        cs = ops.cumsum(sorted_p, axis=-1)
        index_limit = ops.sum(cs < quantile, axis=-1) + 1
        if rand:
            sm = ops.max(ops.where(cs < quantile, cs, ops.zeros_like(cs)), axis=-1)
            sp = ops.min(ops.where(cs >= quantile, cs, ops.full_like(cs, ops.inf)), axis=-1)
            threshold = (quantile - sm) / (sp - sm)
            exclude_last = random.uniform(ops.shape(sp)) > threshold
            index_limit = index_limit - exclude_last
        return [p[:lim] for p, lim in zip(sorted_index, index_limit)]
    return _raps_set

aps_set = raps_set(lambd=0, k_reg=0, rand=False)