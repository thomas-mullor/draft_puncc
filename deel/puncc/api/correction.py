from deel.puncc.typing import TensorLike
from deel.puncc import ops
from typing import TypeAlias, Callable

CorrectionFunction:TypeAlias = Callable[[float|TensorLike], float|TensorLike]

def bonferroni(nvars:int=1)->CorrectionFunction:
    def _bonferroni(alpha: float | TensorLike) -> float | TensorLike:
        if nvars == 1:
            return alpha
        return ops.ones(nvars) * alpha / nvars
    return _bonferroni

def weighted_bonferroni(weights: TensorLike) -> CorrectionFunction:
    def _weighted_bonferroni(alpha: float | TensorLike) -> float | TensorLike:
        # normalization of weights
        w = weights / ops.sum(weights)
        return alpha * w
    return _weighted_bonferroni

def sidak(nvars:int=1)->CorrectionFunction:
    def _sidak(alpha: float | TensorLike) -> float | TensorLike:
        if nvars == 1:
            return alpha
        return ops.ones(nvars) * (1 - (1 - alpha) ** (1 / nvars))
    return _sidak