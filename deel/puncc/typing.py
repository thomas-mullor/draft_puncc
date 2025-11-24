from typing import TYPE_CHECKING, Any, Union, TypeAlias, Protocol, runtime_checkable, Iterable, Callable
from deel.puncc.cloning import clone_model

if TYPE_CHECKING:
    import numpy as _np
    import torch as _torch
    import tensorflow as _tf
    from jax import Array as _JaxArray
    TensorLike: TypeAlias = Union[_np.ndarray, _torch.Tensor, _tf.Tensor, _JaxArray]
else:
    TensorLike = Any

@runtime_checkable
class Predictor(Protocol):
    def __call__(self, X: Iterable[Any]) -> TensorLike:
        ...

@runtime_checkable
class Fitable(Protocol):
    def fit(self, X: Iterable[Any], y: TensorLike):
        ...

@runtime_checkable
class PredictorLike(Protocol):
    def predict(self, X: Iterable[Any]) -> TensorLike:
        ...

class CallablePredictorMixin(Predictor):
    def __call__(self, X: Iterable[Any], *args, **kwargs) -> TensorLike:
        return self.predict(X, *args, **kwargs)
    
class _PredictorAdapter:
    """Wraps a .predict(...) provider into a callable."""
    def __init__(self, model: PredictorLike) -> None:
        self._model = model

    def __call__(self, x: Iterable[Any], /, *args: Any, **kwargs: Any) -> Any:
        return self._model.predict(x, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)
    
    def __setattr__(self, name, value):
        if name == "_model":
            super().__setattr__(name, value)
        else:
            setattr(self._model, name, value)

    def clone(self):
        return _PredictorAdapter(clone_model(self._model))

def make_predictor(model: Union[Predictor, PredictorLike]) -> Predictor:
    if isinstance(model, Predictor):
        return model
    if isinstance(model, PredictorLike):
        predictor = _PredictorAdapter(model)
        return predictor
    raise TypeError("The provided model neither have __call__ nor predict method.")

NCScoreFunction:TypeAlias = Callable[[TensorLike, TensorLike], Iterable[float]]
PredSetFunction:TypeAlias = Callable[[TensorLike, float|TensorLike], Iterable[Any]]
