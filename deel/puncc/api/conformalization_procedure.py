from abc import ABC, abstractmethod
from typing import Any, Iterable
from deel.puncc.typing import Predictor, PredictorLike, make_predictor, TensorLike


class ConformalizationProcedure(ABC):
    __slots__ = ("model",)
    def __init__(self, model:Predictor|PredictorLike):
        self.model = make_predictor(model)

    @abstractmethod
    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike):
        ...

    @abstractmethod
    def predict(self, X_test:Iterable[Any], alpha:float)->tuple[TensorLike, Any]:
        ...

class CRC(ConformalizationProcedure):
    ...

class RCPS(ConformalizationProcedure):
    ...