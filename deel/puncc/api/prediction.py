from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from deel.puncc.typing import Predictor, PredictorLike, TensorLike, make_predictor
from deel.puncc import ops
from deel.puncc.cloning import clone_model

class MultiPredictorStack(Predictor):
    def __init__(self, models:Iterable[Predictor|PredictorLike]):
        self.models = [make_predictor(m) for m in models]

    def clone(self)->MultiPredictorStack:
        return MultiPredictorStack(models=[clone_model(model) for model in self.models])

    def __call__(self, X:Iterable[Any])->TensorLike:
        return ops.stack([model(X) for model in self.models], axis=-1)
    
    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        for model in self.models:
            if callable(getattr(model, "fit", None)):
                model.fit(X_train, y_train)
                continue
            raise NotImplementedError("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        return self

def stack_predictors(*models:Iterable[Predictor|PredictorLike])->MultiPredictorStack:
    return MultiPredictorStack(models=models)

class MeanVarPredictor(MultiPredictorStack):
    def __init__(self, mean_model:Predictor|PredictorLike,
                 var_model:Predictor|PredictorLike):
        super().__init__(models=[mean_model, var_model])

    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike):
        for model in self.models:
            if not callable(getattr(model, "fit", None)):
                raise NotImplementedError("One of the models does not have a fit method. Please provide pretrained models or expose a fit method.")
        self.models[0].fit(X_train, y_train)
        mu_pred = self.models[0](X_train)
        self.models[1].fit(X_train, ops.abs(mu_pred - y_train) )
        return self