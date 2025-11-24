from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable
import pickle
from deel.puncc.typing import Predictor, PredictorLike, TensorLike, NCScoreFunction, PredSetFunction
from deel.puncc._keras import ops
from deel.puncc.api.conformalization_procedure import ConformalizationProcedure

class NoModel(Predictor):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("When loading a ConformalPredictor, the model must be set manually after loading. The model was not saved to avoid issues with model serialization. Please set the model attribute of the loaded ConformalPredictor instance to a valid model before using it.")

class ConformalPredictor(ConformalizationProcedure):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 nc_score_function:NCScoreFunction,
                 pred_set_function: PredSetFunction,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        # Definition of conformal predictor components :
        super().__init__(model=model)
        self.nc_score_function = nc_score_function
        self.pred_set_function = pred_set_function

        self.weight_function = weight_function
        self.fit_function = fit_function

        # Utilities for the calibration procedure :
        self._nc_scores = None

    @property
    def len_calibr(self):
        if self._nc_scores is None:
            return 0
        return len(self._nc_scores)
    
    @property
    def nc_scores(self) -> Iterable[float]:
        if self._nc_scores is None:
            raise RuntimeError("The conformal predictor has not been calibrated yet. Please use `my_predictor.calibrate(X, y)` before performing a prediction or accessing the non conformity scores.")
        return self._nc_scores

    def calibrate(self, X_calib:Iterable[Any],
                  y_calib:TensorLike):
        predictions = self.model(X_calib)
        self._nc_scores = self.nc_score_function(predictions, y_calib)
        return self

    def fit(self,
            X_train:Iterable[Any],
            y_train:TensorLike,
            X_calib:Iterable[Any]|None=None,
            y_calib:TensorLike|None=None):
        if self.fit_function is not None:
            self.fit_function(self.model, X_train, y_train)
        elif callable(getattr(self.model, "fit", None)):
            self.model.fit(X_train, y_train)
        else:
            raise NotImplementedError("The model does not have a fit method and no fit_function was provided. Please provide a pretrained model or a fit_function.")
        if X_calib is not None and y_calib is not None:
            self.calibrate(X_calib, y_calib)
        return self

    def predict(self,
                X_test:Iterable[Any],
                alpha:float,
                correction:Callable|None = None)->tuple[TensorLike, Any]:
        # TODO : apply correction
        prediction = self.model(X_test)
        n = self.len_calibr
        weights = None
        if correction is not None:
            alpha = correction(alpha) # TODO : add kwargs ?
        if self.weight_function is not None:
            weights = self.weight_function(X_test)
        quantile = ops.weighted_quantile(self.nc_scores, (1 - alpha) * (n + 1) / n, axis=0, weights=weights)
        prediction_sets = self.pred_set_function(prediction, quantile)
        return prediction, prediction_sets

    def __getstate__(self):
        state = {}
        if getattr(self, "__dict__", None):
            state = self.__dict__.copy()
        for cls in type(self).mro():
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name == "__dict__":
                    continue
                if hasattr(self, name):
                    state[name] = getattr(self, name)
        # Remove the model from the state to avoid serialization issues
        state["model"] = NoModel()
        return state
    
    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

    def save(self, path:Path|str)->None:
        with open(path, "wb") as f:
            pickle.dump(self.__getstate__(), f)

    @classmethod
    def load(cls, path:Path|str)->ConformalPredictor:
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj

class AutoConformalPredictor(ConformalPredictor):
    nc_score_function:NCScoreFunction
    pred_set_function:PredSetFunction
    def __init__(self, model, weight_function=None, fit_function=None):
        super().__init__(
            model=model,
            nc_score_function=type(self).nc_score_function,
            pred_set_function=type(self).pred_set_function,
            weight_function=weight_function,
            fit_function=fit_function,
        )

class ScoreCalibrator():
    def __init__(self,
                 nc_score_function:NCScoreFunction,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None):
        # Definition of conformal predictor components :
        self.nc_score_function = nc_score_function
        self.weight_function = weight_function

        # Utilities for the calibration procedure :
        self._nc_scores = None

    @property
    def len_calibr(self):
        if self._nc_scores is None:
            return 0
        return len(self._nc_scores)

    @property
    def nc_scores(self) -> Iterable[float]:
        if self._nc_scores is None:
            raise RuntimeError("The conformal predictor has not been calibrated yet. Please use the `calibrate` method before performing a prediction or accessing the non conformity scores.")
        return self._nc_scores

    def calibrate(self, z_calib:Iterable[Any]):
        self._nc_scores = self.nc_score_function(z_calib)
        return self

    def is_conformal(self, z:Iterable[Any], alpha:float)->TensorLike:
        n = self.len_calibr
        weights = None
        if self.weight_function is not None:
            weights = self.weight_function(z)
        quantile = ops.weighted_quantile(self.nc_scores, (1 - alpha) * (n + 1) / n, axis=0, weights=weights)
        test_nonconf_scores = self.nc_score_function(z)
        return test_nonconf_scores <= quantile

