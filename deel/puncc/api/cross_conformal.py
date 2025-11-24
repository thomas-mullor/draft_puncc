from __future__ import annotations
from typing import Any, Iterable, Callable
from deel.puncc.api.conformalization_procedure import ConformalizationProcedure
from deel.puncc.typing import Predictor, PredictorLike, TensorLike
from deel.puncc.api.splitting import KFoldSplitter
from deel.puncc.cloning import clone_model
from deel.puncc.regression import SplitCP
from deel.puncc._keras import ops
import math

# TODO : this could be a daughter class of ConformalPredictor (with some really small adjustments)
class CrossConformalPredictor(ConformalizationProcedure):
    # TODO : see what can be generalized here ?
    ...

class CVPlusRegressor(CrossConformalPredictor):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 K:int=5,
                 random_state:int|None=None,
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        super().__init__(model)
        self.splitter = KFoldSplitter(K=K, shuffle=True)
        self.random_state = random_state
        self.weight_function = weight_function
        self.fit_function = fit_function
        self._conformal_predictors = []

    @property
    def len_calibr(self)->int:
        return sum(cp.len_calibr for cp in self._conformal_predictors)
    
    @property
    def nc_scores(self) -> Iterable[float]:
        all_scores = []
        for n, cp in enumerate(self._conformal_predictors):
            try:
                all_scores.extend(cp.nc_scores)
            except RuntimeError as e:
                raise RuntimeError(f"The conformal predictor number {n} has not been calibrated yet. Please use `my_predictor.fit(X, y)` before performing a prediction or accessing the non conformity scores.") from e
        return all_scores

    def fit(self, X_train:Iterable[Any], y_train:TensorLike)->CVPlusRegressor:
        for X_fit, y_fit, X_calib, y_calib in self.splitter(X_train, y_train):
            self._conformal_predictors.append(SplitCP(clone_model(self.model), weight_function=self.weight_function, fit_function=self.fit_function))
            self._conformal_predictors[-1].fit(X_fit, y_fit, X_calib, y_calib)
            self._conformal_predictors[-1].calibrate(X_calib, y_calib)
        return self

    def calibrate(self, X_calib:Iterable[Any], y_calib:TensorLike):
        raise RuntimeError("Cross-conformal predictors do not require a separate calibration step. Please use the `fit` method to train and calibrate the model.")

    def predict(self, X_test:Iterable[Any], alpha:float)->tuple[TensorLike, Any]:
        print(X_test.shape)
        n = self.len_calibr
        r_l = []
        r_u = []

        # TODO : avoid double loop ? vectorize ? force nc_scores to be more than a simple iterable ?
        for cp in self._conformal_predictors:
            for ricv in cp.nc_scores:
                r_l.append(cp.model(X_test) - ricv)
                r_u.append(cp.model(X_test) + ricv)
        l_stack = ops.stack(r_l, axis=0) # dim (n, b, 1)
        u_stack = ops.stack(r_u, axis=0) # dim (n, b, 1)
        l_stack.sort(axis=0) # sort on n dim
        u_stack.sort(axis=0) # sort on n dim
        l_alpha = l_stack[math.ceil(alpha * (n+1))]
        u_alpha = u_stack[math.ceil((1 - alpha) * (n+1))]
        # TODO : replace None with some aggregation of point predictions of multiple predictors ?
        return None, ops.stack([l_alpha, u_alpha], axis=-1)
        
