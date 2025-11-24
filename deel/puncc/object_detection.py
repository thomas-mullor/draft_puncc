from deel.puncc.api.conformal_predictor import ConformalPredictor
from deel.puncc.typing import Predictor, PredictorLike, TensorLike, NCScoreFunction, PredSetFunction
from typing import Callable, Iterable, Any, Literal
from deel.puncc.api.nonconformity_scores import difference, scaled_bbox_difference
from deel.puncc.api.prediction_sets import constant_bbox, scaled_bbox
from deel.puncc.api.correction import bonferroni

class SplitBoxWise(ConformalPredictor):
    def __init__(self,
                 model:Predictor|PredictorLike,
                 method:Literal["additive", "multiplicative"]="additive",
                 weight_function:Callable[[Iterable[Any]], Iterable[float]]|None = None,
                 fit_function:Callable[[Predictor, Iterable[Any], TensorLike], Predictor]|None = None):
        if method == "additive":
            nc_score_function = difference()
            pred_set_function = constant_bbox()
        elif method == "multiplicative":
            nc_score_function = scaled_bbox_difference()
            pred_set_function = scaled_bbox()
        else:
            raise ValueError(f"Unknown method '{method} for SplitBoxWise'. Supported methods are 'additive' and 'multiplicative'.")
        super().__init__(model=model,
                            nc_score_function=nc_score_function,
                            pred_set_function=pred_set_function,
                            weight_function=weight_function,
                            fit_function=fit_function)
        
    def predict(self,
                X_test:Iterable[Any],
                alpha:float,
                correction:Callable = bonferroni(4))->tuple[TensorLike, Any]:
        return super().predict(X_test, alpha, correction)