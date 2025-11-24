from deel.puncc.api.nonconformity_scores import absolute_difference, scaled_ad, cqr_score
from deel.puncc.api.prediction_sets import constant_interval, scaled_interval, cqr_interval
from deel.puncc.api.conformal_predictor import ConformalPredictor, AutoConformalPredictor


class SplitCP(AutoConformalPredictor):
    nc_score_function=absolute_difference()
    pred_set_function=constant_interval()

class CQR(AutoConformalPredictor):
    nc_score_function=cqr_score()
    pred_set_function=cqr_interval()

# TODO : put the eps somewhere else
class LocallyAdaptiveCP(ConformalPredictor):
    def __init__(self, model,
                 weight_function=None,
                 eps:float=1e-12):
        super().__init__(
            model=model,
            nc_score_function=scaled_ad(eps=eps),
            pred_set_function=scaled_interval(eps=eps),
            weight_function=weight_function,
        )
