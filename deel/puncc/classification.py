from deel.puncc.api.nonconformity_scores import lac_score, aps_score, raps_score
from deel.puncc.api.prediction_sets import lac_set, aps_set, raps_set
from deel.puncc.api.conformal_predictor import AutoConformalPredictor, ConformalPredictor

class LAC(AutoConformalPredictor):
    nc_score_function=lac_score
    pred_set_function=lac_set

class APS(AutoConformalPredictor):
    nc_score_function=aps_score
    pred_set_function=aps_set

class RAPS(ConformalPredictor):
    # TODO : add random state propagation to control randomized tie breaking
    def __init__(self, model, lambd:float=0, k_reg:int=1, rand:bool=False, weight_function=None, fit_function=None):
        nc_score_function = raps_score(lambd=lambd, k_reg=k_reg)
        pred_set_function = raps_set(lambd=lambd, k_reg=k_reg, rand=rand)
        super().__init__(model, nc_score_function, pred_set_function, weight_function, fit_function)