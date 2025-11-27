from __future__ import annotations

import copy
from typing import Any

from deel.puncc.config import get_backend
import sys

# TODO : improve this whole module (preferably before it achieves self-awareness) !
# TODO : add better torch model cloning
# TODO : add warnings where weights can't or must be cloned !
# TODO : praise the mighty GPU spirits for not segfaulting during cloning
# TODO : figure out how to clone advanced models without breaking the space–time continuum
# TODO : ensure this code works on Mondays (historically problematic)
# TODO : rewrite this in fewer hacks and more science
# TODO : investigate why deepcopy behaves like a gremlin after midnight
# TODO : implement a universal clone that also makes coffee
# TODO : write unit tests that also judge our life choices
# TODO : maybe train a deep learning model to clone other models ?
# TODO : optimize for quantum computers (just in case)
# TODO : make the module emit confetti on successful clone
# TODO : add few more TODOs

ML_MODULES = {"torch", "tensorflow", "keras", "sklearn", "transformers", "jax"}

def get_imported_modules() -> set[str]:
    return set(module.split(".")[0] for module in list(sys.modules.keys()) if module)

def get_imported_ml_modules() -> set[str]:
    imported = get_imported_modules()
    return ML_MODULES.intersection(imported)

def get_origin_from_model(obj)->str|None:
    cls = obj.__class__
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__name__", "") or ""
    mro = getattr(cls, "__mro__", ()) or ""

    def mro_has(prefix):
        for c in mro:
            mod = getattr(c, "__module__", "") or ""
            if mod.startswith(prefix):
                return True
        return False

    for orig in ML_MODULES:
        if module.startswith(orig) or name.startswith(orig) or mro_has(orig):
            return orig
    return None


class ModelCannotBeClonedError(RuntimeError):
    poem = """
        Oh weary dev, take heart, take rest,
        This model resists your cloning quest.
        Its weights entwined, a secret kept,
        No copy here, no matter how you prep.

        Perhaps one day the stars align,
        And tensors dance in perfect line.
        But today, dear coder, you must concede,
        Some models are wild, they cannot be freed.
        """
    def __init__(self):
        super().__init__(self.poem)


def clone_model(
    model: Any,
    *,
    clone_weights:bool=False
) -> Any:
    # Check if model has a "clone" or a "copy" method:
    if hasattr(model, "clone") and callable(getattr(model, "clone")):
        return model.clone()
    if hasattr(model, "copy") and callable(getattr(model, "copy")):
        return model.copy()
 
    available_cloners = {
        "sklearn": _clone_sklearn,
        "torch": _clone_torch,
        "keras": _clone_keras,
        "transformers": _clone_hf,
        "tensorflow": _clone_keras,
        "jax": _clone_jax
    }
    # filter cloners with imported modules to avoid unnecessary imports
    available_cloners = {k: v for k, v in available_cloners.items() if k in get_imported_ml_modules()}

    # Try cloner associated to the actually used backend
    first_guess = get_backend()
    if first_guess == "numpy":
        first_guess = "sklearn"
    # Try cloner associated to the model's origin
    second_guess = get_origin_from_model(model)
    for guess in (first_guess, second_guess):
        if guess in available_cloners:
            cloner = available_cloners[guess]
            cloned = cloner(model, clone_weights=clone_weights)
            if cloned is not None:
                return cloned
            available_cloners.pop(guess, None)

    # Try all remaining cloners:
    for cloner in available_cloners.values():
        cloned = cloner(model, clone_weights=clone_weights)
        if cloned is not None:
            return cloned
    try:
        # Fallback to deepcopy if no specific cloner worked
        return copy.deepcopy(model)
    except Exception as e:
        # If even deepcopy fails, raise a custom error
        raise ModelCannotBeClonedError() from e

def _clone_sklearn(model: Any, *, clone_weights:bool=False) -> Any | None:
    try:
        import sklearn.base
    except ImportError:
        return None
    if isinstance(model, getattr(sklearn.base, "BaseEstimator", ())):
        if clone_weights:
            return copy.deepcopy(model)
        return sklearn.base.clone(model)
    return None

def _clone_keras(model: Any, *, clone_weights:bool=False) -> Any | None:
    """
    Clone Keras models with optional recompilation that mirrors optimizer/loss/metrics.
    """
    try:
        import keras
    except ImportError:
        return None

    if isinstance(model, getattr(keras, "Model", ())):
        cloned = keras.models.clone_model(model)
        if clone_weights:
            cloned.set_weights(model.get_weights())
        # TODO : eventually add compilation of the freshly cloned model
        return cloned
    return None


def _clone_torch(model: Any, *, clone_weights) -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    if isinstance(model, getattr(torch.nn, "Module", ())):
        with torch.no_grad():
            cloned = copy.deepcopy(model)
            cloned = cloned.to(model.device)
            cloned.train(model.training)
        return cloned
    return None



def _clone_hf(model: Any) -> Any | None:
    try:
        import transformers
    except ImportError:
        return None

    # PyTorch HF
    if isinstance(model, getattr(transformers, "PreTrainedModel", ())):
        new_m = model.__class__(model.config)
        try:
            import torch
        except ImportError:
            torch = None

        if torch is not None:
            with torch.no_grad():
                new_m.load_state_dict(model.state_dict())
        else:
            new_m.load_state_dict(model.state_dict())
        new_m.train(model.training)
        return new_m

    # TensorFlow HF
    if isinstance(model, getattr(transformers, "TFPreTrainedModel", ())):
        new_m = model.__class__(model.config)
        new_m.set_weights(model.get_weights())
        return new_m

    # Flax HF
    if isinstance(model, getattr(transformers, "FlaxPreTrainedModel", ())):
        dtype = getattr(model, "dtype", None)
        new_m = model.__class__(model.config, dtype=dtype)
        new_m.params = copy.deepcopy(model.params)
        return new_m
    return None

def _clone_jax(model: Any) -> Any | None:
    raise NotImplementedError("JAX model cloning is not yet implemented, please expose a 'clone' method or use non cross conformal methods.")

