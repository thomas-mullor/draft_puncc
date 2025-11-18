# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module provides data splitting schemes.
"""
from abc import ABC, abstractmethod
from typing import Iterable
from typing import List
from typing import Tuple

from deel.puncc._keras import ops, random


class BaseSplitter(ABC):
    @abstractmethod
    def split(self, X:Iterable, y:Iterable|None = None)->Tuple[Iterable]:
        ...

    def __call__(self, *args, **kwargs) -> Tuple[Iterable]:
        return self.split(*args, **kwargs)

class IdSplitter(BaseSplitter):
    def __init__(self, X_fit: Iterable, y_fit: Iterable, X_calib: Iterable, y_calib: Iterable):
        self.X_fit = X_fit
        self.y_fit = y_fit
        self.X_calib = X_calib
        self.y_calib = y_calib

    def split(self, X=None, y=None) -> Tuple[Iterable]:
        return [(self.X_fit, self.y_fit, self.X_calib, self.y_calib)]
    
class RandomSplitter(BaseSplitter):
    def __init__(self, ratio:float=0.8, random_state=None):
        self.ratio = ratio
        self.random_state = random_state

    def split(
        self,
        X: Iterable,
        y: Iterable|None=None,
    ) -> Tuple[Iterable]:
        fit_idxs = random.uniform((len(X), )) > self.ratio
        cal_idxs = ops.logical_not(fit_idxs)
        if y is None:
            return [(X[fit_idxs], X[cal_idxs])]
        return [(X[fit_idxs], y[fit_idxs], X[cal_idxs], y[cal_idxs])]

class KFoldSplitter(BaseSplitter):
    def __init__(self, K: int,
                 shuffle:bool=True,
                 random_state=None) -> None:
        if K < 2:
            raise ValueError(f"K must be >= 2. Provided value: {K}.")
        self.K = K
        self.random_state = random_state
        self.shuffle = shuffle

    def split(
        self,
        X: Iterable,
        y: Iterable|None = None,
    ) -> List[Tuple[Iterable]]:
        
        # TODO : improve this
        n_samples = len(X)
        idxs = ops.arange(n_samples)

        if self.shuffle:
            idxs = random.shuffle(idxs, seed=self.random_state)

        n_min = n_samples // self.K
        r = n_samples % self.K
        folds_length = [n_min + 1] * r + [n_min] * (self.K - r)

        folds = []

        for start, length in zip(ops.cumsum(ops.array([0] + folds_length)), folds_length, strict=False):
            bool_calib_index = ops.array([i in idxs[start:start+length] for i in range(n_samples)])
            if y is None:
                folds.append((X[~bool_calib_index], X[bool_calib_index]))
            else:
                folds.append((X[~bool_calib_index], y[~bool_calib_index],
                            X[bool_calib_index], y[bool_calib_index]))
        return folds
