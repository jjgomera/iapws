#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""IAPWS stubs for numpy"""

from typing import Tuple, List, Any

# With this mypy type definition, we shouldn't be able to do anything
# with the numpy.Float64 except cast them to the Python floats.
class Float64(object):
    def __float__(self) -> float:
        ...

# We can still iterate over the ndarrays however, because this
# definition lies and says ndarray is derived from Tuple.
class ndarray(Tuple[Float64, ...]):
    ...

def linspace(start: Any, stop: Any, num: int = 1) -> ndarray:
    ...

def concatenate(a: List[Any]) -> ndarray:
    ...

def arange(start: Any, stop: Any, step: Any) -> ndarray:
    ...

def log10(Any) -> Float64:
    ...

def log(Any) -> Float64:
    ...

def exp(Any) -> Float64:
    ...

def logspace(start: Any, stop: Any, num: int) -> ndarray:
    ...
