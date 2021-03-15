#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""IAPWS stubs for numpy"""

from typing import Tuple, List, Any

class Float64(float):
    ...

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
