#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""IAPWS stubs for scipy.optimize"""

from typing import Tuple, Dict, Callable, Optional, Literal, Union, Any, overload
import numpy

# With no full_output option, fsolve returns a numpy.ndarray.
@overload
def fsolve(
        func: Callable[..., Any],
        x0: Union[float, int, Tuple[Any], Any])-> numpy.ndarray:
    ...

# If someone specified a False full_output option the return value is the same.
@overload
def fsolve(
        func: Callable[..., Any],
        x0: Union[float, int, Tuple[Any], Any],
        full_output: Literal[False])-> numpy.ndarray:
    ...

# With a True full_option, the return value is a different format.
@overload
def fsolve(
        func: Callable[..., Any],
        x0: Union[float, int, Tuple[Any], Any],
        full_output: Literal[True]
) -> Tuple[numpy.ndarray, Dict[str, Any], int, str]:
    ...


def minimize(fun: Callable[..., Any], x0: Tuple[float],
             bounds: Optional[Tuple[Tuple[float, ...]]] = None,
             jac: Optional[Callable[..., Any]] = None):
    ...


def newton(func: Callable[..., Any], x0: float) -> numpy.Float64:
    ...
