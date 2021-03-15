#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""IAPWS stubs for matplotlib.plylot"""

from typing import Mapping, Tuple, Optional, Any

# Just what we need...

def title(label: str) -> None:
    ...

def xlabel(xlabel: str) -> None:
    ...

def ylabel(ylabel: str) -> None:
    ...

def xlim() -> Tuple[float, float]:
    ...

def ylim() -> Tuple[float, float]:
    ...

def grid(b: Optional[bool] = None) -> None:
    ...

def xscale(value: str) -> None:
    ...

def yscale(value: str) -> None:
    ...

def annotate(txt: str, xy: Tuple[float, float],
             rotation: Optional[float] = None, **kwargs) -> None:
    ...

def plot(x: Any, y: Any, **kwargs) -> None:
    ...

def show() -> None:
    ...
