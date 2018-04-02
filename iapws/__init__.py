#!/usr/bin/python
# -*- coding: utf-8 -*-


from ._iapws import (_Ice, _Sublimation_Pressure, _Melting_Pressure,  # noqa
                     _Viscosity, _ThCond, _Tension, _Dielectric, _Refractive)
from .iapws97 import IAPWS97  # noqa
from .iapws95 import IAPWS95, D2O  # noqa
from .iapws08 import SeaWater  # noqa
from .humidAir import HumidAir  # noqa
from .ammonia import H2ONH3  # noqa

__version__ = "1.3"
