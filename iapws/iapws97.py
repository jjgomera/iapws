#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=too-many-lines, too-many-statements, too-many-locals
# pylint: disable=too-many-instance-attributes, too-many-branches
# pylint: disable=invalid-name

"""IAPWS-IF97 standard implementation

.. image:: images/iapws97.png
    :alt: iapws97

The module implement the fundamental equation for the five regions (rectangular
boxes) and the backward equation (marked in grey).

:class:`IAPWS97`: Global module class with all the functionality integrated

Fundamental equations:
   * :func:`_Region1`
   * :func:`_Region2`
   * :func:`_Region3`
   * :func:`_Region4`
   * :func:`_TSat_P`
   * :func:`_PSat_T`
   * :func:`_Region5`

Backward equations:
   * :func:`_Backward1_T_Ph`
   * :func:`_Backward1_T_Ps`
   * :func:`_Backward1_P_hs`
   * :func:`_Backward2_T_Ph`
   * :func:`_Backward2_T_Ps`
   * :func:`_Backward2_P_hs`
   * :func:`_Backward3_T_Ph`
   * :func:`_Backward3_T_Ps`
   * :func:`_Backward3_P_hs`
   * :func:`_Backward3_v_Ph`
   * :func:`_Backward3_v_Ps`
   * :func:`_Backward3_v_PT`
   * :func:`_Backward4_T_hs`

Boundary equations:
   * :func:`_h13_s`
   * :func:`_h3a_s`
   * :func:`_h1_s`
   * :func:`_t_hs`
   * :func:`_PSat_h`
   * :func:`_h2ab_s`
   * :func:`_h_3ab`
   * :func:`_h2c3b_s`
   * :func:`_hab_s`
   * :func:`_hbc_P`


References:

IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
Thermodynamic Properties of Water and Steam August 2007,
http://www.iapws.org/relguide/IF97-Rev.html

IAPWS, Revised Supplementary Release on Backward Equations for Pressure
as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the IAPWS
Industrial Formulation 1997 for the Thermodynamic Properties of Water and
Steam, http://www.iapws.org/relguide/Supp-PHS12-2014.pdf

IAPWS, Revised Supplementary Release on Backward Equations for the
Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
Industrial Formulation 1997 for the Thermodynamic Properties of Water and
Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf

IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
Region 3, Equations as a Function of h and s for the Region Boundaries, and an
Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997 for
the Thermodynamic Properties of Water and Steam,
http://www.iapws.org/relguide/Supp-phs3-2014.pdf

IAPWS, Revised Supplementary Release on Backward Equations for Specific
Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and
Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf

IAPWS, Revised Advisory Note No. 3: Thermodynamic Derivatives from IAPWS
Formulations, http://www.iapws.org/relguide/Advise3.pdf

Wagner, W; Kretzschmar, H-J: International Steam Tables: Properties of
Water and Steam Based on the Industrial Formulation IAPWS-IF97; Springer, 2008;
doi: 10.1007/978-3-540-74234-0
"""

from __future__ import division
from math import sqrt, log, exp
from scipy.optimize import fsolve, newton
import numpy as np

from . import _iapws97Constants as Const
from ._iapws import R, Tc, Pc, rhoc, Tt, Pt, Tb, Dipole, f_acent
from ._iapws import _Viscosity, _ThCond, _Tension, _Dielectric, _Refractive
from ._utils import getphase, deriv_G, _fase

# Critic properties
sc = 4.41202148223476
hc = 2087.5468451171537

# Pmin = _PSat_T(273.15)   # Minimum pressure
Pmin = 0.000611212677444
# Ps_623 = _PSat_T(623.15)  # P Saturation at 623.15 K, boundary region 1-3
Ps_623 = 16.5291642526


# Boundary Region1-Region3
def _h13_s(s):
    """Define the boundary between Region 1 and 3, h=f(s)

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * s(100MPa,623.15K) ≤ s ≤ s'(623.15K)

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 7

    Examples
    --------
    >>> _h13_s(3.7)
    1632.525047
    >>> _h13_s(3.5)
    1566.104611
    """
    # Check input parameters
    if s < 3.397782955 or s > 3.77828134:
        raise NotImplementedError("Incoming out of bound")

    sigma = s / 3.8
    suma = np.sum(Const.h13_s_n * (sigma - 0.884) ** Const.h13_s_Li * (sigma - 0.864) ** Const.h13_s_Lj)
    return 1700 * suma


# Boundary Region2-Region3
def _P23_T(T):
    """Define the boundary between Region 2 and 3, P=f(T)

    Parameters
    ----------
    T : float
        Temperature, [K]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 5

    Examples
    --------
    >>> _P23_T(623.15)
    16.52916425
    """
    n = (0.34805185628969e3, -0.11671859879975e1, 0.10192970039326e-2)
    return n[0] + n[1] * T + n[2] * T ** 2


def _t_P(P):
    """Define the boundary between Region 2 and 3, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 5

    Examples
    --------
    >>> _t_P(16.52916425)
    623.15
    """
    n = (0.10192970039326e-2, 0.57254459862746e3, 0.1391883977870e2)
    return n[1] + ((P - n[2]) / n[0]) ** 0.5


def _t_hs(h, s):
    """Define the boundary between Region 2 and 3, T=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 5.048096828 ≤ s ≤ 5.260578707
        * 2.563592004e3 ≤ h ≤ 2.812942061e3

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 8

    Examples
    --------
    >>> _t_hs(2600, 5.1)
    713.5259364
    >>> _t_hs(2800, 5.2)
    817.6202120
    """
    # Check input parameters
    if s < 5.048096828 or s > 5.260578707 or \
            h < 2.563592004e3 or h > 2.812942061e3:
        raise NotImplementedError("Incoming out of bound")

    nu = h / 3000
    sigma = s / 5.3
    suma = np.sum(Const.t_hs_n * (nu - 0.727) ** Const.t_hs_Li * (sigma - 0.864) ** Const.t_hs_Lj)
    return 900 * suma


# Saturated line
def _PSat_T(T):
    """Define the saturated line, P=f(T)

    Parameters
    ----------
    T : float
        Temperature, [K]

    Returns
    -------
    P : float
        Pressure, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 273.15 ≤ T ≤ 647.096

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 30

    Examples
    --------
    >>> _PSat_T(500)
    2.63889776
    """
    # Check input parameters
    if T < 273.15 or T > Tc:
        raise NotImplementedError("Incoming out of bound")

    n = (0, 0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
         0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
         -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
         0.65017534844798E+03)
    tita = T + n[9] / (T - n[10])
    A = tita ** 2 + n[1] * tita + n[2]
    B = n[3] * tita ** 2 + n[4] * tita + n[5]
    C = n[6] * tita ** 2 + n[7] * tita + n[8]
    return (2 * C / (-B + (B ** 2 - 4 * A * C) ** 0.5)) ** 4


def _TSat_P(P):
    """Define the saturated line, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    T : float
        Temperature, [K]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 0.00061121 ≤ P ≤ 22.064

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 31

    Examples
    --------
    >>> _TSat_P(10)
    584.149488
    """
    # Check input parameters
    if P < 611.212677 / 1e6 or P > 22.064:
        raise NotImplementedError("Incoming out of bound")

    n = (0, 0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
         0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
         -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
         0.65017534844798E+03)
    beta = P ** 0.25
    E = beta ** 2 + n[3] * beta + n[6]
    F = n[1] * beta ** 2 + n[4] * beta + n[7]
    G = n[2] * beta ** 2 + n[5] * beta + n[8]
    D = 2 * G / (-F - (F ** 2 - 4 * E * G) ** 0.5)
    return (n[10] + D - ((n[10] + D) ** 2 - 4 * (n[9] + n[10] * D)) ** 0.5) / 2


def _PSat_h(h):
    """Define the saturated line, P=f(h) for region 3

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    P : float
        Pressure, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * h'(623.15K) ≤ h ≤ h''(623.15K)

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 10

    Examples
    --------
    >>> _PSat_h(1700)
    17.24175718
    >>> _PSat_h(2400)
    20.18090839
    """
    # Check input parameters
    hmin_Ps3 = _Region1(623.15, Ps_623)["h"]
    hmax_Ps3 = _Region2(623.15, Ps_623)["h"]
    if h < hmin_Ps3 or h > hmax_Ps3:
        raise NotImplementedError("Incoming out of bound")

    nu = h / 2600
    suma = np.sum(Const.PSat_h_n * (nu - 1.02) ** Const.PSat_h_Li * (nu - 0.608) ** Const.PSat_h_Lj)
    return 22 * suma


def _PSat_s(s):
    """Define the saturated line, P=f(s) for region 3

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * s'(623.15K) ≤ s ≤ s''(623.15K)

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 11

    Examples
    --------
    >>> _PSat_s(3.8)
    16.87755057
    >>> _PSat_s(5.2)
    16.68968482
    """
    # Check input parameters
    smin_Ps3 = _Region1(623.15, Ps_623)["s"]
    smax_Ps3 = _Region2(623.15, Ps_623)["s"]
    if s < smin_Ps3 or s > smax_Ps3:
        raise NotImplementedError("Incoming out of bound")

    sigma = s / 5.2
    suma = np.sum(Const.PSat_s_n * (sigma - 1.03) ** Const.PSat_s_Li * (sigma - 0.699) ** Const.PSat_s_Lj)
    return 22 * suma


def _h1_s(s):
    """Define the saturated line boundary between Region 1 and 4, h=f(s)

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * s'(273.15K) ≤ s ≤ s'(623.15K)

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 3

    Examples
    --------
    >>> _h1_s(1)
    308.5509647
    >>> _h1_s(3)
    1198.359754
    """
    # Check input parameters
    if s < -1.545495919e-4 or s > 3.77828134:
        raise NotImplementedError("Incoming out of bound")

    sigma = s / 3.8
    suma = np.sum(Const.h1_s_n * (sigma - 1.09) ** Const.h1_s_Li * (sigma + 0.366e-4) ** Const.h1_s_Lj)
    return 1700 * suma


def _h3a_s(s):
    """Define the saturated line boundary between Region 4 and 3a, h=f(s)

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * s'(623.15K) ≤ s ≤ sc

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 4

    Examples
    --------
    >>> _h3a_s(3.8)
    1685.025565
    >>> _h3a_s(4.2)
    1949.352563
    """
    # Check input parameters
    if s < 3.77828134 or s > 4.41202148223476:
        raise NotImplementedError("Incoming out of bound")

    sigma = s / 3.8
    suma = np.sum(Const.h3a_s_n * (sigma - 1.09) ** Const.h3a_s_Li * (sigma + 0.366e-4) ** Const.h3a_s_Lj)
    return 1700 * suma


def _h2ab_s(s):
    """Define the saturated line boundary between Region 4 and 2a-2b, h=f(s)

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 5.85 ≤ s ≤ s"(273.15K)

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 5

    Examples
    --------
    >>> _h2ab_s(7)
    2723.729985
    >>> _h2ab_s(9)
    2511.861477
    """
    # Check input parameters
    if s < 5.85 or s > 9.155759395:
        raise NotImplementedError("Incoming out of bound")

    sigma1 = s / 5.21
    sigma2 = s / 9.2
    suma = np.sum(Const.h2ab_s_n * (1 / sigma1 - 0.513) ** Const.h2ab_s_Li * (sigma2 - 0.524) ** Const.h2ab_s_Lj)
    return 2800 * exp(suma)


def _h2c3b_s(s):
    """Define the saturated line boundary between Region 4 and 2c-3b, h=f(s)

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * sc ≤ s ≤ 5.85

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 6

    Examples
    --------
    >>> _h2c3b_s(5.5)
    2687.693850
    >>> _h2c3b_s(4.5)
    2144.360448
    """
    # Check input parameters
    if s < 4.41202148223476 or s > 5.85:
        raise NotImplementedError("Incoming out of bound")

    sigma = s / 5.9
    suma = np.sum(Const.h2c3b_s_n * (sigma - 1.02) ** Const.h2c3b_s_Li * (sigma - 0.726) ** Const.h2c3b_s_Lj)
    return 2800 * suma ** 4


# Region 1
def _Region1(T, P):
    """Basic equation for region 1

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties. The available properties are:

            * v: Specific volume, [m³/kg]
            * h: Specific enthalpy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isocoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s]
            * alfav: Cubic expansion coefficient, [1/K]
            * kt: Isothermal compressibility, [1/MPa]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 7

    Examples
    --------
    >>> _Region1(300,3)["v"]
    0.00100215168
    >>> _Region1(300,3)["h"]
    115.331273
    >>> _Region1(300,3)["h"]-3000*_Region1(300,3)["v"]
    112.324818
    >>> _Region1(300,80)["s"]
    0.368563852
    >>> _Region1(300,80)["cp"]
    4.01008987
    >>> _Region1(300,80)["cv"]
    3.91736606
    >>> _Region1(500,3)["w"]
    1240.71337
    >>> _Region1(500,3)["alfav"]
    0.00164118128
    >>> _Region1(500,3)["kt"]
    0.00112892188
    """
    if P < 0:
        P = Pmin

    Tr = 1386 / T
    Pr = P / 16.53

    g = np.sum(Const.Region1_n * (7.1 - Pr) ** Const.Region1_Li * (Tr - 1.222) ** Const.Region1_Lj)
    gp = -np.sum(
        Const.Region1_n_Li_product * (7.1 - Pr) ** Const.Region1_Li_less_1 * (Tr - 1.222) ** Const.Region1_Lj)
    gpp = np.sum(
        Const.Region1_n_Li_product * Const.Region1_Li_less_1 * (7.1 - Pr) ** Const.Region1_Li_less_2 * (
                    Tr - 1.222) ** Const.Region1_Lj)
    gt = np.sum(
        Const.Region1_n_Lj_product * (7.1 - Pr) ** Const.Region1_Li * (Tr - 1.222) ** Const.Region1_Lj_less_1)
    gtt = np.sum(
        Const.Region1_n_Lj_product * Const.Region1_Lj_less_1 * (7.1 - Pr) ** Const.Region1_Li * (Tr - 1.222) ** (
            Const.Region1_Lj_less_2))
    gpt = -np.sum(Const.Region1_n_Li_Lj_product * (7.1 - Pr) ** Const.Region1_Li_less_1 * (Tr - 1.222) ** (
        Const.Region1_Lj_less_1))

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * gp * R * T / P / 1000
    propiedades["h"] = Tr * gt * R * T
    propiedades["s"] = R * (Tr * gt - g)
    propiedades["cp"] = -R * Tr ** 2 * gtt
    propiedades["cv"] = R * (-Tr ** 2 * gtt + (gp - Tr * gpt) ** 2 / gpp)
    propiedades["w"] = sqrt(R * T * 1000 * gp ** 2 / ((gp - Tr * gpt) ** 2 / (Tr ** 2 * gtt) - gpp))
    propiedades["alfav"] = (1 - Tr * gpt / gp) / T
    propiedades["kt"] = -Pr * gpp / gp / P
    propiedades["region"] = 1
    propiedades["x"] = 0
    return propiedades


def _Backward1_T_Ph(P, h):
    """
    Backward equation for region 1, T=f(P,h)
    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]
    Returns
    -------
    T : float
        Temperature, [K]
    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 11
    Examples
    --------
    >>> _Backward1_T_Ph(3,500)
    391.798509
    >>> _Backward1_T_Ph(80,1500)
    611.041229
    """

    Pr = P / 1
    nu = h / 2500
    T = np.sum(Const.Backward1_T_Ph_n * Pr ** Const.Backward1_T_Ph_Li * (nu + 1) ** Const.Backward1_T_Ph_Lj)
    return T


def _Backward1_T_Ps(P, s):
    """Backward equation for region 1, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 13

    Examples
    --------
    >>> _Backward1_T_Ps(3,0.5)
    307.842258
    >>> _Backward1_T_Ps(80,3)
    565.899909
    """

    Pr = P / 1
    sigma = s / 1
    T = np.sum(Const.Backward1_T_Ps_n * Pr ** Const.Backward1_T_Ps_Li * (sigma + 2) ** Const.Backward1_T_Ps_Lj)
    return T


def _Backward1_P_hs(h, s):
    """Backward equation for region 1, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Pressure
    as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of
    Water and Steam, http://www.iapws.org/relguide/Supp-PHS12-2014.pdf, Eq 1

    Examples
    --------
    >>> _Backward1_P_hs(0.001,0)
    0.0009800980612
    >>> _Backward1_P_hs(90,0)
    91.92954727
    >>> _Backward1_P_hs(1500,3.4)
    58.68294423
    """
    nu = h / 3400
    sigma = s / 7.6
    P = np.sum(
        Const.Backward1_P_hs_n * (nu + 0.05) ** Const.Backward1_P_hs_Li * (sigma + 0.05) ** Const.Backward1_P_hs_Lj)
    return 100 * P


# Region 2
def _Region2(T, P):
    """Basic equation for region 2

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties. The available properties are:

            * v: Specific volume, [m³/kg]
            * h: Specific enthalpy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isocoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s]
            * alfav: Cubic expansion coefficient, [1/K]
            * kt: Isothermal compressibility, [1/MPa]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 15-17

    Examples
    --------
    >>> _Region2(700,30)["v"]
    0.00542946619
    >>> _Region2(700,30)["h"]
    2631.49474
    >>> _Region2(700,30)["h"]-30000*_Region2(700,30)["v"]
    2468.61076
    >>> _Region2(700,0.0035)["s"]
    10.1749996
    >>> _Region2(700,0.0035)["cp"]
    2.08141274
    >>> _Region2(700,0.0035)["cv"]
    1.61978333
    >>> _Region2(300,0.0035)["w"]
    427.920172
    >>> _Region2(300,0.0035)["alfav"]
    0.00337578289
    >>> _Region2(300,0.0035)["kt"]
    286.239651
    """
    if P < 0:
        P = Pmin

    Tr = 540 / T
    Pr = P / 1

    go, gop, gopp, got, gott, gopt = Region2_cp0(Tr, Pr)

    gr = np.sum(Const.Region2_nr * Pr ** Const.Region2_Ir * (Tr - 0.5) ** Const.Region2_Jr)
    grp = np.sum(Const.Region2_nr_Ir_product * Pr ** Const.Region2_Ir_less_1 * (Tr - 0.5) ** Const.Region2_Jr)
    grpp = np.sum(
        Const.Region2_nr_Ir_product * Const.Region2_Ir_less_1 * Pr ** Const.Region2_Ir_less_2 * (
                    Tr - 0.5) ** Const.Region2_Jr)
    grt = np.sum(Const.Region2_nr_Jr_product * Pr ** Const.Region2_Ir * (Tr - 0.5) ** Const.Region2_Jr_less_1)
    grtt = np.sum(
        Const.Region2_nr_Jr_product * Const.Region2_Jr_less_1 * Pr ** Const.Region2_Ir *
        (Tr - 0.5) ** Const.Region2_Jr_less_2)
    grpt = np.sum(
        Const.Region2_nr_Ir_Jr_product * Pr ** Const.Region2_Ir_less_1 * (Tr - 0.5) ** Const.Region2_Jr_less_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * (gop + grp) * R * T / P / 1000
    propiedades["h"] = Tr * (got + grt) * R * T
    propiedades["s"] = R * (Tr * (got + grt) - (go + gr))
    propiedades["cp"] = -R * Tr ** 2 * (gott + grtt)
    propiedades["cv"] = R * (-Tr ** 2 * (gott + grtt) - (1 + Pr * grp - Tr * Pr * grpt) ** 2
                             / (1 - Pr ** 2 * grpp))
    propiedades["w"] = (R * T * 1000 * (1 + 2 * Pr * grp + Pr ** 2 * grp ** 2) / (1 - Pr ** 2 * grpp + (
            1 + Pr * grp - Tr * Pr * grpt) ** 2 / Tr ** 2 / (gott + grtt))) ** 0.5
    propiedades["alfav"] = (1 + Pr * grp - Tr * Pr * grpt) / (1 + Pr * grp) / T
    propiedades["kt"] = (1 - Pr ** 2 * grpp) / (1 + Pr * grp) / P
    propiedades["region"] = 2
    propiedades["x"] = 1
    return propiedades


def Region2_cp0(Tr, Pr):
    """Ideal properties for Region 2

    Parameters
    ----------
    Tr : float
        Reduced temperature, [-]
    Pr : float
        Reduced pressure, [-]

    Returns
    -------
    prop : array
        Array with ideal Gibbs energy partial derivatives:

            * g: Ideal Specific Gibbs energy [kJ/kg]
            * gp: ∂g/∂P|T
            * gpp: ∂²g/∂P²|T
            * gt: ∂g/∂T|P
            * gtt: ∂²g/∂T²|P
            * gpt: ∂²g/∂T∂P

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 16

    """
    go = log(Pr)
    gop = Pr ** -1
    gopp = -Pr ** -2
    gopt = 0
    go += np.sum(Const.Region2_cp0_no * Tr ** Const.Region2_cp0_Jo)
    got = np.sum(Const.Region2_cp0_no * Const.Region2_cp0_Jo * Tr ** (Const.Region2_cp0_Jo - 1))
    gott = np.sum(
        Const.Region2_cp0_no * Const.Region2_cp0_Jo * (Const.Region2_cp0_Jo - 1) * Tr ** (Const.Region2_cp0_Jo - 2))
    return go, gop, gopp, got, gott, gopt


def _P_2bc(h):
    """Define the boundary between Region 2b and 2c, P=f(h)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 20

    Examples
    --------
    >>> _P_2bc(3516.004323)
    100.0
    """
    return 905.84278514723 - 0.67955786399241 * h + 1.2809002730136e-4 * h ** 2


def _hbc_P(P):
    """Define the boundary between Region 2b and 2c, h=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 21

    Examples
    --------
    >>> _hbc_P(100)
    3516.004323
    """
    return 0.26526571908428e4 + ((P - 0.45257578905948e1) / 1.2809002730136e-4) ** 0.5


def _hab_s(s):
    """Define the boundary between Region 2a and 2b, h=f(s)

    Parameters
    ----------
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Pressure
    as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of
    Water and Steam, http://www.iapws.org/relguide/Supp-PHS12-2014.pdf, Eq 2

    Examples
    --------
    >>> _hab_s(7)
    3376.437884
    """
    smin = _Region2(_TSat_P(4), 4)["s"]
    smax = _Region2(1073.15, 4)["s"]
    if s < smin:
        h = 0
    elif s > smax:
        h = 5000
    else:
        h = -0.349898083432139e4 + 0.257560716905876e4 * s - \
            0.421073558227969e3 * s ** 2 + 0.276349063799944e2 * s ** 3
    return h


def _Backward2a_T_Ph(P, h):
    """Backward equation for region 2a, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 22

    Examples
    --------
    >>> _Backward2a_T_Ph(0.001,3000)
    534.433241
    >>> _Backward2a_T_Ph(3,4000)
    1010.77577
    """
    Pr = P / 1
    nu = h / 2000
    T = np.sum(Const.Backward2a_T_Ph_n * Pr ** Const.Backward2a_T_Ph_Li * (nu - 2.1) ** Const.Backward2a_T_Ph_Lj)
    return T


def _Backward2b_T_Ph(P, h):
    """Backward equation for region 2b, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 23

    Examples
    --------
    >>> _Backward2b_T_Ph(5,4000)
    1015.31583
    >>> _Backward2b_T_Ph(25,3500)
    875.279054
    """
    Pr = P / 1
    nu = h / 2000
    T = np.sum(
        Const.Backward2b_T_Ph_n * (Pr - 2) ** Const.Backward2b_T_Ph_Li * (nu - 2.6) ** Const.Backward2b_T_Ph_Lj)
    return T


def _Backward2c_T_Ph(P, h):
    """Backward equation for region 2c, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 24

    Examples
    --------
    >>> _Backward2c_T_Ph(40,2700)
    743.056411
    >>> _Backward2c_T_Ph(60,3200)
    882.756860
    """
    Pr = P / 1
    nu = h / 2000
    T = np.sum(
        Const.Backward2c_T_Ph_n * (Pr + 25) ** Const.Backward2c_T_Ph_Li * (nu - 1.8) ** Const.Backward2c_T_Ph_Lj)
    return T


def _Backward2_T_Ph(P, h):
    """Backward equation for region 2, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]
    """
    if P <= 4:
        T = _Backward2a_T_Ph(P, h)
    elif 4 < P <= 6.546699678:
        T = _Backward2b_T_Ph(P, h)
    else:
        hf = _hbc_P(P)
        if h >= hf:
            T = _Backward2b_T_Ph(P, h)
        else:
            T = _Backward2c_T_Ph(P, h)

    if P <= 22.064:
        Tsat = _TSat_P(P)
        T = max(Tsat, T)
    return T


def _Backward2a_T_Ps(P, s):
    """Backward equation for region 2a, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 25

    Examples
    --------
    >>> _Backward2a_T_Ps(0.1,7.5)
    399.517097
    >>> _Backward2a_T_Ps(2.5,8)
    1039.84917
    """
    Pr = P / 1
    sigma = s / 2
    T = np.sum(Const.Backward2a_T_Ps_n * Pr ** Const.Backward2a_T_Ps_Li * (sigma - 2) ** Const.Backward2a_T_Ps_Lj)
    return T


def _Backward2b_T_Ps(P, s):
    """Backward equation for region 2b, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 26

    Examples
    --------
    >>> _Backward2b_T_Ps(8,6)
    600.484040
    >>> _Backward2b_T_Ps(90,6)
    1038.01126
    """
    Pr = P / 1
    sigma = s / 0.7853
    T = np.sum(Const.Backward2b_T_Ps_n * Pr ** Const.Backward2b_T_Ps_Li * (10 - sigma) ** Const.Backward2b_T_Ps_Lj)
    return T


def _Backward2c_T_Ps(P, s):
    """Backward equation for region 2c, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 27

    Examples
    --------
    >>> _Backward2c_T_Ps(20,5.75)
    697.992849
    >>> _Backward2c_T_Ps(80,5.75)
    949.017998
    """
    Pr = P / 1
    sigma = s / 2.9251
    T = np.sum(Const.Backward2c_T_Ps_n * Pr ** Const.Backward2c_T_Ps_Li * (2 - sigma) ** Const.Backward2c_T_Ps_Lj)
    return T


def _Backward2_T_Ps(P, s):
    """Backward equation for region 2, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]
    """
    if P <= 4:
        T = _Backward2a_T_Ps(P, s)
    elif s >= 5.85:
        T = _Backward2b_T_Ps(P, s)
    else:
        T = _Backward2c_T_Ps(P, s)

    if P <= 22.064:
        Tsat = _TSat_P(P)
        T = max(Tsat, T)
    return T


def _Backward2a_P_hs(h, s):
    """Backward equation for region 2a, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Pressure
    as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of
    Water and Steam, http://www.iapws.org/relguide/Supp-PHS12-2014.pdf, Eq 3

    Examples
    --------
    >>> _Backward2a_P_hs(2800,6.5)
    1.371012767
    >>> _Backward2a_P_hs(2800,9.5)
    0.001879743844
    >>> _Backward2a_P_hs(4100,9.5)
    0.1024788997
    """
    nu = h / 4200
    sigma = s / 12
    suma = np.sum(
        Const.Backward2a_P_hs_n * (nu - 0.5) ** Const.Backward2a_P_hs_Li * (sigma - 1.2) ** Const.Backward2a_P_hs_Lj)
    return 4 * suma ** 4


def _Backward2b_P_hs(h, s):
    """Backward equation for region 2b, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Pressure
    as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of
    Water and Steam, http://www.iapws.org/relguide/Supp-PHS12-2014.pdf, Eq 4

    Examples
    --------
    >>> _Backward2b_P_hs(2800,6)
    4.793911442
    >>> _Backward2b_P_hs(3600,6)
    83.95519209
    >>> _Backward2b_P_hs(3600,7)
    7.527161441
    """
    nu = h / 4100
    sigma = s / 7.9
    suma = np.sum(Const.Backward2b_P_hs_n * (nu - 0.6) ** Const.Backward2b_P_hs_Li * (
                sigma - 1.01) ** Const.Backward2b_P_hs_Lj)
    return 100 * suma ** 4


def _Backward2c_P_hs(h, s):
    """Backward equation for region 2c, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Pressure
    as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of
    Water and Steam, http://www.iapws.org/relguide/Supp-PHS12-2014.pdf, Eq 5

    Examples
    --------
    >>> _Backward2c_P_hs(2800,5.1)
    94.39202060
    >>> _Backward2c_P_hs(2800,5.8)
    8.414574124
    >>> _Backward2c_P_hs(3400,5.8)
    83.76903879
    """
    nu = h / 3500
    sigma = s / 5.9
    suma = np.sum(
        Const.Backward2c_P_hs_n * (nu - 0.7) ** Const.Backward2c_P_hs_Li * (sigma - 1.1) ** Const.Backward2c_P_hs_Lj)
    return 100 * suma ** 4


def _Backward2_P_hs(h, s):
    """Backward equation for region 2, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]
    """
    sfbc = 5.85
    hamin = _hab_s(s)
    if h <= hamin:
        P = _Backward2a_P_hs(h, s)
    elif s >= sfbc:
        P = _Backward2b_P_hs(h, s)
    else:
        P = _Backward2c_P_hs(h, s)
    return P


# Region 3
def _Region3(rho, T):
    """Basic equation for region 3

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]

    Returns
    -------
    prop : dict
        Dict with calculated properties. The available properties are:

            * v: Specific volume, [m³/kg]
            * h: Specific enthalpy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isocoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s]
            * alfav: Cubic expansion coefficient, [1/K]
            * kt: Isothermal compressibility, [1/MPa]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 28

    Examples
    --------
    >>> _Region3(500,650)["P"]
    25.5837018
    >>> _Region3(500,650)["h"]
    1863.43019
    >>> p = _Region3(500, 650)
    >>> p["h"]-p["P"]*1000*p["v"]
    1812.26279
    >>> _Region3(200,650)["s"]
    4.85438792
    >>> _Region3(200,650)["cp"]
    44.6579342
    >>> _Region3(200,650)["cv"]
    4.04118076
    >>> _Region3(200,650)["w"]
    383.444594
    >>> _Region3(500,750)["alfav"]
    0.00441515098
    >>> _Region3(500,750)["kt"]
    0.00806710817
    """

    d = rho / rhoc
    Tr = Tc / T

    g = (1.0658070028513 * log(d)) + np.sum(Const.Region3_n * d ** Const.Region3_Li * Tr ** Const.Region3_Lj)
    gd = (1.0658070028513 / d) + np.sum(
        Const.Region3_n_Li_product * d ** Const.Region3_Li_less_1 * Tr ** Const.Region3_Lj)
    gdd = (-1.0658070028513 / d ** 2) + np.sum(
        Const.Region3_n_Li_product * Const.Region3_Li_less_1 * d ** (
            Const.Region3_Li_less_2) * Tr ** Const.Region3_Lj)
    gt = np.sum(Const.Region3_n_Lj_product * d ** Const.Region3_Li * Tr ** Const.Region3_Lj_less_1)
    gtt = np.sum(Const.Region3_n_Lj_product * Const.Region3_Lj_less_1 * d ** Const.Region3_Li * Tr ** (
        Const.Region3_Lj_less_2))
    gdt = np.sum(Const.Region3_n_Li_Lj_product * d ** Const.Region3_Li_less_1 * Tr ** Const.Region3_Lj_less_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = d * gd * R * T * rho / 1000
    propiedades["v"] = 1 / rho
    propiedades["h"] = R * T * (Tr * gt + d * gd)
    propiedades["s"] = R * (Tr * gt - g)
    propiedades["cp"] = R * (-Tr ** 2 * gtt + (d * gd - d * Tr * gdt) ** 2 / (2 * d * gd + d ** 2 * gdd))
    propiedades["cv"] = -R * Tr ** 2 * gtt
    propiedades["w"] = sqrt(R * T * 1000 * (2 * d * gd + d ** 2 * gdd - (d * gd - d * Tr * gdt) ** 2
                                            / Tr ** 2 / gtt))
    propiedades["alfav"] = (gd - Tr * gdt) / (2 * gd + d * gdd) / T
    propiedades["kt"] = 1 / (2 * d * gd + d ** 2 * gdd) / rho / R / T * 1000
    propiedades["region"] = 3
    propiedades["x"] = 1
    return propiedades


def _h_3ab(P):
    """Define the boundary between Region 3a-3b, h=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    h : float
        Specific enthalpy, [kJ/kg]

    Examples
    --------
    >>> _h_3ab(25)
    2095.936454
    """
    return 0.201464004206875e4 + 3.74696550136983 * P - 0.0219921901054187 * P ** 2 + 0.875131686009950e-4 * P ** 3


def _tab_P(P):
    """Define the boundary between Region 3a-3b, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Eq. 2

    Examples
    --------
    >>> _tab_P(40)
    693.0341408
    """
    Pr = P / 1
    T = np.sum(Const.tab_P_n * log(Pr) ** Const.tab_P_Li)
    return T


def _top_P(P):
    """Define the boundary between Region 3o-3p, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Eq. 2

    Examples
    --------
    >>> _top_P(22.8)
    650.0106943
    """
    Pr = P / 1
    T = np.sum(Const.top_P_n * log(Pr) ** Const.top_P_Li)
    return T


def _twx_P(P):
    """Define the boundary between Region 3w-3x, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Eq. 2

    Examples
    --------
    >>> _twx_P(22.3)
    648.2049480
    """
    Pr = P / 1
    T = np.sum(Const.twx_P_n * log(Pr) ** Const.twx_P_Li)
    return T


def _tef_P(P):
    """Define the boundary between Region 3e-3f, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Eq. 3

    Examples
    --------
    >>> _tef_P(40)
    713.9593992
    """
    return 3.727888004 * (P - 22.064) + 647.096


def _txx_P(P, xy):
    """Define the boundary between 3x-3y, T=f(P)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    xy: string
        Subregions options: cd, gh, ij, jk, mn, qu, rx, uv

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Eq. 1

    Examples
    --------
    >>> _txx_P(25,"cd")
    649.3659208
    >>> _txx_P(23,"gh")
    649.8873759
    >>> _txx_P(23,"ij")
    651.5778091
    >>> _txx_P(23,"jk")
    655.8338344
    >>> _txx_P(22.8,"mn")
    649.6054133
    >>> _txx_P(22,"qu")
    645.6355027
    >>> _txx_P(22,"rx")
    648.2622754
    >>> _txx_P(22.3,"uv")
    647.7996121
    """
    ng = {
        "cd": [0.585276966696349e3, 0.278233532206915e1, -0.127283549295878e-1,
               0.159090746562729e-3],
        "gh": [-0.249284240900418e5, 0.428143584791546e4, -0.269029173140130e3,
               0.751608051114157e1, -0.787105249910383e-1],
        "ij": [0.584814781649163e3, -0.616179320924617, 0.260763050899562,
               -0.587071076864459e-2, 0.515308185433082e-4],
        "jk": [0.617229772068439e3, -0.770600270141675e1, 0.697072596851896,
               -0.157391839848015e-1, 0.137897492684194e-3],
        "mn": [0.535339483742384e3, 0.761978122720128e1, -0.158365725441648,
               0.192871054508108e-2],
        "qu": [0.565603648239126e3, 0.529062258221222e1, -0.102020639611016,
               0.122240301070145e-2],
        "rx": [0.584561202520006e3, -0.102961025163669e1, 0.243293362700452,
               -0.294905044740799e-2],
        "uv": [0.528199646263062e3, 0.890579602135307e1, -0.222814134903755,
               0.286791682263697e-2]}

    n = ng[xy]
    Pr = P / 1
    T = 0
    for i, ni in enumerate(n):
        T += ni * Pr ** i
    return T


def _Backward3a_v_Ph(P, h):
    """Backward equation for region 3a, v=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 4

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    Examples
    --------
    >>> _Backward3a_v_Ph(20,1700)
    0.001749903962
    >>> _Backward3a_v_Ph(100,2100)
    0.001676229776
    """
    Pr = P / 100
    nu = h / 2100
    suma = np.sum(
        Const.Backward3a_v_Ph_n * (Pr + 0.128) ** Const.Backward3a_v_Ph_Li * (nu - 0.727) ** Const.Backward3a_v_Ph_Lj)
    return 0.0028 * suma


def _Backward3b_v_Ph(P, h):
    """Backward equation for region 3b, v=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 5

    Examples
    --------
    >>> _Backward3b_v_Ph(20,2500)
    0.006670547043
    >>> _Backward3b_v_Ph(100,2700)
    0.002404234998
    """
    Pr = P / 100
    nu = h / 2800
    suma = np.sum(Const.Backward3b_v_Ph_n * (Pr + 0.0661) ** Const.Backward3b_v_Ph_Li * (
                nu - 0.72) ** Const.Backward3b_v_Ph_Lj)
    return 0.0088 * suma


def _Backward3_v_Ph(P, h):
    """Backward equation for region 3, v=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]
    """
    hf = _h_3ab(P)
    if h <= hf:
        return _Backward3a_v_Ph(P, h)

    return _Backward3b_v_Ph(P, h)


def _Backward3a_T_Ph(P, h):
    """Backward equation for region 3a, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 2

    Examples
    --------
    >>> _Backward3a_T_Ph(20,1700)
    629.3083892
    >>> _Backward3a_T_Ph(100,2100)
    733.6163014
    """
    Pr = P / 100.
    nu = h / 2300.
    suma = np.sum(Const.Backward3a_T_Ph_n * (Pr + 0.240) ** Const.Backward3a_T_Ph_Li * (
                nu - 0.615) ** Const.Backward3a_T_Ph_Lj)
    return 760 * suma


def _Backward3b_T_Ph(P, h):
    """Backward equation for region 3b, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 3

    Examples
    --------
    >>> _Backward3b_T_Ph(20,2500)
    641.8418053
    >>> _Backward3b_T_Ph(100,2700)
    842.0460876
    """
    Pr = P / 100.
    nu = h / 2800.
    suma = np.sum(
        Const.Backward3b_T_Ph_n * (Pr + 0.298) ** Const.Backward3b_T_Ph_Li * (nu - 0.72) ** Const.Backward3b_T_Ph_Lj)
    return 860 * suma


def _Backward3_T_Ph(P, h):
    """Backward equation for region 3, T=f(P,h)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    T : float
        Temperature, [K]
    """
    hf = _h_3ab(P)
    if h <= hf:
        T = _Backward3a_T_Ph(P, h)
    else:
        T = _Backward3b_T_Ph(P, h)
    return T


def _Backward3a_v_Ps(P, s):
    """Backward equation for region 3a, v=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 8

    Examples
    --------
    >>> _Backward3a_v_Ps(20,3.8)
    0.001733791463
    >>> _Backward3a_v_Ps(100,4)
    0.001555893131
    """
    Pr = P / 100
    sigma = s / 4.4
    suma = np.sum(Const.Backward3a_v_Ps_n * (Pr + 0.187) ** Const.Backward3a_v_Ps_Li * (
                sigma - 0.755) ** Const.Backward3a_v_Ps_Lj)
    return 0.0028 * suma


def _Backward3b_v_Ps(P, s):
    """Backward equation for region 3b, v=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 9

    Examples
    --------
    >>> _Backward3b_v_Ps(20,5)
    0.006262101987
    >>> _Backward3b_v_Ps(100,5)
    0.002449610757
    """
    Pr = P / 100
    sigma = s / 5.3
    suma = np.sum(Const.Backward3b_v_Ps_n * (Pr + 0.298) ** Const.Backward3b_v_Ps_Li * (
                sigma - 0.816) ** Const.Backward3b_v_Ps_Lj)
    return 0.0088 * suma


def _Backward3_v_Ps(P, s):
    """Backward equation for region 3, v=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]
    """
    if s <= sc:
        return _Backward3a_v_Ps(P, s)

    return _Backward3b_v_Ps(P, s)


def _Backward3a_T_Ps(P, s):
    """Backward equation for region 3a, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 6

    Examples
    --------
    >>> _Backward3a_T_Ps(20,3.8)
    628.2959869
    >>> _Backward3a_T_Ps(100,4)
    705.6880237
    """
    Pr = P / 100
    sigma = s / 4.4
    suma = np.sum(Const.Backward3a_T_Ps_n * (Pr + 0.240) ** Const.Backward3a_T_Ps_Li * (
                sigma - 0.703) ** Const.Backward3a_T_Ps_Lj)
    return 760 * suma


def _Backward3b_T_Ps(P, s):
    """Backward equation for region 3b, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for the
    Functions T(p,h), v(p,h) and T(p,s), v(p,s) for Region 3 of the IAPWS
    Industrial Formulation 1997 for the Thermodynamic Properties of Water and
    Steam, http://www.iapws.org/relguide/Supp-Tv%28ph,ps%293-2014.pdf, Eq 7

    Examples
    --------
    >>> _Backward3b_T_Ps(20,5)
    640.1176443
    >>> _Backward3b_T_Ps(100,5)
    847.4332825
    """
    Pr = P / 100
    sigma = s / 5.3
    suma = np.sum(Const.Backward3b_T_Ps_n * (Pr + 0.760) ** Const.Backward3b_T_Ps_Li * (
                sigma - 0.818) ** Const.Backward3b_T_Ps_Lj)
    return 860 * suma


def _Backward3_T_Ps(P, s):
    """Backward equation for region 3, T=f(P,s)

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]
    """
    if s <= sc:
        return _Backward3a_T_Ps(P, s)

    return _Backward3b_T_Ps(P, s)


def _Backward3a_P_hs(h, s):
    """Backward equation for region 3a, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 1

    Examples
    --------
    >>> _Backward3a_P_hs(1700,3.8)
    25.55703246
    >>> _Backward3a_P_hs(2000,4.2)
    45.40873468
    >>> _Backward3a_P_hs(2100,4.3)
    60.78123340
    """
    nu = h / 2300
    sigma = s / 4.4
    suma = np.sum(Const.Backward3a_P_hs_n * (nu - 1.01) ** Const.Backward3a_P_hs_Li * (
                sigma - 0.75) ** Const.Backward3a_P_hs_Lj)
    return 99 * suma


def _Backward3b_P_hs(h, s):
    """Backward equation for region 3b, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 1

    Examples
    --------
    >>> _Backward3b_P_hs(2400,4.7)
    63.63924887
    >>> _Backward3b_P_hs(2600,5.1)
    34.34999263
    >>> _Backward3b_P_hs(2700,5.0)
    88.39043281
    """

    nu = h / 2800
    sigma = s / 5.3
    suma = np.sum(Const.Backward3b_P_hs_n * (nu - 0.681) ** Const.Backward3b_P_hs_Li * (
                sigma - 0.792) ** Const.Backward3b_P_hs_Lj)
    return 16.6 / suma


def _Backward3_P_hs(h, s):
    """Backward equation for region 3, P=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    P : float
        Pressure, [MPa]
    """
    if s <= sc:
        return _Backward3a_P_hs(h, s)

    return _Backward3b_P_hs(h, s)


def _Backward3_sat_v_P(P, T, x):
    """Backward equation for region 3 for saturated state, vs=f(P,x)

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    x : integer
        Vapor quality, [-]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    Notes
    -----
    The vapor quality (x) can be 0 (saturated liquid) or 1 (saturated vapour)
    """
    if x == 0:
        if P < 19.00881189:
            region = "c"
        elif P < 21.0434:
            region = "s"
        elif P < 21.9316:
            region = "u"
        else:
            region = "y"
    else:
        if P < 20.5:
            region = "t"
        elif P < 21.0434:
            region = "r"
        elif P < 21.9009:
            region = "x"
        else:
            region = "z"

    return _Backward3x_v_PT(T, P, region)


def _Backward3_v_PT(P, T):
    """Backward equation for region 3, v=f(P,T)

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Table 2 and 10
    """
    if P > 40:
        if T <= _tab_P(P):
            region = "a"
        else:
            region = "b"
    elif 25 < P <= 40:
        tcd = _txx_P(P, "cd")
        tab = _tab_P(P)
        tef = _tef_P(P)
        if T <= tcd:
            region = "c"
        elif tcd < T <= tab:
            region = "d"
        elif tab < T <= tef:
            region = "e"
        else:
            region = "f"
    elif 23.5 < P <= 25:
        tcd = _txx_P(P, "cd")
        tgh = _txx_P(P, "gh")
        tef = _tef_P(P)
        tij = _txx_P(P, "ij")
        tjk = _txx_P(P, "jk")
        if T <= tcd:
            region = "c"
        elif tcd < T <= tgh:
            region = "g"
        elif tgh < T <= tef:
            region = "h"
        elif tef < T <= tij:
            region = "i"
        elif tij < T <= tjk:
            region = "j"
        else:
            region = "k"
    elif 23 < P <= 23.5:
        tcd = _txx_P(P, "cd")
        tgh = _txx_P(P, "gh")
        tef = _tef_P(P)
        tij = _txx_P(P, "ij")
        tjk = _txx_P(P, "jk")
        if T <= tcd:
            region = "c"
        elif tcd < T <= tgh:
            region = "l"
        elif tgh < T <= tef:
            region = "h"
        elif tef < T <= tij:
            region = "i"
        elif tij < T <= tjk:
            region = "j"
        else:
            region = "k"
    elif 22.5 < P <= 23:
        tcd = _txx_P(P, "cd")
        tgh = _txx_P(P, "gh")
        tmn = _txx_P(P, "mn")
        tef = _tef_P(P)
        top = _top_P(P)
        tij = _txx_P(P, "ij")
        tjk = _txx_P(P, "jk")
        if T <= tcd:
            region = "c"
        elif tcd < T <= tgh:
            region = "l"
        elif tgh < T <= tmn:
            region = "m"
        elif tmn < T <= tef:
            region = "n"
        elif tef < T <= top:
            region = "o"
        elif top < T <= tij:
            region = "p"
        elif tij < T <= tjk:
            region = "j"
        else:
            region = "k"
    elif _PSat_T(643.15) < P <= 22.5:
        tcd = _txx_P(P, "cd")
        tqu = _txx_P(P, "qu")
        trx = _txx_P(P, "rx")
        tjk = _txx_P(P, "jk")
        if T <= tcd:
            region = "c"
        elif tcd < T <= tqu:
            region = "q"
        elif tqu < T <= trx:
            # Table 10
            tef = _tef_P(P)
            twx = _twx_P(P)
            tuv = _txx_P(P, "uv")
            if 22.11 < P <= 22.5:
                if T <= tuv:
                    region = "u"
                elif tuv <= T <= tef:
                    region = "v"
                elif tef <= T <= twx:
                    region = "w"
                else:
                    region = "x"
            elif 22.064 < P <= 22.11:
                if T <= tuv:
                    region = "u"
                elif tuv <= T <= tef:
                    region = "y"
                elif tef <= T <= twx:
                    region = "z"
                else:
                    region = "x"
            elif T > _TSat_P(P):
                if _PSat_T(643.15) < P <= 21.90096265:
                    region = "x"
                elif 21.90096265 < P <= 22.064:
                    if T <= twx:
                        region = "z"
                    else:
                        region = "x"
            elif T <= _TSat_P(P):
                if _PSat_T(643.15) < P <= 21.93161551:
                    region = "u"
                elif 21.93161551 < P <= 22.064:
                    if T <= tuv:
                        region = "u"
                    else:
                        region = "y"
        elif trx < T <= tjk:
            region = "r"
        else:
            region = "k"
    elif 20.5 < P <= _PSat_T(643.15):
        tcd = _txx_P(P, "cd")
        Ts = _TSat_P(P)
        tjk = _txx_P(P, "jk")
        if T <= tcd:
            region = "c"
        elif tcd < T <= Ts:
            region = "s"
        elif Ts < T <= tjk:
            region = "r"
        else:
            region = "k"
    elif 19.00881189173929 < P <= 20.5:
        tcd = _txx_P(P, "cd")
        Ts = _TSat_P(P)
        if T <= tcd:
            region = "c"
        elif tcd < T <= Ts:
            region = "s"
        else:
            region = "t"
    elif Ps_623 < P <= 19.00881189173929:
        Ts = _TSat_P(P)
        if T <= Ts:
            region = "c"
        else:
            region = "t"

    return _Backward3x_v_PT(T, P, region)


def _Backward3x_v_PT(T, P, x):
    """Backward equation for region 3x, v=f(P,T)

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    x : char
        Region 3 subregion code

    Returns
    -------
    v : float
        Specific volume, [m³/kg]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations for Specific
    Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the
    IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water
    and Steam, http://www.iapws.org/relguide/Supp-VPT3-2016.pdf, Eq. 4-5

    Examples
    --------
    >>> _Backward3x_v_PT(630,50,"a")
    0.001470853100
    >>> _Backward3x_v_PT(670,80,"a")
    0.001503831359
    >>> _Backward3x_v_PT(710,50,"b")
    0.002204728587
    >>> _Backward3x_v_PT(750,80,"b")
    0.001973692940
    >>> _Backward3x_v_PT(630,20,"c")
    0.001761696406
    >>> _Backward3x_v_PT(650,30,"c")
    0.001819560617
    >>> _Backward3x_v_PT(656,26,"d")
    0.002245587720
    >>> _Backward3x_v_PT(670,30,"d")
    0.002506897702
    >>> _Backward3x_v_PT(661,26,"e")
    0.002970225962
    >>> _Backward3x_v_PT(675,30,"e")
    0.003004627086
    >>> _Backward3x_v_PT(671,26,"f")
    0.005019029401
    >>> _Backward3x_v_PT(690,30,"f")
    0.004656470142
    >>> _Backward3x_v_PT(649,23.6,"g")
    0.002163198378
    >>> _Backward3x_v_PT(650,24,"g")
    0.002166044161
    >>> _Backward3x_v_PT(652,23.6,"h")
    0.002651081407
    >>> _Backward3x_v_PT(654,24,"h")
    0.002967802335
    >>> _Backward3x_v_PT(653,23.6,"i")
    0.003273916816
    >>> _Backward3x_v_PT(655,24,"i")
    0.003550329864
    >>> _Backward3x_v_PT(655,23.5,"j")
    0.004545001142
    >>> _Backward3x_v_PT(660,24,"j")
    0.005100267704
    >>> _Backward3x_v_PT(660,23,"k")
    0.006109525997
    >>> _Backward3x_v_PT(670,24,"k")
    0.006427325645
    >>> _Backward3x_v_PT(646,22.6,"l")
    0.002117860851
    >>> _Backward3x_v_PT(646,23,"l")
    0.002062374674
    >>> _Backward3x_v_PT(648.6,22.6,"m")
    0.002533063780
    >>> _Backward3x_v_PT(649.3,22.8,"m")
    0.002572971781
    >>> _Backward3x_v_PT(649,22.6,"n")
    0.002923432711
    >>> _Backward3x_v_PT(649.7,22.8,"n")
    0.002913311494
    >>> _Backward3x_v_PT(649.1,22.6,"o")
    0.003131208996
    >>> _Backward3x_v_PT(649.9,22.8,"o")
    0.003221160278
    >>> _Backward3x_v_PT(649.4,22.6,"p")
    0.003715596186
    >>> _Backward3x_v_PT(650.2,22.8,"p")
    0.003664754790
    >>> _Backward3x_v_PT(640,21.1,"q")
    0.001970999272
    >>> _Backward3x_v_PT(643,21.8,"q")
    0.002043919161
    >>> _Backward3x_v_PT(644,21.1,"r")
    0.005251009921
    >>> _Backward3x_v_PT(648,21.8,"r")
    0.005256844741
    >>> _Backward3x_v_PT(635,19.1,"s")
    0.001932829079
    >>> _Backward3x_v_PT(638,20,"s")
    0.001985387227
    >>> _Backward3x_v_PT(626,17,"t")
    0.008483262001
    >>> _Backward3x_v_PT(640,20,"t")
    0.006227528101
    >>> _Backward3x_v_PT(644.6,21.5,"u")
    0.002268366647
    >>> _Backward3x_v_PT(646.1,22,"u")
    0.002296350553
    >>> _Backward3x_v_PT(648.6,22.5,"v")
    0.002832373260
    >>> _Backward3x_v_PT(647.9,22.3,"v")
    0.002811424405
    >>> _Backward3x_v_PT(647.5,22.15,"w")
    0.003694032281
    >>> _Backward3x_v_PT(648.1,22.3,"w")
    0.003622226305
    >>> _Backward3x_v_PT(648,22.11,"x")
    0.004528072649
    >>> _Backward3x_v_PT(649,22.3,"x")
    0.004556905799
    >>> _Backward3x_v_PT(646.84,22,"y")
    0.002698354719
    >>> _Backward3x_v_PT(647.05,22.064,"y")
    0.002717655648
    >>> _Backward3x_v_PT(646.89,22,"z")
    0.003798732962
    >>> _Backward3x_v_PT(647.15,22.064,"z")
    0.003701940009
    """
    par = {
        "a": [0.0024, 100, 760, 0.085, 0.817, 1, 1, 1],
        "b": [0.0041, 100, 860, 0.280, 0.779, 1, 1, 1],
        "c": [0.0022, 40, 690, 0.259, 0.903, 1, 1, 1],
        "d": [0.0029, 40, 690, 0.559, 0.939, 1, 1, 4],
        "e": [0.0032, 40, 710, 0.587, 0.918, 1, 1, 1],
        "f": [0.0064, 40, 730, 0.587, 0.891, 0.5, 1, 4],
        "g": [0.0027, 25, 660, 0.872, 0.971, 1, 1, 4],
        "h": [0.0032, 25, 660, 0.898, 0.983, 1, 1, 4],
        "i": [0.0041, 25, 660, 0.910, 0.984, 0.5, 1, 4],
        "j": [0.0054, 25, 670, 0.875, 0.964, 0.5, 1, 4],
        "k": [0.0077, 25, 680, 0.802, 0.935, 1, 1, 1],
        "l": [0.0026, 24, 650, 0.908, 0.989, 1, 1, 4],
        "m": [0.0028, 23, 650, 1.000, 0.997, 1, 0.25, 1],
        "n": [0.0031, 23, 650, 0.976, 0.997, None, None, None],
        "o": [0.0034, 23, 650, 0.974, 0.996, 0.5, 1, 1],
        "p": [0.0041, 23, 650, 0.972, 0.997, 0.5, 1, 1],
        "q": [0.0022, 23, 650, 0.848, 0.983, 1, 1, 4],
        "r": [0.0054, 23, 650, 0.874, 0.982, 1, 1, 1],
        "s": [0.0022, 21, 640, 0.886, 0.990, 1, 1, 4],
        "t": [0.0088, 20, 650, 0.803, 1.020, 1, 1, 1],
        "u": [0.0026, 23, 650, 0.902, 0.988, 1, 1, 1],
        "v": [0.0031, 23, 650, 0.960, 0.995, 1, 1, 1],
        "w": [0.0039, 23, 650, 0.959, 0.995, 1, 1, 4],
        "x": [0.0049, 23, 650, 0.910, 0.988, 1, 1, 1],
        "y": [0.0031, 22, 650, 0.996, 0.994, 1, 1, 4],
        "z": [0.0038, 22, 650, 0.993, 0.994, 1, 1, 4],
    }

    Li = {
        "a": [-12, -12, -12, -10, -10, -10, -8, -8, -8, -6, -5, -5, -5, -4, -3,
              -3, -3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 1, 1, 2, 2],
        "b": [-12, -12, -10, -10, -8, -6, -6, -6, -5, -5, -5, -4, -4, -4, -3,
              -3, -3, -3, -3, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 3, 4, 4],
        "c": [-12, -12, -12, -10, -10, -10, -8, -8, -8, -6, -5, -5, -5, -4, -4,
              -3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3,
              8],
        "d": [-12, -12, -12, -12, -12, -12, -10, -10, -10, -10, -10, -10, -10,
              -8, -8, -8, -8, -6, -6, -5, -5, -5, -5, -4, -4, -4, -3, -3, -2,
              -2, -1, -1, -1, 0, 0, 1, 1, 3],
        "e": [-12, -12, -10, -10, -10, -10, -10, -8, -8, -8, -6, -5, -4, -4,
              -3, -3, -3, -2, -2, -2, -2, -1, 0, 0, 1, 1, 1, 2, 2],
        "f": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7,
              10, 12, 12, 12, 14, 14, 14, 14, 14, 16, 16, 18, 18, 20, 20, 20,
              22, 24, 24, 28, 32],
        "g": [-12, -12, -12, -12, -12, -12, -10, -10, -10, -8, -8, -8, -8, -6,
              -6, -5, -5, -4, -3, -2, -2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1,
              3, 5, 6, 8, 10, 10],
        "h": [-12, -12, -10, -10, -10, -10, -10, -10, -8, -8, -8, -8, -8, -6,
              -6, -6, -5, -5, -5, -4, -4, -3, -3, -2, -1, -1, 0, 1, 1],
        "i": [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 7, 7, 8, 8, 10,
              12, 12, 12, 14, 14, 14, 14, 18, 18, 18, 18, 18, 20, 20, 22, 24,
              24, 32, 32, 36, 36],
        "j": [0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5, 6, 10, 12, 12, 14, 14,
              14, 16, 18, 20, 20, 24, 24, 28, 28],
        "k": [-2, -2, -1, -1, 0, -0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2,
              2, 2, 2, 2, 5, 5, 5, 6, 6, 6, 6, 8, 10, 12],
        "l": [-12, -12, -12, -12, -12, -10, -10, -8, -8, -8, -8, -8, -8, -8,
              -6, -5, -5, -4, -4, -3, -3, -3, -3, -2, -2, -2, -1, -1, -1, 0, 0,
              0, 0, 1, 1, 2, 4, 5, 5, 6, 10, 10, 14],
        "m": [0, 3, 8, 20, 1, 3, 4, 5, 1, 6, 2, 4, 14, 2, 5, 3, 0, 1, 1, 1, 28,
              2, 16, 0, 5, 0, 3, 4, 12, 16, 1, 8, 14, 0, 2, 3, 4, 8, 14, 24],
        "n": [0, 3, 4, 6, 7, 10, 12, 14, 18, 0, 3, 5, 6, 8, 12, 0, 3, 7, 12,
              2, 3, 4, 2, 4, 7, 4, 3, 5, 6, 0, 0, 3, 1, 0, 1, 0, 1, 0, 1],
        "o": [0, 0, 0, 2, 3, 4, 4, 4, 4, 4, 5, 5, 6, 7, 8, 8, 8, 10, 10, 14,
              14, 20, 20, 24],
        "p": [0, 0, 0, 0, 1, 2, 3, 3, 4, 6, 7, 7, 8, 10, 12, 12, 12, 14, 14,
              14, 16, 18, 20, 22, 24, 24, 36],
        "q": [-12, -12, -10, -10, -10, -10, -8, -6, -5, -5, -4, -4, -3, -2,
              -2, -2, -2, -1, -1, -1, 0, 1, 1, 1],
        "r": [-8, -8, -3, -3, -3, -3, -3, 0, 0, 0, 0, 3, 3, 8, 8, 8, 8, 10,
              10, 10, 10, 10, 10, 10, 10, 12, 14],
        "s": [-12, -12, -10, -8, -6, -5, -5, -4, -4, -3, -3, -2, -1, -1, -1, 0,
              0, 0, 0, 1, 1, 3, 3, 3, 4, 4, 4, 5, 14],
        "t": [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 7, 7, 7, 7, 7, 10, 10, 10,
              10, 10, 18, 20, 22, 22, 24, 28, 32, 32, 32, 36],
        "u": [-12, -10, -10, -10, -8, -8, -8, -6, -6, -5, -5, -5, -3, -1, -1,
              -1, -1, 0, 0, 1, 2, 2, 3, 5, 5, 5, 6, 6, 8, 8, 10, 12, 12, 12,
              14, 14, 14, 14],
        "v": [-10, -8, -6, -6, -6, -6, -6, -6, -5, -5, -5, -5, -5, -5, -4, -4,
              -4, -4, -3, -3, -3, -2, -2, -1, -1, 0, 0, 0, 1, 1, 3, 4, 4, 4, 5,
              8, 10, 12, 14],
        "w": [-12, -12, -10, -10, -8, -8, -8, -6, -6, -6, -6, -5, -4, -4, -3,
              -3, -2, -2, -1, -1, -1, 0, 0, 1, 2, 2, 3, 3, 5, 5, 5, 8, 8, 10,
              10],
        "x": [-8, -6, -5, -4, -4, -4, -3, -3, -1, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4,
              5, 5, 5, 6, 8, 8, 8, 8, 10, 12, 12, 12, 12, 14, 14, 14, 14],
        "y": [0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 8, 8, 10, 12],
        "z": [-8, -6, -5, -5, -4, -4, -4, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 6,
              6, 6, 6, 8, 8]}

    Lj = {
        "a": [5, 10, 12, 5, 10, 12, 5, 8, 10, 1, 1, 5, 10, 8, 0, 1, 3, 6, 0,
              2, 3, 0, 1, 2, 0, 1, 0, 2, 0, 2],
        "b": [10, 12, 8, 14, 8, 5, 6, 8, 5, 8, 10, 2, 4, 5, 0, 1, 2, 3, 5, 0,
              2, 5, 0, 2, 0, 1, 0, 2, 0, 2, 0, 1],
        "c": [6, 8, 10, 6, 8, 10, 5, 6, 7, 8, 1, 4, 7, 2, 8, 0, 3, 0, 4, 5, 0,
              1, 2, 0, 1, 2, 0, 2, 0, 1, 3, 7, 0, 7, 1],
        "d": [4, 6, 7, 10, 12, 16, 0, 2, 4, 6, 8, 10, 14, 3, 7, 8, 10, 6, 8, 1,
              2, 5, 7, 0, 1, 7, 2, 4, 0, 1, 0, 1, 5, 0, 2, 0, 6, 0],
        "e": [14, 16, 3, 6, 10, 14, 16, 7, 8, 10, 6, 6, 2, 4, 2, 6, 7, 0, 1,
              3, 4, 0, 0, 1, 0, 4, 6, 0, 2],
        "f": [-3, -2, -1, 0, 1, 2, -1, 1, 2, 3, 0, 1, -5, -2, 0, -3, -8, 1, -6,
              -4, 1, -6, -10, -8, -4, -12, -10, -8, -6, -4, -10, -8, -12, -10,
              -12, -10, -6, -12, -12, -4, -12, -12],
        "g": [7, 12, 14, 18, 22, 24, 14, 20, 24, 7, 8, 10, 12, 8, 22, 7, 20,
              22, 7, 3, 5, 14, 24, 2, 8, 18, 0, 1, 2, 0, 1, 3, 24, 22, 12, 3,
              0, 6],
        "h": [8, 12, 4, 6, 8, 10, 14, 16, 0, 1, 6, 7, 8, 4, 6, 8, 2, 3, 4, 2,
              4, 1, 2, 0, 0, 2, 0, 0, 2],
        "i": [0, 1, 10, -4, -2, -1, 0, 0, -5, 0, -3, -2, -1, -6, -1, 12, -4,
              -3, -6, 10, -8, -12, -6, -4, -10, -8, -4, 5, -12, -10, -8, -6,
              2, -12, -10, -12, -12, -8, -10, -5, -10, -8],
        "j": [-1, 0, 1, -2, -1, 1, -1, 1, -2, -2, 2, -3, -2, 0, 3, -6, -8, -3,
              -10, -8, -5, -10, -12, -12, -10, -12, -6, -12, -5],
        "k": [10, 12, -5, 6, -12, -6, -2, -1, 0, 1, 2, 3, 14, -3, -2, 0, 1, 2,
              -8, -6, -3, -2, 0, 4, -12, -6, -3, -12, -10, -8, -5, -12, -12,
              -10],
        "l": [14, 16, 18, 20, 22, 14, 24, 6, 10, 12, 14, 18, 24, 36, 8, 4, 5,
              7, 16, 1, 3, 18, 20, 2, 3, 10, 0, 1, 3, 0, 1, 2, 12, 0, 16, 1, 0,
              0, 1, 14, 4, 12, 10],
        "m": [0, 0, 0, 2, 5, 5, 5, 5, 6, 6, 7, 8, 8, 10, 10, 12, 14, 14, 18,
              20, 20, 22, 22, 24, 24, 28, 28, 28, 28, 28, 32, 32, 32, 36, 36,
              36, 36, 36, 36, 36],
        "n": [-12, -12, -12, -12, -12, -12, -12, -12, -12, -10, -10, -10, -10,
              -10, -10, -8, -8, -8, -8, -6, -6, -6, -5, -5, -5, -4, -3, -3,
              -3, -2, -1, -1, 0, 1, 1, 2, 4, 5, 6],
        "o": [-12, -4, -1, -1, -10, -12, -8, -5, -4, -1, -4, -3, -8, -12, -10,
              -8, -4, -12, -8, -12, -8, -12, -10, -12],
        "p": [-1, 0, 1, 2, 1, -1, -3, 0, -2, -2, -5, -4, -2, -3, -12, -6, -5,
              -10, -8, -3, -8, -8, -10, -10, -12, -8, -12],
        "q": [10, 12, 6, 7, 8, 10, 8, 6, 2, 5, 3, 4, 3, 0, 1, 2, 4, 0, 1, 2,
              0, 0, 1, 3],
        "r": [6, 14, -3, 3, 4, 5, 8, -1, 0, 1, 5, -6, -2, -12, -10, -8, -5,
              -12, -10, -8, -6, -5, -4, -3, -2, -12, -12],
        "s": [20, 24, 22, 14, 36, 8, 16, 6, 32, 3, 8, 4, 1, 2, 3, 0, 1, 4, 28,
              0, 32, 0, 1, 2, 3, 18, 24, 4, 24],
        "t": [0, 1, 4, 12, 0, 10, 0, 6, 14, 3, 8, 0, 10, 3, 4, 7, 20, 36, 10,
              12, 14, 16, 22, 18, 32, 22, 36, 24, 28, 22, 32, 36, 36],
        "u": [14, 10, 12, 14, 10, 12, 14, 8, 12, 4, 8, 12, 2, -1, 1, 12, 14,
              -3, 1, -2, 5, 10, -5, -4, 2, 3, -5, 2, -8, 8, -4, -12, -4, 4,
              -12, -10, -6, 6],
        "v": [-8, -12, -12, -3, 5, 6, 8, 10, 1, 2, 6, 8, 10, 14, -12, -10, -6,
              10, -3, 10, 12, 2, 4, -2, 0, -2, 6, 10, -12, -10, 3, -6, 3, 10,
              2, -12, -2, -3, 1],
        "w": [8, 14, -1, 8, 6, 8, 14, -4, -3, 2, 8, -10, -1, 3, -10, 3, 1, 2,
              -8, -4, 1, -12, 1, -1, -1, 2, -12, -5, -10, -8, -6, -12, -10,
              -12, -8],
        "x": [14, 10, 10, 1, 2, 14, -2, 12, 5, 0, 4, 10, -10, -1, 6, -12, 0,
              8, 3, -6, -2, 1, 1, -6, -3, 1, 8, -8, -10, -8, -5, -4, -12, -10,
              -8, -6],
        "y": [-3, 1, 5, 8, 8, -4, -1, 4, 5, -8, 4, 8, -6, 6, -2, 1, -8, -2,
              -5, -8],
        "z": [3, 6, 6, 8, 5, 6, 8, -2, 5, 6, 2, -6, 3, 1, 6, -6, -2, -6, -5,
              -4, -1, -8, -4]}

    n = {
        "a": [0.110879558823853e-2, 0.572616740810616e3, -0.767051948380852e5,
              -0.253321069529674e-1, 0.628008049345689e4, 0.234105654131876e6,
              0.216867826045856, -0.156237904341963e3, -0.269893956176613e5,
              -0.180407100085505e-3, 0.116732227668261e-2, 0.266987040856040e2,
              0.282776617243286e5, -0.242431520029523e4, 0.435217323022733e-3,
              -0.122494831387441e-1, 0.179357604019989e1, 0.442729521058314e2,
              -0.593223489018342e-2, 0.453186261685774, 0.135825703129140e1,
              0.408748415856745e-1, 0.474686397863312, 0.118646814997915e1,
              0.546987265727549, 0.195266770452643, -0.502268790869663e-1,
              -0.369645308193377, 0.633828037528420e-2, 0.797441793901017e-1],
        "b": [-0.827670470003621e-1, 0.416887126010565e2, 0.483651982197059e-1,
              -0.291032084950276e5, -0.111422582236948e3, -.202300083904014e-1,
              0.294002509338515e3, 0.140244997609658e3, -0.344384158811459e3,
              0.361182452612149e3, -0.140699677420738e4, -0.202023902676481e-2,
              0.171346792457471e3, -0.425597804058632e1, 0.691346085000334e-5,
              0.151140509678925e-2, -0.416375290166236e-1, -.413754957011042e2,
              -0.506673295721637e2, -0.572212965569023e-3, 0.608817368401785e1,
              0.239600660256161e2, 0.122261479925384e-1, 0.216356057692938e1,
              0.398198903368642, -0.116892827834085, -0.102845919373532,
              -0.492676637589284, 0.655540456406790e-1, -0.240462535078530,
              -0.269798180310075e-1, 0.128369435967012],
        "c": [0.311967788763030e1, 0.276713458847564e5, 0.322583103403269e8,
              -0.342416065095363e3, -0.899732529907377e6, -0.793892049821251e8,
              0.953193003217388e2, 0.229784742345072e4, 0.175336675322499e6,
              0.791214365222792e7, 0.319933345844209e-4, -0.659508863555767e2,
              -0.833426563212851e6, 0.645734680583292e-1, -0.382031020570813e7,
              0.406398848470079e-4, 0.310327498492008e2, -0.892996718483724e-3,
              0.234604891591616e3, 0.377515668966951e4, 0.158646812591361e-1,
              0.707906336241843, 0.126016225146570e2, 0.736143655772152,
              0.676544268999101, -0.178100588189137e2, -0.156531975531713,
              0.117707430048158e2, 0.840143653860447e-1, -0.186442467471949,
              -0.440170203949645e2, 0.123290423502494e7, -0.240650039730845e-1,
              -0.107077716660869e7, 0.438319858566475e-1],
        "d": [-0.452484847171645e-9, .315210389538801e-4, -.214991352047545e-2,
              0.508058874808345e3, -0.127123036845932e8, 0.115371133120497e13,
              -.197805728776273e-15, .241554806033972e-10,
              -.156481703640525e-5, 0.277211346836625e-2, -0.203578994462286e2,
              0.144369489909053e7, -0.411254217946539e11, 0.623449786243773e-5,
              -.221774281146038e2, -0.689315087933158e5, -0.195419525060713e8,
              0.316373510564015e4, 0.224040754426988e7, -0.436701347922356e-5,
              -.404213852833996e-3, -0.348153203414663e3, -0.385294213555289e6,
              0.135203700099403e-6, 0.134648383271089e-3, 0.125031835351736e6,
              0.968123678455841e-1, 0.225660517512438e3, -0.190102435341872e-3,
              -.299628410819229e-1, 0.500833915372121e-2, 0.387842482998411,
              -0.138535367777182e4, 0.870745245971773, 0.171946252068742e1,
              -.326650121426383e-1, 0.498044171727877e4, 0.551478022765087e-2],
        "e": [0.715815808404721e9, -0.114328360753449e12, .376531002015720e-11,
              -0.903983668691157e-4, 0.665695908836252e6, 0.535364174960127e10,
              0.794977402335603e11, 0.922230563421437e2, -0.142586073991215e6,
              -0.111796381424162e7, 0.896121629640760e4, -0.669989239070491e4,
              0.451242538486834e-2, -0.339731325977713e2, -0.120523111552278e1,
              0.475992667717124e5, -0.266627750390341e6, -0.153314954386524e-3,
              0.305638404828265, 0.123654999499486e3, -0.104390794213011e4,
              -0.157496516174308e-1, 0.685331118940253, 0.178373462873903e1,
              -0.544674124878910, 0.204529931318843e4, -0.228342359328752e5,
              0.413197481515899, -0.341931835910405e2],
        "f": [-0.251756547792325e-7, .601307193668763e-5, -.100615977450049e-2,
              0.999969140252192, 0.214107759236486e1, -0.165175571959086e2,
              -0.141987303638727e-2, 0.269251915156554e1, 0.349741815858722e2,
              -0.300208695771783e2, -0.131546288252539e1, -0.839091277286169e1,
              0.181545608337015e-9, -0.591099206478909e-3, 0.152115067087106e1,
              0.252956470663225e-4, 0.100726265203786e-14, -0.14977453386065e1,
              -0.793940970562969e-9, -0.150290891264717e-3, .151205531275133e1,
              0.470942606221652e-5, .195049710391712e-12, -.911627886266077e-8,
              .604374640201265e-3, -.225132933900136e-15, .610916973582981e-11,
              -.303063908043404e-6, -.137796070798409e-4, -.919296736666106e-3,
              .639288223132545e-9, .753259479898699e-6, -0.400321478682929e-12,
              .756140294351614e-8, -.912082054034891e-11, -.237612381140539e-7,
              0.269586010591874e-4, -.732828135157839e-10, .241995578306660e-9,
              -.405735532730322e-3, .189424143498011e-9, -.486632965074563e-9],
        "g": [0.412209020652996e-4, -0.114987238280587e7, 0.948180885032080e10,
              -0.195788865718971e18, 0.4962507048713e25, -0.105549884548496e29,
              -0.758642165988278e12, -.922172769596101e23, .725379072059348e30,
              -0.617718249205859e2, 0.107555033344858e5, -0.379545802336487e8,
              0.228646846221831e12, -0.499741093010619e7, -.280214310054101e31,
              0.104915406769586e7, 0.613754229168619e28, 0.802056715528378e32,
              -0.298617819828065e8, -0.910782540134681e2, 0.135033227281565e6,
              -0.712949383408211e19, -0.104578785289542e37, .304331584444093e2,
              0.593250797959445e10, -0.364174062110798e28, 0.921791403532461,
              -0.337693609657471, -0.724644143758508e2, -0.110480239272601,
              0.536516031875059e1, -0.291441872156205e4, 0.616338176535305e40,
              -0.120889175861180e39, 0.818396024524612e23, 0.940781944835829e9,
              -0.367279669545448e5, -0.837513931798655e16],
        "h": [0.561379678887577e-1, 0.774135421587083e10, 0.111482975877938e-8,
              -0.143987128208183e-2, 0.193696558764920e4, -0.605971823585005e9,
              0.171951568124337e14, -.185461154985145e17, 0.38785116807801e-16,
              -.395464327846105e-13, -0.170875935679023e3, -0.21201062070122e4,
              0.177683337348191e8, 0.110177443629575e2, -0.234396091693313e6,
              -0.656174421999594e7, 0.156362212977396e-4, -0.212946257021400e1,
              0.135249306374858e2, 0.177189164145813, 0.139499167345464e4,
              -0.703670932036388e-2, -0.152011044389648, 0.981916922991113e-4,
              0.147199658618076e-2, 0.202618487025578e2, 0.899345518944240,
              -0.211346402240858, 0.249971752957491e2],
        "i": [0.106905684359136e1, -0.148620857922333e1, 0.259862256980408e15,
              -.446352055678749e-11, -.566620757170032e-6,
              -.235302885736849e-2, -0.269226321968839, 0.922024992944392e1,
              0.357633505503772e-11, -.173942565562222e2, 0.700681785556229e-5,
              -.267050351075768e-3, -.231779669675624e1, -.753533046979752e-12,
              .481337131452891e1, -0.223286270422356e22, -.118746004987383e-4,
              .646412934136496e-2, -0.410588536330937e-9, .422739537057241e20,
              .313698180473812e-12, 0.16439533434504e-23, -.339823323754373e-5,
              -.135268639905021e-1, -.723252514211625e-14, .184386437538366e-8,
              -.463959533752385e-1, -.99226310037675e14, .688169154439335e-16,
              -.222620998452197e-10, -.540843018624083e-7, .345570606200257e-2,
              .422275800304086e11, -.126974478770487e-14, .927237985153679e-9,
              .612670812016489e-13, -.722693924063497e-11,
              -.383669502636822e-3, .374684572410204e-3, -0.931976897511086e5,
              -0.247690616026922e-1, .658110546759474e2],
        "j": [-0.111371317395540e-3, 0.100342892423685e1, 0.530615581928979e1,
              0.179058760078792e-5, -0.728541958464774e-3, -.187576133371704e2,
              0.199060874071849e-2, 0.243574755377290e2, -0.177040785499444e-3,
              -0.25968038522713e-2, -0.198704578406823e3, 0.738627790224287e-4,
              -0.236264692844138e-2, -0.161023121314333e1, 0.622322971786473e4,
              -.960754116701669e-8, -.510572269720488e-10, .767373781404211e-2,
              .663855469485254e-14, -.717590735526745e-9, 0.146564542926508e-4,
              .309029474277013e-11, -.464216300971708e-15,
              -.390499637961161e-13, -.236716126781431e-9,
              .454652854268717e-11, -.422271787482497e-2,
              0.283911742354706e-10, 0.270929002720228e1],
        "k": [-0.401215699576099e9, 0.484501478318406e11, .394721471363678e-14,
              .372629967374147e5, -.369794374168666e-29, -.380436407012452e-14,
              0.475361629970233e-6, -0.879148916140706e-3, 0.844317863844331,
              0.122433162656600e2, -0.104529634830279e3, 0.589702771277429e3,
              -.291026851164444e14, .170343072841850e-5, -0.277617606975748e-3,
              -0.344709605486686e1, 0.221333862447095e2, -0.194646110037079e3,
              .808354639772825e-15, -.18084520914547e-10, -.696664158132412e-5,
              -0.181057560300994e-2, 0.255830298579027e1, 0.328913873658481e4,
              -.173270241249904e-18, -.661876792558034e-6, -.39568892342125e-2,
              .604203299819132e-17, -.400879935920517e-13, .160751107464958e-8,
              .383719409025556e-4, -.649565446702457e-14, -.149095328506e-11,
              0.541449377329581e-8],
        "l": [0.260702058647537e10, -.188277213604704e15, 0.554923870289667e19,
              -.758966946387758e23, .413865186848908e27, -.81503800073806e12,
              -.381458260489955e33, -.123239564600519e-1, 0.226095631437174e8,
              -.49501780950672e12, 0.529482996422863e16, -0.444359478746295e23,
              .521635864527315e35, -0.487095672740742e55, -0.714430209937547e6,
              0.127868634615495, -0.100752127917598e2, 0.777451437960990e7,
              -.108105480796471e25, -.357578581169659e-5, -0.212857169423484e1,
              0.270706111085238e30, -0.695953622348829e33, 0.110609027472280,
              0.721559163361354e2, -0.306367307532219e15, 0.265839618885530e-4,
              0.253392392889754e-1, -0.214443041836579e3, 0.937846601489667,
              0.223184043101700e1, 0.338401222509191e2, 0.494237237179718e21,
              -0.198068404154428, -0.141415349881140e31, -0.993862421613651e2,
              0.125070534142731e3, -0.996473529004439e3, 0.473137909872765e5,
              0.116662121219322e33, -0.315874976271533e16,
              -0.445703369196945e33, 0.642794932373694e33],
        "m": [0.811384363481847, -0.568199310990094e4, -0.178657198172556e11,
              0.795537657613427e32, -0.814568209346872e5, -0.659774567602874e8,
              -.152861148659302e11, -0.560165667510446e12, 0.458384828593949e6,
              -0.385754000383848e14, 0.453735800004273e8, 0.939454935735563e12,
              .266572856432938e28, -0.547578313899097e10, 0.200725701112386e15,
              0.185007245563239e13, 0.185135446828337e9, -0.170451090076385e12,
              0.157890366037614e15, -0.202530509748774e16, 0.36819392618357e60,
              0.170215539458936e18, 0.639234909918741e42, -.821698160721956e15,
              -.795260241872306e24, 0.23341586947851e18, -0.600079934586803e23,
              0.594584382273384e25, 0.189461279349492e40, -.810093428842645e46,
              0.188813911076809e22, 0.111052244098768e36, 0.291133958602503e46,
              -.329421923951460e22, -.137570282536696e26, 0.181508996303902e28,
              -.346865122768353e30, -.21196114877426e38, -0.128617899887675e49,
              0.479817895699239e65],
        "n": [.280967799943151e-38, .614869006573609e-30, .582238667048942e-27,
              .390628369238462e-22, .821445758255119e-20, .402137961842776e-14,
              .651718171878301e-12, -.211773355803058e-7, 0.264953354380072e-2,
              -.135031446451331e-31, -.607246643970893e-23,
              -.402352115234494e-18, -.744938506925544e-16,
              .189917206526237e-12, .364975183508473e-5, .177274872361946e-25,
              -.334952758812999e-18, -.421537726098389e-8,
              -.391048167929649e-1, .541276911564176e-13, .705412100773699e-11,
              .258585887897486e-8, -.493111362030162e-10, -.158649699894543e-5,
              -0.525037427886100, 0.220019901729615e-2, -0.643064132636925e-2,
              0.629154149015048e2, 0.135147318617061e3, 0.240560808321713e-6,
              -.890763306701305e-3, -0.440209599407714e4, -0.302807107747776e3,
              0.159158748314599e4, 0.232534272709876e6, -0.792681207132600e6,
              -.869871364662769e11, .354542769185671e12, 0.400849240129329e15],
        "o": [.128746023979718e-34, -.735234770382342e-11, .28907869214915e-2,
              0.244482731907223, 0.141733492030985e-23, -0.354533853059476e-28,
              -.594539202901431e-17, -.585188401782779e-8, .201377325411803e-5,
              0.138647388209306e1, -0.173959365084772e-4, 0.137680878349369e-2,
              .814897605805513e-14, .425596631351839e-25,
              -.387449113787755e-17, .13981474793024e-12, -.171849638951521e-2,
              0.641890529513296e-21, .118960578072018e-10,
              -.155282762571611e-17, .233907907347507e-7,
              -.174093247766213e-12, .377682649089149e-8,
              -.516720236575302e-10],
        "p": [-0.982825342010366e-4, 0.105145700850612e1, 0.116033094095084e3,
              0.324664750281543e4, -0.123592348610137e4, -0.561403450013495e-1,
              0.856677401640869e-7, 0.236313425393924e3, 0.972503292350109e-2,
              -.103001994531927e1, -0.149653706199162e-8, -.215743778861592e-4,
              -0.834452198291445e1, 0.586602660564988, 0.343480022104968e-25,
              .816256095947021e-5, .294985697916798e-2, 0.711730466276584e-16,
              0.400954763806941e-9, 0.107766027032853e2, -0.409449599138182e-6,
              -.729121307758902e-5, 0.677107970938909e-8, 0.602745973022975e-7,
              -.382323011855257e-10, .179946628317437e-2,
              -.345042834640005e-3],
        "q": [-0.820433843259950e5, 0.473271518461586e11, -.805950021005413e-1,
              0.328600025435980e2, -0.35661702998249e4, -0.172985781433335e10,
              0.351769232729192e8, -0.775489259985144e6, 0.710346691966018e-4,
              0.993499883820274e5, -0.642094171904570, -0.612842816820083e4,
              .232808472983776e3, -0.142808220416837e-4, -0.643596060678456e-2,
              -0.428577227475614e1, 0.225689939161918e4, 0.100355651721510e-2,
              0.333491455143516, 0.109697576888873e1, 0.961917379376452,
              -0.838165632204598e-1, 0.247795908411492e1, -.319114969006533e4],
        "r": [.144165955660863e-2, -.701438599628258e13, -.830946716459219e-16,
              0.261975135368109, 0.393097214706245e3, -0.104334030654021e5,
              0.490112654154211e9, -0.147104222772069e-3, 0.103602748043408e1,
              0.305308890065089e1, -0.399745276971264e7, 0.569233719593750e-11,
              -.464923504407778e-1, -.535400396512906e-17,
              .399988795693162e-12, -.536479560201811e-6, .159536722411202e-1,
              .270303248860217e-14, .244247453858506e-7, -0.983430636716454e-5,
              0.663513144224454e-1, -0.993456957845006e1, 0.546491323528491e3,
              -0.143365406393758e5, 0.150764974125511e6, -.337209709340105e-9,
              0.377501980025469e-8],
        "s": [-0.532466612140254e23, .100415480000824e32, -.191540001821367e30,
              0.105618377808847e17, 0.202281884477061e59, 0.884585472596134e8,
              0.166540181638363e23, -0.313563197669111e6, -.185662327545324e54,
              -.624942093918942e-1, -0.50416072413259e10, 0.187514491833092e5,
              0.121399979993217e-2, 0.188317043049455e1, -0.167073503962060e4,
              0.965961650599775, 0.294885696802488e1, -0.653915627346115e5,
              0.604012200163444e50, -0.198339358557937, -0.175984090163501e58,
              0.356314881403987e1, -0.575991255144384e3, 0.456213415338071e5,
              -.109174044987829e8, 0.437796099975134e34, -0.616552611135792e46,
              0.193568768917797e10, 0.950898170425042e54],
        "t": [0.155287249586268e1, 0.664235115009031e1, -0.289366236727210e4,
              -0.385923202309848e13, -.291002915783761e1, -.829088246858083e12,
              0.176814899675218e1, -0.534686695713469e9, 0.160464608687834e18,
              0.196435366560186e6, 0.156637427541729e13, -0.178154560260006e1,
              -0.229746237623692e16, 0.385659001648006e8, 0.110554446790543e10,
              -.677073830687349e14, -.327910592086523e31, -.341552040860644e51,
              -.527251339709047e21, .245375640937055e24, -0.168776617209269e27,
              .358958955867578e29, -0.656475280339411e36, 0.355286045512301e39,
              .569021454413270e58, -.700584546433113e48, -0.705772623326374e65,
              0.166861176200148e53, -.300475129680486e61, -.668481295196808e51,
              .428432338620678e69, -.444227367758304e72, -.281396013562745e77],
        "u": [0.122088349258355e18, 0.104216468608488e10, -.882666931564652e16,
              .259929510849499e20, 0.222612779142211e15, -0.878473585050085e18,
              -0.314432577551552e22, -.216934916996285e13, .159079648196849e21,
              -.339567617303423e3, 0.884387651337836e13, -0.843405926846418e21,
              0.114178193518022e2, -0.122708229235641e-3, -0.106201671767107e3,
              .903443213959313e25, -0.693996270370852e28, 0.648916718965575e-8,
              0.718957567127851e4, 0.105581745346187e-2, -0.651903203602581e15,
              -0.160116813274676e25, -0.510254294237837e-8, -0.152355388953402,
              0.677143292290144e12, 0.276378438378930e15, 0.116862983141686e-1,
              -.301426947980171e14, 0.169719813884840e-7, 0.104674840020929e27,
              -0.10801690456014e5, -0.990623601934295e-12, 0.536116483602738e7,
              .226145963747881e22, -0.488731565776210e-9, 0.151001548880670e-4,
              -0.227700464643920e5, -0.781754507698846e28],
        "v": [-.415652812061591e-54, .177441742924043e-60,
              -.357078668203377e-54, 0.359252213604114e-25,
              -0.259123736380269e2, 0.594619766193460e5, -0.624184007103158e11,
              0.313080299915944e17, .105006446192036e-8, -0.192824336984852e-5,
              0.654144373749937e6, 0.513117462865044e13, -.697595750347391e19,
              -.103977184454767e29, .119563135540666e-47,
              -.436677034051655e-41, .926990036530639e-29, .587793105620748e21,
              .280375725094731e-17, -0.192359972440634e23, .742705723302738e27,
              -0.517429682450605e2, 0.820612048645469e7, -0.188214882341448e-8,
              .184587261114837e-1, -0.135830407782663e-5, -.723681885626348e17,
              -.223449194054124e27, -.111526741826431e-34,
              .276032601145151e-28, 0.134856491567853e15, 0.652440293345860e-9,
              0.510655119774360e17, -.468138358908732e32, -.760667491183279e16,
              -.417247986986821e-18, 0.312545677756104e14,
              -.100375333864186e15, .247761392329058e27],
        "w": [-.586219133817016e-7, -.894460355005526e11, .531168037519774e-30,
              0.109892402329239, -0.575368389425212e-1, 0.228276853990249e5,
              -.158548609655002e19, .329865748576503e-27,
              -.634987981190669e-24, 0.615762068640611e-8, -.961109240985747e8,
              -.406274286652625e-44, -0.471103725498077e-12, 0.725937724828145,
              .187768525763682e-38, -.103308436323771e4, -0.662552816342168e-1,
              0.579514041765710e3, .237416732616644e-26, .271700235739893e-14,
              -0.9078862134836e2, -0.171242509570207e-36, 0.156792067854621e3,
              0.923261357901470, -0.597865988422577e1, 0.321988767636389e7,
              -.399441390042203e-29, .493429086046981e-7, .812036983370565e-19,
              -.207610284654137e-11, -.340821291419719e-6,
              .542000573372233e-17, -.856711586510214e-12,
              0.266170454405981e-13, 0.858133791857099e-5],
        "x": [.377373741298151e19, -.507100883722913e13, -0.10336322559886e16,
              .184790814320773e-5, -.924729378390945e-3, -0.425999562292738e24,
              -.462307771873973e-12, .107319065855767e22, 0.648662492280682e11,
              0.244200600688281e1, -0.851535733484258e10, 0.169894481433592e22,
              0.215780222509020e-26, -0.320850551367334, -0.382642448458610e17,
              -.275386077674421e-28, -.563199253391666e6, -.326068646279314e21,
              0.397949001553184e14, 0.100824008584757e-6, 0.162234569738433e5,
              -0.432355225319745e11, -.59287424559861e12, 0.133061647281106e1,
              0.157338197797544e7, 0.258189614270853e14, 0.262413209706358e25,
              -.920011937431142e-1, 0.220213765905426e-2, -0.110433759109547e2,
              0.847004870612087e7, -0.592910695762536e9, -0.183027173269660e-4,
              0.181339603516302, -0.119228759669889e4, 0.430867658061468e7],
        "y": [-0.525597995024633e-9, 0.583441305228407e4, -.134778968457925e17,
              .118973500934212e26, -0.159096490904708e27, -.315839902302021e-6,
              0.496212197158239e3, 0.327777227273171e19, -0.527114657850696e22,
              .210017506281863e-16, 0.705106224399834e21, -.266713136106469e31,
              -0.145370512554562e-7, 0.149333917053130e28, -.149795620287641e8,
              -.3818819062711e16, 0.724660165585797e-4, -0.937808169550193e14,
              0.514411468376383e10, -0.828198594040141e5],
        "z": [0.24400789229065e-10, -0.463057430331242e7, 0.728803274777712e10,
              .327776302858856e16, -.110598170118409e10, -0.323899915729957e13,
              .923814007023245e16, 0.842250080413712e-12, 0.663221436245506e12,
              -.167170186672139e15, .253749358701391e4, -0.819731559610523e-20,
              0.328380587890663e12, -0.625004791171543e8, 0.803197957462023e21,
              -.204397011338353e-10, -.378391047055938e4, 0.97287654593862e-2,
              0.154355721681459e2, -0.373962862928643e4, -0.682859011374572e11,
              -0.248488015614543e-3, 0.394536049497068e7]}

    v_, P_, T_, a, b, c, d, e = par[x]

    Pr = P / P_
    Tr = T / T_
    suma = 0
    if x == "n":
        for i, j, ni in zip(Li[x], Lj[x], n[x]):
            suma += ni * (Pr - a) ** i * (Tr - b) ** j
        return v_ * exp(suma)

    for i, j, ni in zip(Li[x], Lj[x], n[x]):
        suma += ni * (Pr - a) ** (c * i) * (Tr - b) ** (j * d)
    return v_ * suma ** e


# Region 4
def _Region4(P, x):
    """Basic equation for region 4

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    x : float
        Vapor quality, [-]

    Returns
    -------
    prop : dict
        Dict with calculated properties. The available properties are:

            * T: Saturated temperature, [K]
            * P: Saturated pressure, [MPa]
            * x: Vapor quality, [-]
            * v: Specific volume, [m³/kg]
            * h: Specific enthalpy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
    """
    T = _TSat_P(P)
    if T > 623.15:
        rhol = 1. / _Backward3_sat_v_P(P, T, 0)
        P1 = _Region3(rhol, T)
        rhov = 1. / _Backward3_sat_v_P(P, T, 1)
        P2 = _Region3(rhov, T)
    else:
        P1 = _Region1(T, P)
        P2 = _Region2(T, P)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = P1["v"] + x * (P2["v"] - P1["v"])
    propiedades["h"] = P1["h"] + x * (P2["h"] - P1["h"])
    propiedades["s"] = P1["s"] + x * (P2["s"] - P1["s"])
    propiedades["cp"] = None
    propiedades["cv"] = None
    propiedades["w"] = None
    propiedades["alfav"] = None
    propiedades["kt"] = None
    propiedades["region"] = 4
    propiedades["x"] = x
    return propiedades


def _Backward4_T_hs(h, s):
    """Backward equation for region 4, T=f(h,s)

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    T : float
        Temperature, [K]

    References
    ----------
    IAPWS, Revised Supplementary Release on Backward Equations p(h,s) for
    Region 3, Equations as a Function of h and s for the Region Boundaries, and
    an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997
    for the Thermodynamic Properties of Water and Steam,
    http://www.iapws.org/relguide/Supp-phs3-2014.pdf. Eq 9

    Examples
    --------
    >>> _Backward4_T_hs(1800,5.3)
    346.8475498
    >>> _Backward4_T_hs(2400,6.0)
    425.1373305
    >>> _Backward4_T_hs(2500,5.5)
    522.5579013
    """

    nu = h / 2800
    sigma = s / 9.2

    suma = np.sum(
        Const.Backward4_T_hs_n * (nu - 0.119) ** Const.Backward4_T_hs_Li * (sigma - 1.07) ** Const.Backward4_T_hs_Lj)
    return 550 * suma


# Region 5
def _Region5(T, P):
    """Basic equation for region 5

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties. The available properties are:

            * v: Specific volume, [m³/kg]
            * h: Specific enthalpy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isocoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s]
            * alfav: Cubic expansion coefficient, [1/K]
            * kt: Isothermal compressibility, [1/MPa]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 32-34

    Examples
    --------
    >>> _Region5(1500,0.5)["v"]
    1.38455090
    >>> _Region5(1500,0.5)["h"]
    5219.76855
    >>> _Region5(1500,0.5)["h"]-500*_Region5(1500,0.5)["v"]
    4527.49310
    >>> _Region5(1500,30)["s"]
    7.72970133
    >>> _Region5(1500,30)["cp"]
    2.72724317
    >>> _Region5(1500,30)["cv"]
    2.19274829
    >>> _Region5(2000,30)["w"]
    1067.36948
    >>> _Region5(2000,30)["alfav"]
    0.000508830641
    >>> _Region5(2000,30)["kt"]
    0.0329193892
    """
    if P < 0:
        P = Pmin

    Tr = 1000 / T
    Pr = P / 1

    go, gop, gopp, got, gott, gopt = Region5_cp0(Tr, Pr)

    gr = np.sum(Const.Region5_nr * Pr ** Const.Region5_Ir * Tr ** Const.Region5_Jr)
    grp = np.sum(Const.Region5_nr_Ir_product * Pr ** Const.Region5_Ir_less_1 * Tr ** Const.Region5_Jr)
    grpp = np.sum(Const.Region5_nr_Ir_product * Const.Region5_Ir_less_1 * Pr ** (
        Const.Region5_Ir_less_2) * Tr ** Const.Region5_Jr)
    grt = np.sum(Const.Region5_nr_Jr_product * Pr ** Const.Region5_Ir * Tr ** Const.Region5_Jr_less_1)
    grtt = np.sum(Const.Region5_nr_Jr_product * Const.Region5_Jr_less_1 * Pr ** Const.Region5_Ir * Tr ** (
        Const.Region5_Jr_less_2))
    grpt = np.sum(Const.Region5_nr_Ir_Jr_product * Pr ** Const.Region5_Ir_less_1 * Tr ** Const.Region5_Jr_less_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * (gop + grp) * R * T / P / 1000
    propiedades["h"] = Tr * (got + grt) * R * T
    propiedades["s"] = R * (Tr * (got + grt) - (go + gr))
    propiedades["cp"] = -R * Tr ** 2 * (gott + grtt)
    propiedades["cv"] = R * (-Tr ** 2 * (gott + grtt) + ((gop + grp) - Tr * (gopt + grpt)) ** 2
                             / (gopp + grpp))
    propiedades["w"] = (R * T * 1000 * (1 + 2 * Pr * grp + Pr ** 2 * grp ** 2) / (1 - Pr ** 2 * grpp + (
            1 + Pr * grp - Tr * Pr * grpt) ** 2 / Tr ** 2 / (gott + grtt))) ** 0.5
    propiedades["alfav"] = (1 + Pr * grp - Tr * Pr * grpt) / (1 + Pr * grp) / T
    propiedades["kt"] = (1 - Pr ** 2 * grpp) / (1 + Pr * grp) / P
    propiedades["region"] = 5
    propiedades["x"] = 1
    return propiedades


def Region5_cp0(Tr, Pr):
    """Ideal properties for Region 5

    Parameters
    ----------
    Tr : float
        Reduced temperature, [-]
    Pr : float
        Reduced pressure, [-]

    Returns
    -------
    prop : array
        Array with ideal Gibbs energy partial derivatives:

            * g: Ideal Specific Gibbs energy, [kJ/kg]
            * gp: [∂g/∂P]T
            * gpp: [∂²g/∂P²]T
            * gt: [∂g/∂T]P
            * gtt: [∂²g/∂T²]P
            * gpt: [∂²g/∂T∂P]

    References
    ----------
    IAPWS, Revised Release on the IAPWS Industrial Formulation 1997 for the
    Thermodynamic Properties of Water and Steam August 2007,
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 33
    """
    gop = Pr ** -1
    gopp = -Pr ** -2
    gopt = 0
    go = log(Pr) + np.sum(Const.Region5_cp0_no * Tr ** Const.Region5_cp0_Jo)
    got = np.sum(Const.Region5_cp0_no_Jo_product * Tr ** Const.Region5_cp0_Jo_less_1)
    gott = np.sum(Const.Region5_cp0_no_Jo_Jo_less_1_product * Tr ** Const.Region5_cp0_Jo_less_2)

    return go, gop, gopp, got, gott, gopt


# Region definitions
def _Bound_TP(T, P):
    """Region definition for input T and P

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    region : float
        IAPWS-97 region code

    References
    ----------
    Wagner, W; Kretzschmar, H-J: International Steam Tables: Properties of
    Water and Steam Based on the Industrial Formulation IAPWS-IF97; Springer,
    2008; doi: 10.1007/978-3-540-74234-0. Fig. 2.3
    """
    region = None
    if 1073.15 < T <= 2273.15 and Pmin <= P <= 50:
        region = 5
    elif Pmin <= P <= Ps_623:
        Tsat = _TSat_P(P)
        if 273.15 <= T <= Tsat:
            region = 1
        elif Tsat < T <= 1073.15:
            region = 2
    elif Ps_623 < P <= 100:
        T_b23 = _t_P(P)
        if 273.15 <= T <= 623.15:
            region = 1
        elif 623.15 < T < T_b23:
            region = 3
        elif T_b23 <= T <= 1073.15:
            region = 2
    return region


def _Bound_Ph(P, h):
    """Region definition for input P y h

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]

    Returns
    -------
    region : float
        IAPWS-97 region code

    References
    ----------
    Wagner, W; Kretzschmar, H-J: International Steam Tables: Properties of
    Water and Steam Based on the Industrial Formulation IAPWS-IF97; Springer,
    2008; doi: 10.1007/978-3-540-74234-0. Fig. 2.5
    """
    region = None
    if Pmin <= P <= Ps_623:
        h14 = _Region1(_TSat_P(P), P)["h"]
        h24 = _Region2(_TSat_P(P), P)["h"]
        h25 = _Region2(1073.15, P)["h"]
        hmin = _Region1(273.15, P)["h"]
        hmax = _Region5(2273.15, P)["h"]
        if hmin <= h <= h14:
            region = 1
        elif h14 < h < h24:
            region = 4
        elif h24 <= h <= h25:
            region = 2
        elif h25 < h <= hmax:
            region = 5
    elif Ps_623 < P < Pc:
        hmin = _Region1(273.15, P)["h"]
        h13 = _Region1(623.15, P)["h"]
        h32 = _Region2(_t_P(P), P)["h"]
        h25 = _Region2(1073.15, P)["h"]
        hmax = _Region5(2273.15, P)["h"]
        if hmin <= h <= h13:
            region = 1
        elif h13 < h < h32:
            try:
                p34 = _PSat_h(h)
            except NotImplementedError:
                p34 = Ps_623
            if P < p34:
                region = 4
            else:
                region = 3
        elif h32 <= h <= h25:
            region = 2
        elif h25 < h <= hmax:
            region = 5
    elif Pc <= P <= 100:
        hmin = _Region1(273.15, P)["h"]
        h13 = _Region1(623.15, P)["h"]
        h32 = _Region2(_t_P(P), P)["h"]
        h25 = _Region2(1073.15, P)["h"]
        hmax = _Region5(2273.15, P)["h"]
        if hmin <= h <= h13:
            region = 1
        elif h13 < h < h32:
            region = 3
        elif h32 <= h <= h25:
            region = 2
        elif P <= 50 and h25 <= h <= hmax:
            region = 5
    return region


def _Bound_Ps(P, s):
    """Region definition for input P and s

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    region : float
        IAPWS-97 region code

    References
    ----------
    Wagner, W; Kretzschmar, H-J: International Steam Tables: Properties of
    Water and Steam Based on the Industrial Formulation IAPWS-IF97; Springer,
    2008; doi: 10.1007/978-3-540-74234-0. Fig. 2.9
    """
    region = None
    if Pmin <= P <= Ps_623:
        smin = _Region1(273.15, P)["s"]
        s14 = _Region1(_TSat_P(P), P)["s"]
        s24 = _Region2(_TSat_P(P), P)["s"]
        s25 = _Region2(1073.15, P)["s"]
        smax = _Region5(2273.15, P)["s"]
        if smin <= s <= s14:
            region = 1
        elif s14 < s < s24:
            region = 4
        elif s24 <= s <= s25:
            region = 2
        elif s25 < s <= smax:
            region = 5
    elif Ps_623 < P < Pc:
        smin = _Region1(273.15, P)["s"]
        s13 = _Region1(623.15, P)["s"]
        s32 = _Region2(_t_P(P), P)["s"]
        s25 = _Region2(1073.15, P)["s"]
        smax = _Region5(2273.15, P)["s"]
        if smin <= s <= s13:
            region = 1
        elif s13 < s < s32:
            try:
                p34 = _PSat_s(s)
            except NotImplementedError:
                smin_Ps3 = _Region1(623.15, Ps_623)["s"]
                if s < smin_Ps3:
                    p34 = Ps_623
                else:
                    p34 = Pc
            if P < p34:
                region = 4
            else:
                region = 3
        elif s32 <= s <= s25:
            region = 2
        elif s25 < s <= smax:
            region = 5
    elif Pc <= P <= 100:
        smin = _Region1(273.15, P)["s"]
        s13 = _Region1(623.15, P)["s"]
        s32 = _Region2(_t_P(P), P)["s"]
        s25 = _Region2(1073.15, P)["s"]
        smax = _Region5(2273.15, P)["s"]
        if smin <= s <= s13:
            region = 1
        elif s13 < s < s32:
            region = 3
        elif s32 <= s <= s25:
            region = 2
        elif P <= 50 and s25 <= s <= smax:
            region = 5
    return region


def _Bound_hs(h, s):
    """Region definition for input h and s

    Parameters
    ----------
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]

    Returns
    -------
    region : float
        IAPWS-97 region code

    References
    ----------
    Wagner, W; Kretzschmar, H-J: International Steam Tables: Properties of
    Water and Steam Based on the Industrial Formulation IAPWS-IF97; Springer,
    2008; doi: 10.1007/978-3-540-74234-0. Fig. 2.14
    """
    region = None
    s13 = _Region1(623.15, 100)["s"]
    s13s = _Region1(623.15, Ps_623)["s"]
    sTPmax = _Region2(1073.15, 100)["s"]
    s2ab = _Region2(1073.15, 4)["s"]

    # Left point in h-s plot
    smin = _Region1(273.15, 100)["s"]
    hmin = _Region1(273.15, Pmin)["h"]

    # Right point in h-s plot
    _Pmax = _Region2(1073.15, Pmin)
    hmax = _Pmax["h"]
    smax = _Pmax["s"]

    # Region 4 left and right point
    _sL = _Region1(273.15, Pmin)
    h4l = _sL["h"]
    s4l = _sL["s"]
    _sV = _Region2(273.15, Pmin)
    h4v = _sV["h"]
    s4v = _sV["s"]

    if smin <= s <= s13:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h1_s(s)
        T = _Backward1_T_Ps(100, s) - 0.0218
        hmax = _Region1(T, 100)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 1

    elif s13 < s <= s13s:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h1_s(s)
        h13 = _h13_s(s)
        v = _Backward3_v_Ps(100, s) * (1 + 9.6e-5)
        T = _Backward3_T_Ps(100, s) - 0.0248
        hmax = _Region3(1 / v, T)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h < h13:
            region = 1
        elif h13 <= h <= hmax:
            region = 3

    elif s13s < s <= sc:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h3a_s(s)
        v = _Backward3_v_Ps(100, s) * (1 + 9.6e-5)
        T = _Backward3_T_Ps(100, s) - 0.0248
        hmax = _Region3(1 / v, T)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 3

    elif sc < s < 5.049096828:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h2c3b_s(s)
        v = _Backward3_v_Ps(100, s) * (1 + 9.6e-5)
        T = _Backward3_T_Ps(100, s) - 0.0248
        hmax = _Region3(1 / v, T)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 3

    elif 5.049096828 <= s < 5.260578707:
        # Specific zone with 2-3 boundary in s shape
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h2c3b_s(s)
        h23max = _Region2(863.15, 100)["h"]
        h23min = _Region2(623.15, Ps_623)["h"]
        T = _Backward2_T_Ps(100, s) - 0.019
        hmax = _Region2(T, 100)["h"]

        if hmin <= h < hs:
            region = 4
        elif hs <= h < h23min:
            region = 3
        elif h23min <= h < h23max:
            if _Backward2c_P_hs(h, s) <= _P23_T(_t_hs(h, s)):
                region = 2
            else:
                region = 3
        elif h23max <= h <= hmax:
            region = 2

    elif 5.260578707 <= s < 5.85:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h2c3b_s(s)
        T = _Backward2_T_Ps(100, s) - 0.019
        hmax = _Region2(T, 100)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif 5.85 <= s < sTPmax:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h2ab_s(s)
        T = _Backward2_T_Ps(100, s) - 0.019
        hmax = _Region2(T, 100)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif sTPmax <= s < s2ab:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h2ab_s(s)
        P = _Backward2_P_hs(h, s)
        hmax = _Region2(1073.15, P)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif s2ab <= s < s4v:
        hmin = h4l + (s - s4l) / (s4v - s4l) * (h4v - h4l)
        hs = _h2ab_s(s)
        P = _Backward2_P_hs(h, s)
        hmax = _Region2(1073.15, P)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif s4v <= s <= smax:
        hmin = _Region2(273.15, Pmin)["h"]
        P = _Backward2a_P_hs(h, s)
        hmax = _Region2(1073.15, P)["h"]
        if Pmin <= P <= 100 and hmin <= h <= hmax:
            region = 2

    # Check region 5
    if not region and \
            _Region5(1073.15, 50)["s"] < s <= _Region5(2273.15, Pmin)["s"] \
            and _Region5(1073.15, 50)["h"] < h <= _Region5(2273.15, Pmin)["h"]:
        def funcion(par):
            return (_Region5(par[0], par[1])["h"] - h,
                    _Region5(par[0], par[1])["s"] - s)

        T, P = fsolve(funcion, [1400, 1])
        if 1073.15 < T <= 2273.15 and Pmin <= P <= 50:
            region = 5

    return region


def prop0(T, P):
    """Ideal gas properties

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties. The available properties are:

            * v: Specific volume, [m³/kg]
            * h: Specific enthalpy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isocoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s]
            * alfav: Cubic expansion coefficient, [1/K]
            * kt: Isothermal compressibility, [1/MPa]
    """
    if T <= 1073.15:
        Tr = 540 / T
        Pr = P / 1.
        go, gop, gopp, got, gott, gopt = Region2_cp0(Tr, Pr)
    else:
        Tr = 1000 / T
        Pr = P / 1.
        go, gop, gopp, got, gott, gopt = Region5_cp0(Tr, Pr)

    p0 = {}
    p0["v"] = Pr * gop * R * T / P / 1000
    p0["h"] = Tr * got * R * T
    p0["s"] = R * (Tr * got - go)
    p0["cp"] = -R * Tr ** 2 * gott
    p0["cv"] = R * (-Tr ** 2 * gott - 1)

    p0["w"] = (R * T * 1000 / (1 + 1 / Tr ** 2 / gott)) ** 0.5
    p0["alfav"] = 1 / T
    p0["xkappa"] = 1 / P
    return p0


class IAPWS97(_fase):
    """Class to model a state of liquid water or steam with the IAPWS-IF97

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]
    x : float
        Vapor quality, [-]
    l : float, optional
        Wavelength of light, for refractive index, [μm]

    Notes
    -----
    Definitions options:

        * T, P: Not valid for two-phases region
        * P, h
        * P, s
        * h, s
        * T, x: Only for two-phases region
        * P, x: Only for two-phases region

    Returns
    -------
    prop : dict
        The calculated instance has the following properties:

            * P: Pressure, [MPa]
            * T: Temperature, [K]
            * g: Specific Gibbs free energy, [kJ/kg]
            * a: Specific Helmholtz free energy, [kJ/kg]
            * v: Specific volume, [m³/kg]
            * rho: Density, [kg/m³]
            * h: Specific enthalpy, [kJ/kg]
            * u: Specific internal energy, [kJ/kg]
            * s: Specific entropy, [kJ/kg·K]
            * cp: Specific isobaric heat capacity, [kJ/kg·K]
            * cv: Specific isochoric heat capacity, [kJ/kg·K]
            * Z: Compression factor, [-]
            * fi: Fugacity coefficient, [-]
            * f: Fugacity, [MPa]

            * gamma: Isoentropic exponent, [-]
            * alfav: Isobaric cubic expansion coefficient, [1/K]
            * xkappa: Isothermal compressibility, [1/MPa]
            * kappas: Adiabatic compresibility, [1/MPa]
            * alfap: Relative pressure coefficient, [1/K]
            * betap: Isothermal stress coefficient, [kg/m³]
            * joule: Joule-Thomson coefficient, [K/MPa]
            * deltat: Isothermal throttling coefficient, [kJ/kg·MPa]
            * region: Region

            * v0: Ideal specific volume, [m³/kg]
            * u0: Ideal specific internal energy, [kJ/kg]
            * h0: Ideal specific enthalpy, [kJ/kg]
            * s0: Ideal specific entropy, [kJ/kg·K]
            * a0: Ideal specific Helmholtz free energy, [kJ/kg]
            * g0: Ideal specific Gibbs free energy, [kJ/kg]
            * cp0: Ideal specific isobaric heat capacity, [kJ/kg·K]
            * cv0: Ideal specific isochoric heat capacity [kJ/kg·K]
            * w0: Ideal speed of sound, [m/s]
            * gamma0: Ideal isoentropic exponent, [-]

            * w: Speed of sound, [m/s]
            * mu: Dynamic viscosity, [Pa·s]
            * nu: Kinematic viscosity, [m²/s]
            * k: Thermal conductivity, [W/m·K]
            * alfa: Thermal diffusivity, [m²/s]
            * sigma: Surface tension, [N/m]
            * epsilon: Dielectric constant, [-]
            * n: Refractive index, [-]
            * Prandt: Prandtl number, [-]
            * Pr: Reduced Pressure, [-]
            * Tr: Reduced Temperature, [-]
            * Hvap: Vaporization heat, [kJ/kg]
            * Svap: Vaporization entropy, [kJ/kg·K]

    Examples
    --------
    >>> water=IAPWS97(T=170+273.15, x=0.5)
    >>> water.Liquid.cp, water.Vapor.cp, water.Liquid.w, water.Vapor.w
    4.3695 2.5985 1418.3 498.78

    >>> water=IAPWS97(T=325+273.15, x=0.5)
    >>> water.P, water.Liquid.v, water.Vapor.v, water.Liquid.h, water.Vapor.h
    12.0505 0.00152830 0.0141887 1493.37 2684.48

    >>> water=IAPWS97(T=50+273.15, P=0.0006112127)
    >>> water.cp0, water.cv0, water.h0, water.s0, water.w0
    1.8714 1.4098 2594.66 9.471 444.93
    """

    M = 18.015257  # kg/kmol
    Pc = Pc
    Tc = Tc
    rhoc = rhoc
    Tt = Tt
    Tb = Tb
    f_accent = f_acent
    dipole = Dipole

    kwargs = {"T": 0.0,
              "P": 0.0,
              "x": None,
              "h": None,
              "s": None,
              "v": 0.0,
              "l": 0.5893}
    status = 0
    msg = "Unknown variables"
    _thermo = ""
    region = None

    Liquid = None
    Vapor = None

    T = None
    P = None
    v = None
    rho = None
    phase = None
    x = None
    Tr = None
    Pr = None
    sigma = None

    v0 = None
    h0 = None
    u0 = None
    s0 = None
    a0 = None
    g0 = None
    cp0 = None
    cv0 = None
    cp0_cv = None
    w0 = None
    gamma0 = None

    h = None
    u = None
    s = None
    a = None
    g = None
    Hvap = None
    Svap = None

    def __init__(self, **kwargs):
        self.kwargs = IAPWS97.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        """Invoke the solver."""
        self.kwargs.update(kwargs)

        if self.calculable:
            self.status = 1
            self.calculo()
            self.msg = "Solved"

    @property
    def calculable(self):
        """Check if class is calculable by its kwargs"""
        self._thermo = ""
        if self.kwargs["T"] and self.kwargs["P"]:
            self._thermo = "TP"
        elif self.kwargs["P"] and self.kwargs["h"] is not None:
            self._thermo = "Ph"
        elif self.kwargs["P"] and self.kwargs["s"] is not None:
            self._thermo = "Ps"
        elif self.kwargs["h"] is not None and self.kwargs["s"] is not None:
            self._thermo = "hs"
        elif self.kwargs["T"] and self.kwargs["x"] is not None:
            self._thermo = "Tx"
        elif self.kwargs["P"] and self.kwargs["x"] is not None:
            self._thermo = "Px"

        # TODO: Add other pairs definitions options
        # elif self.kwargs["P"] and self.kwargs["v"]:
        # self._thermo = "Pv"
        # elif self.kwargs["T"] and self.kwargs["s"] is not None:
        # self._thermo = "Ts"

        return self._thermo

    def calculo(self):
        """Calculate procedure"""
        propiedades = None
        args = (self.kwargs[self._thermo[0]], self.kwargs[self._thermo[1]])
        if self._thermo == "TP":
            T, P = args
            region = _Bound_TP(T, P)
            if region == 1:
                propiedades = _Region1(T, P)
            elif region == 2:
                propiedades = _Region2(T, P)
            elif region == 3:
                if T == Tc and P == Pc:
                    rho = rhoc
                else:
                    vo = _Backward3_v_PT(P, T)

                    def funcion(rho):
                        return _Region3(rho, self.kwargs["T"])["P"] - P

                    rho = newton(funcion, 1 / vo)
                propiedades = _Region3(rho, T)
            elif region == 5:
                propiedades = _Region5(T, P)
            else:
                raise NotImplementedError("Incoming out of bound")

        elif self._thermo == "Ph":
            P, h = args
            region = _Bound_Ph(P, h)
            if region == 1:
                To = _Backward1_T_Ph(P, h)
                T = newton(lambda T: _Region1(T, P)["h"] - h, To)
                propiedades = _Region1(T, P)
            elif region == 2:
                To = _Backward2_T_Ph(P, h)
                T = newton(lambda T: _Region2(T, P)["h"] - h, To)
                propiedades = _Region2(T, P)
            elif region == 3:
                vo = _Backward3_v_Ph(P, h)
                To = _Backward3_T_Ph(P, h)

                def funcion(par):
                    return (_Region3(par[0], par[1])["h"] - h,
                            _Region3(par[0], par[1])["P"] - P)

                rho, T = fsolve(funcion, [1 / vo, To])
                propiedades = _Region3(rho, T)
            elif region == 4:
                T = _TSat_P(P)
                if T <= 623.15:
                    h1 = _Region1(T, P)["h"]
                    h2 = _Region2(T, P)["h"]
                    x = (h - h1) / (h2 - h1)
                    propiedades = _Region4(P, x)
                else:
                    h1 = _Region4(P, 0)["h"]
                    h2 = _Region4(P, 1)["h"]
                    x = (h - h1) / (h2 - h1)
                    propiedades = _Region4(P, x)
            elif region == 5:
                T = newton(lambda T: _Region5(T, P)["h"] - h, 1500)
                propiedades = _Region5(T, P)
            else:
                raise NotImplementedError("Incoming out of bound")

        elif self._thermo == "Ps":
            P, s = args
            region = _Bound_Ps(P, s)
            if region == 1:
                To = _Backward1_T_Ps(P, s)
                T = newton(lambda T: _Region1(T, P)["s"] - s, To)
                propiedades = _Region1(T, P)
            elif region == 2:
                To = _Backward2_T_Ps(P, s)
                T = newton(lambda T: _Region2(T, P)["s"] - s, To)
                propiedades = _Region2(T, P)
            elif region == 3:
                vo = _Backward3_v_Ps(P, s)
                To = _Backward3_T_Ps(P, s)

                def funcion(par):
                    return (_Region3(par[0], par[1])["s"] - s,
                            _Region3(par[0], par[1])["P"] - P)

                rho, T = fsolve(funcion, [1 / vo, To])
                propiedades = _Region3(rho, T)
            elif region == 4:
                T = _TSat_P(P)
                if T <= 623.15:
                    s1 = _Region1(T, P)["s"]
                    s2 = _Region2(T, P)["s"]
                    x = (s - s1) / (s2 - s1)
                    propiedades = _Region4(P, x)
                else:
                    s1 = _Region4(P, 0)["s"]
                    s2 = _Region4(P, 1)["s"]
                    x = (s - s1) / (s2 - s1)
                    propiedades = _Region4(P, x)
            elif region == 5:
                T = newton(lambda T: _Region5(T, P)["s"] - s, 1500)
                propiedades = _Region5(T, P)
            else:
                raise NotImplementedError("Incoming out of bound")

        elif self._thermo == "hs":
            h, s = args
            region = _Bound_hs(h, s)
            if region == 1:
                Po = _Backward1_P_hs(h, s)
                To = _Backward1_T_Ph(Po, h)

                def funcion(par):
                    return (_Region1(par[0], par[1])["h"] - h,
                            _Region1(par[0], par[1])["s"] - s)

                T, P = fsolve(funcion, [To, Po])
                propiedades = _Region1(T, P)
            elif region == 2:
                Po = _Backward2_P_hs(h, s)
                To = _Backward2_T_Ph(Po, h)

                def funcion(par):
                    return (_Region2(par[0], par[1])["h"] - h,
                            _Region2(par[0], par[1])["s"] - s)

                T, P = fsolve(funcion, [To, Po])
                propiedades = _Region2(T, P)
            elif region == 3:
                P = _Backward3_P_hs(h, s)
                vo = _Backward3_v_Ph(P, h)
                To = _Backward3_T_Ph(P, h)

                def funcion(par):
                    return (_Region3(par[0], par[1])["h"] - h,
                            _Region3(par[0], par[1])["s"] - s)

                rho, T = fsolve(funcion, [1 / vo, To])
                propiedades = _Region3(rho, T)
            elif region == 4:
                if round(s - sc, 6) == 0 and round(h - hc, 6) == 0:
                    propiedades = _Region3(rhoc, Tc)

                else:
                    To = _Backward4_T_hs(h, s)
                    if To < 273.15 or To > Tc:
                        To = 300

                    def funcion(par):
                        if par[1] < 0:
                            par[1] = 0
                        elif par[1] > 1:
                            par[1] = 1
                        if par[0] < 273.15:
                            par[0] = 273.15
                        elif par[0] > Tc:
                            par[0] = Tc

                        Po = _PSat_T(par[0])
                        liquid = _Region1(par[0], Po)
                        vapor = _Region2(par[0], Po)
                        hl = liquid["h"]
                        sl = liquid["s"]
                        hv = vapor["h"]
                        sv = vapor["s"]
                        return (hv * par[1] + hl * (1 - par[1]) - h,
                                sv * par[1] + sl * (1 - par[1]) - s)

                    T, x = fsolve(funcion, [To, 0.5])
                    P = _PSat_T(T)

                    if Pt <= P < Pc and 0 < x < 1:
                        propiedades = _Region4(P, x)
                    elif Pt <= P <= Ps_623 and x == 0:
                        propiedades = _Region1(T, P)
            elif region == 5:
                def funcion(par):
                    return (_Region5(par[0], par[1])["h"] - h,
                            _Region5(par[0], par[1])["s"] - s)

                T, P = fsolve(funcion, [1400, 1])
                propiedades = _Region5(T, P)
            else:
                raise NotImplementedError("Incoming out of bound")

        elif self._thermo == "Px":
            P, x = args
            T = _TSat_P(P)
            if Pt <= P < Pc and 0 < x < 1:
                propiedades = _Region4(P, x)
            elif Pt <= P <= Ps_623 and x == 0:
                propiedades = _Region1(T, P)
            elif Pt <= P <= Ps_623 and x == 1:
                propiedades = _Region2(T, P)
            elif Ps_623 < P < Pc and x in (0, 1):
                def funcion(rho):
                    return _Region3(rho, T)["P"] - P

                rhoo = 1. / _Backward3_sat_v_P(P, T, x)
                rho = fsolve(funcion, rhoo)[0]
                propiedades = _Region3(rho, T)
            elif P == Pc and 0 <= x <= 1:
                propiedades = _Region3(rhoc, Tc)
            else:
                raise NotImplementedError("Incoming out of bound")
            self.sigma = _Tension(T)
            propiedades["x"] = x

        elif self._thermo == "Tx":
            T, x = args
            P = _PSat_T(T)
            if 273.15 <= T < Tc and 0 < x < 1:
                propiedades = _Region4(P, x)
            elif 273.15 <= T <= 623.15 and x == 0:
                propiedades = _Region1(T, P)
            elif 273.15 <= T <= 623.15 and x == 1:
                propiedades = _Region2(T, P)
            elif 623.15 < T < Tc and x in (0, 1):
                rho = 1. / _Backward3_sat_v_P(P, T, x)
                propiedades = _Region3(rho, T)
            elif T == Tc and 0 <= x <= 1:
                propiedades = _Region3(rhoc, Tc)
            else:
                raise NotImplementedError("Incoming out of bound")
            self.sigma = _Tension(T)
            propiedades["x"] = x

        self.x = propiedades["x"]
        self.region = propiedades["region"]

        self.T = propiedades["T"]
        self.P = propiedades["P"]
        self.v = propiedades["v"]
        self.rho = 1 / self.v
        self.phase = getphase(self.Tc, self.Pc, self.T, self.P, self.x,
                              self.region)
        self.Tr = self.T / self.Tc
        self.Pr = self.P / self.Pc

        # Ideal properties
        cp0 = prop0(self.T, self.P)
        self.v0 = cp0["v"]
        self.h0 = cp0["h"]
        self.u0 = self.h0 - self.P * 1000 * self.v0
        self.s0 = cp0["s"]
        self.a0 = self.u0 - self.T * self.s0
        self.g0 = self.h0 - self.T * self.s0

        self.cp0 = cp0["cp"]
        self.cv0 = cp0["cv"]
        self.cp0_cv = self.cp0 / self.cv0
        self.w0 = cp0["w"]
        self.gamma0 = self.cp0_cv

        self.Liquid = _fase()
        self.Vapor = _fase()
        if self.x == 0:
            # only liquid phase
            self.fill(self, propiedades)
            self.fill(self.Liquid, propiedades)
            self.sigma = _Tension(self.T)
        elif self.x == 1:
            # only vapor phase
            self.fill(self, propiedades)
            self.fill(self.Vapor, propiedades)
        else:
            # two phases
            liquido = _Region1(self.T, self.P)
            self.fill(self.Liquid, liquido)
            vapor = _Region2(self.T, self.P)
            self.fill(self.Vapor, vapor)

            self.h = propiedades["h"]
            self.u = self.h - self.P * 1000 * self.v
            self.s = propiedades["s"]
            self.a = self.u - self.T * self.s
            self.g = self.h - self.T * self.s
            self.sigma = _Tension(self.T)

            self.Hvap = vapor["h"] - liquido["h"]
            self.Svap = vapor["s"] - liquido["s"]

    def fill(self, fase, estado):
        """Fill phase properties"""
        fase.v = estado["v"]
        fase.rho = 1 / fase.v

        fase.h = estado["h"]
        fase.s = estado["s"]
        fase.u = fase.h - self.P * 1000 * fase.v
        fase.a = fase.u - self.T * fase.s
        fase.g = fase.h - self.T * fase.s

        fase.cv = estado["cv"]
        fase.cp = estado["cp"]
        fase.cp_cv = fase.cp / fase.cv
        fase.w = estado["w"]

        fase.Z = self.P * fase.v / R * 1000 / self.T
        fase.alfav = estado["alfav"]
        fase.xkappa = estado["kt"]
        fase.kappas = -1 / fase.v * self.derivative("v", "P", "s", fase)

        fase.joule = self.derivative("T", "P", "h", fase)
        fase.deltat = self.derivative("h", "P", "T", fase)
        fase.gamma = -fase.v/self.P \
            * self.derivative("P", "v", "T", fase)*fase.cp_cv

        fase.alfap = fase.alfav / self.P / fase.xkappa
        fase.betap = -1 / self.P * self.derivative("P", "v", "T", fase)

        fase.fi = exp((fase.g - self.g0) / R / self.T)
        fase.f = self.P * fase.fi

        fase.mu = _Viscosity(fase.rho, self.T)
        # Use industrial formulation for critical enhancement in thermal
        # conductivity calculation
        fase.drhodP_T = self.derivative("rho", "P", "T", fase)
        fase.k = _ThCond(fase.rho, self.T, fase)

        fase.nu = fase.mu / fase.rho
        fase.alfa = fase.k / 1000 / fase.rho / fase.cp
        try:
            fase.epsilon = _Dielectric(fase.rho, self.T)
        except NotImplementedError:
            fase.epsilon = None
        fase.Prandt = fase.mu * fase.cp * 1000 / fase.k
        try:
            fase.n = _Refractive(fase.rho, self.T, self.kwargs["l"])
        except NotImplementedError:
            fase.n = None

    def derivative(self, z, x, y, fase):
        """
        Wrapper derivative for custom derived properties
        where x, y, z can be: P, T, v, u, h, s, g, a
        """
        return deriv_G(self, z, x, y, fase)


class IAPWS97_PT(IAPWS97):
    """Derivated class for direct P and T input"""

    def __init__(self, P, T):
        IAPWS97.__init__(self, T=T, P=P)


class IAPWS97_Ph(IAPWS97):
    """Derivated class for direct P and h input"""

    def __init__(self, P, h):
        IAPWS97.__init__(self, P=P, h=h)


class IAPWS97_Ps(IAPWS97):
    """Derivated class for direct P and s input"""

    def __init__(self, P, s):
        IAPWS97.__init__(self, P=P, s=s)


class IAPWS97_Px(IAPWS97):
    """Derivated class for direct P and x input"""

    def __init__(self, P, x):
        IAPWS97.__init__(self, P=P, x=x)


class IAPWS97_Tx(IAPWS97):
    """Derivated class for direct T and x input"""

    def __init__(self, T, x):
        IAPWS97.__init__(self, T=T, x=x)
