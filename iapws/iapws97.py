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

    n = Const.h13_s_n
    Li = Const.h13_s_Li
    Lj = Const.h13_s_Lj
    sigma = s / 3.8
    suma = np.sum(n * (sigma - 0.884) ** Li * (sigma - 0.864) ** Lj)
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
    n = Const.t_hs_n
    Li = Const.t_hs_Li
    Lj = Const.t_hs_Lj
    suma = np.sum(n * (nu - 0.727) ** Li * (sigma - 0.864) ** Lj)
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

    n = Const.PSat_h_n
    Li = Const.PSat_h_Li
    Lj = Const.PSat_h_Lj
    nu = h / 2600
    suma = np.sum(n * (nu - 1.02) ** Li * (nu - 0.608) ** Lj)
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

    n = Const.PSat_s_n
    Li = Const.PSat_s_Li
    Lj = Const.PSat_s_Lj

    sigma = s / 5.2
    suma = np.sum(n * (sigma - 1.03) ** Li * (sigma - 0.699) ** Lj)
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

    n = Const.h1_s_n
    Li = Const.h1_s_Li
    Lj = Const.h1_s_Lj

    sigma = s / 3.8
    suma = np.sum(n * (sigma - 1.09) ** Li * (sigma + 0.366e-4) ** Lj)
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

    n = Const.h3a_s_n
    Li = Const.h3a_s_Li
    Lj = Const.h3a_s_Lj

    sigma = s / 3.8
    suma = np.sum(n * (sigma - 1.09) ** Li * (sigma + 0.366e-4) ** Lj)
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

    n = Const.h2ab_s_n
    Li = Const.h2ab_s_Li
    Lj = Const.h2ab_s_Lj

    sigma1 = s / 5.21
    sigma2 = s / 9.2
    suma = np.sum(n * (1 / sigma1 - 0.513) ** Li * (sigma2 - 0.524) ** Lj)
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

    n = Const.h2c3b_s_n
    Li = Const.h2c3b_s_Li
    Lj = Const.h2c3b_s_Lj

    sigma = s / 5.9
    suma = np.sum(n * (sigma - 1.02) ** Li * (sigma - 0.726) ** Lj)
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

    n = Const.Region1_n
    Li = Const.Region1_Li
    Lj = Const.Region1_Lj
    Li_1 = Const.Region1_Li_less_1
    Li_2 = Const.Region1_Li_less_2
    Lj_1 = Const.Region1_Lj_less_1
    Lj_2 = Const.Region1_Lj_less_2

    g = np.sum(n * (7.1-Pr)**Li * (Tr-1.222)**Lj)
    gp = -np.sum(n * Li * (7.1-Pr)**Li_1 * (Tr-1.222)**Lj)
    gpp = np.sum(n * Li * Li_1 * (7.1-Pr)**Li_2 * (Tr-1.222)**Lj)
    gt = np.sum(n * Lj * (7.1-Pr)**Li * (Tr-1.222)**Lj_1)
    gtt = np.sum(n * Lj * Lj_1 * (7.1-Pr)**Li * (Tr-1.222)**Lj_2)
    gpt = -np.sum(n * Li * Lj * (7.1-Pr)**Li_1 * (Tr-1.222)**Lj_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * gp * R * T / P / 1000
    propiedades["h"] = Tr * gt * R * T
    propiedades["s"] = R * (Tr * gt - g)
    propiedades["cp"] = -R * Tr ** 2 * gtt
    propiedades["cv"] = R * (-Tr ** 2 * gtt + (gp - Tr * gpt) ** 2 / gpp)
    propiedades["w"] = sqrt(R * T * 1000 * gp ** 2 / \
                            ((gp - Tr * gpt) ** 2 / (Tr ** 2 * gtt) - gpp))
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

    n = Const.Backward1_T_Ph_n
    Li = Const.Backward1_T_Ph_Li
    Lj = Const.Backward1_T_Ph_Lj

    Pr = P / 1
    nu = h / 2500
    T = np.sum(n * Pr ** Li * (nu + 1) ** Lj)
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

    n = Const.Backward1_T_Ps_n
    Li = Const.Backward1_T_Ps_Li
    Lj = Const.Backward1_T_Ps_Lj

    Pr = P / 1
    sigma = s / 1
    T = np.sum(n * Pr ** Li * (sigma + 2) ** Lj)
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

    n = Const.Backward1_P_hs_n
    Li = Const.Backward1_P_hs_Li
    Lj = Const.Backward1_P_hs_Lj

    nu = h / 3400
    sigma = s / 7.6
    P = np.sum(n * (nu + 0.05) ** Li * (sigma + 0.05) ** Lj)
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

    n = Const.Region2_n
    Li = Const.Region2_Li
    Lj = Const.Region2_Lj
    Li_1 = Const.Region2_Li_less_1
    Li_2 = Const.Region2_Li_less_2
    Lj_1 = Const.Region2_Lj_less_1
    Lj_2 = Const.Region2_Lj_less_2

    Tr = 540 / T
    Pr = P / 1

    go, gop, gopp, got, gott, gopt = Region2_cp0(Tr, Pr)

    gr = np.sum(n * Pr**Li * (Tr-0.5)**Lj)
    grp = np.sum(n * Li * Pr**Li_1 * (Tr-0.5)**Lj)
    grpp = np.sum(n * Li * Li_1 * Pr**Li_2 * (Tr-0.5)**Lj)
    grt = np.sum(n * Lj * Pr**Li * (Tr-0.5)**Lj_1)
    grtt = np.sum(n * Lj * Lj_1 * Pr**Li * (Tr-0.5)**Lj_2)
    grpt = np.sum(n * Li * Lj * Pr**Li_1 * (Tr-0.5)**Lj_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * (gop + grp) * R * T / P / 1000
    propiedades["h"] = Tr * (got + grt) * R * T
    propiedades["s"] = R * (Tr * (got + grt) - (go + gr))
    propiedades["cp"] = -R * Tr ** 2 * (gott + grtt)
    propiedades["cv"] = R * (-Tr**2*(gott+grtt) - (1 + Pr*grp - Tr*Pr*grpt)**2
                             / (1 - Pr ** 2 * grpp))
    propiedades["w"] = (R*T*1000*(1 + 2*Pr*grp + Pr**2*grp**2)
                        / (1 - Pr**2*grpp + (1 + Pr*grp - Tr*Pr*grpt)**2
                           / Tr**2 / (gott + grtt))) ** 0.5
    propiedades["alfav"] = (1 + Pr * grp - Tr * Pr * grpt) / (1 + Pr * grp) / T
    propiedades["kt"] = (1 - Pr ** 2 * grpp) / (1 + Pr * grp) / P
    propiedades["region"] = 2
    propiedades["x"] = 1
    return propiedades


def Region2_cp0(Tr, Pr, meta=False):
    """Ideal properties for Region 2

    Parameters
    ----------
    Tr : float
        Reduced temperature, [-]
    Pr : float
        Reduced pressure, [-]
    meta : boolean
        Boolean to set the calculation from metastable region

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
    if meta:
        no = Const.Region2_cp0_no_meta
    else:
        no = Const.Region2_cp0_no
    Jo = Const.Region2_cp0_Jo

    go = log(Pr)
    gop = Pr ** -1
    gopp = -Pr ** -2
    gopt = 0
    go += np.sum(no * Tr**Jo)
    got = np.sum(no * Jo * Tr**(Jo-1))
    gott = np.sum(no * Jo * (Jo-1) * Tr**(Jo-2))
    return go, gop, gopp, got, gott, gopt


def _Region2_meta(T, P):
    """Basic equation for region 2 in the metaestable region

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
    http://www.iapws.org/relguide/IF97-Rev.html, Eq 18-19

    Examples
    --------
    >>> _Region2_meta(450, 1)["v"]
    0.192516540
    >>> _Region2_meta(450, 1)["h"]
    2768.81115
    >>> _Region2_meta(450, 1)["h"]-1000*_Region2_meta(450, 1)["v"]
    2576.29461
    >>> _Region2_meta(450, 1)["s"]
    6.56660377
    >>> _Region2_meta(450, 1)["cp"]
    2.76349265
    >>> _Region2_meta(450, 1)["w"]
    498.408101
    """
    if P < 0:
        P = Pmin

    n = Const.Region2_nr_m
    Ir = Const.Region2_Ir_m
    Jr = Const.Region2_Jr_m
    Ir_1 = Const.Region2_Li_less_1_m
    Ir_2 = Const.Region2_Li_less_2_m
    Jr_1 = Const.Region2_Lj_less_1_m
    Jr_2 = Const.Region2_Lj_less_2_m

    Tr = 540 / T
    Pr = P / 1

    go, gop, gopp, got, gott, gopt = Region2_cp0(Tr, Pr, True)

    gr = np.sum(n * Pr**Ir * (Tr-0.5)**Jr)
    grp = np.sum(n * Ir * Pr**Ir_1 * (Tr-0.5)**Jr)
    grpp = np.sum(n * Ir * Ir_1 * Pr**Ir_2 * (Tr-0.5)**Jr)
    grt = np.sum(n * Jr * Pr**Ir * (Tr-0.5)**Jr_1)
    grtt = np.sum(n * Jr * Jr_1 * Pr**Ir * (Tr-0.5)**Jr_2)
    grpt = np.sum(n * Ir * Jr * Pr**Ir_1 * (Tr-0.5)**Jr_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * (gop + grp) * R * T / P / 1000
    propiedades["h"] = Tr * (got + grt) * R * T
    propiedades["s"] = R * (Tr * (got + grt) - (go + gr))
    propiedades["cp"] = -R * Tr ** 2 * (gott + grtt)
    propiedades["cv"] = R * (-Tr**2*(gott+grtt) - (1+Pr*grp-Tr*Pr*grpt)**2
                             / (1-Pr**2*grpp))
    propiedades["w"] = (R*T*1000*(1 + 2*Pr*grp + Pr**2*grp**2)
                        / (1-Pr**2*grpp + (1+Pr*grp-Tr*Pr*grpt)**2 / Tr**2
                           / (gott + grtt))) ** 0.5
    propiedades["alfav"] = (1 + Pr * grp - Tr * Pr * grpt) / (1 + Pr * grp) / T
    propiedades["kt"] = (1 - Pr ** 2 * grpp) / (1 + Pr * grp) / P
    propiedades["region"] = 2
    propiedades["x"] = 1
    return propiedades


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
    return 0.26526571908428e4 + ((P-4.5257578905948) / 1.2809002730136e-4)**0.5


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
    n = Const.Backward2a_T_Ph_n
    Li = Const.Backward2a_T_Ph_Li
    Lj = Const.Backward2a_T_Ph_Lj

    Pr = P / 1
    nu = h / 2000
    T = np.sum(n * Pr**Li * (nu-2.1)**Lj)
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
    n = Const.Backward2b_T_Ph_n
    Li = Const.Backward2b_T_Ph_Li
    Lj = Const.Backward2b_T_Ph_Lj

    Pr = P / 1
    nu = h / 2000
    T = np.sum(n * (Pr-2)**Li * (nu-2.6)**Lj)
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
    n = Const.Backward2c_T_Ph_n
    Li = Const.Backward2c_T_Ph_Li
    Lj = Const.Backward2c_T_Ph_Lj

    Pr = P / 1
    nu = h / 2000
    T = np.sum(n * (Pr+25)**Li * (nu-1.8)**Lj)
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
    n = Const.Backward2a_T_Ps_n
    Li = Const.Backward2a_T_Ps_Li
    Lj = Const.Backward2a_T_Ps_Lj

    Pr = P / 1
    sigma = s / 2
    T = np.sum(n * Pr**Li * (sigma-2)**Lj)
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
    n = Const.Backward2b_T_Ps_n
    Li = Const.Backward2b_T_Ps_Li
    Lj = Const.Backward2b_T_Ps_Lj

    Pr = P / 1
    sigma = s / 0.7853
    T = np.sum(n * Pr**Li * (10-sigma)**Lj)
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
    n = Const.Backward2c_T_Ps_n
    Li = Const.Backward2c_T_Ps_Li
    Lj = Const.Backward2c_T_Ps_Lj

    Pr = P / 1
    sigma = s / 2.9251
    T = np.sum(n * Pr**Li * (2-sigma)**Lj)
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
    n = Const.Backward2a_P_hs_n
    Li = Const.Backward2a_P_hs_Li
    Lj = Const.Backward2a_P_hs_Lj

    nu = h / 4200
    sigma = s / 12
    suma = np.sum(n * (nu-0.5)**Li * (sigma-1.2)**Lj)
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
    n = Const.Backward2b_P_hs_n
    Li = Const.Backward2b_P_hs_Li
    Lj = Const.Backward2b_P_hs_Lj

    nu = h / 4100
    sigma = s / 7.9
    suma = np.sum(n * (nu-0.6)**Li * (sigma-1.01)**Lj)
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
    n = Const.Backward2c_P_hs_n
    Li = Const.Backward2c_P_hs_Li
    Lj = Const.Backward2c_P_hs_Lj

    nu = h / 3500
    sigma = s / 5.9
    suma = np.sum(n * (nu-0.7)**Li * (sigma-1.1)**Lj)
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

    n = Const.Region3_n
    Li = Const.Region3_Li
    Lj = Const.Region3_Lj
    Li_1 = Const.Region3_Li_less_1
    Li_2 = Const.Region3_Li_less_2
    Lj_1 = Const.Region3_Lj_less_1
    Lj_2 = Const.Region3_Lj_less_2

    g = (1.0658070028513 * log(d)) + np.sum(n * d**Li * Tr**Lj)
    gd = (1.0658070028513 / d) + np.sum(n * Li * d**Li_1 * Tr**Lj)
    gdd = (-1.0658070028513/d**2) + np.sum(n * Li * Li_1 * d**Li_2 * Tr**Lj)
    gt = np.sum(n * Lj * d**Li * Tr**Lj_1)
    gtt = np.sum(n * Lj * Lj_1 * d**Li * Tr**Lj_2)
    gdt = np.sum(n * Li * Lj * d**Li_1 * Tr**Lj_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = d * gd * R * T * rho / 1000
    propiedades["v"] = 1 / rho
    propiedades["h"] = R * T * (Tr * gt + d * gd)
    propiedades["s"] = R * (Tr * gt - g)
    propiedades["cp"] = R * (-Tr**2*gtt+(d*gd-d*Tr*gdt)**2/(2*d*gd+d**2*gdd))
    propiedades["cv"] = -R * Tr ** 2 * gtt
    propiedades["w"] = sqrt(R*T*1000 * (2*d*gd + d**2*gdd
                                        - (d*gd - d*Tr*gdt)**2 / Tr**2 / gtt))
    propiedades["alfav"] = (gd - Tr * gdt) / (2 * gd + d * gdd) / T
    propiedades["kt"] = 1 / (2 * d * gd + d ** 2 * gdd) / rho / R / T * 1000
    propiedades["region"] = 3

    propiedades["x"] = 1
    if T < Tc and propiedades["P"] < Pc:
        t_sat = _TSat_P(propiedades["P"])
        if T < t_sat:
            propiedades["x"] = 0

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
    h = 0.201464004206875e4 + 3.74696550136983*P - 0.0219921901054187*P**2 \
        + 0.875131686009950e-4*P**3
    return h


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
    return 3.727888004*(P-22.064) + 647.096


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
    n = Const.Backward3a_v_Ph_n
    Li = Const.Backward3a_v_Ph_Li
    Lj = Const.Backward3a_v_Ph_Lj

    Pr = P / 100
    nu = h / 2100
    suma = np.sum(n * (Pr+0.128)**Li * (nu-0.727)**Lj)
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
    n = Const.Backward3b_v_Ph_n
    Li = Const.Backward3b_v_Ph_Li
    Lj = Const.Backward3b_v_Ph_Lj

    Pr = P / 100
    nu = h / 2800
    suma = np.sum(n * (Pr+0.0661)**Li * (nu-0.72)**Lj)
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
    n = Const.Backward3a_T_Ph_n
    Li = Const.Backward3a_T_Ph_Li
    Lj = Const.Backward3a_T_Ph_Lj

    Pr = P / 100.
    nu = h / 2300.
    suma = np.sum(n * (Pr+0.240)**Li * (nu-0.615)**Lj)
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
    n = Const.Backward3b_T_Ph_n
    Li = Const.Backward3b_T_Ph_Li
    Lj = Const.Backward3b_T_Ph_Lj

    Pr = P / 100.
    nu = h / 2800.
    suma = np.sum(n * (Pr+0.298)**Li * (nu-0.72)**Lj)
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
    n = Const.Backward3a_v_Ps_n
    Li = Const.Backward3a_v_Ps_Li
    Lj = Const.Backward3a_v_Ps_Lj

    Pr = P / 100
    sigma = s / 4.4
    suma = np.sum(n * (Pr+0.187)**Li * (sigma-0.755)**Lj)
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
    n = Const.Backward3b_v_Ps_n
    Li = Const.Backward3b_v_Ps_Li
    Lj = Const.Backward3b_v_Ps_Lj

    Pr = P / 100
    sigma = s / 5.3
    suma = np.sum(n * (Pr+0.298)**Li * (sigma-0.816)**Lj)
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
    n = Const.Backward3a_T_Ps_n
    Li = Const.Backward3a_T_Ps_Li
    Lj = Const.Backward3a_T_Ps_Lj

    Pr = P / 100
    sigma = s / 4.4
    suma = np.sum(n * (Pr+0.240)**Li * (sigma-0.703)**Lj)
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
    n = Const.Backward3b_T_Ps_n
    Li = Const.Backward3b_T_Ps_Li
    Lj = Const.Backward3b_T_Ps_Lj

    Pr = P / 100
    sigma = s / 5.3
    suma = np.sum(n * (Pr+0.760)**Li * (sigma-0.818)**Lj)
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
    n = Const.Backward3a_P_hs_n
    Li = Const.Backward3a_P_hs_Li
    Lj = Const.Backward3a_P_hs_Lj

    nu = h / 2300
    sigma = s / 4.4
    suma = np.sum(n * (nu-1.01)**Li * (sigma-0.75)**Lj)
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
    n = Const.Backward3b_P_hs_n
    Li = Const.Backward3b_P_hs_Li
    Lj = Const.Backward3b_P_hs_Lj

    nu = h / 2800
    sigma = s / 5.3
    suma = np.sum(n * (nu-0.681)**Li * (sigma-0.792)**Lj)
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
    par = Const.Backward3_v_PT_par
    Li = Const.Backward3_v_PT_Li
    Lj = Const.Backward3_v_PT_Lj
    n = Const.Backward3_v_PT_n

    v_, P_, T_, a, b, c, d, e = par[x]

    Pr = P / P_
    Tr = T / T_
    if x == "n":
        return v_ * exp(np.sum(n[x] * (Pr-a)**Li[x] * (Tr-b)**Lj[x]))

    return v_ * np.sum(n[x] * (Pr-a)**(c*Li[x]) * (Tr-b)**(Lj[x]*d))**e


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
    n = Const.Backward4_T_hs_n
    Li = Const.Backward4_T_hs_Li
    Lj = Const.Backward4_T_hs_Lj

    nu = h / 2800
    sigma = s / 9.2

    suma = np.sum(n * (nu-0.119)**Li * (sigma-1.07)**Lj)
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
    n = Const.Region5_n
    Li = Const.Region5_Li
    Lj = Const.Region5_Lj
    Li_1 = Const.Region5_Li_less_1
    Li_2 = Const.Region5_Li_less_2
    Lj_1 = Const.Region5_Lj_less_1
    Lj_2 = Const.Region5_Lj_less_2

    if P < 0:
        P = Pmin

    Tr = 1000 / T
    Pr = P / 1

    go, gop, gopp, got, gott, gopt = Region5_cp0(Tr, Pr)

    gr = np.sum(n * Pr**Li * Tr**Lj)
    grp = np.sum(n * Li * Pr**Li_1 * Tr**Lj)
    grpp = np.sum(n * Li * Li_1 * Pr**Li_2 * Tr**Lj)
    grt = np.sum(n * Lj * Pr**Li * Tr**Lj_1)
    grtt = np.sum(n * Lj * Lj_1 * Pr**Li * Tr**Lj_2)
    grpt = np.sum(n * Li * Lj * Pr**Li_1 * Tr**Lj_1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr * (gop + grp) * R * T / P / 1000
    propiedades["h"] = Tr * (got + grt) * R * T
    propiedades["s"] = R * (Tr * (got + grt) - (go + gr))
    propiedades["cp"] = -R * Tr ** 2 * (gott + grtt)
    propiedades["cv"] = R * (-Tr**2*(gott+grtt) + ((gop+grp)-Tr*(gopt+grpt))**2
                             / (gopp+grpp))
    propiedades["w"] = (R*T*1000*(1 + 2*Pr*grp + Pr**2*grp**2) / (
        1 - Pr**2*grpp + (1+Pr*grp-Tr*Pr*grpt)**2 / Tr**2 / (gott+grtt)))**0.5
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
    no = Const.Region5_cp0_no
    Jo = Const.Region5_cp0_Jo
    Jo_1 = Const.Region5_cp0_Jo_less_1
    Jo_2 = Const.Region5_cp0_Jo_less_2

    gop = Pr ** -1
    gopp = -Pr ** -2
    gopt = 0
    go = log(Pr) + np.sum(no * Tr**Jo)
    got = np.sum(no * Jo * Tr**Jo_1)
    gott = np.sum(no * Jo * Jo_1 * Tr**Jo_2)

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
    Gas = None

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
        self.Gas = self.Vapor
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
            if 623.15 < self.T <= Tc:
                rhol = 1. / _Backward3_sat_v_P(self.P, self.T, 0)
                liquido = _Region3(rhol, self.T)
                rhov = 1. / _Backward3_sat_v_P(self.P, self.T, 1)
                vapor = _Region3(rhov, self.T)
            else:
                liquido = _Region1(self.T, self.P)
                vapor = _Region2(self.T, self.P)
            self.fill(self.Liquid, liquido)
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
