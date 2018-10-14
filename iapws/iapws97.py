#!/usr/bin/python
# -*- coding: utf-8 -*-
"""IAPWS-IF97 standard implementation

.. image:: https://raw.githubusercontent.com/jjgomera/iapws/master/images/iapws97.png
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
    ------
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

    sigma = s/3.8
    I = [0, 1, 1, 3, 5, 6]
    J = [0, -2, 2, -12, -4, -3]
    n = [0.913965547600543, -0.430944856041991e-4, 0.603235694765419e2,
         0.117518273082168e-17, 0.220000904781292, -0.690815545851641e2]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (sigma-0.884)**i * (sigma-0.864)**j
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
    n = [0.34805185628969e3, -0.11671859879975e1, 0.10192970039326e-2]
    return n[0]+n[1]*T+n[2]*T**2


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
    n = [0.10192970039326e-2, 0.57254459862746e3, 0.1391883977870e2]
    return n[1]+((P-n[2])/n[0])**0.5


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
    ------
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

    nu = h/3000
    sigma = s/5.3
    I = [-12, -10, -8, -4, -3, -2, -2, -2, -2, 0, 1, 1, 1, 3, 3, 5, 6, 6, 8, 8,
         8, 12, 12, 14, 14]
    J = [10, 8, 3, 4, 3, -6, 2, 3, 4, 0, -3, -2, 10, -2, -1, -5, -6, -3, -8,
         -2, -1, -12, -1, -12, 1]
    n = [0.629096260829810e-3, -0.823453502583165e-3, 0.515446951519474e-7,
         -0.117565945784945e1, 0.348519684726192e1, -0.507837382408313e-11,
         -0.284637670005479e1, -0.236092263939673e1, 0.601492324973779e1,
         0.148039650824546e1, 0.360075182221907e-3, -0.126700045009952e-1,
         -0.122184332521413e7, 0.149276502463272, 0.698733471798484,
         -0.252207040114321e-1, 0.147151930985213e-1, -0.108618917681849e1,
         -0.936875039816322e-3, 0.819877897570217e2, -0.182041861521835e3,
         0.261907376402688e-5, -0.291626417025961e5, 0.140660774926165e-4,
         0.783237062349385e7]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-0.727)**i * (sigma-0.864)**j
    return 900*suma


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
    ------
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

    n = [0, 0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
         0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
         -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
         0.65017534844798E+03]
    tita = T+n[9]/(T-n[10])
    A = tita**2+n[1]*tita+n[2]
    B = n[3]*tita**2+n[4]*tita+n[5]
    C = n[6]*tita**2+n[7]*tita+n[8]
    return (2*C/(-B+(B**2-4*A*C)**0.5))**4


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
    ------
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
    if P < 611.212677/1e6 or P > 22.064:
        raise NotImplementedError("Incoming out of bound")

    n = [0, 0.11670521452767E+04, -0.72421316703206E+06, -0.17073846940092E+02,
         0.12020824702470E+05, -0.32325550322333E+07, 0.14915108613530E+02,
         -0.48232657361591E+04, 0.40511340542057E+06, -0.23855557567849E+00,
         0.65017534844798E+03]
    beta = P**0.25
    E = beta**2+n[3]*beta+n[6]
    F = n[1]*beta**2+n[4]*beta+n[7]
    G = n[2]*beta**2+n[5]*beta+n[8]
    D = 2*G/(-F-(F**2-4*E*G)**0.5)
    return (n[10]+D-((n[10]+D)**2-4*(n[9]+n[10]*D))**0.5)/2


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
    ------
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

    nu = h/2600
    I = [0, 1, 1, 1, 1, 5, 7, 8, 14, 20, 22, 24, 28, 36]
    J = [0, 1, 3, 4, 36, 3, 0, 24, 16, 16, 3, 18, 8, 24]
    n = [0.600073641753024, -0.936203654849857e1, 0.246590798594147e2,
         -0.107014222858224e3, -0.915821315805768e14, -0.862332011700662e4,
         -0.235837344740032e2, 0.252304969384128e18, -0.389718771997719e19,
         -0.333775713645296e23, 0.356499469636328e11, -0.148547544720641e27,
         0.330611514838798e19, 0.813641294467829e38]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-1.02)**i * (nu-0.608)**j
    return 22*suma


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
    ------
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

    sigma = s/5.2
    I = [0, 1, 1, 4, 12, 12, 16, 24, 28, 32]
    J = [0, 1, 32, 7, 4, 14, 36, 10, 0, 18]
    n = [0.639767553612785, -0.129727445396014e2, -0.224595125848403e16,
         0.177466741801846e7, 0.717079349571538e10, -0.378829107169011e18,
         -0.955586736431328e35, 0.187269814676188e24, 0.119254746466473e12,
         0.110649277244882e37]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (sigma-1.03)**i * (sigma-0.699)**j
    return 22*suma


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
    ------
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

    sigma = s/3.8
    I = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 12, 12, 14, 14, 16, 20,
         20, 22, 24, 28, 32, 32]
    J = [14, 36, 3, 16, 0, 5, 4, 36, 4, 16, 24, 18, 24, 1, 4, 2, 4, 1, 22, 10,
         12, 28, 8, 3, 0, 6, 8]
    n = [0.332171191705237, 0.611217706323496e-3, -0.882092478906822e1,
         -0.455628192543250, -0.263483840850452e-4, -0.223949661148062e2,
         -0.428398660164013e1, -0.616679338856916, -0.146823031104040e2,
         0.284523138727299e3, -0.113398503195444e3, 0.115671380760859e4,
         0.395551267359325e3, -0.154891257229285e1, 0.194486637751291e2,
         -0.357915139457043e1, -0.335369414148819e1, -0.664426796332460,
         0.323321885383934e5, 0.331766744667084e4, -0.223501257931087e5,
         0.573953875852936e7, 0.173226193407919e3, -0.363968822121321e-1,
         0.834596332878346e-6, 0.503611916682674e1, 0.655444787064505e2]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (sigma-1.09)**i * (sigma+0.366e-4)**j
    return 1700*suma


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
    ------
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

    sigma = s/3.8
    I = [0, 0, 0, 0, 2, 3, 4, 4, 5, 5, 6, 7, 7, 7, 10, 10, 10, 32, 32]
    J = [1, 4, 10, 16, 1, 36, 3, 16, 20, 36, 4, 2, 28, 32, 14, 32, 36, 0, 6]
    n = [0.822673364673336, 0.181977213534479, -0.112000260313624e-1,
         -0.746778287048033e-3, -0.179046263257381, 0.424220110836657e-1,
         -0.341355823438768, -0.209881740853565e1, -0.822477343323596e1,
         -0.499684082076008e1, 0.191413958471069, 0.581062241093136e-1,
         -0.165505498701029e4, 0.158870443421201e4, -0.850623535172818e2,
         -0.317714386511207e5, -0.945890406632871e5, -0.139273847088690e-5,
         0.631052532240980]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (sigma-1.09)**i * (sigma+0.366e-4)**j
    return 1700*suma


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
    ------
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

    sigma1 = s/5.21
    sigma2 = s/9.2
    I = [1, 1, 2, 2, 4, 4, 7, 8, 8, 10, 12, 12, 18, 20, 24, 28, 28, 28, 28, 28,
         32, 32, 32, 32, 32, 36, 36, 36, 36, 36]
    J = [8, 24, 4, 32, 1, 2, 7, 5, 12, 1, 0, 7, 10, 12, 32, 8, 12, 20, 22, 24,
         2, 7, 12, 14, 24, 10, 12, 20, 22, 28]
    n = [-0.524581170928788e3, -0.926947218142218e7, -0.237385107491666e3,
         0.210770155812776e11, -0.239494562010986e2, 0.221802480294197e3,
         -0.510472533393438e7, 0.124981396109147e7, 0.200008436996201e10,
         -0.815158509791035e3, -0.157612685637523e3, -0.114200422332791e11,
         0.662364680776872e16, -0.227622818296144e19, -0.171048081348406e32,
         0.660788766938091e16, 0.166320055886021e23, -0.218003784381501e30,
         -0.787276140295618e30, 0.151062329700346e32, 0.795732170300541e7,
         0.131957647355347e16, -0.325097068299140e24, -0.418600611419248e26,
         0.297478906557467e35, -0.953588761745473e20, 0.166957699620939e25,
         -0.175407764869978e33, 0.347581490626396e35, -0.710971318427851e39]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (1/sigma1-0.513)**i * (sigma2-0.524)**j
    return 2800*exp(suma)


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
    ------
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

    sigma = s/5.9
    I = [0, 0, 0, 1, 1, 5, 6, 7, 8, 8, 12, 16, 22, 22, 24, 36]
    J = [0, 3, 4, 0, 12, 36, 12, 16, 2, 20, 32, 36, 2, 32, 7, 20]
    n = [0.104351280732769e1, -0.227807912708513e1, 0.180535256723202e1,
         0.420440834792042, -0.105721244834660e6, 0.436911607493884e25,
         -0.328032702839753e12, -0.678686760804270e16, 0.743957464645363e4,
         -0.356896445355761e20, 0.167590585186801e32, -0.355028625419105e38,
         0.396611982166538e12, -0.414716268484468e41, 0.359080103867382e19,
         -0.116994334851995e41]

    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (sigma-1.02)**i * (sigma-0.726)**j
    return 2800*suma**4


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

    I = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4,
         4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32]
    J = [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4, 0,
         6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41]
    n = [0.14632971213167, -0.84548187169114, -0.37563603672040e1,
         0.33855169168385e1, -0.95791963387872, 0.15772038513228,
         -0.16616417199501e-1, 0.81214629983568e-3, 0.28319080123804e-3,
         -0.60706301565874e-3, -0.18990068218419e-1, -0.32529748770505e-1,
         -0.21841717175414e-1, -0.52838357969930e-4, -0.47184321073267e-3,
         -0.30001780793026e-3, 0.47661393906987e-4, -0.44141845330846e-5,
         -0.72694996297594e-15, -0.31679644845054e-4, -0.28270797985312e-5,
         -0.85205128120103e-9, -0.22425281908000e-5, -0.65171222895601e-6,
         -0.14341729937924e-12, -0.40516996860117e-6, -0.12734301741641e-8,
         -0.17424871230634e-9, -0.68762131295531e-18, 0.14478307828521e-19,
         0.26335781662795e-22, -0.11947622640071e-22, 0.18228094581404e-23,
         -0.93537087292458e-25]
    Tr = 1386/T
    Pr = P/16.53
    g = gp = gpp = gt = gtt = gpt = 0
    for i, j, ni in zip(I, J, n):
        g += ni * (7.1-Pr)**i * (Tr-1.222)**j
        gp -= ni*i * (7.1-Pr)**(i-1) * (Tr-1.222)**j
        gpp += ni*i*(i-1) * (7.1-Pr)**(i-2) * (Tr-1.222)**j
        gt += ni*j * (7.1-Pr)**i * (Tr-1.222)**(j-1)
        gtt += ni*j*(j-1) * (7.1-Pr)**i * (Tr-1.222)**(j-2)
        gpt -= ni*i*j * (7.1-Pr)**(i-1) * (Tr-1.222)**(j-1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr*gp*R*T/P/1000
    propiedades["h"] = Tr*gt*R*T
    propiedades["s"] = R*(Tr*gt-g)
    propiedades["cp"] = -R*Tr**2*gtt
    propiedades["cv"] = R*(-Tr**2*gtt+(gp-Tr*gpt)**2/gpp)
    propiedades["w"] = sqrt(R*T*1000*gp**2/((gp-Tr*gpt)**2/(Tr**2*gtt)-gpp))
    propiedades["alfav"] = (1-Tr*gpt/gp)/T
    propiedades["kt"] = -Pr*gpp/gp/P
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
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6]
    J = [0, 1, 2, 6, 22, 32, 0, 1, 2, 3, 4, 10, 32, 10, 32, 10, 32, 32, 32, 32]
    n = [-0.23872489924521e3, 0.40421188637945e3, 0.11349746881718e3,
         -0.58457616048039e1, -0.15285482413140e-3, -0.10866707695377e-5,
         -0.13391744872602e2, 0.43211039183559e2, -0.54010067170506e2,
         0.30535892203916e2, -0.65964749423638e1, 0.93965400878363e-2,
         0.11573647505340e-6, -0.25858641282073e-4, -0.40644363084799e-8,
         0.66456186191635e-7, 0.80670734103027e-10, -0.93477771213947e-12,
         0.58265442020601e-14, -0.15020185953503e-16]

    Pr = P/1
    nu = h/2500
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (nu+1)**j
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
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4]
    J = [0, 1, 2, 3, 11, 31, 0, 1, 2, 3, 12, 31, 0, 1, 2, 9, 31, 10, 32, 32]
    n = [0.17478268058307e3, 0.34806930892873e2, 0.65292584978455e1,
         0.33039981775489, -0.19281382923196e-6, -0.24909197244573e-22,
         -0.26107636489332, 0.22592965981586, -0.64256463395226e-1,
         0.78876289270526e-2, 0.35672110607366e-9, 0.17332496994895e-23,
         0.56608900654837e-3, -0.32635483139717e-3, 0.44778286690632e-4,
         -0.51322156908507e-9, -0.42522657042207e-25, 0.26400441360689e-12,
         0.78124600459723e-28, -0.30732199903668e-30]

    Pr = P/1
    sigma = s/1
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (sigma+2)**j
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
    I = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 5]
    J = [0, 1, 2, 4, 5, 6, 8, 14, 0, 1, 4, 6, 0, 1, 10, 4, 1, 4, 0]
    n = [-0.691997014660582, -0.183612548787560e2, -0.928332409297335e1,
         0.659639569909906e2, -0.162060388912024e2, 0.450620017338667e3,
         0.854680678224170e3, 0.607523214001162e4, 0.326487682621856e2,
         -0.269408844582931e2, -0.319947848334300e3, -0.928354307043320e3,
         0.303634537455249e2, -0.650540422444146e2, -0.430991316516130e4,
         -0.747512324096068e3, 0.730000345529245e3, 0.114284032569021e4,
         -0.436407041874559e3]

    nu = h/3400
    sigma = s/7.6
    P = 0
    for i, j, ni in zip(I, J, n):
        P += ni * (nu+0.05)**i * (sigma+0.05)**j
    return 100*P


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

    Tr = 540/T
    Pr = P/1

    go, gop, gopp, got, gott, gopt = Region2_cp0(Tr, Pr)

    Ir = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 7,
          7, 7, 8, 8, 9, 10, 10, 10, 16, 16, 18, 20, 20, 20, 21, 22, 23, 24,
          24, 24]
    Jr = [0, 1, 2, 3, 6, 1, 2, 4, 7, 36, 0, 1, 3, 6, 35, 1, 2, 3, 7, 3, 16, 35,
          0, 11, 25, 8, 36, 13, 4, 10, 14, 29, 50, 57, 20, 35, 48, 21, 53, 39,
          26, 40, 58]
    nr = [-0.0017731742473212999, -0.017834862292357999, -0.045996013696365003,
          -0.057581259083432, -0.050325278727930002, -3.3032641670203e-05,
          -0.00018948987516315, -0.0039392777243355001, -0.043797295650572998,
          -2.6674547914087001e-05, 2.0481737692308999e-08,
          4.3870667284435001e-07, -3.2277677238570002e-05, -0.0015033924542148,
          -0.040668253562648998, -7.8847309559367001e-10,
          1.2790717852285001e-08, 4.8225372718507002e-07,
          2.2922076337661001e-06, -1.6714766451061001e-11,
          -0.0021171472321354998, -23.895741934103999, -5.9059564324270004e-18,
          -1.2621808899101e-06, -0.038946842435739003, 1.1256211360459e-11,
          -8.2311340897998004, 1.9809712802088e-08, 1.0406965210174e-19,
          -1.0234747095929e-13, -1.0018179379511e-09, -8.0882908646984998e-11,
          0.10693031879409, -0.33662250574170999, 8.9185845355420999e-25,
          3.0629316876231997e-13, -4.2002467698208001e-06,
          -5.9056029685639003e-26, 3.7826947613457002e-06,
          -1.2768608934681e-15, 7.3087610595061e-29, 5.5414715350778001e-17,
          -9.4369707241209998e-07]

    gr = grp = grpp = grt = grtt = grpt = 0
    for i, j, ni in zip(Ir, Jr, nr):
        gr += ni * Pr**i * (Tr-0.5)**j
        grp += ni*i * Pr**(i-1) * (Tr-0.5)**j
        grpp += ni*i*(i-1) * Pr**(i-2) * (Tr-0.5)**j
        grt += ni*j * Pr**i * (Tr-0.5)**(j-1)
        grtt += ni*j*(j-1) * Pr**i * (Tr-0.5)**(j-2)
        grpt += ni*i*j * Pr**(i-1) * (Tr-0.5)**(j-1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr*(gop+grp)*R*T/P/1000
    propiedades["h"] = Tr*(got+grt)*R*T
    propiedades["s"] = R*(Tr*(got+grt)-(go+gr))
    propiedades["cp"] = -R*Tr**2*(gott+grtt)
    propiedades["cv"] = R*(-Tr**2*(gott+grtt)-(1+Pr*grp-Tr*Pr*grpt)**2 /
                           (1-Pr**2*grpp))
    propiedades["w"] = (R*T*1000*(1+2*Pr*grp+Pr**2*grp**2)/(1-Pr**2*grpp+(
        1+Pr*grp-Tr*Pr*grpt)**2/Tr**2/(gott+grtt)))**0.5
    propiedades["alfav"] = (1+Pr*grp-Tr*Pr*grpt)/(1+Pr*grp)/T
    propiedades["kt"] = (1-Pr**2*grpp)/(1+Pr*grp)/P
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
    Jo = [0, 1, -5, -4, -3, -2, -1, 2, 3]
    no = [-0.96927686500217E+01, 0.10086655968018E+02, -0.56087911283020E-02,
          0.71452738081455E-01, -0.40710498223928E+00, 0.14240819171444E+01,
          -0.43839511319450E+01, -0.28408632460772E+00, 0.21268463753307E-01]
    go = log(Pr)
    gop = Pr**-1
    gopp = -Pr**-2
    got = gott = gopt = 0
    for j, ni in zip(Jo, no):
        go += ni * Tr**j
        got += ni*j * Tr**(j-1)
        gott += ni*j*(j-1) * Tr**(j-2)
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
    return 905.84278514723-0.67955786399241*h+1.2809002730136e-4*h**2


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
    return 0.26526571908428e4+((P-0.45257578905948e1)/1.2809002730136e-4)**0.5


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
        h = -0.349898083432139e4 + 0.257560716905876e4*s - \
            0.421073558227969e3*s**2+0.276349063799944e2*s**3
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
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
         3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7]
    J = [0, 1, 2, 3, 7, 20, 0, 1, 2, 3, 7, 9, 11, 18, 44, 0, 2, 7, 36, 38, 40,
         42, 44, 24, 44, 12, 32, 44, 32, 36, 42, 34, 44, 28]
    n = [0.10898952318288e4, 0.84951654495535e3, -0.10781748091826e3,
         0.33153654801263e2, -0.74232016790248e1, 0.11765048724356e2,
         0.18445749355790e1, -0.41792700549624e1, 0.62478196935812e1,
         -0.17344563108114e2, -0.20058176862096e3, 0.27196065473796e3,
         -0.45511318285818e3, 0.30919688604755e4, 0.25226640357872e6,
         -0.61707422868339e-2, -0.31078046629583, 0.11670873077107e2,
         0.12812798404046e9, -0.98554909623276e9, 0.28224546973002e10,
         -0.35948971410703e10, 0.17227349913197e10, -0.13551334240775e5,
         0.12848734664650e8, 0.13865724283226e1, 0.23598832556514e6,
         -0.13105236545054e8, 0.73999835474766e4, -0.55196697030060e6,
         0.37154085996233e7, 0.19127729239660e5, -0.41535164835634e6,
         -0.62459855192507e2]

    Pr = P/1
    nu = h/2000
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (nu-2.1)**j
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
    I = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
         3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 6, 7, 7, 9, 9]
    J = [0, 1, 2, 12, 18, 24, 28, 40, 0, 2, 6, 12, 18, 24, 28, 40, 2, 8, 18,
         40, 1, 2, 12, 24, 2, 12, 18, 24, 28, 40, 18, 24, 40, 28, 2, 28, 1, 40]
    n = [0.14895041079516e4, 0.74307798314034e3, -0.97708318797837e2,
         0.24742464705674e1, -0.63281320016026, 0.11385952129658e1,
         -0.47811863648625, 0.85208123431544e-2, 0.93747147377932,
         0.33593118604916e1, 0.33809355601454e1, 0.16844539671904,
         0.73875745236695, -0.47128737436186, 0.15020273139707,
         -0.21764114219750e-2, -0.21810755324761e-1, -0.10829784403677,
         -0.46333324635812e-1, 0.71280351959551e-4, 0.11032831789999e-3,
         0.18955248387902e-3, 0.30891541160537e-2, 0.13555504554949e-2,
         0.28640237477456e-6, -0.10779857357512e-4, -0.76462712454814e-4,
         0.14052392818316e-4, -0.31083814331434e-4, -0.10302738212103e-5,
         0.28217281635040e-6, 0.12704902271945e-5, 0.73803353468292e-7,
         -0.11030139238909e-7, -0.81456365207833e-13, -0.25180545682962e-10,
         -0.17565233969407e-17, 0.86934156344163e-14]

    Pr = P/1
    nu = h/2000
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * (Pr-2)**i * (nu-2.6)**j
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
    I = [-7, -7, -6, -6, -5, -5, -2, -2, -1, -1, 0, 0, 1, 1, 2, 6, 6, 6, 6, 6,
         6, 6, 6]
    J = [0, 4, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 4, 8, 4, 0, 1, 4, 10, 12, 16,
         20, 22]
    n = [-0.32368398555242e13, 0.73263350902181e13, 0.35825089945447e12,
         -0.58340131851590e12, -0.10783068217470e11, 0.20825544563171e11,
         0.61074783564516e6, 0.85977722535580e6, -0.25745723604170e5,
         0.31081088422714e5, 0.12082315865936e4, 0.48219755109255e3,
         0.37966001272486e1, -0.10842984880077e2, -0.45364172676660e-1,
         0.14559115658698e-12, 0.11261597407230e-11, -0.17804982240686e-10,
         0.12324579690832e-6, -0.11606921130984e-5, 0.27846367088554e-4,
         -0.59270038474176e-3, 0.12918582991878e-2]

    Pr = P/1
    nu = h/2000
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * (Pr+25)**i * (nu-1.8)**j
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
    I = [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.25, -1.25, -1.25, -1.0, -1.0,
         -1.0, -1.0, -1.0, -1.0, -0.75, -0.75, -0.5, -0.5, -0.5, -0.5, -0.25,
         -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5]
    J = [-24, -23, -19, -13, -11, -10, -19, -15, -6, -26, -21, -17, -16, -9,
         -8, -15, -14, -26, -13, -9, -7, -27, -25, -11, -6, 1, 4, 8, 11, 0, 1,
         5, 6, 10, 14, 16, 0, 4, 9, 17, 7, 18, 3, 15, 5, 18]
    n = [-0.39235983861984e6, 0.51526573827270e6, 0.40482443161048e5,
         -0.32193790923902e3, 0.96961424218694e2, -0.22867846371773e2,
         -0.44942914124357e6, -0.50118336020166e4, 0.35684463560015,
         0.44235335848190e5, -0.13673388811708e5, 0.42163260207864e6,
         0.22516925837475e5, 0.47442144865646e3, -0.14931130797647e3,
         -0.19781126320452e6, -0.23554399470760e5, -0.19070616302076e5,
         0.55375669883164e5, 0.38293691437363e4, -0.60391860580567e3,
         0.19363102620331e4, 0.42660643698610e4, -0.59780638872718e4,
         -0.70401463926862e3, 0.33836784107553e3, 0.20862786635187e2,
         0.33834172656196e-1, -0.43124428414893e-4, 0.16653791356412e3,
         -0.13986292055898e3, -0.78849547999872, 0.72132411753872e-1,
         -0.59754839398283e-2, -0.12141358953904e-4, 0.23227096733871e-6,
         -0.10538463566194e2, 0.20718925496502e1, -0.72193155260427e-1,
         0.20749887081120e-6, -0.18340657911379e-1, 0.29036272348696e-6,
         0.21037527893619, 0.25681239729999e-3, -0.12799002933781e-1,
         -0.82198102652018e-5]

    Pr = P/1
    sigma = s/2
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (sigma-2)**j
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
    I = [-6, -6, -5, -5, -4, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1,
         -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
         4, 4, 5, 5, 5]
    J = [0, 11, 0, 11, 0, 1, 11, 0, 1, 11, 12, 0, 1, 6, 10, 0, 1, 5, 8, 9, 0,
         1, 2, 4, 5, 6, 9, 0, 1, 2, 3, 7, 8, 0, 1, 5, 0, 1, 3, 0, 1, 0, 1, 2]
    n = [0.31687665083497e6, 0.20864175881858e2, -0.39859399803599e6,
         -0.21816058518877e2, 0.22369785194242e6, -0.27841703445817e4,
         0.99207436071480e1, -0.75197512299157e5, 0.29708605951158e4,
         -0.34406878548526e1, 0.38815564249115, 0.17511295085750e5,
         -0.14237112854449e4, 0.10943803364167e1, 0.89971619308495,
         -0.33759740098958e4, 0.47162885818355e3, -0.19188241993679e1,
         0.41078580492196, -0.33465378172097, 0.13870034777505e4,
         -0.40663326195838e3, 0.41727347159610e2, 0.21932549434532e1,
         -0.10320050009077e1, 0.35882943516703, 0.52511453726066e-2,
         0.12838916450705e2, -0.28642437219381e1, 0.56912683664855,
         -0.99962954584931e-1, -0.32632037778459e-2, 0.23320922576723e-3,
         -0.15334809857450, 0.29072288239902e-1, 0.37534702741167e-3,
         0.17296691702411e-2, -0.38556050844504e-3, -0.35017712292608e-4,
         -0.14566393631492e-4, 0.56420857267269e-5, 0.41286150074605e-7,
         -0.20684671118824e-7, 0.16409393674725e-8]

    Pr = P/1
    sigma = s/0.7853
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (10-sigma)**j
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
    I = [-2, -2, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5,
         5, 6, 6, 7, 7, 7, 7, 7]
    J = [0, 1, 0, 0, 1, 2, 3, 0, 1, 3, 4, 0, 1, 2, 0, 1, 5, 0, 1, 4, 0, 1, 2,
         0, 1, 0, 1, 3, 4, 5]
    n = [0.90968501005365e3, 0.24045667088420e4, -0.59162326387130e3,
         0.54145404128074e3, -0.27098308411192e3, 0.97976525097926e3,
         -0.46966772959435e3, 0.14399274604723e2, -0.19104204230429e2,
         0.53299167111971e1, -0.21252975375934e2, -0.31147334413760,
         0.60334840894623, -0.42764839702509e-1, 0.58185597255259e-2,
         -0.14597008284753e-1, 0.56631175631027e-2, -0.76155864584577e-4,
         0.22440342919332e-3, -0.12561095013413e-4, 0.63323132660934e-6,
         -0.20541989675375e-5, 0.36405370390082e-7, -0.29759897789215e-8,
         0.10136618529763e-7, 0.59925719692351e-11, -0.20677870105164e-10,
         -0.20874278181886e-10, 0.10162166825089e-9, -0.16429828281347e-9]

    Pr = P/1
    sigma = s/2.9251
    T = 0
    for i, j, ni in zip(I, J, n):
        T += ni * Pr**i * (2-sigma)**j
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
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3,
         3, 4, 5, 5, 6, 7]
    J = [1, 3, 6, 16, 20, 22, 0, 1, 2, 3, 5, 6, 10, 16, 20, 22, 3, 16, 20, 0,
         2, 3, 6, 16, 16, 3, 16, 3, 1]
    n = [-0.182575361923032e-1, -0.125229548799536, 0.592290437320145,
         0.604769706185122e1, 0.238624965444474e3, -0.298639090222922e3,
         0.512250813040750e-1, -0.437266515606486, 0.413336902999504,
         -0.516468254574773e1, -0.557014838445711e1, 0.128555037824478e2,
         0.114144108953290e2, -0.119504225652714e3, -0.284777985961560e4,
         0.431757846408006e4, 0.112894040802650e1, 0.197409186206319e4,
         0.151612444706087e4, 0.141324451421235e-1, 0.585501282219601,
         -0.297258075863012e1, 0.594567314847319e1, -0.623656565798905e4,
         0.965986235133332e4, 0.681500934948134e1, -0.633207286824489e4,
         -0.558919224465760e1, 0.400645798472063e-1]

    nu = h/4200
    sigma = s/12
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-0.5)**i * (sigma-1.2)**j
    return 4*suma**4


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
    I = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6,
         6, 6, 7, 7, 8, 8, 8, 8, 12, 14]
    J = [0, 1, 2, 4, 8, 0, 1, 2, 3, 5, 12, 1, 6, 18, 0, 1, 7, 12, 1, 16, 1,
         12, 1, 8, 18, 1, 16, 1, 3, 14, 18, 10, 16]
    n = [0.801496989929495e-1, -0.543862807146111, 0.337455597421283,
         0.890555451157450e1, 0.313840736431485e3, 0.797367065977789,
         -0.121616973556240e1, 0.872803386937477e1, -0.169769781757602e2,
         -0.186552827328416e3, 0.951159274344237e5, -0.189168510120494e2,
         -0.433407037194840e4, 0.543212633012715e9, 0.144793408386013,
         0.128024559637516e3, -0.672309534071268e5, 0.336972380095287e8,
         -0.586634196762720e3, -0.221403224769889e11, 0.171606668708389e4,
         -0.570817595806302e9, -0.312109693178482e4, -0.207841384633010e7,
         0.305605946157786e13, 0.322157004314333e4, 0.326810259797295e12,
         -0.144104158934487e4, 0.410694867802691e3, 0.109077066873024e12,
         -0.247964654258893e14, 0.188801906865134e10, -0.123651009018773e15]

    nu = h/4100
    sigma = s/7.9
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-0.6)**i * (sigma-1.01)**j
    return 100*suma**4


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
    I = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5,
         5, 5, 5, 6, 6, 10, 12, 16]
    J = [0, 1, 2, 3, 4, 8, 0, 2, 5, 8, 14, 2, 3, 7, 10, 18, 0, 5, 8, 16, 18,
         18, 1, 4, 6, 14, 8, 18, 7, 7, 10]
    n = [0.112225607199012, -0.339005953606712e1, -0.320503911730094e2,
         -0.197597305104900e3, -0.407693861553446e3, 0.132943775222331e5,
         0.170846839774007e1, 0.373694198142245e2, 0.358144365815434e4,
         0.423014446424664e6, -0.751071025760063e9, 0.523446127607898e2,
         -0.228351290812417e3, -0.960652417056937e6, -0.807059292526074e8,
         0.162698017225669e13, 0.772465073604171, 0.463929973837746e5,
         -0.137317885134128e8, 0.170470392630512e13, -0.251104628187308e14,
         0.317748830835520e14, 0.538685623675312e2, -0.553089094625169e5,
         -0.102861522421405e7, 0.204249418756234e13, 0.273918446626977e9,
         -0.263963146312685e16, -0.107890854108088e10, -0.296492620980124e11,
         -0.111754907323424e16]

    nu = h/3500
    sigma = s/5.9
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-0.7)**i * (sigma-1.1)**j
    return 100*suma**4


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

    I = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4,
         4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9, 9, 10, 10, 11]
    J = [0, 1, 2, 7, 10, 12, 23, 2, 6, 15, 17, 0, 2, 6, 7, 22, 26, 0, 2, 4, 16,
         26, 0, 2, 4, 26, 1, 3, 26, 0, 2, 26, 2, 26, 2, 26, 0, 1, 26]
    n = [-0.15732845290239e2, 0.20944396974307e2, -0.76867707878716e1,
         0.26185947787954e1, -0.28080781148620e1, 0.12053369696517e1,
         -0.84566812812502e-2, -0.12654315477714e1, -0.11524407806681e1,
         0.88521043984318, -0.64207765181607, 0.38493460186671,
         -0.85214708824206, 0.48972281541877e1, -0.30502617256965e1,
         0.39420536879154e-1, 0.12558408424308, -0.27999329698710,
         0.13899799569460e1, -0.20189915023570e1, -0.82147637173963e-2,
         -0.47596035734923, 0.43984074473500e-1, -0.44476435428739,
         0.90572070719733, .70522450087967, .10770512626332, -0.32913623258954,
         -0.50871062041158, -0.22175400873096e-1, 0.94260751665092e-1,
         0.16436278447961, -0.13503372241348e-1, -0.14834345352472e-1,
         0.57922953628084e-3, 0.32308904703711e-2, 0.80964802996215e-4,
         -0.16557679795037e-3, -0.44923899061815e-4]

    d = rho/rhoc
    Tr = Tc/T
    g = 1.0658070028513*log(d)
    gd = 1.0658070028513/d
    gdd = -1.0658070028513/d**2
    gt = gtt = gdt = 0
    for i, j, ni in zip(I, J, n):
        g += ni * d**i * Tr**j
        gd += ni*i * d**(i-1) * Tr**j
        gdd += ni*i*(i-1) * d**(i-2) * Tr**j
        gt += ni*j * d**i * Tr**(j-1)
        gtt += ni*j*(j-1) * d**i * Tr**(j-2)
        gdt += ni*i*j * d**(i-1) * Tr**(j-1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = d*gd*R*T*rho/1000
    propiedades["v"] = 1/rho
    propiedades["h"] = R*T*(Tr*gt+d*gd)
    propiedades["s"] = R*(Tr*gt-g)
    propiedades["cp"] = R*(-Tr**2*gtt+(d*gd-d*Tr*gdt)**2/(2*d*gd+d**2*gdd))
    propiedades["cv"] = -R*Tr**2*gtt
    propiedades["w"] = sqrt(R*T*1000*(2*d*gd+d**2*gdd-(d*gd-d*Tr*gdt)**2 /
                                      Tr**2/gtt))
    propiedades["alfav"] = (gd-Tr*gdt)/(2*gd+d*gdd)/T
    propiedades["kt"] = 1/(2*d*gd+d**2*gdd)/rho/R/T*1000
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
    return 0.201464004206875e4 + 3.74696550136983*P - \
        0.0219921901054187*P**2+0.875131686009950e-4*P**3


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
    I = [0, 1, 2, -1, -2]
    n = [0.154793642129415e4, -0.187661219490113e3, 0.213144632222113e2,
         -0.191887498864292e4, 0.918419702359447e3]

    Pr = P/1
    T = 0
    for i, ni in zip(I, n):
        T += ni * log(Pr)**i
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
    I = [0, 1, 2, -1, -2]
    n = [0.969461372400213e3, -0.332500170441278e3, 0.642859598466067e2,
         0.773845935768222e3, -0.152313732937084e4]

    Pr = P/1
    T = 0
    for i, ni in zip(I, n):
        T += ni * log(Pr)**i
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
    I = [0, 1, 2, -1, -2]
    n = [0.728052609145380e1, 0.973505869861952e2, 0.147370491183191e2,
         0.329196213998375e3, 0.873371668682417e3]

    Pr = P/1
    T = 0
    for i, ni in zip(I, n):
        T += ni * log(Pr)**i
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
    return 3.727888004*(P-22.064)+647.096


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
    Pr = P/1
    T = 0
    for i, ni in enumerate(n):
        T += ni * Pr**i
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
    I = [-12, -12, -12, -12, -10, -10, -10, -8, -8, -6, -6, -6, -4, -4, -3, -2,
         -2, -1, -1, -1, -1, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 8]
    J = [6, 8, 12, 18, 4, 7, 10, 5, 12, 3, 4, 22, 2, 3, 7, 3, 16, 0, 1, 2, 3,
         0, 1, 0, 1, 2, 0, 2, 0, 2, 2, 2]
    n = [0.529944062966028e-2, -0.170099690234461, 0.111323814312927e2,
         -0.217898123145125e4, -0.506061827980875e-3, 0.556495239685324,
         -0.943672726094016e1, -0.297856807561527, 0.939353943717186e2,
         0.192944939465981e-1, 0.421740664704763, -0.368914126282330e7,
         -0.737566847600639e-2, -0.354753242424366, -0.199768169338727e1,
         0.115456297059049e1, 0.568366875815960e4, 0.808169540124668e-2,
         0.172416341519307, 0.104270175292927e1, -0.297691372792847,
         0.560394465163593, 0.275234661176914, -0.148347894866012,
         -0.651142513478515e-1, -0.292468715386302e1, 0.664876096952665e-1,
         0.352335014263844e1, -0.146340792313332e-1, -0.224503486668184e1,
         0.110533464706142e1, -0.408757344495612e-1]

    Pr = P/100
    nu = h/2100
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (Pr+0.128)**i * (nu-0.727)**j
    return 0.0028*suma


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
    I = [-12, -12, -8, -8, -8, -8, -8, -8, -6, -6, -6, -6, -6, -6, -4, -4, -4,
         -3, -3, -2, -2, -1, -1, -1, -1, 0, 1, 1, 2, 2]
    J = [0, 1, 0, 1, 3, 6, 7, 8, 0, 1, 2, 5, 6, 10, 3, 6, 10, 0, 2, 1, 2, 0,
         1, 4, 5, 0, 0, 1, 2, 6]
    n = [-0.225196934336318e-8, 0.140674363313486e-7, 0.233784085280560e-5,
         -0.331833715229001e-4, 0.107956778514318e-2, -0.271382067378863,
         0.107202262490333e1, -0.853821329075382, -0.215214194340526e-4,
         0.769656088222730e-3, -0.431136580433864e-2, 0.453342167309331,
         -0.507749535873652, -0.100475154528389e3, -0.219201924648793,
         -0.321087965668917e1, 0.607567815637771e3, 0.557686450685932e-3,
         0.187499040029550, 0.905368030448107e-2, 0.285417173048685,
         0.329924030996098e-1, 0.239897419685483, 0.482754995951394e1,
         -0.118035753702231e2, 0.169490044091791, -0.179967222507787e-1,
         0.371810116332674e-1, -0.536288335065096e-1, 0.160697101092520e1]

    Pr = P/100
    nu = h/2800
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (Pr+0.0661)**i * (nu-0.72)**j
    return 0.0088*suma


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
    else:
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
    I = [-12, -12, -12, -12, -12, -12, -12, -12, -10, -10, -10, -8, -8, -8, -8,
         -5, -3, -2, -2, -2, -1, -1, 0, 0, 1, 3, 3, 4, 4, 10, 12]
    J = [0, 1, 2, 6, 14, 16, 20, 22, 1, 5, 12, 0, 2, 4, 10, 2, 0, 1, 3, 4, 0,
         2, 0, 1, 1, 0, 1, 0, 3, 4, 5]
    n = [-0.133645667811215e-6, 0.455912656802978e-5, -0.146294640700979e-4,
         0.639341312970080e-2, 0.372783927268847e3, -0.718654377460447e4,
         0.573494752103400e6, -0.267569329111439e7, -0.334066283302614e-4,
         -0.245479214069597e-1, 0.478087847764996e2, 0.764664131818904e-5,
         0.128350627676972e-2, 0.171219081377331e-1, -0.851007304583213e1,
         -0.136513461629781e-1, -0.384460997596657e-5, 0.337423807911655e-2,
         -0.551624873066791, 0.729202277107470, -0.992522757376041e-2,
         -.119308831407288, .793929190615421, .454270731799386,
         .20999859125991, -0.642109823904738e-2, -0.235155868604540e-1,
         0.252233108341612e-2, -0.764885133368119e-2, 0.136176427574291e-1,
         -0.133027883575669e-1]

    Pr = P/100.
    nu = h/2300.
    suma = 0
    for i, j, n in zip(I, J, n):
        suma += n*(Pr+0.240)**i*(nu-0.615)**j
    return 760*suma


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
    I = [-12, -12, -10, -10, -10, -10, -10, -8, -8, -8, -8, -8, -6, -6, -6, -4,
         -4, -3, -2, -2, -1, -1, -1, -1, -1, -1, 0, 0, 1, 3, 5, 6, 8]
    J = [0, 1, 0, 1, 5, 10, 12, 0, 1, 2, 4, 10, 0, 1, 2, 0, 1, 5, 0, 4, 2, 4,
         6, 10, 14, 16, 0, 2, 1, 1, 1, 1, 1]
    n = [0.323254573644920e-4, -0.127575556587181e-3, -0.475851877356068e-3,
         0.156183014181602e-2, 0.105724860113781, -0.858514221132534e2,
         0.724140095480911e3, 0.296475810273257e-2, -0.592721983365988e-2,
         -0.126305422818666e-1, -0.115716196364853, 0.849000969739595e2,
         -0.108602260086615e-1, 0.154304475328851e-1, 0.750455441524466e-1,
         0.252520973612982e-1, -0.602507901232996e-1, -0.307622221350501e1,
         -0.574011959864879e-1, 0.503471360939849e1, -0.925081888584834,
         0.391733882917546e1, -0.773146007130190e2, 0.949308762098587e4,
         -0.141043719679409e7, 0.849166230819026e7, 0.861095729446704,
         0.323346442811720, 0.873281936020439, -0.436653048526683,
         0.286596714529479, -0.131778331276228, 0.676682064330275e-2]

    Pr = P/100.
    nu = h/2800.
    suma = 0
    for i, j, n in zip(I, J, n):
        suma += n*(Pr+0.298)**i*(nu-0.72)**j
    return 860*suma


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
    I = [-12, -12, -12, -10, -10, -10, -10, -8, -8, -8, -8, -6, -5, -4, -3, -3,
         -2, -2, -1, -1, 0, 0, 0, 1, 2, 4, 5, 6]
    J = [10, 12, 14, 4, 8, 10, 20, 5, 6, 14, 16, 28, 1, 5, 2, 4, 3, 8, 1, 2, 0,
         1, 3, 0, 0, 2, 2, 0]
    n = [0.795544074093975e2, -0.238261242984590e4, 0.176813100617787e5,
         -0.110524727080379e-2, -0.153213833655326e2, 0.297544599376982e3,
         -0.350315206871242e8, 0.277513761062119, -0.523964271036888,
         -0.148011182995403e6, 0.160014899374266e7, 0.170802322663427e13,
         0.246866996006494e-3, 0.165326084797980e1, -0.118008384666987,
         0.253798642355900e1, 0.965127704669424, -0.282172420532826e2,
         0.203224612353823, 0.110648186063513e1, 0.526127948451280,
         0.277000018736321, 0.108153340501132e1, -0.744127885357893e-1,
         0.164094443541384e-1, -0.680468275301065e-1, 0.257988576101640e-1,
         -0.145749861944416e-3]

    Pr = P/100
    sigma = s/4.4
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (Pr+0.187)**i * (sigma-0.755)**j
    return 0.0028*suma


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
    I = [-12, -12, -12, -12, -12, -12, -10, -10, -10, -10, -8, -5, -5, -5, -4,
         -4, -4, -4, -3, -2, -2, -2, -2, -2, -2, 0, 0, 0, 1, 1, 2]
    J = [0, 1, 2, 3, 5, 6, 0, 1, 2, 4, 0, 1, 2, 3, 0, 1, 2, 3, 1, 0, 1, 2, 3,
         4, 12, 0, 1, 2, 0, 2, 2]
    n = [0.591599780322238e-4, -0.185465997137856e-2, 0.104190510480013e-1,
         0.598647302038590e-2, -0.771391189901699, 0.172549765557036e1,
         -0.467076079846526e-3, 0.134533823384439e-1, -0.808094336805495e-1,
         0.508139374365767, 0.128584643361683e-2, -0.163899353915435e1,
         0.586938199318063e1, -0.292466667918613e1, -0.614076301499537e-2,
         0.576199014049172e1, -0.121613320606788e2, 0.167637540957944e1,
         -0.744135838773463e1, 0.378168091437659e-1, 0.401432203027688e1,
         0.160279837479185e2, 0.317848779347728e1, -0.358362310304853e1,
         -0.115995260446827e7, 0.199256573577909, -0.122270624794624,
         -0.191449143716586e2, -0.150448002905284e-1, 0.146407900162154e2,
         -0.327477787188230e1]

    Pr = P/100
    sigma = s/5.3
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (Pr+0.298)**i * (sigma-0.816)**j
    return 0.0088*suma


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
    else:
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
    I = [-12, -12, -10, -10, -10, -10, -8, -8, -8, -8, -6, -6, -6, -5, -5, -5,
         -4, -4, -4, -2, -2, -1, -1, 0, 0, 0, 1, 2, 2, 3, 8, 8, 10]
    J = [28, 32, 4, 10, 12, 14, 5, 7, 8, 28, 2, 6, 32, 0, 14, 32, 6, 10, 36, 1,
         4, 1, 6, 0, 1, 4, 0, 0, 3, 2, 0, 1, 2]
    n = [0.150042008263875e10, -0.159397258480424e12, 0.502181140217975e-3,
         -0.672057767855466e2, 0.145058545404456e4, -0.823889534888890e4,
         -0.154852214233853, 0.112305046746695e2, -0.297000213482822e2,
         0.438565132635495e11, 0.137837838635464e-2, -0.297478527157462e1,
         0.971777947349413e13, -0.571527767052398e-4, 0.288307949778420e5,
         -0.744428289262703e14, 0.128017324848921e2, -0.368275545889071e3,
         0.664768904779177e16, 0.449359251958880e-1, -0.422897836099655e1,
         -0.240614376434179, -0.474341365254924e1, 0.724093999126110,
         0.923874349695897, 0.399043655281015e1, 0.384066651868009e-1,
         -0.359344365571848e-2, -0.735196448821653, 0.188367048396131,
         0.141064266818704e-3, -0.257418501496337e-2, 0.123220024851555e-2]

    Pr = P/100
    sigma = s/4.4
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (Pr+0.240)**i * (sigma-0.703)**j
    return 760*suma


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
    I = [-12, -12, -12, -12, -8, -8, -8, -6, -6, -6, -5, -5, -5, -5, -5, -4,
         -3, -3, -2, 0, 2, 3, 4, 5, 6, 8, 12, 14]
    J = [1, 3, 4, 7, 0, 1, 3, 0, 2, 4, 0, 1, 2, 4, 6, 12, 1, 6, 2, 0, 1, 1, 0,
         24, 0, 3, 1, 2]
    n = [0.527111701601660, -0.401317830052742e2, 0.153020073134484e3,
         -0.224799398218827e4, -0.193993484669048, -0.140467557893768e1,
         0.426799878114024e2, 0.752810643416743, 0.226657238616417e2,
         -0.622873556909932e3, -0.660823667935396, 0.841267087271658,
         -0.253717501764397e2, 0.485708963532948e3, 0.880531517490555e3,
         0.265015592794626e7, -0.359287150025783, -0.656991567673753e3,
         0.241768149185367e1, 0.856873461222588, 0.655143675313458,
         -0.213535213206406, 0.562974957606348e-2, -0.316955725450471e15,
         -0.699997000152457e-3, 0.119845803210767e-1, 0.193848122022095e-4,
         -0.215095749182309e-4]

    Pr = P/100
    sigma = s/5.3
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (Pr+0.760)**i * (sigma-0.818)**j
    return 860*suma


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
    sc = 4.41202148223476
    if s <= sc:
        T = _Backward3a_T_Ps(P, s)
    else:
        T = _Backward3b_T_Ps(P, s)
    return T


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
    I = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 10, 10,
         14, 18, 20, 22, 22, 24, 28, 28, 32, 32]
    J = [0, 1, 5, 0, 3, 4, 8, 14, 6, 16, 0, 2, 3, 0, 1, 4, 5, 28, 28, 24, 1,
         32, 36, 22, 28, 36, 16, 28, 36, 16, 36, 10, 28]
    n = [0.770889828326934e1, -0.260835009128688e2, 0.267416218930389e3,
         0.172221089496844e2, -0.293542332145970e3, 0.614135601882478e3,
         -0.610562757725674e5, -0.651272251118219e8, 0.735919313521937e5,
         -0.116646505914191e11, 0.355267086434461e2, -0.596144543825955e3,
         -0.475842430145708e3, 0.696781965359503e2, 0.335674250377312e3,
         0.250526809130882e5, 0.146997380630766e6, 0.538069315091534e20,
         0.143619827291346e22, 0.364985866165994e20, -0.254741561156775e4,
         0.240120197096563e28, -0.393847464679496e30, 0.147073407024852e25,
         -0.426391250432059e32, 0.194509340621077e39, 0.666212132114896e24,
         0.706777016552858e34, 0.175563621975576e42, 0.108408607429124e29,
         0.730872705175151e44, 0.159145847398870e25, 0.377121605943324e41]

    nu = h/2300
    sigma = s/4.4
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-1.01)**i * (sigma-0.75)**j
    return 99*suma


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
    I = [-12, -12, -12, -12, -12, -10, -10, -10, -10, -8, -8, -6, -6, -6, -6,
         -5, -4, -4, -4, -3, -3, -3, -3, -2, -2, -1, 0, 2, 2, 5, 6, 8, 10, 14,
         14]
    J = [2, 10, 12, 14, 20, 2, 10, 14, 18, 2, 8, 2, 6, 7, 8, 10, 4, 5, 8, 1, 3,
         5, 6, 0, 1, 0, 3, 0, 1, 0, 1, 1, 1, 3, 7]
    n = [0.125244360717979e-12, -0.126599322553713e-1, 0.506878030140626e1,
         0.317847171154202e2, -0.391041161399932e6, -0.975733406392044e-10,
         -0.186312419488279e2, 0.510973543414101e3, 0.373847005822362e6,
         0.299804024666572e-7, 0.200544393820342e2, -0.498030487662829e-5,
         -0.102301806360030e2, 0.552819126990325e2, -0.206211367510878e3,
         -0.794012232324823e4, 0.782248472028153e1, -0.586544326902468e2,
         0.355073647696481e4, -0.115303107290162e-3, -0.175092403171802e1,
         0.257981687748160e3, -0.727048374179467e3, 0.121644822609198e-3,
         0.393137871762692e-1, 0.704181005909296e-2, -0.829108200698110e2,
         -0.265178818131250, 0.137531682453991e2, -0.522394090753046e2,
         0.240556298941048e4, -0.227361631268929e5, 0.890746343932567e5,
         -0.239234565822486e8, 0.568795808129714e10]

    nu = h/2800
    sigma = s/5.3
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-0.681)**i * (sigma-0.792)**j
    return 16.6/suma


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
    sc = 4.41202148223476
    if s <= sc:
        return _Backward3a_P_hs(h, s)
    else:
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

    I = {
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

    J = {
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

    I = I[x]
    J = J[x]
    n = n[x]
    v_, P_, T_, a, b, c, d, e = par[x]

    Pr = P/P_
    Tr = T/T_
    suma = 0
    if x == "n":
        for i, j, ni in zip(I, J, n):
            suma += ni * (Pr-a)**i * (Tr-b)**j
        return v_*exp(suma)
    else:
        for i, j, ni in zip(I, J, n):
            suma += ni * (Pr-a)**(c*i) * (Tr-b)**(j*d)
        return v_*suma**e


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
        rhol = 1./_Backward3_sat_v_P(P, T, 0)
        P1 = _Region3(rhol, T)
        rhov = 1./_Backward3_sat_v_P(P, T, 1)
        P2 = _Region3(rhov, T)
    else:
        P1 = _Region1(T, P)
        P2 = _Region2(T, P)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = P1["v"]+x*(P2["v"]-P1["v"])
    propiedades["h"] = P1["h"]+x*(P2["h"]-P1["h"])
    propiedades["s"] = P1["s"]+x*(P2["s"]-P1["s"])
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
    I = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6, 6,
         8, 10, 10, 12, 14, 14, 16, 16, 18, 18, 18, 20, 28]
    J = [0, 3, 12, 0, 1, 2, 5, 0, 5, 8, 0, 2, 3, 4, 0, 1, 1, 2, 4, 16, 6, 8,
         22, 1, 20, 36, 24, 1, 28, 12, 32, 14, 22, 36, 24, 36]
    n = [0.179882673606601, -0.267507455199603, 0.116276722612600e1,
         0.147545428713616, -0.512871635973248, 0.421333567697984,
         0.563749522189870, 0.429274443819153, -0.335704552142140e1,
         0.108890916499278e2, -0.248483390456012, 0.304153221906390,
         -0.494819763939905, 0.107551674933261e1, 0.733888415457688e-1,
         0.140170545411085e-1, -0.106110975998808, 0.168324361811875e-1,
         0.125028363714877e1, 0.101316840309509e4, -0.151791558000712e1,
         0.524277865990866e2, 0.230495545563912e5, 0.249459806365456e-1,
         0.210796467412137e7, 0.366836848613065e9, -0.144814105365163e9,
         -0.179276373003590e-2, 0.489955602100459e10, 0.471262212070518e3,
         -0.829294390198652e11, -0.171545662263191e4, 0.355777682973575e7,
         0.586062760258436e12, -0.129887635078195e8, 0.317247449371057e11]

    nu = h/2800
    sigma = s/9.2
    suma = 0
    for i, j, ni in zip(I, J, n):
        suma += ni * (nu-0.119)**i * (sigma-1.07)**j
    return 550*suma


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

    Tr = 1000/T
    Pr = P/1

    go, gop, gopp, got, gott, gopt = Region5_cp0(Tr, Pr)

    Ir = [1, 1, 1, 2, 2, 3]
    Jr = [1, 2, 3, 3, 9, 7]
    nr = [0.15736404855259e-2, 0.90153761673944e-3, -0.50270077677648e-2,
          0.22440037409485e-5, -0.41163275453471e-5, 0.37919454822955e-7]
    gr = grp = grpp = grt = grtt = grpt = 0
    for i, j, ni in zip(Ir, Jr, nr):
        gr += ni * Pr**i * Tr**j
        grp += ni*i * Pr**(i-1) * Tr**j
        grpp += ni*i*(i-1) * Pr**(i-2) * Tr**j
        grt += ni*j * Pr**i * Tr**(j-1)
        grtt += ni*j*(j-1) * Pr**i * Tr**(j-2)
        grpt += ni*i*j * Pr**(i-1) * Tr**(j-1)

    propiedades = {}
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = Pr*(gop+grp)*R*T/P/1000
    propiedades["h"] = Tr*(got+grt)*R*T
    propiedades["s"] = R*(Tr*(got+grt)-(go+gr))
    propiedades["cp"] = -R*Tr**2*(gott+grtt)
    propiedades["cv"] = R*(-Tr**2*(gott+grtt)+((gop+grp)-Tr*(gopt+grpt))**2 /
                           (gopp+grpp))
    propiedades["w"] = (R*T*1000*(1+2*Pr*grp+Pr**2*grp**2)/(1-Pr**2*grpp+(
        1+Pr*grp-Tr*Pr*grpt)**2/Tr**2/(gott+grtt)))**0.5
    propiedades["alfav"] = (1+Pr*grp-Tr*Pr*grpt)/(1+Pr*grp)/T
    propiedades["kt"] = (1-Pr**2*grpp)/(1+Pr*grp)/P
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
    Jo = [0, 1, -3, -2, -1, 2]
    no = [-0.13179983674201e2, 0.68540841634434e1, -0.24805148933466e-1,
          0.36901534980333, -0.31161318213925e1, -0.32961626538917]
    go = log(Pr)
    gop = Pr**-1
    gopp = -Pr**-2
    got = gott = gopt = 0
    for j, ni in zip(Jo, no):
        go += ni * Tr**j
        got += ni*j * Tr**(j-1)
        gott += ni*j*(j-1) * Tr**(j-2)

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
                p34 = Pc
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
    s13s = _Region1(623.15,  Ps_623)["s"]
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
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h1_s(s)
        T = _Backward1_T_Ps(100, s)-0.0218
        hmax = _Region1(T, 100)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 1

    elif s13 < s <= s13s:
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h1_s(s)
        h13 = _h13_s(s)
        v = _Backward3_v_Ps(100, s)*(1+9.6e-5)
        T = _Backward3_T_Ps(100, s)-0.0248
        hmax = _Region3(1/v, T)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h < h13:
            region = 1
        elif h13 <= h <= hmax:
            region = 3

    elif s13s < s <= sc:
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h3a_s(s)
        v = _Backward3_v_Ps(100, s)*(1+9.6e-5)
        T = _Backward3_T_Ps(100, s)-0.0248
        hmax = _Region3(1/v, T)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 3

    elif sc < s < 5.049096828:
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h2c3b_s(s)
        v = _Backward3_v_Ps(100, s)*(1+9.6e-5)
        T = _Backward3_T_Ps(100, s)-0.0248
        hmax = _Region3(1/v, T)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 3

    elif 5.049096828 <= s < 5.260578707:
        # Specific zone with 2-3 boundary in s shape
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h2c3b_s(s)
        h23max = _Region2(863.15, 100)["h"]
        h23min = _Region2(623.15, Ps_623)["h"]
        T = _Backward2_T_Ps(100, s)-0.019
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
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h2c3b_s(s)
        T = _Backward2_T_Ps(100, s)-0.019
        hmax = _Region2(T, 100)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif 5.85 <= s < sTPmax:
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h2ab_s(s)
        T = _Backward2_T_Ps(100, s)-0.019
        hmax = _Region2(T, 100)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif sTPmax <= s < s2ab:
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
        hs = _h2ab_s(s)
        P = _Backward2_P_hs(h, s)
        hmax = _Region2(1073.15, P)["h"]
        if hmin <= h < hs:
            region = 4
        elif hs <= h <= hmax:
            region = 2

    elif s2ab <= s < s4v:
        hmin = h4l+(s-s4l)/(s4v-s4l)*(h4v-h4l)
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
            return (_Region5(par[0], par[1])["h"]-h,
                    _Region5(par[0], par[1])["s"]-s)
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
        Tr = 540/T
        Pr = P/1.
        go, gop, gopp, got, gott, gopt = Region2_cp0(Tr, Pr)
    else:
        Tr = 1000/T
        Pr = P/1.
        go, gop, gopp, got, gott, gopt = Region5_cp0(Tr, Pr)

    prop0 = {}
    prop0["v"] = Pr*gop*R*T/P/1000
    prop0["h"] = Tr*got*R*T
    prop0["s"] = R*(Tr*got-go)
    prop0["cp"] = -R*Tr**2*gott
    prop0["cv"] = R*(-Tr**2*gott-1)

    prop0["w"] = (R*T*1000/(1+1/Tr**2/gott))**0.5
    prop0["alfav"] = 1/T
    prop0["xkappa"] = 1/P
    return prop0


class IAPWS97(object):
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
        Wavelength of light, for refractive index, [nm]

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
    kwargs = {"T": 0.0,
              "P": 0.0,
              "x": None,
              "h": None,
              "s": None,
              "v": 0.0,
              "l": 0.5893}
    status = 0
    msg = "Unknown variables"

    def __init__(self, **kwargs):
        self.kwargs = IAPWS97.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
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
        # TODO: Add other pairs definitions options
        # elif self.kwargs["P"] and self.kwargs["v"]:
            # self._thermo = "Pv"
        # elif self.kwargs["T"] and self.kwargs["s"] is not None:
            # self._thermo = "Ts"
        elif self.kwargs["h"] is not None and self.kwargs["s"] is not None:
            self._thermo = "hs"
        elif self.kwargs["T"] and self.kwargs["x"] is not None:
            self._thermo = "Tx"
        elif self.kwargs["P"] and self.kwargs["x"] is not None:
            self._thermo = "Px"
        return self._thermo

    def calculo(self):
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
                        return _Region3(rho, self.kwargs["T"])["P"]-P
                    rho = newton(funcion, 1/vo)
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
                T = newton(lambda T: _Region1(T, P)["h"]-h, To)
                propiedades = _Region1(T, P)
            elif region == 2:
                To = _Backward2_T_Ph(P, h)
                T = newton(lambda T: _Region2(T, P)["h"]-h, To)
                propiedades = _Region2(T, P)
            elif region == 3:
                vo = _Backward3_v_Ph(P, h)
                To = _Backward3_T_Ph(P, h)

                def funcion(par):
                    return (_Region3(par[0], par[1])["h"]-h,
                            _Region3(par[0], par[1])["P"]-P)
                rho, T = fsolve(funcion, [1/vo, To])
                propiedades = _Region3(rho, T)
            elif region == 4:
                T = _TSat_P(P)
                if T <= 623.15:
                    h1 = _Region1(T, P)["h"]
                    h2 = _Region2(T, P)["h"]
                    x = (h-h1)/(h2-h1)
                    propiedades = _Region4(P, x)
                else:
                    vo = _Backward3_v_Ph(P, h)
                    To = _Backward3_T_Ph(P, h)

                    def funcion(par):
                        return (_Region3(par[0], par[1])["h"]-h,
                                _Region3(par[0], par[1])["P"]-P)
                    rho, T = fsolve(funcion, [1/vo, To])
                    propiedades = _Region3(rho, T)
            elif region == 5:
                T = newton(lambda T: _Region5(T, P)["h"]-h, 1500)
                propiedades = _Region5(T, P)
            else:
                raise NotImplementedError("Incoming out of bound")

        elif self._thermo == "Ps":
            P, s = args
            region = _Bound_Ps(P, s)
            if region == 1:
                To = _Backward1_T_Ps(P, s)
                T = newton(lambda T: _Region1(T, P)["s"]-s, To)
                propiedades = _Region1(T, P)
            elif region == 2:
                To = _Backward2_T_Ps(P, s)
                T = newton(lambda T: _Region2(T, P)["s"]-s, To)
                propiedades = _Region2(T, P)
            elif region == 3:
                vo = _Backward3_v_Ps(P, s)
                To = _Backward3_T_Ps(P, s)

                def funcion(par):
                    return (_Region3(par[0], par[1])["s"]-s,
                            _Region3(par[0], par[1])["P"]-P)
                rho, T = fsolve(funcion, [1/vo, To])
                propiedades = _Region3(rho, T)
            elif region == 4:
                T = _TSat_P(P)
                if T <= 623.15:
                    s1 = _Region1(T, P)["s"]
                    s2 = _Region2(T, P)["s"]
                    x = (s-s1)/(s2-s1)
                    propiedades = _Region4(P, x)
                else:
                    vo = _Backward3_v_Ps(P, s)
                    To = _Backward3_T_Ps(P, s)

                    def funcion(par):
                        return (_Region3(par[0], par[1])["s"]-s,
                                _Region3(par[0], par[1])["P"]-P)
                    rho, T = fsolve(funcion, [1/vo, To])
                    propiedades = _Region3(rho, T)
            elif region == 5:
                T = newton(lambda T: _Region5(T, P)["s"]-s, 1500)
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
                    return (_Region1(par[0], par[1])["h"]-h,
                            _Region1(par[0], par[1])["s"]-s)
                T, P = fsolve(funcion, [To, Po])
                propiedades = _Region1(T, P)
            elif region == 2:
                Po = _Backward2_P_hs(h, s)
                To = _Backward2_T_Ph(Po, h)

                def funcion(par):
                    return (_Region2(par[0], par[1])["h"]-h,
                            _Region2(par[0], par[1])["s"]-s)
                T, P = fsolve(funcion, [To, Po])
                propiedades = _Region2(T, P)
            elif region == 3:
                P = _Backward3_P_hs(h, s)
                vo = _Backward3_v_Ph(P, h)
                To = _Backward3_T_Ph(P, h)

                def funcion(par):
                    return (_Region3(par[0], par[1])["h"]-h,
                            _Region3(par[0], par[1])["s"]-s)
                rho, T = fsolve(funcion, [1/vo, To])
                propiedades = _Region3(rho, T)
            elif region == 4:
                if round(s-sc, 6) == 0 and round(h-hc, 6) == 0:
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
                        return (hv*par[1]+hl*(1-par[1])-h,
                                sv*par[1]+sl*(1-par[1])-s)
                    T, x = fsolve(funcion, [To, 0.5])
                    P = _PSat_T(T)

                    if Pt <= P < Pc and 0 < x < 1:
                        propiedades = _Region4(P, x)
                    elif Pt <= P <= Ps_623 and x == 0:
                        propiedades = _Region1(T, P)
            elif region == 5:
                def funcion(par):
                    return (_Region5(par[0], par[1])["h"]-h,
                            _Region5(par[0], par[1])["s"]-s)
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
                    return _Region3(rho, T)["P"]-P
                rhoo = 1./_Backward3_sat_v_P(P, T, x)
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
                rho = 1./_Backward3_sat_v_P(P, T, x)
                propiedades = _Region3(rho, T)
            elif T == Tc and 0 <= x <= 1:
                propiedades = _Region3(rhoc, Tc)
            else:
                raise NotImplementedError("Incoming out of bound")
            self.sigma = _Tension(T)
            propiedades["x"] = x

        self.M = 18.015257  # kg/kmol
        self.Pc = Pc
        self.Tc = Tc
        self.rhoc = rhoc
        self.Tt = Tt
        self.Tb = Tb
        self.f_accent = f_acent
        self.dipole = Dipole

        self.x = propiedades["x"]
        self.region = propiedades["region"]
        self.name = "water"
        self.synonim = "R-718"
        self.CAS = "7732-18-5"

        self.T = propiedades["T"]
        self.P = propiedades["P"]
        self.v = propiedades["v"]
        self.rho = 1/self.v
        self.phase = getphase(self.Tc, self.Pc, self.T, self.P, self.x,
                              self.region)
        self.Tr = self.T/self.Tc
        self.Pr = self.P/self.Pc

        # Ideal properties
        cp0 = prop0(self.T, self.P)
        self.v0 = cp0["v"]
        self.h0 = cp0["h"]
        self.u0 = self.h0-self.P*1000*self.v0
        self.s0 = cp0["s"]
        self.a0 = self.u0-self.T*self.s0
        self.g0 = self.h0-self.T*self.s0

        self.cp0 = cp0["cp"]
        self.cv0 = cp0["cv"]
        self.cp0_cv = self.cp0/self.cv0
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
            self.u = self.h-self.P*1000*self.v
            self.s = propiedades["s"]
            self.a = self.u-self.T*self.s
            self.g = self.h-self.T*self.s
            self.sigma = _Tension(self.T)

            self.Hvap = vapor["h"]-liquido["h"]
            self.Svap = vapor["s"]-liquido["s"]

    def fill(self, fase, estado):
        fase.v = estado["v"]
        fase.rho = 1/fase.v

        fase.h = estado["h"]
        fase.s = estado["s"]
        fase.u = fase.h-self.P*1000*fase.v
        fase.a = fase.u-self.T*fase.s
        fase.g = fase.h-self.T*fase.s

        fase.cv = estado["cv"]
        fase.cp = estado["cp"]
        fase.cp_cv = fase.cp/fase.cv
        fase.w = estado["w"]

        fase.Z = self.P*fase.v/R*1000/self.T
        fase.alfav = estado["alfav"]
        fase.xkappa = estado["kt"]
        fase.kappas = -1/fase.v*self.derivative("v", "P", "s", fase)

        fase.joule = self.derivative("T", "P", "h", fase)
        fase.deltat = self.derivative("h", "P", "T", fase)
        fase.gamma = -fase.v/self.P*self.derivative("P", "v", "s", fase)

        fase.alfap = fase.alfav/self.P/fase.xkappa
        fase.betap = -1/self.P*self.derivative("P", "v", "T", fase)

        fase.fi = exp((fase.g-self.g0)/R/self.T)
        fase.f = self.P*fase.fi

        fase.mu = _Viscosity(fase.rho, self.T)
        # Use industrial formulation for critical enhancement in thermal
        # conductivity calculation
        fase.drhodP_T = self.derivative("rho", "P", "T", fase)
        fase.k = _ThCond(fase.rho, self.T, fase)

        fase.nu = fase.mu/fase.rho
        fase.alfa = fase.k/1000/fase.rho/fase.cp
        try:
            fase.epsilon = _Dielectric(fase.rho, self.T)
        except NotImplementedError:
            fase.epsilon = None
        fase.Prandt = fase.mu*fase.cp*1000/fase.k
        try:
            fase.n = _Refractive(fase.rho, self.T, self.kwargs["l"])
        except NotImplementedError:
            fase.n = None

    def derivative(self, z, x, y, fase):
        """Wrapper derivative for custom derived properties
        where x, y, z can be: P, T, v, u, h, s, g, a"""
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
