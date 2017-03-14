#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Guideline on the IAPWS Formulation 2001 for the Thermodynamic Properties of
Ammonia-Water Mistures
"""


from __future__ import division
from math import exp, log
import warnings

from ._iapws import M as Mw
from .iapws95 import MEoS, IAPWS95


Ma = 28.96546  # g/mol
R = 8.314472  # J/molK


def _virial(T):
    """Virial equations for humid air

    Parameters
    ----------
    T : float
        Temperature [K]

    Returns
    -------
    prop : float
        dictionary with critical coefficient
        Baa: Second virial coefficient of dry air [m³/mol]
        Baw: Second air-water cross virial coefficient [m³/mol]
        Bww: Second virial coefficient of water [m³/mol]
        Caaa: Third virial coefficient of dry air [m⁶/mol]
        Caaw: Third air-water cross virial coefficient [m⁶/mol]
        Caww: Third air-water cross virial coefficient [m⁶/mol]
        Cwww: Third virial coefficient of dry air [m⁶/mol]
        Bawt: dBaw/dT [m³/molK]
        Bawtt: d²Baw/dT² [m³/molK²]
        Caawt: dCaaw/dT [m⁶/molK]
        Caawtt: d²Caaw/dT² [m⁶/molK²]
        Cawwt: dCaww/dT [m⁶/molK]
        Cawwtt: d²Caww/dT² [m⁶/molK²]

    Raises
    ------
    Warning : If T isn't in range of validity
        * Baa: 60 ≤ T ≤ 2000
        * Baw: 130 ≤ T ≤ 2000
        * Bww: 130 ≤ T ≤ 1273
        * Caaa: 60 ≤ T ≤ 2000
        * Caaw: 193 ≤ T ≤ 493
        * Caww: 173 ≤ T ≤ 473
        * Cwww: 130 ≤ T ≤ 1273

    Examples
    --------
    >>> _virial(200)["Baa"]
    -3.92722567e-5

    References
    ----------
    IAPWS, Guideline on a Virial Equation for the Fugacity of H2O in Humid Air,
    http://www.iapws.org/relguide/VirialFugacity.html
    IAPWS, Guideline on an Equation of State for Humid Air in Contact with
    Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
    Thermodynamic Properties of Seawater, Table 10,
    http://www.iapws.org/relguide/SeaAir.html
    """
    # Check input parameters
    if T < 60 or T > 2000:
        warnings.warn("Baa out of validity range")
    if T < 130 or T > 2000:
        warnings.warn("Baw out of validity range")
    if T < 130 or T > 1273:
        warnings.warn("Bww out of validity range")
    if T < 60 or T > 2000:
        warnings.warn("Caaa out of validity range")
    if T < 193 or T > 493:
        warnings.warn("Caaw out of validity range")
    if T < 173 or T > 473:
        warnings.warn("Caww out of validity range")
    if T < 130 or T > 1273:
        warnings.warn("Cwww out of validity range")

    T_ = T/100
    tau = IAPWS95.Tc/T

    # Table 1
    # Reorganizated to easy use in equations
    tb = [-0.5, 0.875, 1, 4, 6, 12, 7]
    nb = [0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
          -0.66856572307965, 0.20433810950965, -0.66212605039687e-4,
          -0.10793600908932]
    bc = [0.5, 0.75, 1, 5, 1, 9, 10]
    nc = [0.31802509345418, -0.26145533859358, -0.19232721156002,
          -0.25709043003438, 0.17611491008752e-1, 0.22132295167546,
          -0.40247669763528]
    bc2 = [4, 6, 12]
    nc2 = [-0.66856572307965, 0.20433810950965, -0.66212605039687e-4]

    # Table 2
    ai = [3.5, 3.5]
    bi = [0.85, 0.95]
    Bi = [0.2, 0.2]
    ni = [-0.14874640856724, 0.31806110878444]
    Ci = [28, 32]
    Di = [700, 800]
    Ai = [0.32, 0.32]
    betai = [0.3, 0.3]

    # Eq 5
    sum1 = sum([n*tau**t for n, t in zip(nb, tb)])
    sum2 = 0
    for n, b, B, A, C, D in zip(ni, bi, Bi, Ai, Ci, Di):
        sum2 += n*((A+1-tau)**2+B)**b*exp(-C-D*(tau-1)**2)
    Bww = Mw/IAPWS95.rhoc*(sum1+sum2)

    # Eq 6
    sum1 = sum([n*tau**t for n, t in zip(nc, bc)])
    sum2 = sum([n*tau**t for n, t in zip(nc2, bc2)])
    sum3 = 0
    for a, b, B, n, C, D, A, beta in zip(ai, bi, Bi, ni, Ci, Di, Ai, betai):
        Tita = A+1-tau
        sum3 += n*(C*(Tita**2+B)-b*(A*Tita/beta+B*a))*(Tita**2+B)**(b-1) * \
            exp(-C-D*(tau-1)**2)
    Cwww = 2*(Mw/IAPWS95.rhoc)**2*(sum1-sum2+2*sum3)

    # Table 3
    ai = [0.482737e-3, 0.105678e-2, -0.656394e-2, 0.294442e-1, -0.319317e-1]
    bi = [-10.728876, 34.7802, -38.3383, 33.406]
    ci = [66.5687, -238.834, -176.755]
    di = [-0.237, -1.048, -3.183]

    Baw = 1e-6*sum([c*T_**d for c, d in zip(ci, di)])                  # Eq 7
    Caaw = 1e-6*sum([a/T_**i for i, a in enumerate(ai)])               # Eq 8
    Caww = -1e-6*exp(sum([b/T_**i for i, b in enumerate(bi)]))         # Eq 9

    # Eq T56
    Bawt = 1e-6*T_/T*sum([c*d*T_**(d-1) for c, d in zip(ci, di)])
    # Eq T57
    Bawtt = 1e-6*T_**2/T**2*sum(
        [c*d*(d-1)*T_**(d-2) for c, d in zip(ci, di)])
    # Eq T59
    Caawt = -1e-6*T_/T*sum([i*a*T_**(-i-1) for i, a in enumerate(ai)])
    # Eq T60
    Caawtt = 1e-6*T_**2/T**2*sum(
        [i*(i+1)*a*T_**(-i-2) for i, a in enumerate(ai)])
    # Eq T62
    Cawwt = 1e-6*T_/T*sum([i*b*T_**(-i-1) for i, b in enumerate(bi)]) * \
        exp(sum([b/T_**i for i, b in enumerate(bi)]))
    # Eq T63
    Cawwtt = -1e-6*T_**2/T**2*((
        sum([i*(i+1)*b*T_**(-i-2) for i, b in enumerate(bi)]) +
        sum([i*b*T_**(-i-1) for i, b in enumerate(bi)])**2) *
        exp(sum([b/T_**i for i, b in enumerate(bi)])))

    # Table 4
    # Reorganizated to easy use in equations
    ji = [0, 0.33, 1.01, 1.6, 3.6, 3.5]
    ni = [0.118160747229, 0.713116392079, -0.161824192067e1, -0.101365037912,
          -0.146629609713, 0.148287891978e-1]
    tau = 132.6312/T

    Baa = 1/10.4477*sum([n*tau**j for j, n in zip(ji, ni)])          # Eq 10
    Caaa = 2/10.4477**2*(0.714140178971e-1+0.101365037912*tau**1.6)  # Eq 11

    prop = {}
    prop["Baa"] = Baa/1000
    prop["Baw"] = Baw
    prop["Bww"] = Bww/1000
    prop["Caaa"] = Caaa/1e6
    prop["Caaw"] = Caaw
    prop["Caww"] = Caww
    prop["Cwww"] = Cwww/1e6
    prop["Bawt"] = Bawt
    prop["Bawtt"] = Bawtt
    prop["Caawt"] = Caawt
    prop["Caawtt"] = Caawtt
    prop["Cawwt"] = Cawwt
    prop["Cawwtt"] = Cawwtt
    return prop


def _fugacity(T, P, x):
    """Fugacity equation for humid air

    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [MPa]
    x : float
        Mole fraction of water-vapor [-]

    Returns
    -------
    fv : float
        fugacity coefficient [MPa]

    Raises
    ------
    NotImplementedError : If input isn't in range of validity
        * 193 ≤ T ≤ 473
        * 0 ≤ P ≤ 5
        * 0 ≤ x ≤ 1
        Really the xmax is the xsaturation but isn't implemented

    Examples
    --------
    >>> _fugacity(300, 1, 0.1)
    0.0884061686

    References
    ----------
    IAPWS, Guideline on a Virial Equation for the Fugacity of H2O in Humid Air,
    http://www.iapws.org/relguide/VirialFugacity.html
    """
    # Check input parameters
    if T < 193 or T > 473 or P < 0 or P > 5 or x < 0 or x > 1:
        raise(NotImplementedError("Input not in range of validity"))

    R = 8.314462  # J/molK

    # Virial coefficients
    vir = _virial(T)

    # Eq 3
    beta = x*(2-x)*vir["Bww"]+(1-x)**2*(2*vir["Baw"]-vir["Baa"])

    # Eq 4
    gamma = x**2*(3-2*x)*vir["Cwww"] + \
        (1-x)**2*(6*x*vir["Caww"]+3*(1-2*x)*vir["Caaw"]-2*(1-x)*vir["Caaa"]) +\
        (x**2*vir["Bww"]+2*x*(1-x)*vir["Baw"]+(1-x)**2*vir["Baa"]) * \
        (x*(3*x-4)*vir["Bww"]+2*(1-x)*(3*x-2)*vir["Baw"]+3*(1-x)**2*vir["Baa"])

    # Eq 2
    fv = x*P*exp(beta*P*1e6/R/T+0.5*gamma*(P*1e6/R/T)**2)
    return fv
