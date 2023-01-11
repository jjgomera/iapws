#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-branches, too-many-statements
# pylint: disable=too-many-arguments

"""
Miscelaneous internal utilities. This module include:

    * :func:`getphase`: Get phase string of state
    * :class:`_fase`: Base class to define a phase state
    * :func:`deriv_H`: Calculate generic partial derivative with a fundamental
      Helmholtz free energy equation of state
    * :func:`deriv_G`: Calculate generic partial derivative with a fundamental
      Gibbs free energy equation of state
"""

from __future__ import division


def getphase(Tc, Pc, T, P, x, region):
    """Return fluid phase string name

    Parameters
    ----------
    Tc : float
        Critical temperature, [K]
    Pc : float
        Critical pressure, [MPa]
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    x : float
        Quality, [-]
    region: int
        Region number, used only for IAPWS97 region definition

    Returns
    -------
    phase : str
        Phase name
    """
    # Avoid round problem
    P = round(P, 8)
    T = round(T, 8)
    if P > Pc and T > Tc:
        phase = "Supercritical fluid"
    elif T > Tc:
        phase = "Gas"
    elif P > Pc:
        phase = "Compressible liquid"
    elif P == Pc and T == Tc:
        phase = "Critical point"
    elif region == 4 and x == 1:
        phase = "Saturated vapor"
    elif region == 4 and x == 0:
        phase = "Saturated liquid"
    elif region == 4:
        phase = "Two phases"
    elif x == 1:
        phase = "Vapour"
    elif x == 0:
        phase = "Liquid"
    return phase


class _fase(object):
    """Class to implement a null phase"""

    v = None
    rho = None

    h = None
    s = None
    u = None
    a = None
    g = None

    cp = None
    cv = None
    cp_cv = None
    w = None
    Z = None
    fi = None
    f = None

    mu = None
    k = None
    nu = None
    Prandt = None
    epsilon = None
    alfa = None
    n = None

    alfap = None
    betap = None
    joule = None
    Gruneisen = None
    alfav = None
    kappa = None
    betas = None
    gamma = None
    Kt = None
    kt = None
    Ks = None
    ks = None
    dpdT_rho = None
    dpdrho_T = None
    drhodT_P = None
    drhodP_T = None
    dhdT_rho = None
    dhdT_P = None
    dhdrho_T = None
    dhdrho_P = None
    dhdP_T = None
    dhdP_rho = None

    Z_rho = None
    IntP = None
    hInput = None


def deriv_H(state, z, x, y, fase):
    r"""Calculate generic partial derivative
    :math:`\left.\frac{\partial z}{\partial x}\right|_{y}` from a fundamental
    helmholtz free energy equation of state

    Parameters
    ----------
    state : any python object
        Only need to define P and T properties, non phase specific properties
    z : str
        Name of variables in numerator term of derivatives
    x : str
        Name of variables in denominator term of derivatives
    y : str
        Name of constant variable in partial derivaritive
    fase : any python object
        Define phase specific properties (v, cv, alfap, s, betap)

    Notes
    -----
    x, y and z can be the following values:

        * P: Pressure
        * T: Temperature
        * v: Specific volume
        * rho: Density
        * u: Internal Energy
        * h: Enthalpy
        * s: Entropy
        * g: Gibbs free energy
        * a: Helmholtz free energy

    Returns
    -------
    deriv : float
        ∂z/∂x|y

    References
    ----------
    IAPWS, Revised Advisory Note No. 3: Thermodynamic Derivatives from IAPWS
    Formulations, http://www.iapws.org/relguide/Advise3.pdf
    """
    # We use the relation between rho and v and his partial derivative
    # ∂v/∂b|c = -1/ρ² ∂ρ/∂b|c
    # ∂a/∂v|c = -ρ² ∂a/∂ρ|c
    mul = 1
    if z == "rho":
        mul = -fase.rho**2
        z = "v"
    if x == "rho":
        mul = -1/fase.rho**2
        x = "v"
    if y == "rho":
        y = "v"

    if x == "P":
        dTdx = state.P*1000*fase.alfap
        dvdx = -state.P*1000*fase.betap
    elif x == "T":
        dTdx = 1
        dvdx = 0
    elif x == "v":
        dTdx = 0
        dvdx = 1
    elif x == "u":
        dTdx = fase.cv
        dvdx = state.P*1000*(state.T*fase.alfap-1)
    elif x == "h":
        dTdx = fase.cv+state.P*1000*fase.v*fase.alfap
        dvdx = state.P*1000*(state.T*fase.alfap-fase.v*fase.betap)
    elif x == "s":
        dTdx = fase.cv/state.T
        dvdx = state.P*1000*fase.alfap
    elif x == "g":
        dTdx = state.P*1000*fase.v*fase.alfap-fase.s
        dvdx = -state.P*1000*fase.v*fase.betap
    elif x == "a":
        dTdx = -fase.s
        dvdx = -state.P*1000

    if y == "P":
        dTdy = state.P*1000*fase.alfap
        dvdy = -state.P*1000*fase.betap
    elif y == "T":
        dTdy = 1
        dvdy = 0
    elif y == "v":
        dTdy = 0
        dvdy = 1
    elif y == "u":
        dTdy = fase.cv
        dvdy = state.P*1000*(state.T*fase.alfap-1)
    elif y == "h":
        dTdy = fase.cv+state.P*1000*fase.v*fase.alfap
        dvdy = state.P*1000*(state.T*fase.alfap-fase.v*fase.betap)
    elif y == "s":
        dTdy = fase.cv/state.T
        dvdy = state.P*1000*fase.alfap
    elif y == "g":
        dTdy = state.P*1000*fase.v*fase.alfap-fase.s
        dvdy = -state.P*1000*fase.v*fase.betap
    elif y == "a":
        dTdy = -fase.s
        dvdy = -state.P*1000

    if z == "P":
        dTdz = state.P*1000*fase.alfap
        dvdz = -state.P*1000*fase.betap
    elif z == "T":
        dTdz = 1
        dvdz = 0
    elif z == "v":
        dTdz = 0
        dvdz = 1
    elif z == "u":
        dTdz = fase.cv
        dvdz = state.P*1000*(state.T*fase.alfap-1)
    elif z == "h":
        dTdz = fase.cv+state.P*1000*fase.v*fase.alfap
        dvdz = state.P*1000*(state.T*fase.alfap-fase.v*fase.betap)
    elif z == "s":
        dTdz = fase.cv/state.T
        dvdz = state.P*1000*fase.alfap
    elif z == "g":
        dTdz = state.P*1000*fase.v*fase.alfap-fase.s
        dvdz = -state.P*1000*fase.v*fase.betap
    elif z == "a":
        dTdz = -fase.s
        dvdz = -state.P*1000

    deriv = (dvdz*dTdy-dTdz*dvdy)/(dvdx*dTdy-dTdx*dvdy)
    return mul*deriv


def deriv_G(state, z, x, y, fase):
    r"""Calculate generic partial derivative
    :math:`\left.\frac{\partial z}{\partial x}\right|_{y}` from a fundamental
    Gibbs free energy equation of state

    Parameters
    ----------
    state : any python object
        Only need to define P and T properties, non phase specific properties
    z : str
        Name of variables in numerator term of derivatives
    x : str
        Name of variables in denominator term of derivatives
    y : str
        Name of constant variable in partial derivaritive
    fase : any python object
        Define phase specific properties (v, cp, alfav, s, xkappa)

    Notes
    -----
    x, y and z can be the following values:

        * P: Pressure
        * T: Temperature
        * v: Specific volume
        * rho: Density
        * u: Internal Energy
        * h: Enthalpy
        * s: Entropy
        * g: Gibbs free energy
        * a: Helmholtz free energy

    Returns
    -------
    deriv : float
        ∂z/∂x|y

    References
    ----------
    IAPWS, Revised Advisory Note No. 3: Thermodynamic Derivatives from IAPWS
    Formulations, http://www.iapws.org/relguide/Advise3.pdf
    """
    mul = 1
    if z == "rho":
        mul = -fase.rho**2
        z = "v"
    if x == "rho":
        mul = -1/fase.rho**2
        x = "v"

    if x == "P":
        dPdx = 1.0
        dTdx = 0.0
    elif x == "T":
        dPdx = 0.0
        dTdx = 1.0
    elif x == "v":
        dPdx = -fase.v*fase.xkappa
        dTdx = fase.v*fase.alfav
    elif x == "u":
        dPdx = fase.v*(state.P*1000.0*fase.xkappa-state.T*fase.alfav)
        dTdx = fase.cp-state.P*1000.0*fase.v*fase.alfav
    elif x == "h":
        dPdx = fase.v*(1.0-state.T*fase.alfav)
        dTdx = fase.cp
    elif x == "s":
        dPdx = -fase.v * fase.alfav
        dTdx = fase.cp/state.T
    elif x == "g":
        dPdx = fase.v
        dTdx = -fase.s
    elif x == "a":
        dPdx = state.P*1000.0*fase.v*fase.xkappa
        dTdx = -state.P * 1000.0 * fase.v * fase.alfav - fase.s
    else:
        raise ValueError("x must be one of P, T, v, u, h, s, g, a")

    if y == "P":
        dPdy = 1.0
        dTdy = 0.0
    elif y == "T":
        dPdy = 0.0
        dTdy = 1.0
    elif y == "v":
        dPdy = -fase.v*fase.xkappa
        dTdy = fase.v*fase.alfav
    elif y == "u":
        dPdy = fase.v*(state.P*1000.0*fase.xkappa-state.T*fase.alfav)
        dTdy = fase.cp-state.P*1000.0*fase.v*fase.alfav
    elif y == "h":
        dPdy = fase.v*(1.0-state.T*fase.alfav)
        dTdy = fase.cp
    elif y == "s":
        dPdy = -fase.v * fase.alfav
        dTdy = fase.cp/state.T
    elif y == "g":
        dPdy = fase.v
        dTdy = -fase.s
    elif y == "a":
        dPdy = state.P*1000.0*fase.v*fase.xkappa
        dTdy = -state.P * 1000.0 * fase.v * fase.alfav - fase.s
    else:
        raise ValueError("y must be one of P, T, v, u, h, s, g, a")

    if z == "P":
        dPdz = 1.0
        dTdz = 0.0
    elif z == "T":
        dPdz = 0.0
        dTdz = 1.0
    elif z == "v":
        dPdz = -fase.v*fase.xkappa
        dTdz = fase.v*fase.alfav
    elif z == "u":
        dPdz = fase.v*(state.P*1000.0*fase.xkappa-state.T*fase.alfav)
        dTdz = fase.cp-state.P*1000.0*fase.v*fase.alfav
    elif z == "h":
        dPdz = fase.v*(1.0-state.T*fase.alfav)
        dTdz = fase.cp
    elif z == "s":
        dPdz = -fase.v * fase.alfav
        dTdz = fase.cp/state.T
    elif z == "g":
        dPdz = fase.v
        dTdz = -fase.s
    elif z == "a":
        dPdz = state.P*1000.0*fase.v*fase.xkappa
        dTdz = -state.P * 1000.0 * fase.v * fase.alfav - fase.s
    else:
        raise ValueError("z must be one of P, T, v, u, h, s, g, a")

    deriv = (dPdz * dTdy - dTdz * dPdy) / (dPdx * dTdy - dTdx * dPdy)
    return mul*deriv
