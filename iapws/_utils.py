#!/usr/bin/python
# -*- coding: utf-8 -*-
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

    dT = {"P": state.P*1000*fase.alfap,
          "T": 1,
          "v": 0,
          "u": fase.cv,
          "h": fase.cv+state.P*1000*fase.v*fase.alfap,
          "s": fase.cv/state.T,
          "g": state.P*1000*fase.v*fase.alfap-fase.s,
          "a": -fase.s}
    dv = {"P": -state.P*1000*fase.betap,
          "T": 0,
          "v": 1,
          "u": state.P*1000*(state.T*fase.alfap-1),
          "h": state.P*1000*(state.T*fase.alfap-fase.v*fase.betap),
          "s": state.P*1000*fase.alfap,
          "g": -state.P*1000*fase.v*fase.betap,
          "a": -state.P*1000}
    deriv = (dv[z]*dT[y]-dT[z]*dv[y])/(dv[x]*dT[y]-dT[x]*dv[y])
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

    dT = {"P": 0,
          "T": 1,
          "v": fase.v*fase.alfav,
          "u": fase.cp-state.P*1000*fase.v*fase.alfav,
          "h": fase.cp,
          "s": fase.cp/state.T,
          "g": -fase.s,
          "a": -state.P*1000*fase.v*fase.alfav-fase.s}
    dP = {"P": 1,
          "T": 0,
          "v": -fase.v*fase.xkappa,
          "u": fase.v*(state.P*1000*fase.xkappa-state.T*fase.alfav),
          "h": fase.v*(1-state.T*fase.alfav),
          "s": -fase.v*fase.alfav,
          "g": fase.v,
          "a": state.P*1000*fase.v*fase.xkappa}
    deriv = (dP[z]*dT[y]-dT[z]*dP[y])/(dP[x]*dT[y]-dT[x]*dP[y])
    return mul*deriv
