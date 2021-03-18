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
from typing import Optional, Any


def getphase(Tc: float, Pc: float, T: float, P: float, x: float, region: int) -> str:
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
    """
    Class to implement a null phase.

    IAPWS95 and IAPWS97 both implement Liquid and Gas/Vapor phasee in
    addition to being phases themselves.  Confusingly minor
    differences between derived classes impose different constraints
    on this class.
    """

    def __init__(self) -> None:
        # One always computed form the other
        self.v = float('nan')
        self.rho = float('nan')

        self.h = float('nan')
        self.s = float('nan')

        self.cv = float('nan')
        self.alfap = float('nan')
        self.betap = float('nan')
        self.cp = float('nan')
        self.kappa = float('nan')
        self.alfav = float('nan')

        self.g = float('nan')
        self.fi = float('nan')

        self.w = float('nan')
        self.Z = float('nan')

        self.drhodP_T = float('nan')
        self.mu = float('nan')
        self.cp_cv = float('nan')
        self.k = float('nan')

        self.epsilon: Optional[float] = None
        self.n: Optional[float] = None

        # --------------------------------------------
        # Calculated identically between 95 and 97
        self.u = float('nan')
        self.a = float('nan')
        self.nu = float('nan')
        self.Prandt = float('nan')
        self.alfa = float('nan')
        self.f = float('nan')

        # Calculated similarly, but not identically?
        self.joule = float('nan')
        self.gamma = float('nan')
        self.deltat = float('nan')

        # Calculated on 95 only from earlier variables and self.M
        self.rhoM = float('nan')
        self.hM = float('nan')
        self.sM = float('nan')
        self.uM = float('nan')
        self.aM = float('nan')
        self.gM = float('nan')
        self.cvM = float('nan')
        self.cpM = float('nan')
        self.Z_rho = float('nan')

        # Derivatives calculated only in IAPWS95
        self.dpdT_rho = float('nan')
        self.dpdrho_T = float('nan')
        self.drhodT_P = float('nan')
        self.dhdT_rho = float('nan')
        self.dhdT_P = float('nan')
        self.dhdrho_T = float('nan')
        self.dhdrho_P = float('nan')
        self.dhdP_T = float('nan')
        self.dhdP_rho = float('nan')
        self.kt = float('nan')
        self.ks = float('nan')
        self.Ks = float('nan')
        self.Kt = float('nan')
        self.betas = float('nan')
        self.Gruneisen = float('nan')
        self.IntP = float('nan')
        self.hInput = float('nan')

        # Properties added because various methods set/access them?
        self.xkappa = float('nan')
        self.kappas = float('nan')


def deriv_H(state: Any, z: str, x: str, y: str, fase: _fase) -> float:
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
    mul = 1.0
    if z == "rho":
        assert(isinstance(fase.rho, float))
        mul = -fase.rho**2
        z = "v"
    if x == "rho":
        assert(isinstance(fase.rho, float))
        mul = -1/fase.rho**2
        x = "v"
    if y == "rho":
        y = "v"

    assert(isinstance(fase.alfap, float))
    assert(isinstance(fase.betap, float))
    assert(isinstance(fase.v, float))
    assert(isinstance(fase.cv, float))
    assert(isinstance(fase.s, float))

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


class HelmholtzDerivatives(object):
    """Helmholtz free energy and derivatives."""

    def __init__(self, fi: float, fit: float, fid: float,
                 fitt: float, fidd: float, fidt: float) -> None:
        self.fi = fi
        self.fit = fit
        self.fid = fid
        self.fitt = fitt
        self.fidd = fidd
        self.fidt = fidt


def deriv_G(state: Any, z: str, x: str, y: str, fase: _fase) -> float:
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
    mul = 1.0
    if z == "rho":
        assert(isinstance(fase.rho, float))
        mul = -fase.rho**2
        z = "v"
    if x == "rho":
        assert(isinstance(fase.rho, float))
        mul = -1/fase.rho**2
        x = "v"

    assert(isinstance(fase.alfav, float))
    assert(isinstance(fase.v, float))
    assert(isinstance(fase.cp, float))
    assert(isinstance(fase.s, float))
    assert(isinstance(fase.xkappa, float))

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
