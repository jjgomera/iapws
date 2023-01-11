#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes, too-many-boolean-expressions
# pylint: disable=too-many-locals

"""
IAPWS standard for Seawater IAPWS08 and related functionality. The module
include:

:class:`SeaWater`: Global module class with all the functionality integrated

Other functionality:
   * :func:`_Tb`: Boiling temperature of seawater
   * :func:`_Tf`: Freezing temperature of seawater
   * :func:`_Triple`: Triple point properties of seawater
   * :func:`_OsmoticPressure`: Osmotic pressure of seawater
   * :func:`_ThCond_SeaWater`: Thermal conductivity of seawater
   * :func:`_Tension_SeaWater`: Surface tension of seawater
   * :func:`_solNa2SO4`: Solubility of sodium sulfate in aqueous mixtures of
     sodium chloride and sulfuric acid
   * :func:`_critNaCl`: Critical locus of aqueous solutions of sodium chloride
"""

from __future__ import division
from math import exp, log
import warnings

from scipy.optimize import fsolve

from .iapws95 import IAPWS95
from .iapws97 import IAPWS97, _Region1, _Region2, _TSat_P
from ._iapws import _ThCond, Tc, Pc, rhoc, _Ice, _Tension
from ._utils import deriv_G


# Constants
Rm = 8.314472
Sn = 0.03516504
S_ = Sn*40/35
Ms = 31.4038218
T_ = 40
P_ = 100
Po = 0.101325
To = 273.15


class SeaWater(object):
    """
    Class to model seawater with standard IAPWS-08

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    S : float
        Salinity, [kg/kg]

    fast : bool, default False
        Use the Supplementary release SR7-09 to speed up the calculation
    IF97 : bool, default False
        Use the Advisory Note No. 5 with industrial formulation

    Returns
    -------
    rho : float
        Density, [kg/m³]
    v : float
        Specific volume, [m³/kg]
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kg·K]
    u : float
        Specific internal energy, [kJ/kg]
    g : float
        Specific Gibbs free energy, [kJ/kg]
    a : float
        Specific Helmholtz free energy, [kJ/kg]
    cp : float
        Specific isobaric heat capacity, [kJ/kg·K]
    cv : float
        Specific isochoric heat capacity, [kJ/kg·K]
    gt : float
        Derivative Gibbs energy with temperature, [kJ/kg·K]
    gp : float
        Derivative Gibbs energy with pressure, [m³/kg]
    gtt : float
        Derivative Gibbs energy with temperature square, [kJ/kg·K²]
    gtp : float
        Derivative Gibbs energy with pressure and temperature, [m³/kg·K]
    gpp : float
        Derivative Gibbs energy with temperature square, [m³/kg·MPa]
    gs : float
        Derivative Gibbs energy with salinity, [kJ/kg]
    gsp : float
        Derivative Gibbs energy with salinity and pressure, [m³/kg]
    alfav : float
        Thermal expansion coefficient, [1/K]
    betas : float
        Isentropic temperature-pressure coefficient, [K/MPa]
    xkappa : float
        Isothermal compressibility, [1/MPa]
    ks : float
        Isentropic compressibility, [1/MPa]
    w : float
        Sound Speed, [m/s]

    k : float
        Thermal conductivity, [W/m·K]
    sigma: float
        Surface tension, [N/m]

    m : float
        Molality of seawater, [mol/kg]
    mu : float
        Relative chemical potential, [kJ/kg]
    muw : float
        Chemical potential of H2O, [kJ/kg]
    mus : float
        Chemical potential of sea salt, [kJ/kg]
    osm : float
        Osmotic coefficient, [-]
    haline : float
        Haline contraction coefficient, [kg/kg]

    Notes
    -----
    :class:`Warning` if input isn't in limit:

        * 261 ≤ T ≤ 353
        * 0 < P ≤ 100
        * 0 ≤ S ≤ 0.12

    References
    ----------
    IAPWS, Release on the IAPWS Formulation 2008 for the Thermodynamic
    Properties of Seawater, http://www.iapws.org/relguide/Seawater.html

    IAPWS, Supplementary Release on a Computationally Efficient Thermodynamic
    Formulation for Liquid Water for Oceanographic Use,
    http://www.iapws.org/relguide/OceanLiquid.html

    IAPWS, Guideline on the Thermal Conductivity of Seawater,
    http://www.iapws.org/relguide/Seawater-ThCond.html

    IAPWS, Guideline on the Surface Tension of Seawater,
    http://www.iapws.org/relguide/Seawater-Surf.html

    IAPWS, Revised Advisory Note No. 3: Thermodynamic Derivatives from IAPWS
    Formulations, http://www.iapws.org/relguide/Advise3.pdf

    IAPWS,  Advisory Note No. 5: Industrial Calculation of the Thermodynamic
    Properties of Seawater, http://www.iapws.org/relguide/Advise5.html

    Examples
    --------
    >>> salt = iapws.SeaWater(T=300, P=1, S=0.04)
    >>> salt.rho
    1026.7785717245113
    >>> salt.gs
    88.56221805501536
    >>> salt.haline
    0.7311487666026304
    """

    kwargs = {"T": 0.0,
              "P": 0.0,
              "S": None,
              "fast": False,
              "IF97": False}
    status = 0
    msg = "Undefined"

    T = None
    P = None
    rho = None
    v = None
    s = None
    cp = None
    cv = None
    h = None
    u = None
    a = None
    alfav = None
    betas = None
    xkappa = None
    ks = None
    w = None
    k = None

    sigma = None
    m = None
    mu = None
    muw = None
    mus = None
    osm = None
    haline = None

    def __init__(self, **kwargs):
        """Constructor, initinialice kwargs"""
        self.kwargs = SeaWater.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        """Make instance callable to can add input parameter one to one"""
        self.kwargs.update(kwargs)

        if self.kwargs["T"] and self.kwargs["P"] and \
                self.kwargs["S"] is not None:
            self.status = 1
            self.calculo()
            self.msg = ""

    def calculo(self):
        """Calculate procedure"""
        T = self.kwargs["T"]
        P = self.kwargs["P"]
        S = self.kwargs["S"]

        self.m = S/(1-S)/Ms
        if self.kwargs["fast"] and T <= 313.15:
            pw = self._waterSupp(T, P)
        elif self.kwargs["IF97"]:
            pw = self._waterIF97(T, P)
        else:
            pw = self._water(T, P)
        ps = self.saline(T, P, S)

        prop = {}
        for key in ps:
            prop[key] = pw[key]+ps[key]
            self.__setattr__(key, prop[key])

        self.T = T
        self.P = P
        self.rho = 1./prop["gp"]
        self.v = prop["gp"]
        self.s = -prop["gt"]
        self.cp = -T*prop["gtt"]
        self.cv = T*(prop["gtp"]**2/prop["gpp"]-prop["gtt"])
        self.h = prop["g"]-T*prop["gt"]
        self.u = prop["g"]-T*prop["gt"]-P*1000*prop["gp"]
        self.a = prop["g"]-P*1000*prop["gp"]
        self.alfav = prop["gtp"]/prop["gp"]
        self.betas = -prop["gtp"]/prop["gtt"]
        self.xkappa = -prop["gpp"]/prop["gp"]
        self.ks = (prop["gtp"]**2-prop["gtt"]*prop["gpp"])/prop["gp"] / \
            prop["gtt"]
        self.w = prop["gp"]*(prop["gtt"]*1000/(
            prop["gtp"]**2 - prop["gtt"]*1000*prop["gpp"]*1e-6))**0.5

        # Thermal conductivity calculation
        if "thcond" in pw:
            kw = pw["thcond"]
        else:
            kw = _ThCond(1/pw["gp"], T)
        try:
            self.k = _ThCond_SeaWater(T, P, S)+kw
        except NotImplementedError:
            self.k = None

        # Surface tension calculation
        try:
            self.sigma = _Tension_SeaWater(T, S)
        except NotImplementedError:
            pass

        if S:
            self.mu = prop["gs"]
            self.muw = prop["g"]-S*prop["gs"]
            self.mus = prop["g"]+(1-S)*prop["gs"]
            self.osm = -(ps["g"]-S*prop["gs"])/self.m/Rm/T
            self.haline = -prop["gsp"]/prop["gp"]

    def derivative(self, z, x, y):
        """
        Wrapper derivative for custom derived properties
        where x, y, z can be: P, T, v, u, h, s, g, a
        """
        return deriv_G(self, z, x, y, self)

    @classmethod
    def _water(cls, T, P):
        """Get properties of pure water, Table4 pag 8"""
        water = IAPWS95(P=P, T=T)
        prop = {}
        prop["g"] = water.h-T*water.s
        prop["gt"] = -water.s
        prop["gp"] = 1./water.rho
        prop["gtt"] = -water.cp/T
        prop["gtp"] = water.betas*water.cp/T
        prop["gpp"] = -1e6/(water.rho*water.w)**2-water.betas**2*1e3*water.cp/T
        prop["gs"] = 0
        prop["gsp"] = 0
        prop["thcond"] = water.k
        return prop

    @classmethod
    def _waterIF97(cls, T, P):
        water = IAPWS97(P=P, T=T)
        betas = water.derivative("T", "P", "s", water)
        prop = {}
        prop["g"] = water.h-T*water.s
        prop["gt"] = -water.s
        prop["gp"] = 1./water.rho
        prop["gtt"] = -water.cp/T
        prop["gtp"] = betas*water.cp/T
        prop["gpp"] = -1e6/(water.rho*water.w)**2-betas**2*1e3*water.cp/T
        prop["gs"] = 0
        prop["gsp"] = 0
        return prop

    @classmethod
    def _waterSupp(cls, T, P):
        """
        Get properties of pure water using the supplementary release SR7-09,
        Table4 pag 6
        """
        tau = (T-273.15)/40
        pi = (P-0.101325)/100

        J = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
             3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7]
        K = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2,
             3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1]
        G = [0.101342743139674e3, 0.100015695367145e6, -0.254457654203630e4,
             0.284517778446287e3, -0.333146754253611e2, 0.420263108803084e1,
             -0.546428511471039, 0.590578347909402e1, -0.270983805184062e3,
             0.776153611613101e3, -0.196512550881220e3, 0.289796526294175e2,
             -0.213290083518327e1, -0.123577859330390e5, 0.145503645404680e4,
             -0.756558385769359e3, 0.273479662323528e3, -0.555604063817218e2,
             0.434420671917197e1, 0.736741204151612e3, -0.672507783145070e3,
             0.499360390819152e3, -0.239545330654412e3, 0.488012518593872e2,
             -0.166307106208905e1, -0.148185936433658e3, 0.397968445406972e3,
             -0.301815380621876e3, 0.152196371733841e3, -0.263748377232802e2,
             0.580259125842571e2, -0.194618310617595e3, 0.120520654902025e3,
             -0.552723052340152e2, 0.648190668077221e1, -0.189843846514172e2,
             0.635113936641785e2, -0.222897317140459e2, 0.817060541818112e1,
             0.305081646487967e1, -0.963108119393062e1]

        g, gt, gp, gtt, gtp, gpp = 0, 0, 0, 0, 0, 0
        for j, k, gi in zip(J, K, G):
            g += gi*tau**j*pi**k
            if j >= 1:
                gt += gi*j*tau**(j-1)*pi**k
            if k >= 1:
                gp += k*gi*tau**j*pi**(k-1)
            if j >= 2:
                gtt += j*(j-1)*gi*tau**(j-2)*pi**k
            if j >= 1 and k >= 1:
                gtp += j*k*gi*tau**(j-1)*pi**(k-1)
            if k >= 2:
                gpp += k*(k-1)*gi*tau**j*pi**(k-2)

        prop = {}
        prop["g"] = g*1e-3
        prop["gt"] = gt/40*1e-3
        prop["gp"] = gp/100*1e-6
        prop["gtt"] = gtt/40**2*1e-3
        prop["gtp"] = gtp/40/100*1e-6
        prop["gpp"] = gpp/100**2*1e-6
        prop["gs"] = 0
        prop["gsp"] = 0
        return prop

    @classmethod
    def saline(cls, T, P, S):
        """Eq 4"""
        # Check input in range of validity
        if T <= 261 or T > 353 or P <= 0 or P > 100 or S < 0 or S > 0.12:
            warnings.warn("Incoming out of bound")

        X = (S/S_)**0.5
        tau = (T-273.15)/40
        pi = (P-0.101325)/100

        Li = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 2, 3, 4, 2, 3, 4, 2, 3, 4,
              2, 4, 2, 2, 3, 4, 5, 2, 3, 4, 2, 3, 2, 3, 2, 3, 2, 3, 4, 2, 3, 2,
              3, 2, 2, 2, 3, 4, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2]
        Lj = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
              5, 5, 6, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 1, 1, 2,
              2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 3, 4, 0, 0, 1, 2, 3, 0, 1, 2]
        Lk = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
              2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5]
        G = [0.581281456626732e4, 0.141627648484197e4, -0.243214662381794e4,
             0.202580115603697e4, -0.109166841042967e4, 0.374601237877840e3,
             -0.485891069025409e2, 0.851226734946706e3, 0.168072408311545e3,
             -0.493407510141682e3, 0.543835333000098e3, -0.196028306689776e3,
             0.367571622995805e2, 0.880031352997204e3, -0.430664675978042e2,
             -0.685572509204491e2, -0.225267649263401e3, -0.100227370861875e2,
             0.493667694856254e2, 0.914260447751259e2, 0.875600661808945,
             -0.171397577419788e2, -0.216603240875311e2, 0.249697009569508e1,
             0.213016970847183e1, -0.331049154044839e4, 0.199459603073901e3,
             -0.547919133532887e2, 0.360284195611086e2, 0.729116529735046e3,
             -0.175292041186547e3, -0.226683558512829e2, -0.860764303783977e3,
             0.383058066002476e3, 0.694244814133268e3, -0.460319931801257e3,
             -0.297728741987187e3, 0.234565187611355e3, 0.384794152978599e3,
             -0.522940909281335e2, -0.408193978912261e1, -0.343956902961561e3,
             0.831923927801819e2, 0.337409530269367e3, -0.541917262517112e2,
             -0.204889641964903e3, 0.747261411387560e2, -0.965324320107458e2,
             0.680444942726459e2, -0.301755111971161e2, 0.124687671116248e3,
             -0.294830643494290e2, -0.178314556207638e3, 0.256398487389914e2,
             0.113561697840594e3, -0.364872919001588e2, 0.158408172766824e2,
             -0.341251932441282e1, -0.316569643860730e2, 0.442040358308000e2,
             -0.111282734326413e2, -0.262480156590992e1, 0.704658803315449e1,
             -0.792001547211682e1]

        g, gt, gp, gtt, gtp, gpp, gs, gsp = 0, 0, 0, 0, 0, 0, 0, 0

        # Calculate only for some salinity
        if S != 0:
            for i, j, k, gi in zip(Li, Lj, Lk, G):
                if i == 1:
                    g += gi*X**2*log(X)*tau**j*pi**k
                    gs += gi*(2*log(X)+1)*tau**j*pi**k
                else:
                    g += gi*X**i*tau**j*pi**k
                    gs += i*gi*X**(i-2)*tau**j*pi**k
                if j >= 1:
                    if i == 1:
                        gt += gi*X**2*log(X)*j*tau**(j-1)*pi**k
                    else:
                        gt += gi*X**i*j*tau**(j-1)*pi**k
                if k >= 1:
                    gp += k*gi*X**i*tau**j*pi**(k-1)
                    gsp += i*k*gi*X**(i-2)*tau**j*pi**(k-1)
                if j >= 2:
                    gtt += j*(j-1)*gi*X**i*tau**(j-2)*pi**k
                if j >= 1 and k >= 1:
                    gtp += j*k*gi*X**i*tau**(j-1)*pi**(k-1)
                if k >= 2:
                    gpp += k*(k-1)*gi*X**i*tau**j*pi**(k-2)

        prop = {}
        prop["g"] = g*1e-3
        prop["gt"] = gt/40*1e-3
        prop["gp"] = gp/100*1e-6
        prop["gtt"] = gtt/40**2*1e-3
        prop["gtp"] = gtp/40/100*1e-6
        prop["gpp"] = gpp/100**2*1e-6
        prop["gs"] = gs/S_/2*1e-3
        prop["gsp"] = gsp/S_/2/100*1e-6
        return prop


def _Tb(P, S):
    """Procedure to calculate the boiling temperature of seawater

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    S : float
        Salinity, [kg/kg]

    Returns
    -------
    Tb : float
        Boiling temperature, [K]

    References
    ----------
    IAPWS,  Advisory Note No. 5: Industrial Calculation of the Thermodynamic
    Properties of Seawater, http://www.iapws.org/relguide/Advise5.html, Eq 7
    """
    def f(T):
        pw = _Region1(T, P)
        gw = pw["h"]-T*pw["s"]

        pv = _Region2(T, P)
        gv = pv["h"]-T*pv["s"]

        ps = SeaWater.saline(T, P, S)
        return -ps["g"]+S*ps["gs"]-gw+gv

    try:
        to = _TSat_P(P)
    except NotImplementedError:
        to = 300
    rinput = fsolve(f, to, full_output=True)
    Tb = fsolve(f, to)[0]
    if rinput[2] == 1:
        return Tb


def _Tf(P, S):
    """Procedure to calculate the freezing temperature of seawater

    Parameters
    ----------
    P : float
        Pressure, [MPa]
    S : float
        Salinity, [kg/kg]

    Returns
    -------
    Tf : float
        Freezing temperature, [K]

    References
    ----------
    IAPWS,  Advisory Note No. 5: Industrial Calculation of the Thermodynamic
    Properties of Seawater, http://www.iapws.org/relguide/Advise5.html, Eq 12
    """
    def f(T):
        T = float(T)
        pw = _Region1(T, P)
        gw = pw["h"]-T*pw["s"]

        gih = _Ice(T, P)["g"]

        ps = SeaWater.saline(T, P, S)
        return -ps["g"]+S*ps["gs"]-gw+gih

    try:
        to = _TSat_P(P)
    except NotImplementedError:
        to = 300
    rinput = fsolve(f, to, full_output=True)
    Tf = fsolve(f, to)[0]
    if rinput[2] == 1:
        return Tf


def _Triple(S):
    """Procedure to calculate the triple point pressure and temperature for
    seawater

    Parameters
    ----------
    S : float
        Salinity, [kg/kg]

    Returns
    -------
    prop : dict
        Dictionary with the triple point properties:

            * Tt: Triple point temperature, [K]
            * Pt: Triple point pressure, [MPa]

    References
    ----------
    IAPWS,  Advisory Note No. 5: Industrial Calculation of the Thermodynamic
    Properties of Seawater, http://www.iapws.org/relguide/Advise5.html, Eq 7
    """
    def f(parr):
        T, P = parr
        pw = _Region1(T, P)
        gw = pw["h"]-T*pw["s"]

        pv = _Region2(T, P)
        gv = pv["h"]-T*pv["s"]

        gih = _Ice(T, P)["g"]
        ps = SeaWater.saline(T, P, S)

        return -ps["g"]+S*ps["gs"]-gw+gih, -ps["g"]+S*ps["gs"]-gw+gv

    Tt, Pt = fsolve(f, [273, 6e-4])

    prop = {}
    prop["Tt"] = Tt
    prop["Pt"] = Pt
    return prop


def _OsmoticPressure(T, P, S):
    """Procedure to calculate the osmotic pressure of seawater

    Parameters
    ----------
    T : float
        Tmperature, [K]
    P : float
        Pressure, [MPa]
    S : float
        Salinity, [kg/kg]

    Returns
    -------
    Posm : float
        Osmotic pressure, [MPa]

    References
    ----------
    IAPWS,  Advisory Note No. 5: Industrial Calculation of the Thermodynamic
    Properties of Seawater, http://www.iapws.org/relguide/Advise5.html, Eq 15
    """
    pw = _Region1(T, P)
    gw = pw["h"]-T*pw["s"]

    def f(Posm):
        pw2 = _Region1(T, P+Posm)
        gw2 = pw2["h"]-T*pw2["s"]
        ps = SeaWater.saline(T, P+Posm, S)
        return -ps["g"]+S*ps["gs"]-gw+gw2

    Posm = fsolve(f, 0)[0]
    return Posm


def _ThCond_SeaWater(T, P, S):
    """Equation for the thermal conductivity of seawater

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    S : float
        Salinity, [kg/kg]

    Returns
    -------
    k : float
        Thermal conductivity excess relative to that of the pure water, [W/mK]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 273.15 ≤ T ≤ 523.15
        * 0 ≤ P ≤ 140
        * 0 ≤ S ≤ 0.17

    Examples
    --------
    >>> _ThCond_Seawater(293.15, 0.1, 0.035)
    -0.00418604

    References
    ----------
    IAPWS, Guideline on the Thermal Conductivity of Seawater,
    http://www.iapws.org/relguide/Seawater-ThCond.html
    """
    # Check input parameters
    if T < 273.15 or T > 523.15 or P < 0 or P > 140 or S < 0 or S > 0.17:
        raise NotImplementedError("Incoming out of bound")

    # Eq 4
    a1 = -7.180891e-5+1.831971e-7*P
    a2 = 1.048077e-3-4.494722e-6*P

    # Eq 5
    b1 = 1.463375e-1+9.208586e-4*P
    b2 = -3.086908e-3+1.798489e-5*P

    a = a1*exp(a2*(T-273.15))  # Eq 2
    b = b1*exp(b2*(T-273.15))  # Eq 3

    # Eq 1
    DL = a*(1000*S)**(1+b)
    return DL


def _Tension_SeaWater(T, S):
    """Equation for the surface tension of seawater

    Parameters
    ----------
    T : float
        Temperature, [K]
    S : float
        Salinity, [kg/kg]

    Returns
    -------
    σ : float
        Surface tension, [N/m]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 0 ≤ S ≤ 0.131  for 274.15 ≤ T ≤ 365.15
        * 0 ≤ S ≤ 0.038  for 248.15 ≤ T ≤ 274.15

    Examples
    --------
    >>> _Tension_Seawater(253.15, 0.035)
    -0.07922517961

    References
    ----------
    IAPWS, Guideline on the Surface Tension of Seawater,
    http://www.iapws.org/relguide/Seawater-Surf.html
    """
    # Check input parameters
    if 248.15 < T < 274.15:
        if S < 0 or S > 0.038:
            raise NotImplementedError("Incoming out of bound")
    elif 274.15 < T < 365.15:
        if S < 0 or S > 0.131:
            raise NotImplementedError("Incoming out of bound")
    else:
        raise NotImplementedError("Incoming out of bound")

    sw = _Tension(T)
    sigma = sw*(1+3.766e-1*S+2.347e-3*S*(T-273.15))
    return sigma


def _solNa2SO4(T, mH2SO4, mNaCl):
    """Equation for the solubility of sodium sulfate in aqueous mixtures of
    sodium chloride and sulfuric acid

    Parameters
    ----------
    T : float
        Temperature, [K]
    mH2SO4 : float
        Molality of sufuric acid, [mol/kg(water)]
    mNaCl : float
        Molality of sodium chloride, [mol/kg(water)]

    Returns
    -------
    S : float
        Molal solutility of sodium sulfate, [mol/kg(water)]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 523.15 ≤ T ≤ 623.15
        * 0 ≤ mH2SO4 ≤ 0.75
        * 0 ≤ mNaCl ≤ 2.25

    Examples
    --------
    >>> _solNa2SO4(523.15, 0.25, 0.75)
    2.68

    References
    ----------
    IAPWS, Solubility of Sodium Sulfate in Aqueous Mixtures of Sodium Chloride
    and Sulfuric Acid from Water to Concentrated Solutions,
    http://www.iapws.org/relguide/na2so4.pdf
    """
    # Check input parameters
    if T < 523.15 or T > 623.15 or mH2SO4 < 0 or mH2SO4 > 0.75 or \
            mNaCl < 0 or mNaCl > 2.25:
        raise NotImplementedError("Incoming out of bound")

    A00 = -0.8085987*T+81.4613752+0.10537803*T*log(T)
    A10 = 3.4636364*T-281.63322-0.46779874*T*log(T)
    A20 = -6.0029634*T+480.60108+0.81382854*T*log(T)
    A30 = 4.4540258*T-359.36872-0.60306734*T*log(T)
    A01 = 0.4909061*T-46.556271-0.064612393*T*log(T)
    A02 = -0.002781314*T+1.722695+0.0000013319698*T*log(T)
    A03 = -0.014074108*T+0.99020227+0.0019397832*T*log(T)
    A11 = -0.87146573*T+71.808756+0.11749585*T*log(T)

    S = A00 + A10*mH2SO4 + A20*mH2SO4**2 + A30*mH2SO4**3 + A01*mNaCl + \
        A02*mNaCl**2 + A03*mNaCl**3 + A11*mH2SO4*mNaCl

    return S


def _critNaCl(x):
    """Equation for the critical locus of aqueous solutions of sodium chloride

    Parameters
    ----------
    x : float
        Mole fraction of NaCl, [-]

    Returns
    -------
    prop : dict
        A dictionary withe the properties:

            * Tc: critical temperature, [K]
            * Pc: critical pressure, [MPa]
            * rhoc: critical density, [kg/m³]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 0 ≤ x ≤ 0.12

    Examples
    --------
    >>> _critNaCl(0.1)
    975.571016

    References
    ----------
    IAPWS, Revised Guideline on the Critical Locus of Aqueous Solutions of
    Sodium Chloride, http://www.iapws.org/relguide/critnacl.html
    """
    # Check input parameters
    if x < 0 or x > 0.12:
        raise NotImplementedError("Incoming out of bound")

    T1 = Tc*(1 + 2.3e1*x - 3.3e2*x**1.5 - 1.8e3*x**2)
    T2 = Tc*(1 + 1.757e1*x - 3.026e2*x**1.5 + 2.838e3*x**2 - 1.349e4*x**2.5
             + 3.278e4*x**3 - 3.674e4*x**3.5 + 1.437e4*x**4)
    f1 = (abs(10000*x-10-1)-abs(10000*x-10+1))/4+0.5
    f2 = (abs(10000*x-10+1)-abs(10000*x-10-1))/4+0.5

    # Eq 1
    tc = f1*T1+f2*T2

    # Eq 7
    rc = rhoc*(1 + 1.7607e2*x - 2.9693e3*x**1.5 + 2.4886e4*x**2
               - 1.1377e5*x**2.5 + 2.8847e5*x**3 - 3.8195e5*x**3.5
               + 2.0633e5*x**4)

    # Eq 8
    DT = tc-Tc
    pc = Pc*(1+9.1443e-3*DT+5.1636e-5*DT**2-2.5360e-7*DT**3+3.6494e-10*DT**4)

    prop = {}
    prop["Tc"] = tc
    prop["rhoc"] = rc
    prop["Pc"] = pc
    return prop
