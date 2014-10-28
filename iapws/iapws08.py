#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
# IAPWS for seawater
###############################################################################

from __future__ import division
from math import log

from .iapws95 import IAPWS95


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
    Incoming properties:
        T   -   Temperature, K
        P   -   Pressure, MPa
        S   -   Salinity, kg/kg

    Calculated properties:
        T: Temperature, K
        P: Pressure, MPa
        rho: Density, kg/m³
        v: Specific volume, m³/kg
        h: Specific enthalpy, kJ/kg
        s: Specific entropy, kJ/kg·K
        u: Specific internal energy, kJ/kg
        g: Specific Gibbs free energy, kJ/kg
        a: Specific Helmholtz free energy, kJ/kg
        cp: Specific isobaric heat capacity, kJ/kg·K

        gt: Derivative Gibbs energy with temperature, kJ/kg·K
        gp: Derivative Gibbs energy with pressure, m³/kg
        gtt: Derivative Gibbs energy with temperature square, kJ/kg·K²
        gtp: Derivative Gibbs energy with pressure and temperature, m³/kg·K
        gpp: Derivative Gibbs energy with temperature square, m³/kg·MPa
        gs: Derivative Gibbs energy with salinity, kJ/kg
        gsp: Derivative Gibbs energy with salinity and pressure, m³/kg

        alfa: Thermal expansion coefficient, 1/K
        betas: Isentropic temperature-pressure coefficient, K/MPa
        kt: Isothermal compressibility, 1/MPa
        ks: Isentropic compressibility, 1/MPa
        w: Sound Speed, m/s

        mu: Relative chemical potential, kJ/kg
        muw: Chemical potential of H2O, kJ/kg
        mus: Chemical potential of sea salt, kJ/kg
        osm: Osmotic coefficient, [-]
        haline: Haline contraction coefficient, kg/kg
    """
    kwargs = {"T": 0.0,
              "P": 0.0,
              "S": None}
    status = 0
    msg = "Undefined"

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

        m = S/(1-S)/Ms
        pw = self._water(T, P)
        ps = self._saline(T, P, S)

        prop = {}
        for key in pw:
            prop[key] = pw[key]+ps[key]
            self.__setattr__(key, prop[key])

        self.T = T
        self.P = P
        self.rho = 1./prop["gp"]
        self.v = prop["gp"]
        self.s = -prop["gt"]
        self.cp = -T*prop["gtt"]
        self.h = prop["g"]-T*prop["gt"]
        self.u = prop["g"]-T*prop["gt"]-P*1000*prop["gp"]
        self.a = prop["g"]-P*1000*prop["gp"]
        self.alfa = prop["gtp"]/prop["gp"]
        self.betas = -prop["gtp"]/prop["gtt"]
        self.kt = -prop["gpp"]/prop["gp"]
        self.ks = (prop["gtp"]**2-prop["gt"]*prop["gpp"])/prop["gp"]/prop["gtt"]
        self.w = prop["gp"]*(prop["gtt"]*1000/(prop["gtp"]**2 -
                             prop["gtt"]*1000*prop["gpp"]*1e-6))**0.5

        self.mu = prop["gs"]
        self.muw = prop["g"]-S*prop["gs"]
        self.mus = prop["g"]+(1-S)*prop["gs"]
        self.osm = -(ps["g"]-S*prop["gs"])/m/Rm/T
        self.haline = -prop["gsp"]/prop["gp"]

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
        return prop

    @classmethod
    def _saline(cls, T, P, S):
        """Eq 4"""
        S_ = 0.03516504*40/35
        X = (S/S_)**0.5
        tau = (T-273.15)/40
        pi = (P-0.101325)/100

        I = [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 2, 3, 4, 2, 3, 4, 2, 3, 4,
             2, 4, 2, 2, 3, 4, 5, 2, 3, 4, 2, 3, 2, 3, 2, 3, 2, 3, 4, 2, 3, 2,
             3, 2, 2, 2, 3, 4, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2]
        J = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
             5, 5, 6, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 1, 1, 2,
             2, 3, 4, 0, 0, 0, 1, 1, 2, 2, 3, 4, 0, 0, 1, 2, 3, 0, 1, 2]
        K = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
        for i, j, k, gi in zip(I, J, K, G):
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
