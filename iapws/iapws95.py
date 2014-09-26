#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################################################
# Implemented multiparameter equation of state
#   o   IAPWS-95  implementation
#   o   Heavy water formulation 2005
#############################################################################

import os

from scipy import exp, log, sinh, cosh, tanh, __version__
if int(__version__.split(".")[1]) < 10:
    from scipy.constants import Bolzmann as Boltzmann
else:
    from scipy.constants import Boltzmann
from scipy.optimize import fsolve

from iapws import _fase


Tref = 298.15
Pref = 101325.
so = 0
ho = 0


class MEoS(object):
    """
    General implementation of multiparameter equation of state
    From this derived all child class specified per individual compounds

    Incoming properties:
        T   -   Temperature, K
        P   -   Pressure, MPa
        rho -   Density, kg/m3
        v   -   Specific volume, m3/kg
        h   -   Specific enthalpy, kJ/kg
        s   -   Specific entropy, kJ/kg·K
        u   -   Specific internal energy, kJ/kg·K
        x   -   Quality
    It needs two incoming properties

    Calculated properties:
        P         -   Pressure, MPa
        T         -   Temperature, K
        g         -   Specific Gibbs free energy, kJ/kg
        a         -   Specific Helmholtz free energy, kJ/kg
        v         -   Specific volume, m³/kg
        rho       -   Density, kg/m³
        h         -   Specific enthalpy, kJ/kg
        u         -   Specific internal energy, kJ/kg
        s         -   Specific entropy, kJ/kg·K
        x         -   Quatity
        cp        -   Specific isobaric heat capacity, kJ/kg·K
        cv        -   Specific isochoric heat capacity, kJ/kg·K
        Z         -   Compression factor
        f         -   Fugacity, MPa
        fi        -   Fugacity coefficient
        gamma     -   Isoentropic exponent
        alfav     -   Thermal expansion coefficient, 1/K
        kappa     -   Isothermal compressibility, 1/MPa
        alfap     -   Relative pressure coefficient, 1/K
        betap     -   Isothermal stress coefficient, kg/m³
        betas     -   Isoentropic temperature-pressure coefficient
        joule     -   Joule-Thomson coefficient, K/MPa
        dhdP_T    -   Isothermal throttling coefficient, kJ/kg·MPa
        n         -   Isentropic Expansion Coefficient
        kt        -   Isothermal Expansion Coefficient
        ks        -   Adiabatic Compressibility, 1/MPa
        Ks        -   Adiabatic bulk modulus, MPa
        Kt        -   Isothermal bulk modulus, MPa

        virialB   -   Second virial coefficient
        virialC   -   Third virial coefficient

        mu        -   Dynamic viscosity, Pa·s
        nu        -   Kinematic viscosity, m²/s
        k         -   Thermal conductivity, W/m·K
        sigma     -   Surface tension, N/m
        alfa      -   Thermal diffusivity, m²/s
        Pr        -   Prandtl number

    """

    _surface = None
    _vapor_Pressure = None
    _liquid_Density = None
    _vapor_Density = None

    kwargs = {"T": 0.0,
              "P": 0.0,
              "rho": 0.0,
              "v": 0.0,
              "h": 0.0,
              "s": 0.0,
              "u": 0.0,
              "x": None,
              "v": 0.0,

              "eq": 0,
              "visco": 0,
              "thermal": 0,
              "ref": None,
              "recursion": True}

    def __init__(self, **kwargs):
        """Constructor, define common constant and initinialice kwargs"""
        self._constants = self.eq[self.kwargs["eq"]]
        self.R = self._constants["R"]/self.M
        self.Zc = self.Pc/self.rhoc/self.R/self.Tc
        self.kwargs = MEoS.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        """Make instance callable to can add input parameter one to one"""
        self.kwargs.update(kwargs)

        if self.calculable:
            self.status = 1
            self.calculo()
            self.msg = ""

    @property
    def calculable(self):
        """Check if inputs are enough to define state"""
        self._mode = ""
        if self.kwargs["T"] and self.kwargs["P"]:
            self._mode = "TP"
        elif self.kwargs["T"] and self.kwargs["rho"]:
            self._mode = "Trho"
        elif self.kwargs["P"] and self.kwargs["h"] is not None:
            self._mode = "Ph"
        elif self.kwargs["P"] and self.kwargs["s"] is not None:
            self._mode = "Ps"
        elif self.kwargs["P"] and self.kwargs["v"]:
            self._mode = "Pv"
        elif self.kwargs["T"] and self.kwargs["s"] is not None:
            self._mode = "Ts"
        elif self.kwargs["T"] and self.kwargs["x"] is not None:
            self._mode = "Tx"
        return bool(self._mode)

    def calculo(self):
        """Calculate procedure"""
        T = self.kwargs["T"]
        rho = self.kwargs["rho"]
        P = self.kwargs["P"]
        v = self.kwargs["v"]
        s = self.kwargs["s"]
        h = self.kwargs["h"]
        u = self.kwargs["u"]
        x = self.kwargs["x"]
        recursion = self.kwargs["recursion"]
        eq = self.kwargs["eq"]
        visco = self.kwargs["visco"]
        thermal = self.kwargs["thermal"]
        ref = self.kwargs["ref"]

        self._eq = self._Helmholtz
        self._constants = self.eq[eq]
        self.R = self._constants["R"]/self.M

        propiedades = None
        if v and not rho:
            rho = 1./v

        if T and x is not None:
            if self.Tt > T or self.Tc < T:
                raise ValueError("Wrong input values")

            self.T = T
            self.x = x
            self.Tr = self.T/self.Tc

            rhol, rhov, Ps = self._saturation()
            self.P = Ps

            self.Liquido = self.__class__(T=T, rho=rhol, recursion=False)
            self.Gas = self.__class__(T=T, rho=rhov, recursion=False)

            self.v = self.Liquido.v*(1-x)+self.Gas.v*x
            self.rho = 1./self.v
            self.s = self.Liquido.s*(1-x)+self.Gas.s*x
            self.h = self.Liquido.h*(1-x)+self.Gas.h*x
            self.u = self.Liquido.u*(1-x)+self.Gas.u*x
            self.a = self.Liquido.a*(1-x)+self.Gas.a*x
            self.g = self.Liquido.g*(1-x)+self.Gas.g*x
            self.Z = None
            self.cp = None
            self.cv = None
            self.w = None
            self.alfap = None
            self.betap = None
            self.alfav = None
            self.kappa = None
            self.joule = None
            self.dhdP_T = None
            self.betas = None
            self.gamma = None
            self.fi = self.Liquido.fi
            self.f = self.fi*self.P
            self.virialB = self.Liquido.virialB
            self.virialC = self.Liquido.virialC

        else:
            if T and P:
                if T < self.Tc and P < self.Pc and \
                        self._Vapor_Pressure(T) < P:
                    rhoo = self._Liquid_Density(T)
                elif T < self.Tc and P < self.Pc:
                    rhoo = self._Vapor_Density(T)
                else:
                    rhoo = self.rhoc
                rho = fsolve(lambda rho: self._eq(rho, T)["P"]-P, rhoo)
                propiedades = self._eq(rho[0], T)
            elif T and rho:
                propiedades = self._eq(rho, T)
            elif T and h is not None:
                rho = fsolve(lambda rho: self._eq(rho, T)["h"]-h, 200)
                propiedades = self._eq(rho[0], T)
            elif T and s is not None:
                rho = fsolve(lambda rho: self._eq(rho, T)["s"]-s, 200)
                propiedades = self._eq(rho[0], T)
            elif T and u is not None:
                def funcion(rho):
                    par = self._eq(rho, T)
                    return par["h"]-par["P"]/1000*par["v"]-u
                rho = fsolve(funcion, 200)
                propiedades = self._eq(rho[0], T)

            elif P and rho:
                T = fsolve(lambda T: self._eq(rho, T)["P"]-P, 600)
                propiedades = self._eq(rho, T[0])
            elif P and h is not None:
                rho, T = fsolve(lambda par: (self._eq(par[0], par[1])["P"]-P, self._eq(par[0], par[1])["h"]-h), [200, 600])
                propiedades = self._eq(rho[0], T[0])
            elif P and s is not None:
                rho, T = fsolve(lambda par: (self._eq(par[0], par[1])["P"]-P, self._eq(par[0], par[1])["s"]-s), [200, 600])
                propiedades = self._eq(rho[0], T[0])
            elif P and u is not None:
                def funcion(parr):
                    par = self._eq(parr[0], parr[1])
                    return par["h"]-par["P"]/1000*par["v"]-u, par["P"]-P
                rho, T = fsolve(funcion, [200, 600])
                propiedades = self._eq(rho[0], T[0])

            elif rho and h is not None:
                T = fsolve(lambda T: self._eq(rho, T)["h"]-h, 600)
                propiedades = self._eq(rho, T[0])
            elif rho and s is not None:
                T = fsolve(lambda T: self._eq(rho, T)["s"]-s, 600)
                propiedades = self._eq(rho, T[0])
            elif rho and u is not None:
                def funcion(T):
                    par = self._eq(rho, T)
                    return par["h"]-par["P"]/1000*par["v"]-u
                T = fsolve(funcion, 600)
                propiedades = self._eq(rho, T[0])

            elif h is not None and s is not None:
                rho, T = fsolve(lambda par: (self._eq(par[0], par[1])["h"]-h, self._eq(par[0], par[1])["s"]-s), [200, 600])
                propiedades = self._eq(rho[0], T[0])
            elif h is not None and u is not None:
                def funcion(parr):
                    par = self._eq(parr[0], parr[1])
                    return par["h"]-par["P"]/1000*par["v"]-u, par["h"]-h
                rho, T = fsolve(funcion, [200, 600])
                propiedades = self._eq(rho[0], T[0])

            elif s is not None and u is not None:
                def funcion(parr):
                    par = self._eq(parr[0], parr[1])
                    return par["h"]-par["P"]/1000*par["v"]-u, par["s"]-s
                rho, T = fsolve(funcion, [200, 600])
                propiedades = self._eq(rho[0], T[0])

            if propiedades["P"]<=0:
                raise ValueError ("Wrong input values")

        self.T = T
        self.Tr = T/self.Tc
        self.P = propiedades["P"]
        self.Pr = self.P/self.Pc


        self.Liquido = _fase()
        self.Gas = _fase()
        if self.x < 1:            # liquid phase
            liquido = _Region1(self.T, self.P.MPa)
            self.fill(self.Liquido, liquido)
            self.Liquido.epsilon = unidades.Tension(_Tension(self.T))
        if self.x > 0:            # vapor phase
            vapor = _Region2(self.T, self.P.MPa)
            self.fill(self.Gas, vapor)
        if self.x in (0, 1):        # single phase
            self.fill(self, propiedades)
        else:
            self.h = unidades.Enthalpy(self.x*self.Gas.h+(1-self.x)*self.Liquido.h)
            self.s = unidades.SpecificHeat(self.x*self.Gas.s+(1-self.x)*self.Liquido.s)
            self.u = unidades.SpecificHeat(self.x*self.Gas.u+(1-self.x)*self.Liquido.u)
            self.a = unidades.Enthalpy(self.x*self.Gas.a+(1-self.x)*self.Liquido.a)
            self.g = unidades.Enthalpy(self.x*self.Gas.g+(1-self.x)*self.Liquido.g)

            self.cv = unidades.SpecificHeat(None)
            self.cp = unidades.SpecificHeat(None)
            self.cp_cv = unidades.Dimensionless(None)
            self.w = unidades.Speed(None)


    def getphase(self, fld):
        """Return fluid phase"""
        # check if fld above critical pressure
        if fld["P"] > self.Pc.MPa:
            # check if fld above critical pressure
            if fld["T"] > self.Tc:
                return QApplication.translate("pychemqt", "Supercritical fluid")
            else:
                return QApplication.translate("pychemqt", "Compressible liquid")
        # check if fld above critical pressure
        elif fld["T"] > self.Tc:
            return QApplication.translate("pychemqt", "Gas")
        # check quality
        if fld["x"] >= 1.:
            if self.kwargs["x"] == 1.:
                return QApplication.translate("pychemqt", "Saturated vapor")
            else:
                return QApplication.translate("pychemqt", "Vapor")
        elif 0 < fld["x"] < 1:
            return QApplication.translate("pychemqt", "Two phases")
        elif fld["x"] <= 0.:
            if self.kwargs["x"] == 0.:
                return QApplication.translate("pychemqt", "Saturated liquid")
            else:
                return QApplication.translate("pychemqt", "Liquid")


            self.T = T
            self.Tr = T/self.Tc
            self.s = propiedades["s"]-so
            self.P = propiedades["P"]
            self.Pr = self.P/self.Pc
            self.v = propiedades["v"]
            self.rho = 1/self.v
            self.h = propiedades["h"]-ho
            self.u = self.h-self.P*self.v
            self.a = self.u-self.T*self.s
            self.g = self.h-self.T*self.s
            self.Z = self.P*self.v/self.T/self.R/1000
            self.cp = propiedades["cp"]
            self.cv = propiedades["cv"]
            self.w = propiedades["w"]
            self.alfap = propiedades["alfap"]
            self.betap = propiedades["betap"]

            self.cp0 = self._Cp0(self._constants["cp"])
            self.gamma = -self.v/self.P*self.derivative("P", "v", "s")
            self.fi = propiedades["fugacity"]
            self.f = self.fi*self.P
            self.virialB = propiedades["B"]/self.rhoc
            self.virialC = propiedades["C"]/self.rhoc**2

            self.cp_cv = self.cp/self.cv
            self.cp0_cv = self.cp0/self.cv
            self.alfav = self.derivative("v", "T", "P")/self.v
            self.kappa = -self.derivative("v", "P", "T")/self.v
            self.joule = self.derivative("T", "P", "h")
            self.betas = self.derivative("T", "P", "s")
            self.n = -self.v/self.P*self.derivative("P", "v", "s")
            self.kt = -self.v/self.P*self.derivative("P", "v", "T")
            self.ks = -self.derivative("v", "P", "s")/self.v
            self.Kt = -self.v*self.derivative("P", "v", "s")
            self.Ks = -self.v*self.derivative("P", "v", "T")
            self.Gruneisen = self.v/self.cv*self.derivative("P", "T", "v")
            self.dhdp_rho = self.derivative("h", "P", "v")
            self.dhdT_rho = self.derivative("h", "T", "v")
            self.dhdT_P = self.derivative("h", "T", "P")
            self.dhdP_T = self.derivative("h", "P", "T")
#            self.dpdT=propiedades["dpdT"]
            self.dpdT = self.derivative("P", "T", "rho")*1e-3

            self.invT = -1/self.T
            self.Z_rho = (self.Z-1)/self.rho
            self.InternalPressure = self.T*self.derivative("P", "T", "rho")*1e-3-self.P

#            print propiedades["B"]/self.rho/self.M, propiedades["C"]/self.rho**2/self.M**2
#            print -self.rho*self.cp*self.derivative("P", "rho", "T")/self.derivative("P", "T", "rho"), self.rho*self.derivative("h", "rho", "P")
#            print self.derivative("h", "rho", "T")

#            print self.gamma, self.cp0_cv
#            print self.virialB, self.virialC
#            print self.gamma, self.cp_cv
#            print propiedades["dpdT"], self.derivative("P", "T", "v")

            if recursion:
                if self.Tt < self.T < self.Tc and 0 < self.P < self.Pc:
                    rhol, rhov, Ps = self._saturation()
                    self.Liquido = _fase()
                    self.Gas = _fase()

                    vapor = self._eq(rhov, self.T)
                    liquido = self._eq(rhol, self.T)
                    self.Hvap = vapor["h"]-liquido["h"]

                    if self.rho > rhol:
                        self.x = 0
                    elif self.rho < rhov:
                        self.x = 1
                    else:
                        self.x = (self.s-liquido["s"])/(vapor["s"]-liquido["s"])

                    if self.x < 1:
                        self.Liquido = self.__class__(T=T, rho=rhol, recursion=False)
                    if self.x > 0:
                        self.Gas = self.__class__(T=T, rho=rhov, recursion=False)

                else:
                    self.Gas = self.__class__(T=self.T, rho=self.rho, recursion=False)
                    self.Liquido = _fase()
                    self.x = 1
                    self.Hvap = 0

#            self.mu = unidades.Viscosity(self._Viscosity())
#            self.k = unidades.ThermalConductivity(self._ThCond())
#            self.sigma = unidades.Tension(self._Surface())
#            self.epsilon = self._Dielectric()

#            self.nu = self.mu/self.rho
#            self.Prandt = self.mu*self.cp/self.k
#            self.alfa = self.k/self.rho/self.cp

    def fill(self, fase, estado):
        """Fill phase properties"""
        fase.M = self.M
        fase.v = unidades.SpecificVolume(estado["v"])
        fase.rho = unidades.Density(1/fase.v)
        fase.Z = unidades.Dimensionless(self.P*fase.v/R/1000*self.M/self.T)

        fase.h = unidades.Enthalpy(estado["h"], "kJkg")
        fase.s = unidades.SpecificHeat(estado["s"], "kJkgK")
        fase.u = unidades.Enthalpy(fase.h-self.P*fase.v)
        fase.a = unidades.Enthalpy(fase.u-self.T*fase.s)
        fase.g = unidades.Enthalpy(fase.h-self.T*fase.s)

        fase.cv = unidades.SpecificHeat(estado["cv"], "kJkgK")
        fase.cp = unidades.SpecificHeat(estado["cp"], "kJkgK")
        fase.cp_cv = unidades.Dimensionless(fase.cp/fase.cv)
        fase.w = unidades.Speed(estado["w"])

        fase.mu = unidades.Viscosity(_Viscosity(fase.rho, self.T))
        fase.k = unidades.ThermalConductivity(_ThCond(fase.rho, self.T))
        fase.nu = unidades.Diffusivity(fase.mu/fase.rho)
        fase.dielec = unidades.Dimensionless(_Dielectric(fase.rho, self.T))
        fase.Prandt = unidades.Dimensionless(fase.mu*fase.cp*1000/fase.k)

#        fase.joule=unidades.TemperaturePressure(self.derivative("T", "P", "h"))
        fase.alfav = unidades.InvTemperature(estado["alfav"])
        fase.xkappa = unidades.InvPressure(estado["kt"], "MPa")

#        self.alfa=self.k/1000/self.rho/self.cp
#        self.n=_Refractive(self.rho, self.T)
#        self.joule=self.derivative("T", "P", "h")
#        self.deltat=self.derivative("h", "P", "T")
#        self.gamma=-self.v/self.P/1000*self.derivative("P", "v", "s")

#        if self.region==3:
#            self.alfap=estado["alfap"]
#            self.betap=estado["betap"]
#        else:
#            self.alfap=fase.alfav/self.P/fase.kt
#            self.betap=-1/self.P/1000*self.derivative("P", "v", "T")

        cp0 = prop0(self.T, self.P.MPa)
        fase.v0 = unidades.SpecificVolume(cp0.v)
        fase.h0 = unidades.Enthalpy(cp0.h)
        fase.u0 = unidades.Enthalpy(fase.h0-self.P*fase.v0)
        fase.s0 = unidades.SpecificHeat(cp0.s)
        fase.a0 = unidades.Enthalpy(fase.u0-self.T*fase.s0)
        fase.g0 = unidades.Enthalpy(fase.h0-self.T*fase.s0)
        fase.cp0 = unidades.SpecificHeat(cp0.cp)
        fase.cv0 = unidades.SpecificHeat(cp0.cv)
        fase.cp0_cv = unidades.Dimensionless(fase.cp0/fase.cv0)
        fase.w0 = cp0.w
        fase.gamma0 = cp0.gamma
        fase.f = unidades.Pressure(self.P*exp((fase.g-fase.g0)/R/self.T))


    def _saturation(self, T=None):
        """Akasaka (2008) "A Reliable and Useful Method to Determine the Saturation State from Helmholtz Energy Equations of State", Journal of Thermal Science and Technology, 3, 442-451
        http://dx.doi.org/10.1299/jtst.3.442"""
        if not T:
            T = self.T

        Ps = self._Vapor_Pressure(T)
        rhoL = self._Liquid_Density(T)
        rhoG = self._Vapor_Density(T, Ps)
        g = 1000.
        erroro = 1e6
        rholo = rhoL
        rhogo = rhoG
        while True:
            deltaL = rhoL/self.rhoc
            deltaG = rhoG/self.rhoc
            liquido = self._eq(rhoL, T)
            vapor = self._eq(rhoG, T)
            Jl = deltaL*(1+deltaL*liquido["fird"])
            Jv = deltaG*(1+deltaG*vapor["fird"])
            Kl = deltaL*liquido["fird"]+liquido["fir"]+log(deltaL)
            Kv = deltaG*vapor["fird"]+vapor["fir"]+log(deltaG)
            Jdl = 1+2*deltaL*liquido["fird"]+deltaL**2*liquido["firdd"]
            Jdv = 1+2*deltaG*vapor["fird"]+deltaG**2*vapor["firdd"]
            Kdl = 2*liquido["fird"]+deltaL*liquido["firdd"]+1/deltaL
            Kdv = 2*vapor["fird"]+deltaG*vapor["firdd"]+1/deltaG
            Delta = Jdv*Kdl-Jdl*Kdv
            error = abs(Kv-Kl)+abs(Jv-Jl)

            if error < 1e-12:
                break
            elif error > erroro:
                rhoL = rholo
                rhoG = rhogo
                g = g/2.
            else:
                erroro = error
                rholo = rhoL
                rhogo = rhoG
                rhoL = rhoL+g/Delta*((Kv-Kl)*Jdv-(Jv-Jl)*Kdv)
                rhoG = rhoG+g/Delta*((Kv-Kl)*Jdl-(Jv-Jl)*Kdl)
        Ps = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(liquido["fir"]-vapor["fir"]+log(deltaL/deltaG))
        return rhoL, rhoG, Ps

    def _Helmholtz(self, rho, T):
        """Implementación general de la ecuación de estado Setzmann-Wagner, ecuación de estado de multiparámetros basada en la energía libre de Helmholtz"""
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        delta = rho/rhoc
        tau = Tc/T

        fio, fiot, fiott, fiod, fiodd, fiodt = self._phi0(self._constants["cp"], tau, delta, Tref, Pref)
        fir, firt, firtt, fird, firdd, firdt, firdtt, B, C=self._phir(tau, delta)

        propiedades = {}
        propiedades["fir"] = fir
        propiedades["fird"] = fird
        propiedades["firdd"] = firdd

        propiedades["T"] = T
        propiedades["P"] = (1+delta*fird)*self.R*T*rho
        propiedades["v"] = 1./rho
        propiedades["h"] = self.R*T*(1+tau*(fiot+firt)+delta*fird)
        propiedades["s"] = self.R*(tau*(fiot+firt)-fio-fir)
        propiedades["cv"] = -self.R*tau**2*(fiott+firtt)
        propiedades["cp"] = self.R*(-tau**2*(fiott+firtt)+(1+delta*fird-delta*tau*firdt)**2/(1+2*delta*fird+delta**2*firdd))
        propiedades["w"] = (self.R*T*(1+2*delta*fird+delta**2*firdd-(1+delta*fird-delta*tau*firdt)**2/tau**2/(fiott+firtt)))**0.5
        propiedades["alfap"] = (1-delta*tau*firdt/(1+delta*fird))/T
        propiedades["betap"] = rho*(1+(delta*fird+delta**2*firdd)/(1+delta*fird))
        propiedades["fugacity"] = exp(fir+delta*fird-log(1+delta*fird))
        propiedades["B"] = B
        propiedades["C"] = C
        propiedades["dpdrho"] = self.R*T*(1+2*delta*fird+delta**2*firdd)
        propiedades["dpdT"] = self.R*rho*(1+delta*fird+delta*tau*firdt)
#        propiedades["cps"] = propiedades["cv"] Add cps from Argon pag.27

        return propiedades


    def _phi0(self, cp, tau, delta, To, Po):
        R = cp.get("R", self._constants["R"])/self.M*1000
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        rho0 = Po/R/To
        tau0 = Tc/To
        delta0 = rho0/rhoc
        co = cp["ao"]-1
        ti = [-x for x in cp["pow"]]
        ci = [-n/(t*(t+1))*Tc**t for n, t in zip(cp["an"], cp["pow"])]
        titao = [fi/Tc for fi in cp["exp"]]
        hyp = [fi/Tc for fi in cp["hyp"]]
        cI = -(1+co)/tau0
        cII = co*(1-log(tau0))-log(delta0)
#        for c, t in zip(ci, ti):
#            cI-=c*t*tau0**(t-1)
#            cII+=c*(t-1)*tau0**t
#        for ao, tita in zip(cp["ao_exp"], titao):
#            cI-=ao*tita*(1/(1-exp(-tita*tau0))-1)
#            cII+=ao*tita*(tau0*(1/(1-exp(-tita*tau0))-1)-log(1-exp(-tita*tau0)))
#        if cp["ao_hyp"]:
#            for i in [0, 2]:
#                cI-=cp["ao_hyp"][i]*hyp[i]/(tanh(hyp[i]*tau0))
#                cII+=cp["ao_hyp"][i]*(hyp[i]*tau0/tanh(hyp[i]*tau0)-log(abs(sinh(hyp[i]*tau0))))
#            for i in [1, 3]:
#                cI+=cp["ao_hyp"][i]*hyp[i]*tanh(hyp[i]*tau0)
#                cII-=cp["ao_hyp"][i]*(hyp[i]*tau0*tanh(hyp[i]*tau0)-log(abs(cosh(hyp[i]*tau0))))

        Fi0 = {"ao_log": [1,  co],
               "pow": [0, 1] + ti,
               "ao_pow": [cII, cI] + ci,
               "ao_exp": cp["ao_exp"],
               "titao": titao,
               "ao_hyp": cp["ao_hyp"],
               "hyp": hyp}

        # FIXME: Reference estate
        T = self._constants.get("Tref", self.Tc)/tau
        rho = delta*self.rhoc

        fio=Fi0["ao_log"][0]*log(delta)+Fi0["ao_log"][1]*log(tau)
        fiot=+Fi0["ao_log"][1]/tau
        fiott=-Fi0["ao_log"][1]/tau**2

        fiod = 1/delta
        fiodd = -1/delta**2
        fiodt = 0

        for n, t in zip(Fi0["ao_pow"], Fi0["pow"]):
            fio += n*tau**t
            if t != 0:
                fiot += t*n*tau**(t-1)
            if t not in [0, 1]:
                fiott += n*t*(t-1)*tau**(t-2)

        for n, t in zip(Fi0["ao_exp"], Fi0["titao"]):
            fio += n*log(1-exp(-tau*t))
            fiot += n*t*((1-exp(-t*tau))**-1-1)
            fiott -= n*t**2*exp(-t*tau)*(1-exp(-t*tau))**-2

        if Fi0["ao_hyp"]:
            for i in [0, 2]:
                fio += Fi0["ao_hyp"][i]*log(abs(sinh(Fi0["hyp"][i]*tau)))
                fiot += Fi0["ao_hyp"][i]*Fi0["hyp"][i]/tanh(Fi0["hyp"][i]*tau)
                fiott -= Fi0["ao_hyp"][i]*Fi0["hyp"][i]**2/sinh(Fi0["hyp"][i]*tau)**2

            for i in [1, 3]:
                fio -= Fi0["ao_hyp"][i]*log(abs(cosh(Fi0["hyp"][i]*tau)))
                fiot -= Fi0["ao_hyp"][i]*Fi0["hyp"][i]*tanh(Fi0["hyp"][i]*tau)
                fiott -= Fi0["ao_hyp"][i]*Fi0["hyp"][i]**2/cosh(Fi0["hyp"][i]*tau)**2

        R_ = cp.get("R", self._constants["R"])
        factor = R_/self._constants["R"]
        return factor*fio, factor*fiot, factor*fiott, factor*fiod, factor*fiodd, factor*fiodt

    def _phir(self, tau, delta):
        delta_0 = 1e-50

        fir = fird = firdd = firt = firtt = firdt = firdtt = B = C = 0

        # Polinomial terms
        nr1 = self._constants.get("nr1", [])
        d1 = self._constants.get("d1", [])
        t1 = self._constants.get("t1", [])
        for i in range(len(nr1)):
            fir += nr1[i]*delta**d1[i]*tau**t1[i]
            fird += nr1[i]*d1[i]*delta**(d1[i]-1)*tau**t1[i]
            firdd += nr1[i]*d1[i]*(d1[i]-1)*delta**(d1[i]-2)*tau**t1[i]
            firt += nr1[i]*t1[i]*delta**d1[i]*tau**(t1[i]-1)
            firtt += nr1[i]*t1[i]*(t1[i]-1)*delta**d1[i]*tau**(t1[i]-2)
            firdt += nr1[i]*t1[i]*d1[i]*delta**(d1[i]-1)*tau**(t1[i]-1)
            firdtt += nr1[i]*t1[i]*d1[i]*(t1[i]-1)*delta**(d1[i]-1)*tau**(t1[i]-2)
            B += nr1[i]*d1[i]*delta_0**(d1[i]-1)*tau**t1[i]
            C += nr1[i]*d1[i]*(d1[i]-1)*delta_0**(d1[i]-2)*tau**t1[i]

        # Exponential terms
        nr2 = self._constants.get("nr2", [])
        d2 = self._constants.get("d2", [])
        g2 = self._constants.get("gamma2", [])
        t2 = self._constants.get("t2", [])
        c2 = self._constants.get("c2", [])
        for i in range(len(nr2)):
            fir += nr2[i]*delta**d2[i]*tau**t2[i]*exp(-g2[i]*delta**c2[i])
            fird += nr2[i]*exp(-g2[i]*delta**c2[i])*delta**(d2[i]-1)*tau**t2[i]*(d2[i]-g2[i]*c2[i]*delta**c2[i])
            firdd += nr2[i]*exp(-g2[i]*delta**c2[i])*delta**(d2[i]-2)*tau**t2[i]*((d2[i]-g2[i]*c2[i]*delta**c2[i])*(d2[i]-1-g2[i]*c2[i]*delta**c2[i])-g2[i]**2*c2[i]**2*delta**c2[i])
            firt += nr2[i]*t2[i]*delta**d2[i]*tau**(t2[i]-1)*exp(-g2[i]*delta**c2[i])
            firtt += nr2[i]*t2[i]*(t2[i]-1)*delta**d2[i]*tau**(t2[i]-2)*exp(-g2[i]*delta**c2[i])
            firdt += nr2[i]*t2[i]*delta**(d2[i]-1)*tau**(t2[i]-1)*(d2[i]-g2[i]*c2[i]*delta**c2[i])*exp(-g2[i]*delta**c2[i])
            firdtt += nr2[i]*t2[i]*(t2[i]-1)*delta**(d2[i]-1)*tau**(t2[i]-2)*(d2[i]-g2[i]*c2[i]*delta**c2[i])*exp(-g2[i]*delta**c2[i])
            B += nr2[i]*exp(-g2[i]*delta_0**c2[i])*delta_0**(d2[i]-1)*tau**t2[i]*(d2[i]-g2[i]*c2[i]*delta_0**c2[i])
            C += nr2[i]*exp(-g2[i]*delta_0**c2[i])*(delta_0**(d2[i]-2)*tau**t2[i]*((d2[i]-g2[i]*c2[i]*delta_0**c2[i])*(d2[i]-1-g2[i]*c2[i]*delta_0**c2[i])-g2[i]**2*c2[i]**2*delta_0**c2[i]))

        # Gaussian terms
        nr3 = self._constants.get("nr3", [])
        d3 = self._constants.get("d3", [])
        t3 = self._constants.get("t3", [])
        a3 = self._constants.get("alfa3", [])
        e3 = self._constants.get("epsilon3", [])
        b3 = self._constants.get("beta3", [])
        g3 = self._constants.get("gamma3", [])
        for i in range(len(nr3)):
            exp1 = self._constants.get("exp1", [2]*len(nr3))
            exp2 = self._constants.get("exp2", [2]*len(nr3))
            fir += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])
            fird += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(d3[i]/delta-2*a3[i]*(delta-e3[i]))
            firdd += nr3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(-2*a3[i]*delta**d3[i]+4*a3[i]**2*delta**d3[i]*(delta-e3[i])**exp1[i]-4*d3[i]*a3[i]*delta**2*(delta-e3[i])+d3[i]*2*delta)
            firt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(t3[i]/tau-2*b3[i]*(tau-g3[i]))
            firtt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*((t3[i]/tau-2*b3[i]*(tau-g3[i]))**exp2[i]-t3[i]/tau**2-2*b3[i])
            firdt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(t3[i]/tau-2*b3[i]*(tau-g3[i]))*(d3[i]/delta-2*a3[i]*(delta-e3[i]))
            firdtt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*((t3[i]/tau-2*b3[i]*(tau-g3[i]))**exp2[i]-t3[i]/tau**2-2*b3[i])*(d3[i]/delta-2*a3[i]*(delta-e3[i]))
            B += nr3[i]*delta_0**d3[i]*tau**t3[i]*exp(-a3[i]*(delta_0-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(d3[i]/delta_0-2*a3[i]*(delta_0-e3[i]))
            C += nr3[i]*tau**t3[i]*exp(-a3[i]*(delta_0-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(-2*a3[i]*delta_0**d3[i]+4*a3[i]**2*delta_0**d3[i]*(delta_0-e3[i])**exp1[i]-4*d3[i]*a3[i]*delta_0**2*(delta_0-e3[i])+d3[i]*2*delta_0)

        # Non analitic terms
        nr4 = self._constants.get("nr4", [])
        a4 = self._constants.get("a4", [])
        b4 = self._constants.get("beta4", [])
        A = self._constants.get("A", [])
        B = self._constants.get("B", [])
        C = self._constants.get("C", [])
        D = self._constants.get("D", [])
        b = self._constants.get("b", [])
        for i in range(len(nr4)):
            Tita = (1-tau)+A[i]*((delta-1)**2)**(1/2/b4[i])
            F = exp(-C[i]*(delta-1)**2-D[i]*(tau-1)**2)
            Fd = -2*C[i]*F*(delta-1)
            Fdd = 2*C[i]*F*(2*C[i]*(delta-1)**2-1)
            Ft = -2*D[i]*F*(tau-1)
            Ftt = 2*D[i]*F*(2*D[i]*(tau-1)**2-1)
            Fdt = 4*C[i]*D[i]*F*(delta-1)*(tau-1)
            Fdtt = 4*C[i]*D[i]*F*(delta-1)*(2*D[i]*(tau-1)**2-1)

            Delta = Tita**2+B[i]*((delta-1)**2)**a4[i]
            Deltad = (delta-1)*(A[i]*Tita*2/b4[i]*((delta-1)**2)**(1/2/b4[i]-1)+2*B[i]*a4[i]*((delta-1)**2)**(a4[i]-1))
            Deltadd = Deltad/(delta-1)+(delta-1)**2*(4*B[i]*a4[i]*(a4[i]-1)*((delta-1)**2)**(a4[i]-2)+2*A[i]**2/b4[i]**2*(((delta-1)**2)**(1/2/b4[i]-1))**2+A[i]*Tita*4/b4[i]*(1/2/b4[i]-1)*((delta-1)**2)**(1/2/b4[i]-2))

            DeltaBd = b[i]*Delta**(b[i]-1)*Deltad
            DeltaBdd = b[i]*(Delta**(b[i]-1)*Deltadd+(b[i]-1)*Delta**(b[i]-2)*Deltad**2)
            DeltaBt = -2*Tita*b[i]*Delta**(b[i]-1)
            DeltaBtt = 2*b[i]*Delta**(b[i]-1)+4*Tita**2*b[i]*(b[i]-1)*Delta**(b[i]-2)
            DeltaBdt = -A[i]*b[i]*2/b4[i]*Delta**(b[i]-1)*(delta-1)*((delta-1)**2)**(1/2/b4[i]-1)-2*Tita*b[i]*(b[i]-1)*Delta**(b[i]-2)*Deltad
            DeltaBdtt = 2*b[i]*(b[i]-1)*Delta**(b[i]-2)*(Deltad*(1+2*Tita**2*(b[i]-2)/Delta)+4*Tita*A[i]*(delta-1)/b4[i]*((delta-1)**2)**(1/2/b4[i]-1))

            fir += nr4[i]*Delta**b[i]*delta*F
            fird += nr4[i]*(Delta**b[i]*(F+delta*Fd)+DeltaBd*delta*F)
            firdd += nr4[i]*(Delta**b[i]*(2*Fd+delta*Fdd)+2*DeltaBd*(F+delta*Fd)+DeltaBdd*delta*F)
            firt += nr4[i]*delta*(DeltaBt*F+Delta**b[i]*delta*Ft)
            firtt += nr4[i]*delta*(DeltaBtt*F+2*DeltaBt*Ft+Delta**b[i]*Ftt)
            firdt += nr4[i]*(Delta**b[i]*(Ft+delta*Fdt)+delta*DeltaBd*Ft+DeltaBt*(F+delta*Fd)+DeltaBdt*delta*F)
            firdtt += nr4[i]*((DeltaBtt*F+2*DeltaBt*Ft+Delta**b[i]*Ftt)+delta*(DeltaBdtt*F+DeltaBtt*Fd+2*DeltaBdt*Ft+2*DeltaBt*Fdt+DeltaBt*Ftt+Delta**b[i]*Fdtt))

            Tita_virial = (1-tau)+A[i]*((delta_0-1)**2)**(1/2/b4[i])
            Delta_Virial = Tita_virial**2+B[i]*((delta_0-1)**2)**a4[i]
            Deltad_Virial = (delta_0-1)*(A[i]*Tita_virial*2/b4[i]*((delta_0-1)**2)**(1/2/b4[i]-1)+2*B[i]*a4[i]*((delta_0-1)**2)**(a4[i]-1))
            Deltadd_Virial = Deltad_Virial/(delta_0-1)+(delta_0-1)**2*(4*B[i]*a4[i]*(a4[i]-1)*((delta_0-1)**2)**(a4[i]-2)+2*A[i]**2/b4[i]**2*(((delta_0-1)**2)**(1/2/b4[i]-1))**2+A[i]*Tita_virial*4/b4[i]*(1/2/b4[i]-1)*((delta_0-1)**2)**(1/2/b4[i]-2))
            DeltaBd_Virial = b[i]*Delta_Virial**(b[i]-1)*Deltad_Virial
            DeltaBdd_Virial = b[i]*(Delta_Virial**(b[i]-1)*Deltadd_Virial+(b[i]-1)*Delta_Virial**(b[i]-2)*Deltad_Virial**2)
            F_virial = exp(-C[i]*(delta_0-1)**2-D[i]*(tau-1)**2)
            Fd_virial = -2*C[i]*F_virial*(delta_0-1)
            Fdd_virial = 2*C[i]*F_virial*(2*C[i]*(delta_0-1)**2-1)

            B += nr4[i]*(Delta_Virial**b[i]*(F_virial+delta_0*Fd_virial)+DeltaBd_Virial*delta_0*F_virial)
            C += nr4[i]*(Delta_Virial**b[i]*(2*Fd_virial+delta_0*Fdd_virial)+2*DeltaBd_Virial*(F_virial+delta_0*Fd_virial)+DeltaBdd_Virial*delta_0*F_virial)

        # Hard sphere term
        if self._constants.get("Fi", None):
            f = self._constants["Fi"]
            n = 0.1617
            a = 0.689
            gamma = 0.3674
            X = n*delta/(a+(1-a)/tau**gamma)
            Xd = n/(a+(1-a)/tau**gamma)
            Xt = n*delta*(1-a)*gamma/tau**(gamma+1)/(a+(1-a)/tau**gamma)**2
            Xdt = n*(1-a)*gamma/tau**(gamma+1)/(a+(1-a)/tau**gamma)**2
            Xtt = -n*delta*((1-a)*gamma/tau**(gamma+2)*((gamma+1)*(a+(1-a)/tau**gamma)-2*gamma*(1-a)/tau**gamma))/(a+(1-a)/tau**gamma)**3
            Xdtt = -n*((1-a)*gamma/tau**(gamma+2)*((gamma+1)*(a+(1-a)/tau**gamma)-2*gamma*(1-a)/tau**gamma))/(a+(1-a)/tau**gamma)**3

            ahdX = -(f**2-1)/(1-X)+(f**2+3*f+X*(f**2-3*f))/(1-X)**3
            ahdXX = -(f**2-1)/(1-X)**2+(3*(f**2+3*f)+(f**2-3*f)*(1+2*X))/(1-X)**4
            ahdXXX = -2*(f**2-1)/(1-X)**3+6*(2*(f**2+3*f)+(f**2-3*f)*(1+X))/(1-X)**5

            fir += (f**2-1)*log(1-X)+((f**2+3*f)*X-3*f*X**2)/(1-X)**2
            fird += ahdX*Xd
            firdd += ahdXX*Xd**2
            firt += ahdX*Xt
            firtt += ahdXX*Xt**2+ahdX*Xtt
            firdt += ahdXX*Xt*Xd+ahdX*Xdt
            firdtt += ahdXXX*Xt**2*Xd+ahdXX*(Xtt*Xd+2*Xdt*Xt)*ahdX*Xdtt

            X_virial = n*delta_0/(a+(1-a)/tau**gamma)
            ahdX_virial = -(f**2-1)/(1-X_virial)+(f**2+3*f+X_virial*(f**2-3*f))/(1-X_virial)**3
            ahdXX_virial = -(f**2-1)/(1-X_virial)**2+(3*(f**2+3*f)+(f**2-3*f)*(1+2*X_virial))/(1-X_virial)**4
            B += ahdX_virial*Xd
            C += ahdXX_virial*Xd**2
        return fir, firt, firtt, fird, firdd, firdt, firdtt, B, C

    def _Cp0(self, cp, T=False):
        if not T:
            T = self.T
        tau = self.Tc/T
        cpo = cp["ao"]
        for a, t in zip(cp["an"], cp["pow"]):
            cpo += a*T**t
        for m, tita in zip(cp["ao_exp"], cp["exp"]):
            cpo += m*(tita/T)**2*exp(tita/T)/(1-exp(tita/T))**2
        if cp["ao_hyp"]:
            for i in [0, 2]:
                cpo += cp["ao_hyp"][i]*(cp["hyp"][i]/T/(sinh(cp["hyp"][i]/T)))**2
            for i in [1, 3]:
                cpo += cp["ao_hyp"][i]*(cp["hyp"][i]/T/(cosh(cp["hyp"][i]/T)))**2
        return cpo/self.M*1000


    def derivative(self, z, x, y):
        """Calculate generic partial derivative: (δz/δx)y where x, y, z can be: P, T, v, rho, u, h, s, g, a"""
        dT = {"P": self.P*self.alfap,
              "T": 1,
              "v": 0,
              "rho": 0,
              "u": self.cv,
              "h": self.cv+self.P*self.v*self.alfap,
              "s": self.cv/self.T,
              "g": self.P*self.v*self.alfap-self.s,
              "a": -self.s}
        dv = {"P": -self.P*self.betap,
              "T": 0,
              "v": 1,
              "rho": -1,
              "u": self.P*(self.T*self.alfap-1),
              "h": self.P*(self.T*self.alfap-self.v*self.betap),
              "s": self.P*self.alfap,
              "g": -self.P*self.v*self.betap,
              "a": -self.P}
        return (dv[z]*dT[y]-dT[z]*dv[y])/(dv[x]*dT[y]-dT[x]*dv[y])


    def _Vapor_Pressure(self, T):
        eq = self._vapor_Pressure["eq"]
        Tita = 1-T/self.Tc
        if eq in [2, 4, 6]:
            Tita = Tita**0.5
        suma = sum([n*Tita**x for n, x in zip(
            self._vapor_Pressure["ao"], self._vapor_Pressure["exp"])])
        if eq in [1, 2]:
            Pr = suma+1
        elif eq in [3, 4]:
            Pr = exp(suma)
        else:
            Pr = exp(self.Tc/T*suma)
        Pv = Pr*self.Pc
        return Pv

    def _Liquid_Density(self, T=None):
        if not T:
            T = self.T
        eq = self._liquid_Density["eq"]
        Tita = 1-T/self.Tc
        if eq in [2, 4, 6]:
            Tita = Tita**(1./3)
        suma = sum([n*Tita**x for n, x in zip(
            self._liquid_Density["ao"], self._liquid_Density["exp"])])
        if eq in [1, 2]:
            Pr = suma+1
        elif eq in [3, 4]:
            Pr = exp(suma)
        else:
            Pr = exp(self.Tc/T*suma)
        rho = Pr*self.rhoc
        return rho

    def _Vapor_Density(self, T=None, P=None):
        eq = self._vapor_Density["eq"]
        Tita = 1-T/self.Tc
        if eq in [2, 4, 6]:
            Tita = Tita**(1./3)
        suma = sum([n*Tita**x for n, x in zip(
            self._vapor_Density["ao"], self._vapor_Density["exp"])])
        if eq in [1, 2]:
            Pr = suma+1
        elif eq in [3, 4]:
            Pr = exp(suma)
        else:
            Pr = exp(self.Tc/T*suma)
        rho = Pr*self.rhoc
        return rho


    def _Surface(self, T=None):
        """Equation for the surface tension"""
        if self.Tt <= self.T <= self.Tc and self._surface:
            if not T:
                T = self.T
            tau = 1-T/self.Tc
            tension = 0
            for sigma, n in zip(self._surface["sigma"],
                                self._surface["exp"]):
                tension += sigma*tau**n
            sigma = tension
        else:
            sigma = None
        return sigma



    @classmethod
    def __test__(cls):
        """Implement test unit"""
        pruebas = {}
        for i, test in enumerate(cls._test):
            prueba = ">>> for value1 in {}:".format(test["value1"])+os.linesep
            prueba += "...  for value2 in {}:".format(test["value2"])+os.linesep
            prueba += "...   fluido={}({}=value1, {}=value2)".format(
                cls.__name__, test["var1"], test["var2"])+os.linesep
            prueba += "...   print("+"'"+"{: .5g} "*len(test["prop"])+"'"+".format("
            for propiedad, unidad in zip(test["prop"], test["unit"]):
                prueba += "fluido.{}.{}, ".format(propiedad, unidad)
            prueba += "))"+os.linesep
            prueba += test["result"]
        pruebas[str(i)] = prueba
        return pruebas


class IAPWS95(MEoS):
    """Multiparameter equation of state for water (including IAPWS95)

    >>> water=H2O(T=300, rho=996.5560)
    >>> print "%0.10f %0.8f %0.5f %0.9f" % (water.P.MPa, water.cv.kJkgK, water.w, water.s.kJkgK)
    0.0992418352 4.13018112 1501.51914 0.393062643

    >>> water=H2O(T=500, rho=0.435)
    >>> print "%0.10f %0.8f %0.5f %0.9f" % (water.P.MPa, water.cv.kJkgK, water.w, water.s.kJkgK)
    0.0999679423 1.50817541 548.31425 7.944882714

    >>> water=H2O(T=900., P=700)
    >>> print "%0.4f %0.8f %0.5f %0.8f" % (water.rho, water.cv.kJkgK, water.w, water.s.kJkgK)
    870.7690 2.66422350 2019.33608 4.17223802
    """
    name = "water"
    CASNumber = "7732-18-5"
    formula = "H2O"
    synonym = "R-718"
    Tc = 647.096
    rhoc = 322.
    Pc = 22.064
    M = 18.015268  # g/mol
    Tt = 273.16
    Tb = 373.1243
    f_acent = 0.3443
    momentoDipolar = 1.855

    CP1 = {"ao": 4.00632,
           "an": [], "pow": [],
           "ao_exp": [0.012436, 0.97315, 1.27950, 0.96956, 0.24873],
           "exp": [833, 2289, 5009, 5982, 17800],
           "ao_hyp": [], "hyp": []}

    helmholtz = {
        "__type__": "Helmholtz",
        "__name__": u"Helmholtz equation of state for water of Wagner and Pruß (2002).",
        "__doc__":  u"""Wagner, W., Pruß, A. The IAPWS formulation 1995 for the thermodyamic properties of ordinary water substance for general and scientific use. J. Phys. Chem. Ref. Data 31 (2002), 387 – 535.""",
        "R": 8.314371357587,
        "cp": CP1,

        "nr1": [0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
                0.31802509345418, -0.26145533859358, -0.78199751687981e-2,
                0.88089493102134e-2],
        "d1": [1, 1, 1, 2, 2, 3, 4],
        "t1": [-0.5, 0.875, 1, 0.5, 0.75, 0.375, 1],

        "nr2": [-0.66856572307965, 0.20433810950965, -0.66212605039687e-4,
                -0.19232721156002, -0.25709043003438, 0.16074868486251,
                -0.4009282892587e-1, 0.39343422603254e-6, -0.75941377088144e-5,
                0.56250979351888e-3, -0.15608652257135e-4, 0.11537996422951e-8,
                0.36582165144204e-6, -0.13251180074668e-11, -0.62639586912454e-9,
                -0.10793600908932, 0.17611491008752e-1, 0.22132295167546,
                -0.40247669763528, 0.58083399985759, 0.49969146990806e-2,
                -0.31358700712549e-1, -0.74315929710341, 0.47807329915480,
                0.20527940895948e-1, -0.13636435110343, 0.14180634400617e-1,
                0.83326504880713e-2, -0.29052336009585e-1, 0.38615085574206e-1,
                -0.20393486513704e-1, -0.16554050063734e-2, 0.19955571979541e-2,
                0.15870308324157e-3, -0.16388568342530e-4, 0.43613615723811e-1,
                0.34994005463765e-1, -0.76788197844621e-1, 0.22446277332006e-1,
                -0.62689710414685e-4, -0.55711118565645e-9, -0.19905718354408,
                0.31777497330738, -0.11841182425981],
        "c2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 6, 6,
               6, 6],
        "d2": [1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10, 11, 13, 15, 1, 2, 2, 2, 3,
               4, 4, 4, 5, 6, 6, 7, 9, 9, 9, 9, 9, 10, 10, 12, 3, 4, 4, 5, 14,
               3, 6, 6, 6],
        "t2": [4, 6, 12, 1, 5, 4, 2, 13, 9, 3, 4, 11, 4, 13, 1, 7, 1, 9, 10,
               10, 3, 7, 10, 10, 6, 10, 10, 1, 2, 3, 4, 8, 6, 9, 8, 16, 22, 23,
               23, 10, 50, 44, 46, 50],
        "gamma2": [1]*44,

        "nr3": [-0.31306260323435e2, 0.31546140237781e2, -0.25213154341695e4],
        "d3": [3]*3,
        "t3": [0, 1, 4],
        "alfa3": [20]*3,
        "beta3": [150, 150, 250],
        "gamma3": [1.21, 1.21, 1.25],
        "epsilon3": [1.]*3,

        "nr4": [-0.14874640856724, 0.31806110878444],
        "a4": [3.5, 3.5],
        "b": [0.85, 0.95],
        "B": [0.2, 0.2],
        "C": [28, 32],
        "D": [700, 800],
        "A": [0.32, .32],
        "beta4": [0.3, 0.3]}

    eq = helmholtz,

    _surface = {"sigma": [0.2358, -0.147375], "exp": [1.256, 2.256]}
    _vapor_Pressure = {
        "eq": 6,
        "ao": [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719,
               1.80122502],
        "exp": [2, 3, 6, 7, 8, 15]}
    _liquid_Density = {
        "eq": 2,
        "ao": [1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352,
               -6.74694450e5],
        "exp": [1, 2, 5, 16, 43, 110]}
    _vapor_Density = {
        "eq": 4,
        "ao": [-2.0315024, -2.6830294, -5.38626492, -17.2991605, -44.7586581,
               -63.9201063],
        "exp": [1, 2, 4, 9, 18.5, 35.5]}


class D2O(MEoS):
    """Multiparameter equation of state for heavy water

    >>> water=D2O(T=300, rho=996.5560)
    >>> print "%0.10f %0.8f %0.5f %0.9f" % (water.P.MPa, water.cv.kJkgK, water.w, water.s.kJkgK)
    0.0992418352 4.13018112 1501.51914 0.393062643
    """
    name = "heavy water"
    CASNumber = "7789-20-0"
    formula = "D2O"
    synonym = "deuterium oxide"
    Tc = 643.89
    rhoc = 357.992
    Pc = 21.671
    M = 20.0275  # g/mol
    Tt = 276.97
    Tb = 374.563
    f_acent = 0.364
    momentoDipolar = 1.9

    CP1 = {"ao": 0.39176485e1,
           "an": [-0.31123915e-3, 0.41173363e-5, -0.28943955e-8,
                  0.63278791e-12, 0.78728740],
           "pow": [1.00, 2.00, 3.00, 4.00, -0.99],
           "ao_exp": [],
           "exp": [],
           "ao_hyp": [], "hyp": []}

    helmholtz1 = {
        "__type__": "Helmholtz",
        "__name__": u"Helmholtz equation of state for heavy water of Hill et al. (1982).",
        "__doc__":  u"""Hill, P.G., MacMillan, R.D.C., and Lee, V., "A Fundamental Equation of State for Heavy Water," J. Phys. Chem. Ref. Data, 11(1):1-14, 1982.""",
        "R": 8.3143565,
        "cp": CP1,

        "nr1": [-0.384820628204e3, 0.108213047259e4, -0.110768260635e4,
                0.164668954246e4, -0.137959852228e4, 0.598964185629e3,
                -0.100451752702e3, 0.419192736351e3, -0.107279987867e4,
                0.653852283544e3, -0.984305985655e3, 0.845444459339e3,
                -0.376799930490e3, 0.644512590492e2, -0.214911115714e3,
                0.531113962967e3, -0.135454224420e3, 0.202814416558e3,
                -0.178293865031e3, 0.818739394970e2, -0.143312594493e2,
                0.651202383207e2, -0.171227351208e3, 0.100859921516e2,
                -0.144684680657e2, 0.128871134847e2, -0.610605957134e1,
                0.109663804408e1, -0.115734899702e2, 0.374970075409e2,
                0.897967147669, -0.527005883203e1, 0.438084681795e-1,
                0.406772082680, -0.965258571044e-2, -0.119044600379e-1],
        "d1": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
               4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        "t1": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6,
               0, 1, 2, 3, 4, 5, 6, 0, 1, 0, 1, 0, 1, 0, 1],

        "nr2": [0.382589102341e3, -0.106406466204e4, 0.105544952919e4,
                -0.157579942855e4, 0.132703387531e4, -0.579348879870e3,
                0.974163902526e2, 0.286799294226e3, -0.127543020847e4,
                0.275802674911e4, -0.381284331492e4, 0.293755152012e4,
                -0.117858249946e4, 0.186261198012e3],
        "c2": [1]*14,
        "d2": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
        "t2": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6],
        "gamma2": [1.5394]*14}

    eq = helmholtz1,

    _surface = {"sigma": [0.238, -0.152082], "exp": [1.25, 2.25]}
    _vapor_Pressure = {
        "eq": 5,
        "ao": [-0.80236e1, 0.23957e1, -0.42639e2, 0.99569e2, -0.62135e2],
        "exp": [1.0, 1.5, 2.75, 3.0, 3.2]}
    _liquid_Density = {
        "eq": 1,
        "ao": [0.26406e1, 0.97090e1, -0.18058e2, 0.87202e1, -0.74487e1],
        "exp": [0.3678, 1.9, 2.2, 2.63, 7.3]}
    _vapor_Density = {
        "eq": 3,
        "ao": [-0.37651e1, -0.38673e2, 0.73024e2, -0.13251e3, 0.75235e2, -0.70412e2],
        "exp": [0.409, 1.766, 2.24, 3.04, 3.42, 6.9]}




if __name__ == "__main__":
#    import doctest
#    doctest.testmod()


#    water=IAPWS95(T=300., P=0.1)
#    print "%0.1f %0.2f %0.4f %0.9f %0.9f %0.5f %0.4f %0.9f %0.4f %0.4f %0.9f %0.6f" % (water.T, water.P.MPa, water.rho, water.kappa, water.alfav, water.n, water.kt, water.ks, water.Kt.MPa, water.Ks.MPa, water.deltat, water.Gruneisen)
#    print -water.derivative("P", "v", "T")/1e6, water.betap
#        alfap        -   Relative pressure coefficient, 1/K
#        betap       -   Isothermal stress coefficient, kg/m³
#        betas       -   Isoentropic temperature-pressure coefficient

    water=IAPWS95(T=300., x=0.5)
#    print water.x
#    print water.Liquido.Z
#    print water.Gas.Z
#    print water.Z
    print water.Liquido.cp, water.Gas.cp

#    p=unidades.Pressure(1, "atm")
#    water1=H2O(T=400, P=p.MPa)
#    water2=H2O(T=450, P=p.MPa)
#    print "%0.10f %0.8f %0.5f %0.9f" % (water.P.MPa, water.cv.kJkgK, water.w, water.s.kJkgK)
#    0.0992418352 4.13018112 1501.51914 0.393062643
#    print water2.h.MJkg-water1.h.MJkg

#    agua=H2O(T=350, x=0.5)
#    print "%0.1f %0.4f %0.3f %0.3f %0.5f %0.4f %0.2f" % (aire.T, aire.rho, aire.h.kJkg, aire.s.kJkgK, aire.cv.kJkgK, aire.cp.kJkgK, aire.w), aire.P.MPa



#    aire=IAPWS95(T=300, P=1., visco=0)
#    aire2=IAPWS95(T=300, P=1., visco=1)
#    print  aire.T, aire.P.MPa, aire.rho, aire.k.mWmK, aire.mu.muPas
#    print  aire2.T, aire2.P.MPa, aire2.rho, aire2.k.mWmK, aire2.mu.muPas

#    aire=D2O(T=300, P=1., visco=0)
#    print  aire.P.MPa, aire.rho, aire.mu.muPas, aire.k.mWmK
#
#    aire=IAPWS95(T=298.15, P=0.101325)
#    print "%0.2f %0.6f %0.10f %0.3f %0.3f %0.5f %0.4f %0.2f" % (aire.T, aire.P.MPa, aire.rho, aire.h.kJkg, aire.s.kJkgK, aire.cv.kJkgK, aire.cp.kJkgK, aire.w)
#
#    aire=IAPWS95(T=500, P=1)
#    print "%0.2f %0.6f %0.10f %0.3f %0.3f %0.5f %0.4f %0.2f" % (aire.T, aire.P.MPa, aire.rho, aire.h.kJkg, aire.s.kJkgK, aire.cv.kJkgK, aire.cp.kJkgK, aire.w)
#    print aire.T, aire.P.MPa, aire.rho, aire.h.kJkg, aire.s.kJkgK

#    water=IAPWS95(T=300, P=0.101325)
#    print "%0.1f %0.6f %0.5f %0.4f %0.4f" % (water.T, water.P, water.rho, water.h, water.s)
#    print water.cp, water.x

#    agua=H2O(T=298.15, P=0.101325, visco=0, thermal=0)
#    print  agua.P.MPa, agua.rho


