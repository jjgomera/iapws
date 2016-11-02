#!/usr/bin/python
# -*- coding: utf-8 -*-


###############################################################################
# Implemented multiparameter equation of state
#   o   IAPWS-95  implementation
#   o   Heavy water formulation 2005
###############################################################################


from scipy import exp, log
from scipy.optimize import fsolve

from ._iapws import _fase, getphase
from ._iapws import _Viscosity, _ThCond, _Dielectric, _Refractive, _Tension
from .iapws97 import _TSat_P


class MEoS(_fase):
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
        l   -   Opcional parameter to light wavelength for Refractive index

    It needs two incoming properties

    Calculated properties:
        P         -   Pressure, MPa
        Pr        -   Reduce pressure
        T         -   Temperature, K
        Tr        -   Reduced temperature
        x         -   Quality
        v         -   Specific volume, m³/kg
        rho       -   Density, kg/m³
        h         -   Specific enthalpy, kJ/kg
        s         -   Specific entropy, kJ/kg·K
        u         -   Specific internal energy, kJ/kg
        g         -   Specific Gibbs free energy, kJ/kg
        a         -   Specific Helmholtz free energy, kJ/kg
        cp        -   Specific isobaric heat capacity, kJ/kg·K
        cv        -   Specific isochoric heat capacity, kJ/kg·K
        cp_cv     -   Heat capacity ratio
        w         -   Speed of sound, m/s
        Z         -   Compression factor
        fi        -   Fugacity coefficient
        f         -   Fugacity, MPa
        gamma     -   Isoentropic exponent
        Hvap      -   Vaporization heat, kJ/kg
        alfav     -   Thermal expansion coefficient (Volume expansivity), 1/K
        kappa     -   Isothermal compressibility, 1/MPa
        alfap     -   Relative pressure coefficient, 1/K
        betap     -   Isothermal stress coefficient, kg/m³
        betas     -   Isoentropic temperature-pressure coefficient
        joule     -   Joule-Thomson coefficient, K/MPa
        Gruneisen -   Gruneisen parameter
        virialB   -   Second virial coefficient, m³/kg
        virialC   -   Third virial coefficient, m⁶/kg²
        dpdT_rho  -   Derivatives, dp/dT at constant rho, MPa/K
        dpdrho_T  -   Derivatives, dp/drho at constant T, MPa·m³/kg
        drhodT_P  -   Derivatives, drho/dT at constant P, kg/m³·K
        drhodP_T  -   Derivatives, drho/dP at constant T, kg/m³·MPa
        dhdT_rho  -   Derivatives, dh/dT at constant rho, kJ/kg·K
        dhdP_T    -   Isothermal throttling coefficient, kJ/kg·MPa
        dhdT_P    -   Derivatives, dh/dT at constant P, kJ/kg·K
        dhdrho_T  -   Derivatives, dh/drho at constant T, kJ·m³/kg²
        dhdrho_P  -   Derivatives, dh/drho at constant P, kJ·m³/kg²
        dhdP_rho  -   Derivatives, dh/dP at constant rho, kJ/kg·MPa
        kt        -   Isothermal Expansion Coefficient
        ks        -   Adiabatic Compressibility, 1/MPa
        Ks        -   Adiabatic bulk modulus, MPa
        Kt        -   Isothermal bulk modulus, MPa

        Z_rho     -   (Z-1) over the density, m³/kg
        IntP      -   Internal pressure
        invT      -   Negative reciprocal temperature
        hInput    -   Specific heat input, kJ/kg
        mu        -   Dynamic viscosity, Pa·s
        nu        -   Kinematic viscosity, m²/s
        k         -   Thermal conductivity, W/m·K
        sigma     -   Surface tension, N/m
        alfa      -   Thermal diffusivity, m²/s
        Pramdt    -   Prandtl number
        epsilon   -   Dielectric constant
        n         -   Refractive index

        v0        -   Ideal gas Specific volume, m³/kg
        rho0      -   Ideal gas Density, kg/m³
        h0        -   Ideal gas Specific enthalpy, kJ/kg
        u0        -   Ideal gas Specific internal energy, kJ/kg
        s0        -   Ideal gas Specific entropy, kJ/kg·K
        a0        -   Ideal gas Specific Helmholtz free energy, kJ/kg
        g0        -   Ideal gas Specific Gibbs free energy, kJ/kg
        cp0       -   Ideal gas Specific isobaric heat capacity, kJ/kg·K
        cv0       -   Ideal gas Specific isochoric heat capacity, kJ/kg·K
        cp0_cv    -   Ideal gas Heat capacity ratio
        gamma0    -   Ideal gas Isoentropic exponent

    """

    CP = None
    _surface = None
    _vapor_Pressure = None
    _liquid_Density = None
    _vapor_Density = None

    kwargs = {"T": 0.0,
              "P": 0.0,
              "rho": 0.0,
              "v": 0.0,
              "h": None,
              "s": None,
              "u": None,
              "x": None,
              "l": 0.5893}
    status = 0
    msg = "Undefined"

    def __init__(self, **kwargs):
        """Constructor, define common constant and initinialice kwargs"""
        self.R = self._constants["R"]/self.M
        self.Zc = self.Pc/self.rhoc/self.R/self.Tc
        self.kwargs = MEoS.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        """Make instance callable to can add input parameter one to one"""
        if kwargs.get("v", 0):
            kwargs["rho"] = 1./kwargs["v"]
            del kwargs["v"]
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
        elif self.kwargs["T"] and self.kwargs["h"] is not None:
            self._mode = "Th"
        elif self.kwargs["T"] and self.kwargs["s"] is not None:
            self._mode = "Ts"
        elif self.kwargs["T"] and self.kwargs["u"] is not None:
            self._mode = "Tu"
        elif self.kwargs["P"] and self.kwargs["rho"]:
            self._mode = "Prho"
        elif self.kwargs["P"] and self.kwargs["h"] is not None:
            self._mode = "Ph"
        elif self.kwargs["P"] and self.kwargs["s"] is not None:
            self._mode = "Ps"
        elif self.kwargs["P"] and self.kwargs["u"] is not None:
            self._mode = "Pu"
        elif self.kwargs["rho"] and self.kwargs["h"] is not None:
            self._mode = "rhoh"
        elif self.kwargs["rho"] and self.kwargs["s"] is not None:
            self._mode = "rhos"
        elif self.kwargs["rho"] and self.kwargs["u"] is not None:
            self._mode = "rhou"
        elif self.kwargs["h"] is not None and self.kwargs["s"] is not None:
            self._mode = "hs"
        elif self.kwargs["h"] is not None and self.kwargs["u"] is not None:
            self._mode = "hu"
        elif self.kwargs["s"] is not None and self.kwargs["u"] is not None:
            self._mode = "su"

        elif self.kwargs["T"] and self.kwargs["x"] is not None:
            self._mode = "Tx"
        elif self.kwargs["P"] and self.kwargs["x"] is not None:
            self._mode = "Px"
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

        self.R = self._constants["R"]/self.M

        propiedades = None
        if v and not rho:
            rho = 1./v

        if x is None:
            # Method with iteration necessary to get x
            if self._mode == "TP":
                if T < self.Tc and P < self.Pc and \
                        self._Vapor_Pressure(T) < P:
                    rhoo = self._Liquid_Density(T)
                elif T < self.Tc and P < self.Pc:
                    rhoo = self._Vapor_Density(T)
                else:
                    rhoo = self.rhoc*3
                rho = fsolve(
                    lambda rho: self._Helmholtz(rho, T)["P"]-P*1000, rhoo)

            elif self._mode == "Th":
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                hv = self._Helmholtz(rhov, T)["h"]
                if h > hv:
                    rhoo = rhov
                else:
                    rhoo = 1000
                rho = fsolve(lambda rho: self._Helmholtz(rho, T)["h"]-h, rhoo)

            elif self._mode == "Ts":
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                sv = self._Helmholtz(rhov, T)["s"]
                if s > sv:
                    rhoo = rhov
                else:
                    rhoo = 1000
                rho = fsolve(lambda rho: self._Helmholtz(rho, T)["s"]-s, rhoo)

            elif self._mode == "Tu":
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                Ps = self._Vapor_Pressure(T)
                vapor = self._Helmholtz(rhov, T)
                uv = vapor["h"]-Ps*vapor["v"]
                if u > uv:
                    rhoo = rhov
                else:
                    rhoo = 1000

                def funcion(rho):
                    par = self._Helmholtz(rho, T)
                    return par["h"]-par["P"]*par["v"]-u
                rho = fsolve(funcion, rhoo)

            elif self._mode == "Prho":
                T = fsolve(lambda T: self._Helmholtz(rho, T)["P"]-P*1000, 600)
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T[0] == 600 or rhov <= rho <= rhol:
                    def funcion(T):
                        rhol, rhov, Ps = self._saturation(T)
                        return Ps-P*1000
                    T = fsolve(funcion, 600)

            elif self._mode == "Ph":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["P"]-P*1000, par["h"]-h
                rho, T = fsolve(funcion, [1000, 300])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == 1000 or rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return Ps-P*1000, vapor["h"]*x+liquido["h"]*(1-x)-h
                    rho, T = fsolve(funcion, [2., 500.])

            elif self._mode == "Ps":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["P"]-P*1000, par["s"]-s
                rho, T = fsolve(funcion, [2., 400])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == 2. or rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return Ps-P*1000, vapor["s"]*x+liquido["s"]*(1-x)-s
                    rho, T = fsolve(funcion, [2., 500.])

            elif self._mode == "Pu":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-par["P"]*par["v"]-u, par["P"]-P*1000
                rho, T = fsolve(funcion, [1000, 600])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == 1000 or rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        vu = vapor["h"]-Ps/rhov
                        lu = liquido["h"]-Ps/rhol
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return Ps-P*1000, vu*x+lu*(1-x)-u
                    rho, T = fsolve(funcion, [2., 500.])

            elif self._mode == "rhoh":
                T = fsolve(lambda T: self._Helmholtz(rho, T)["h"]-h, 600)
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T[0] == 600 or rhov <= rho <= rhol:
                    def funcion(T):
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return vapor["h"]*x+liquido["h"]*(1-x)-h
                    T = fsolve(funcion, 500.)

            elif self._mode == "rhos":
                T = fsolve(lambda T: self._Helmholtz(rho, T)["s"]-s, 600)
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T[0] == 600 or rhov <= rho <= rhol:
                    def funcion(T):
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return vapor["s"]*x+liquido["s"]*(1-x)-s
                    T = fsolve(funcion, 500.)

            elif self._mode == "rhou":
                def funcion(T):
                    par = self._Helmholtz(rho, T)
                    return par["h"]-par["P"]/1000*par["v"]-u
                T = fsolve(funcion, 600)
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T[0] == 600 or rhov <= rho <= rhol:
                    def funcion(T):
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        vu = vapor["h"]-Ps/rhov
                        lu = liquido["h"]-Ps/rhol
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return vu*x+lu*(1-x)-u
                    T = fsolve(funcion, 500.)

            elif self._mode == "hs":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-h, par["s"]-s
                rho, T = fsolve(funcion, [1000, 300])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return (vapor["h"]*x+liquido["h"]*(1-x)-h,
                                vapor["s"]*x+liquido["s"]*(1-x)-s)
                    rho, T = fsolve(funcion, [0.5, 400.])

            elif self._mode == "hu":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-par["P"]/1000*par["v"]-u, par["h"]-h
                rho, T = fsolve(funcion, [1000, 600])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == 600 or rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        vu = vapor["h"]-Ps/rhov
                        lu = liquido["h"]-Ps/rhol
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return (vapor["h"]*x+liquido["h"]*(1-x)-h,
                                vu*x+lu*(1-x)-u)
                    rho, T = fsolve(funcion, [2., 500.])

            elif self._mode == "su":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-par["P"]/1000*par["v"]-u, par["s"]-s
                rho, T = fsolve(funcion, [1000, 600])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == 600 or rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        vu = vapor["h"]-Ps/rhov
                        lu = liquido["h"]-Ps/rhol
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return (vapor["s"]*x+liquido["s"]*(1-x)-s,
                                vu*x+lu*(1-x)-u)
                    rho, T = fsolve(funcion, [2., 500.])

            rho = float(rho)
            T = float(T)
            propiedades = self._Helmholtz(rho, T)
            if T <= self.Tc:
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rhol > rho > rhov:
                    rhol, rhov, Ps = self._saturation(T)
                    x = (1/rho-1/rhol)/(1/rhov-1/rhol)
                    if x < 0:
                        x = 0
                    elif x > 1:
                        x = 1
                    P = Ps/1000
                elif rho <= rhov:
                    x = 1
                elif rho >= rhol:
                    x = 0

                vapor = self._Helmholtz(rhov, T)
                liquido = self._Helmholtz(rhol, T)

            elif T > self.Tc:
                x = 1
            else:
                raise NotImplementedError("Incoming out of bound")

            if not P:
                P = propiedades["P"]/1000.

        elif self._mode == "Tx":
            # Check input T in saturation range
            if self.Tt > T or self.Tc < T:
                raise ValueError("Wrong input values")

            rhol, rhov, Ps = self._saturation(T)
            vapor = self._Helmholtz(rhov, T)
            liquido = self._Helmholtz(rhol, T)
            if x == 0:
                propiedades = liquido
            elif x == 1:
                propiedades = vapor
            P = Ps/1000.

        elif self._mode == "Px":
            # Iterate over saturation routine to get T

            def funcion(T):
                rhol = self._Liquid_Density(T)
                rhog = self._Vapor_Density(T)

                deltaL = rhol/self.rhoc
                deltaG = rhog/self.rhoc
                liquido = self._Helmholtz(rhol, T)
                vapor = self._Helmholtz(rhog, T)
                Ps = self.R*T*rhol*rhog/(rhol-rhog)*(
                    liquido["fir"]-vapor["fir"]+log(deltaL/deltaG))
                return Ps/1000-P

            To = _TSat_P(P)
            T = fsolve(funcion, To)[0]
            rhol, rhov, Ps = self._saturation(T)
            vapor = self._Helmholtz(rhov, T)
            liquido = self._Helmholtz(rhol, T)
            if x == 0:
                propiedades = liquido
            elif x == 1:
                propiedades = vapor

        self.T = T
        self.Tr = T/self.Tc
        self.P = P
        self.Pr = self.P/self.Pc
        self.x = x
        if self._mode in ["Tx", "Px"] or 0 < x < 1:
            region = 4
        else:
            region = 0
        self.phase = getphase(self.Tc, self.Pc, self.T, self.P, self.x, region)

        self.Liquid = _fase()
        self.Gas = _fase()
        if x == 0:
            # liquid phase
            self.fill(self.Liquid, propiedades)
            self.fill(self, propiedades)
        elif x == 1:
            # vapor phase
            self.fill(self.Gas, propiedades)
            self.fill(self, propiedades)
        else:
            self.fill(self.Liquid, liquido)
            self.fill(self.Gas, vapor)

            self.v = x*self.Gas.v+(1-x)*self.Liquid.v
            self.rho = 1./self.v

            self.h = x*self.Gas.h+(1-x)*self.Liquid.h
            self.s = x*self.Gas.s+(1-x)*self.Liquid.s
            self.u = x*self.Gas.u+(1-x)*self.Liquid.u
            self.a = x*self.Gas.a+(1-x)*self.Liquid.a
            self.g = x*self.Gas.g+(1-x)*self.Liquid.g

            self.Z = x*self.Gas.Z+(1-x)*self.Liquid.Z
            self.f = x*self.Gas.f+(1-x)*self.Liquid.f

            self.Z_rho = x*self.Gas.Z_rho+(1-x)*self.Liquid.Z_rho
            self.IntP = x*self.Gas.IntP+(1-x)*self.Liquid.IntP

        # Calculate special properties useful only for one phase
        if self._mode in ("Px", "Tx") or (x < 1 and self.Tt <= T <= self.Tc):
            if self.name == "water":
                self.sigma = _Tension(T)
            else:
                self.sigma = self._Tension(T)
        else:
            self.sigma = None

        if 0 < x < 1:
            self.virialB = vapor["B"]/self.rhoc
            self.virialC = vapor["C"]/self.rhoc**2
        else:
            self.virialB = propiedades["B"]/self.rhoc
            self.virialC = propiedades["C"]/self.rhoc**2

        if self.Tt <= T <= self.Tc:
            self.Hvap = vapor["h"]-liquido["h"]
        else:
            self.Hvap = None
        self.invT = -1/self.T

        # Ideal properties
        cp0 = self._prop0(self.rho, self.T)
        self.v0 = cp0.v
        self.rho0 = 1./self.v0
        self.h0 = cp0.h
        self.u0 = self.h0-self.P*self.v0
        self.s0 = cp0.s
        self.a0 = self.u0-self.T*self.s0
        self.g0 = self.h0-self.T*self.s0
        self.cp0 = cp0.cp
        self.cv0 = cp0.cv
        self.cp0_cv = self.cp0/self.cv0
        self.gamma0 = -self.v0/self.P/1000*self.derivative("P", "v", "s", cp0)

    def fill(self, fase, estado):
        """Fill phase properties"""
        fase.v = estado["v"]
        fase.rho = 1/fase.v

        fase.h = estado["h"]
        fase.s = estado["s"]
        fase.u = fase.h-self.P*1000*fase.v
        fase.a = fase.u-self.T*fase.s
        fase.g = fase.h-self.T*fase.s

        fase.Z = self.P*fase.v/self.T/self.R*1e3
        fase.fi = estado["fugacity"]
        fase.f = fase.fi*self.P
        fase.cp = estado["cp"]
        fase.cv = estado["cv"]
        fase.cp_cv = fase.cp/fase.cv
        fase.w = estado["w"]

        fase.alfap = estado["alfap"]
        fase.betap = estado["betap"]

        fase.joule = self.derivative("T", "P", "h", fase)*1e3
        fase.Gruneisen = fase.v/fase.cv*self.derivative("P", "T", "v", fase)
        fase.alfav = self.derivative("v", "T", "P", fase)/fase.v
        fase.kappa = -self.derivative("v", "P", "T", fase)/fase.v*1e3
        fase.betas = self.derivative("T", "P", "s", fase)

        fase.gamma = -fase.v/self.P*self.derivative("P", "v", "s", fase)*1e-3
        fase.kt = -fase.v/self.P*self.derivative("P", "v", "T", fase)*1e-3
        fase.ks = -self.derivative("v", "P", "s", fase)/fase.v*1e3
        fase.Kt = -fase.v*self.derivative("P", "v", "s", fase)*1e-3
        fase.Ks = -fase.v*self.derivative("P", "v", "T", fase)*1e-3
        fase.dhdT_rho = self.derivative("h", "T", "rho", fase)
        fase.dhdT_P = self.derivative("h", "T", "P", fase)
        fase.dhdP_T = self.derivative("h", "P", "T", fase)*1e3
        fase.dhdP_rho = self.derivative("h", "P", "rho", fase)*1e3
        fase.dhdrho_T = estado["dhdrho"]
        fase.dhdrho_P = estado["dhdrho"]+fase.dhdT_rho/estado["drhodt"]
        fase.dpdT_rho = self.derivative("P", "T", "rho", fase)*1e-3
        fase.dpdrho_T = estado["dpdrho"]*1e-3
        fase.drhodP_T = 1/estado["dpdrho"]*1e3
        fase.drhodT_P = estado["drhodt"]

        fase.Z_rho = (fase.Z-1)/fase.rho
        fase.IntP = self.T*self.derivative("P", "T", "rho", fase)*1e-3-self.P
        fase.hInput = fase.v*self.derivative("h", "v", "P", fase)

        fase.mu = self._visco(fase.rho, self.T, fase)
        fase.k = self._thermo(fase.rho, self.T, fase)
        fase.nu = fase.mu/fase.rho
        fase.alfa = fase.k/1000/fase.rho/fase.cp
        fase.Prandt = fase.mu*fase.cp*1000/fase.k
        if self.name == "water":
            fase.epsilon = _Dielectric(fase.rho, self.T)
            fase.n = _Refractive(fase.rho, self.T, self.kwargs["l"])
        else:
            fase.epsilon = None
            fase.n = None

    def _saturation(self, T=None):
        """Saturation calculation for two phase search"""
        if not T:
            T = self.T

        rhoLo = self._Liquid_Density(T)
        rhoGo = self._Vapor_Density(T)

        def f(parr):
            rhol, rhog = parr
            deltaL = rhol/self.rhoc
            deltaG = rhog/self.rhoc
            liquido = self._Helmholtz(rhol, T)
            vapor = self._Helmholtz(rhog, T)
            Jl = deltaL*(1+deltaL*liquido["fird"])
            Jv = deltaG*(1+deltaG*vapor["fird"])
            Kl = deltaL*liquido["fird"]+liquido["fir"]+log(deltaL)
            Kv = deltaG*vapor["fird"]+vapor["fir"]+log(deltaG)
            return Kv-Kl, Jv-Jl

        rhoL, rhoG = fsolve(f, [rhoLo, rhoGo])
        if rhoL == rhoG:
            Ps = self.Pc
        else:
            liquido = self._Helmholtz(rhoL, T)
            vapor = self._Helmholtz(rhoG, T)
            deltaL = rhoL/self.rhoc
            deltaG = rhoG/self.rhoc

            Ps = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                liquido["fir"]-vapor["fir"]+log(deltaL/deltaG))
        return rhoL, rhoG, Ps

#    def _saturation2(self, T):
#        """Akasaka (2008) "A Reliable and Useful Method to Determine the
#        Saturation State from Helmholtz Energy Equations of State", Journal of
#        Thermal Science and Technology, 3, 442-451
#        http://dx.doi.org/10.1299/jtst.3.442"""
#
#        rhoL = self._Liquid_Density(T)
#        rhoG = self._Vapor_Density(T)
#        g = 500.
#        erroro = 1e6
#        rholo = rhoL
#        rhogo = rhoG
#        contador = 0
#        while True:
#            contador += 1
#            deltaL = rhoL/self.rhoc
#            deltaG = rhoG/self.rhoc
#            liquido = self._Helmholtz(rhoL, T)
#            vapor = self._Helmholtz(rhoG, T)
#            Jl = deltaL*(1+deltaL*liquido["fird"])
#            Jv = deltaG*(1+deltaG*vapor["fird"])
#            Kl = deltaL*liquido["fird"]+liquido["fir"]+log(deltaL)
#            Kv = deltaG*vapor["fird"]+vapor["fir"]+log(deltaG)
#            Jdl = 1+2*deltaL*liquido["fird"]+deltaL**2*liquido["firdd"]
#            Jdv = 1+2*deltaG*vapor["fird"]+deltaG**2*vapor["firdd"]
#            Kdl = 2*liquido["fird"]+deltaL*liquido["firdd"]+1/deltaL
#            Kdv = 2*vapor["fird"]+deltaG*vapor["firdd"]+1/deltaG
#            Delta = Jdv*Kdl-Jdl*Kdv
#            error = abs(Kv-Kl)+abs(Jv-Jl)
#            if error < 1e-12 or contador > 100:
#                break
#            elif error > erroro:
#                rhoL = rholo
#                rhoG = rhogo
#                g = g*0.5
#            else:
#                erroro = error
#                rholo = rhoL
#                rhogo = rhoG
#                rhoL = rhoL+g/Delta*((Kv-Kl)*Jdv-(Jv-Jl)*Kdv)
#                rhoG = rhoG+g/Delta*((Kv-Kl)*Jdl-(Jv-Jl)*Kdl)
#        if error > 1e-3:
#            print("Iteration don´t converge, residual error %g" % error)
#
#        Ps = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
#            liquido["fir"]-vapor["fir"]+log(deltaL/deltaG))
#        return rhoL, rhoG, Ps

    def _Helmholtz(self, rho, T):
        """Calculated properties, table 3 pag 10"""
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        delta = rho/rhoc
        tau = Tc/T
        fio, fiot, fiott, fiod, fiodd, fiodt = self._phi0(tau, delta)
        fir, firt, firtt, fird, firdd, firdt, firdtt, B, C = self._phir(
            tau, delta)

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
        propiedades["cp"] = self.R*(
            -tau**2*(fiott+firtt) + (1+delta*fird-delta*tau*firdt)**2/(
                1+2*delta*fird+delta**2*firdd))
        propiedades["w"] = (
            self.R*1000*T*(1+2*delta*fird+delta**2*firdd - (
                1+delta*fird-delta*tau*firdt)**2/tau**2/(fiott+firtt)))**0.5
        propiedades["alfap"] = (1-delta*tau*firdt/(1+delta*fird))/T
        propiedades["betap"] = rho*(
            1+(delta*fird+delta**2*firdd)/(1+delta*fird))
        propiedades["fugacity"] = exp(fir+delta*fird-log(1+delta*fird))
        propiedades["B"] = B
        propiedades["C"] = C
        propiedades["dpdrho"] = self.R*T*(1+2*delta*fird+delta**2*firdd)
        propiedades["drhodt"] = -rho*(1+delta*fird-delta*tau*firdt) / \
            (T*(1+2*delta*fird+delta**2*firdd))
        propiedades["dhdrho"] = self.R*T/rho * \
            (tau*delta*(fiodt+firdt)+delta*fird+delta**2*firdd)
#        dbt=-phi11/rho/t
#        propiedades["cps"] = propiedades["cv"] Add cps from Argon pag.27

        return propiedades

    def _prop0(self, rho, T):
        """Ideal gas properties"""
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        delta = rho/rhoc
        tau = Tc/T
        fio, fiot, fiott, fiod, fiodd, fiodt = self._phi0(tau, delta)

        propiedades = _fase()
        propiedades.v = self.R*T/self.P/1000
        propiedades.h = self.R*T*(1+tau*fiot)
        propiedades.s = self.R*(tau*fiot-fio)
        propiedades.cv = -self.R*tau**2*fiott
        propiedades.cp = self.R*(-tau**2*fiott+1)
        propiedades.alfap = 1/T
        propiedades.betap = rho
        return propiedades

    def _phi0(self, tau, delta):
        Fi0 = self.Fi0

        fio = Fi0["ao_log"][0]*log(delta)+Fi0["ao_log"][1]*log(tau)
        fiot = +Fi0["ao_log"][1]/tau
        fiott = -Fi0["ao_log"][1]/tau**2

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

        return fio, fiot, fiott, fiod, fiodd, fiodt

    def _phir(self, tau, delta):
        delta_0 = 1e-200

        fir = fird = firdd = firt = firtt = firdt = firdtt = B = C = 0

        # Polinomial terms
        nr1 = self._constants.get("nr1", [])
        d1 = self._constants.get("d1", [])
        t1 = self._constants.get("t1", [])
        for n, d, t in zip(nr1, d1, t1):
            fir += n*delta**d*tau**t
            fird += n*d*delta**(d-1)*tau**t
            firdd += n*d*(d-1)*delta**(d-2)*tau**t
            firt += n*t*delta**d*tau**(t-1)
            firtt += n*t*(t-1)*delta**d*tau**(t-2)
            firdt += n*t*d*delta**(d-1)*tau**(t-1)
            firdtt += n*t*d*(t-1)*delta**(d-1)*tau**(t-2)
            B += n*d*delta_0**(d-1)*tau**t
            C += n*d*(d-1)*delta_0**(d-2)*tau**t

        # Exponential terms
        nr2 = self._constants.get("nr2", [])
        d2 = self._constants.get("d2", [])
        g2 = self._constants.get("gamma2", [])
        t2 = self._constants.get("t2", [])
        c2 = self._constants.get("c2", [])
        for n, d, g, t, c in zip(nr2, d2, g2, t2, c2):
            fir += n*delta**d*tau**t*exp(-g*delta**c)
            fird += n*exp(-g*delta**c)*delta**(d-1)*tau**t*(d-g*c*delta**c)
            firdd += n*exp(-g*delta**c)*delta**(d-2)*tau**t * \
                ((d-g*c*delta**c)*(d-1-g*c*delta**c)-g**2*c**2*delta**c)
            firt += n*t*delta**d*tau**(t-1)*exp(-g*delta**c)
            firtt += n*t*(t-1)*delta**d*tau**(t-2)*exp(-g*delta**c)
            firdt += n*t*delta**(d-1)*tau**(t-1)*(d-g*c*delta**c)*exp(
                -g*delta**c)
            firdtt += n*t*(t-1)*delta**(d-1)*tau**(t-2)*(d-g*c*delta**c) * \
                exp(-g*delta**c)
            B += n*exp(-g*delta_0**c)*delta_0**(d-1)*tau**t*(d-g*c*delta_0**c)
            C += n*exp(-g*delta_0**c)*(delta_0**(d-2)*tau**t*(
                (d-g*c*delta_0**c)*(d-1-g*c*delta_0**c)-g**2*c**2*delta_0**c))

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
            fir += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(
                delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])
            fird += nr3[i]*delta**d3[i]*tau**t3[i]*exp(
                -a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(
                    d3[i]/delta-2*a3[i]*(delta-e3[i]))
            firdd += nr3[i]*tau**t3[i]*exp(
                -a3[i]*(delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(
                    -2*a3[i]*delta**d3[i]+4*a3[i]**2*delta**d3[i]*(
                        delta-e3[i])**exp1[i]-4*d3[i]*a3[i]*delta**2*(
                            delta-e3[i])+d3[i]*2*delta)
            firt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(
                delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(
                    t3[i]/tau-2*b3[i]*(tau-g3[i]))
            firtt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(
                delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(
                    (t3[i]/tau-2*b3[i]*(tau-g3[i]))**exp2[i]-t3[i]/tau**2 -
                    2*b3[i])
            firdt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(
                delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(
                    t3[i]/tau-2*b3[i]*(tau-g3[i]))*(d3[i]/delta-2*a3[i]*(
                        delta-e3[i]))
            firdtt += nr3[i]*delta**d3[i]*tau**t3[i]*exp(-a3[i]*(
                delta-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*((
                    t3[i]/tau-2*b3[i]*(tau-g3[i]))**exp2[i]-t3[i]/tau**2-2 *
                    b3[i])*(d3[i]/delta-2*a3[i]*(delta-e3[i]))
            B += nr3[i]*delta_0**d3[i]*tau**t3[i]*exp(-a3[i]*(
                delta_0-e3[i])**exp1[i]-b3[i]*(tau-g3[i])**exp2[i])*(
                    d3[i]/delta_0-2*a3[i]*(delta_0-e3[i]))
            C += nr3[i]*tau**t3[i]*exp(-a3[i]*(delta_0-e3[i])**exp1[i]-b3[i]*(
                tau-g3[i])**exp2[i])*(
                    -2*a3[i]*delta_0**d3[i]+4*a3[i]**2*delta_0**d3[i]*(
                        delta_0-e3[i])**exp1[i]-4*d3[i]*a3[i]*delta_0**2*(
                            delta_0-e3[i])+d3[i]*2*delta_0)

        # Non analitic terms
        nr4 = self._constants.get("nr4", [])
        a4 = self._constants.get("a4", [])
        b = self._constants.get("b4", [])
        A = self._constants.get("A", [])
        Bi = self._constants.get("B", [])
        Ci = self._constants.get("C", [])
        D = self._constants.get("D", [])
        bt = self._constants.get("beta4", [])
        for i in range(len(nr4)):
            Tita = (1-tau)+A[i]*((delta-1)**2)**(0.5/bt[i])
            F = exp(-Ci[i]*(delta-1)**2-D[i]*(tau-1)**2)
            Fd = -2*Ci[i]*F*(delta-1)
            Fdd = 2*Ci[i]*F*(2*Ci[i]*(delta-1)**2-1)
            Ft = -2*D[i]*F*(tau-1)
            Ftt = 2*D[i]*F*(2*D[i]*(tau-1)**2-1)
            Fdt = 4*Ci[i]*D[i]*F*(delta-1)*(tau-1)
            Fdtt = 4*Ci[i]*D[i]*F*(delta-1)*(2*D[i]*(tau-1)**2-1)

            Delta = Tita**2+Bi[i]*((delta-1)**2)**a4[i]
            Deltad = (delta-1)*(A[i]*Tita*2/bt[i]*((delta-1)**2)**(
                0.5/bt[i]-1)+2*Bi[i]*a4[i]*((delta-1)**2)**(a4[i]-1))
            if delta == 1:
                Deltadd = 0
            else:
                Deltadd = Deltad/(delta-1)+(delta-1)**2*(4*Bi[i]*a4[i]*(
                    a4[i]-1)*((delta-1)**2)**(a4[i]-2)+2*A[i]**2/bt[i]**2*(((
                        delta-1)**2)**(0.5/bt[i]-1))**2+A[i]*Tita*4/bt[i]*(
                            0.5/bt[i]-1)*((delta-1)**2)**(0.5/bt[i]-2))

            DeltaBd = b[i]*Delta**(b[i]-1)*Deltad
            DeltaBdd = b[i]*(Delta**(b[i]-1)*Deltadd+(b[i]-1)*Delta**(
                b[i]-2)*Deltad**2)
            DeltaBt = -2*Tita*b[i]*Delta**(b[i]-1)
            DeltaBtt = 2*b[i]*Delta**(b[i]-1)+4*Tita**2*b[i]*(
                b[i]-1)*Delta**(b[i]-2)
            DeltaBdt = -A[i]*b[i]*2/bt[i]*Delta**(b[i]-1)*(delta-1)*((
                delta-1)**2)**(0.5/bt[i]-1)-2*Tita*b[i]*(b[i]-1)*Delta**(
                    b[i]-2)*Deltad
            DeltaBdtt = 2*b[i]*(b[i]-1)*Delta**(b[i]-2)*(Deltad*(
                1+2*Tita**2*(b[i]-2)/Delta)+4*Tita*A[i]*(delta-1)/bt[i]*((
                    delta-1)**2)**(0.5/bt[i]-1))

            fir += nr4[i]*Delta**b[i]*delta*F
            fird += nr4[i]*(Delta**b[i]*(F+delta*Fd)+DeltaBd*delta*F)
            firdd += nr4[i]*(Delta**b[i]*(2*Fd+delta*Fdd)+2*DeltaBd*(
                F+delta*Fd)+DeltaBdd*delta*F)
            firt += nr4[i]*delta*(DeltaBt*F+Delta**b[i]*Ft)
            firtt += nr4[i]*delta*(DeltaBtt*F+2*DeltaBt*Ft+Delta**b[i]*Ftt)
            firdt += nr4[i]*(Delta**b[i]*(Ft+delta*Fdt)+delta*DeltaBd*Ft +
                             DeltaBt*(F+delta*Fd)+DeltaBdt*delta*F)
            firdtt += nr4[i]*((DeltaBtt*F+2*DeltaBt*Ft+Delta**b[i]*Ftt)+delta*(
                DeltaBdtt*F+DeltaBtt*Fd+2*DeltaBdt*Ft+2*DeltaBt*Fdt+DeltaBt *
                Ftt+Delta**b[i]*Fdtt))

            Tita_ = (1-tau)+A[i]*((delta_0-1)**2)**(0.5/bt[i])
            Delta_ = Tita_**2+Bi[i]*((delta_0-1)**2)**a4[i]
            Deltad_ = (delta_0-1)*(A[i]*Tita_*2/bt[i]*((delta_0-1)**2)**(
                0.5/bt[i]-1)+2*Bi[i]*a4[i]*((delta_0-1)**2)**(a4[i]-1))
            Deltadd_ = Deltad_/(delta_0-1)+(delta_0-1)**2*(
                4*Bi[i]*a4[i]*(a4[i]-1)*((delta_0-1)**2)**(
                    a4[i]-2)+2*A[i]**2/bt[i]**2*(((delta_0-1)**2)**(
                        0.5/bt[i]-1))**2+A[i]*Tita_*4/bt[i]*(0.5/bt[i]-1)*((
                            delta_0-1)**2)**(0.5/bt[i]-2))
            DeltaBd_ = b[i]*Delta_**(b[i]-1)*Deltad_
            DeltaBdd_ = b[i]*(Delta_**(b[i]-1)*Deltadd_+(b[i]-1)*Delta_**(
                b[i]-2)*Deltad_**2)
            F_ = exp(-Ci[i]*(delta_0-1)**2-D[i]*(tau-1)**2)
            Fd_ = -2*Ci[i]*F_*(delta_0-1)
            Fdd_ = 2*Ci[i]*F_*(2*Ci[i]*(delta_0-1)**2-1)

            B += nr4[i]*(Delta_**b[i]*(F_+delta_0*Fd_)+DeltaBd_*delta_0*F_)
            C += nr4[i]*(Delta_**b[i]*(2*Fd_+delta_0*Fdd_)+2*DeltaBd_*(
                F_+delta_0*Fd_)+DeltaBdd_*delta_0*F_)

        return fir, firt, firtt, fird, firdd, firdt, firdtt, B, C

    def derivative(self, z, x, y, fase):
        """Calculate generic partial derivative: (δz/δx)y
        where x, y, z can be: P, T, v, u, h, s, g, a"""
        dT = {"P": self.P*1000*fase.alfap,
              "T": 1,
              "v": 0,
              "rho": 0,
              "u": fase.cv,
              "h": fase.cv+self.P*1000*fase.v*fase.alfap,
              "s": fase.cv/self.T,
              "g": self.P*1000*fase.v*fase.alfap-fase.s,
              "a": -fase.s}
        dv = {"P": -self.P*1000*fase.betap,
              "T": 0,
              "v": 1,
              "rho": -1,
              "u": self.P*1000*(self.T*fase.alfap-1),
              "h": self.P*1000*(self.T*fase.alfap-fase.v*fase.betap),
              "s": self.P*1000*fase.alfap,
              "g": -self.P*1000*fase.v*fase.betap,
              "a": -self.P*1000}
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

    def _Vapor_Density(self, T=None):
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

    def _Tension(self, T):
        """Equation for the surface tension"""
        tau = 1-T/self.Tc
        tension = 0
        for sigma, n in zip(self._surface["sigma"],
                            self._surface["exp"]):
            tension += sigma*tau**n
        return tension


class IAPWS95(MEoS):
    """Multiparameter equation of state for water (including IAPWS95)

    >>> water=IAPWS95(T=300, rho=996.5560)
    >>> print("%0.10f %0.8f %0.5f %0.9f" % ( \
        water.P, water.cv, water.w, water.s))
    0.0992418350 4.13018112 1501.51914 0.393062643

    >>> water=IAPWS95(T=500, rho=0.435)
    >>> print("%0.10f %0.8f %0.5f %0.9f" % ( \
        water.P, water.cv, water.w, water.s))
    0.0999679423 1.50817541 548.31425 7.944882714

    >>> water=IAPWS95(T=900., P=700)
    >>> print("%0.4f %0.8f %0.5f %0.8f" % ( \
        water.rho, water.cv, water.w, water.s))
    870.7690 2.66422350 2019.33608 4.17223802

    >>> water=IAPWS95(T=300., P=0.1)
    >>> print("%0.2f %0.5f %0.2f %0.2f %0.5f %0.4f %0.1f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.cp, water.w, \
        water.virialB))
    300.00 0.10000 996.56 112.65 0.39306 4.1806 1501.5 -0.066682

    >>> water=IAPWS95(T=500., P=0.1)
    >>> print("%0.2f %0.5f %0.5f %0.1f %0.4f %0.4f %0.2f %0.7f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.cp, water.w, \
        water.virialB))
    500.00 0.10000 0.43514 2928.6 7.9447 1.9813 548.31 -0.0094137

    >>> water=IAPWS95(T=450., x=0.5)
    >>> print("%0.2f %0.5f %0.4f %0.1f %0.4f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.virialB))
    450.00 0.93220 9.5723 1761.8 4.3589 -0.013028

    >>> water=IAPWS95(P=1.5, rho=1000.)
    >>> print("%0.2f %0.4f %0.1f %0.3f %0.5f %0.4f %0.1f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.cp, water.w, \
        water.virialB))
    286.44 1.5000 1000.0 57.253 0.19931 4.1855 1462.1 -0.085566

    >>> water=IAPWS95(h=3000, s=8.)
    >>> print("%0.2f %0.5f %0.5f %0.1f %0.4f %0.4f %0.2f %0.7f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.cp, water.w, \
        water.virialB))
    536.24 0.11970 0.48547 3000.0 8.0000 1.9984 567.04 -0.0076606

    >>> water=IAPWS95(h=150, s=0.4)
    >>> print("%0.2f %0.5f %0.2f %0.2f %0.5f %0.4f %0.1f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.cp, water.w, \
        water.virialB))
    301.27 35.50549 1011.48 150.00 0.40000 4.0932 1564.1 -0.065238

    >>> water=IAPWS95(T=450., rho=300)
    >>> print("%0.2f %0.5f %0.2f %0.2f %0.4f %0.6f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB))
    450.00 0.93220 300.00 770.82 2.1568 0.010693 -0.013028

    >>> water=IAPWS95(rho=300., P=0.1)
    >>> print("%0.2f %0.5f %0.2f %0.2f %0.4f %0.7f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB))
    372.76 0.10000 300.00 420.56 1.3110 0.0013528 -0.025144

    >>> water=IAPWS95(h=1500., P=0.1)
    >>> print("%0.2f %0.5f %0.4f %0.1f %0.4f %0.5f %0.6f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB))
    372.76 0.10000 1.2303 1500.0 4.2068 0.47952 -0.025144

    >>> water=IAPWS95(s=5., P=3.5)
    >>> print("%0.2f %0.4f %0.3f %0.1f %0.4f %0.5f %0.7f" % ( \
        water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB))
    515.71 3.5000 25.912 2222.8 5.0000 0.66921 -0.0085877

    >>> water=IAPWS95(T=500., u=900)
    >>> print("%0.2f %0.2f %0.2f %0.2f %0.1f %0.4f %0.4f %0.1f %0.7f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.cp, \
        water.w, water.virialB))
    500.00 108.21 903.62 900.00 1019.8 2.4271 4.1751 1576.0 -0.0094137

    >>> water=IAPWS95(P=0.3, u=1550.)
    >>> print("%0.2f %0.5f %0.4f %0.1f %0.1f %0.4f %0.5f %0.6f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    406.67 0.30000 3.3029 1550.0 1640.8 4.3260 0.49893 -0.018263

    >>> water=IAPWS95(rho=300, h=1000.)
    >>> print("%0.2f %0.4f %0.2f %0.2f %0.1f %0.4f %0.6f %0.7f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    494.92 2.3991 300.00 992.00 1000.0 2.6315 0.026071 -0.0097064

    >>> water=IAPWS95(rho=30, s=8.)
    >>> print("%0.2f %0.3f %0.3f %0.1f %0.1f %0.4f %0.4f %0.2f %0.9f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.cp, \
        water.w, water.virialB))
    1562.42 21.671 30.000 4628.5 5350.9 8.0000 2.7190 943.53 0.000047165

    >>> water=IAPWS95(rho=30, s=4.)
    >>> print("%0.2f %0.4f %0.3f %0.1f %0.1f %0.4f %0.5f %0.7f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    495.00 2.4029 30.000 1597.3 1677.4 4.0000 0.39218 -0.0097015

    >>> water=IAPWS95(rho=300, u=1000.)
    >>> print("%0.2f %0.4f %0.3f %0.1f %0.1f %0.4f %0.5f %0.7f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    496.44 2.4691 300.000 1000.0 1008.2 2.6476 0.02680 -0.0096173

    >>> water=IAPWS95(s=3., h=1000.)
    >>> print("%0.2f %0.6f %0.5f %0.2f %0.1f %0.4f %0.5f %0.6f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    345.73 0.034850 0.73526 952.60 1000.0 3.0000 0.29920 -0.034124

    >>> water=IAPWS95(u=995., h=1000.)
    >>> print("%0.2f %0.4f %0.2f %0.2f %0.1f %0.4f %0.5f %0.6f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    501.89 2.7329 546.58 995.00 1000.0 2.6298 0.00866 -0.009308

    >>> water=IAPWS95(u=1000., s=3.)
    >>> print("%0.2f %0.6f %0.5f %0.2f %0.1f %0.4f %0.5f %0.6f" % ( \
        water.T, water.P, water.rho, water.u, water.h, water.s, water.x, \
        water.virialB))
    371.24 0.094712 1.99072 1000.00 1047.6 3.0000 0.28144 -0.025543

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

    Fi0 = {"ao_log": [1, 3.00632],
           "pow": [0, 1],
           "ao_pow": [-8.3204464837497, 6.6832105275932],
           "ao_exp": [0.012436, 0.97315, 1.2795, 0.96956, 0.24873],
           "titao": [1.28728967, 3.53734222, 7.74073708, 9.24437796,
                     27.5075105]}

    _constants = {
        "R": 8.314371357587,

        "nr1": [0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
                0.31802509345418, -0.26145533859358, -0.78199751687981e-2,
                0.88089493102134e-2],
        "d1": [1, 1, 1, 2, 2, 3, 4],
        "t1": [-0.5, 0.875, 1, 0.5, 0.75, 0.375, 1],

        "nr2": [-0.66856572307965, 0.20433810950965, -0.66212605039687e-4,
                -0.19232721156002, -0.25709043003438, 0.16074868486251,
                -0.4009282892587e-1, 0.39343422603254e-6, -0.75941377088144e-5,
                0.56250979351888e-3, -0.15608652257135e-4, 0.11537996422951e-8,
                .36582165144204e-6, -.13251180074668e-11, -.62639586912454e-9,
                -0.10793600908932, 0.17611491008752e-1, 0.22132295167546,
                -0.40247669763528, 0.58083399985759, 0.49969146990806e-2,
                -0.31358700712549e-1, -0.74315929710341, 0.47807329915480,
                0.20527940895948e-1, -0.13636435110343, 0.14180634400617e-1,
                0.83326504880713e-2, -0.29052336009585e-1, 0.38615085574206e-1,
                -0.20393486513704e-1, -0.16554050063734e-2, .19955571979541e-2,
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
        "b4": [0.85, 0.95],
        "B": [0.2, 0.2],
        "C": [28, 32],
        "D": [700, 800],
        "A": [0.32, .32],
        "beta4": [0.3, 0.3]}

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

    def _visco(self, rho, T, fase):
        ref = IAPWS95()
        estado = ref._Helmholtz(rho, 1.5*647.096)
        drho = 1/estado["dpdrho"]*1e3
        return _Viscosity(rho, T, fase, drho)

    def _thermo(self, rho, T, fase):
        ref = IAPWS95()
        estado = ref._Helmholtz(rho, 1.5*647.096)
        drho = 1/estado["dpdrho"]*1e3
        return _ThCond(rho, T, fase, drho)


class IAPWS95_PT(IAPWS95):
    """Derivated class for direct P and T input"""
    def __init__(self, P, T):
        IAPWS95.__init__(self, T=T, P=P)


class IAPWS95_Ph(IAPWS95):
    """Derivated class for direct P and h input"""
    def __init__(self, P, h):
        IAPWS95.__init__(self, P=P, h=h)


class IAPWS95_Ps(IAPWS95):
    """Derivated class for direct P and s input"""
    def __init__(self, P, s):
        IAPWS95.__init__(self, P=P, s=s)


class IAPWS95_Pv(IAPWS95):
    """Derivated class for direct P and v input"""
    def __init__(self, P, v):
        IAPWS95.__init__(self, P=P, v=v)


class IAPWS95_Tx(IAPWS95):
    """Derivated class for direct T and x input"""
    def __init__(self, T, x):
        IAPWS95.__init__(self, T=T, x=x)


class D2O(MEoS):
    """Multiparameter equation of state for heavy water

    >>> water=D2O(T=300, rho=996.5560)
    >>> print("%0.10f %0.8f %0.5f" % ( \
        water.P, water.Liquid.cv, water.Liquid.w))
    0.0030675947 4.21191157 5332.04871
    """
    name = "heavy water"
    CASNumber = "7789-20-0"
    formula = "D2O"
    synonym = "deuterium oxide"
    Tc = 643.847
    rhoc = 356.0
    Pc = 21.671
    M = 20.027508  # g/mol
    Tt = 276.97
    Tb = 374.563
    f_acent = 0.364
    momentoDipolar = 1.9

    Fi0 = {'ao_log': [1, 2.9176485],
           'ao_pow': [-5.60420745, 5.4495718, 0.100195196505025,
                      -0.2844660508898171, 0.06437609920676933,
                      -0.005436994367359454],
           'pow': [0, 1, -1.0, -2.0, -3.0, -4.0],
           'ao_exp': [], 'titao': []}

    _constants = {
        "R": 8.3143565,
        "rhoref": 358.,

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
        "ao": [-0.37651e1, -0.38673e2, 0.73024e2, -0.13251e3, 0.75235e2,
               -0.70412e2],
        "exp": [0.409, 1.766, 2.24, 3.04, 3.42, 6.9]}

    @classmethod
    def _visco(cls, rho, T, fase=None):
        Tr = T/643.847
        rhor = rho/358.0

        no = [1.0, 0.940695, 0.578377, -0.202044]
        fi0 = Tr**0.5/sum([n/Tr**i for i, n in enumerate(no)])

        Li = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 5, 0, 1, 2, 3, 0, 1, 3,
              5, 0, 1, 5, 3]
        Lj = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
              4, 5, 5, 5, 6]
        Lij = [0.4864192, -0.2448372, -0.8702035, 0.8716056, -1.051126,
               0.3458395, 0.3509007, 1.315436, 1.297752, 1.353448, -0.2847572,
               -1.037026, -1.287846, -0.02148229, 0.07013759, 0.4660127,
               0.2292075, -0.4857462, 0.01641220, -0.02884911, 0.1607171,
               -.009603846, -.01163815, -.008239587, 0.004559914, -0.003886659]

        arr = [lij*(1./Tr-1)**i*(rhor-1)**j for i, j, lij in zip(Li, Lj, Lij)]
        fi1 = exp(rhor*sum(arr))

        return 55.2651e-6*fi0*fi1

    @classmethod
    def _thermo(cls, rho, T, fase=None):
        rhor = rho/358
        Tr = T/643.847
        tau = Tr/(abs(Tr-1.1)+1.1)

        no = [1.0, 37.3223, 22.5485, 13.0465, 0.0, -2.60735]
        Lo = sum([Li*Tr**i for i, Li in enumerate(no)])

        nr = [483.656, -191.039, 73.0358, -7.57467]
        Lr = -167.31*(1-exp(-2.506*rhor))+sum(
            [Li*rhor**(i+1) for i, Li in enumerate(nr)])

        f1 = exp(0.144847*Tr-5.64493*Tr**2)
        f2 = exp(-2.8*(rhor-1)**2)-0.080738543*exp(-17.943*(rhor-0.125698)**2)
        f3 = 1+exp(60*(tau-1)+20)
        f4 = 1+exp(100*(tau-1)+15)
        Lc = 35429.6*f1*f2*(1+f2**2*(5e9*f1**4/f3+3.5*f2/f4))

        Ll = -741.112*f1**1.2*(1-exp(-(rhor/2.5)**10))

        return 0.742128e-3*(Lo+Lr+Lc+Ll)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
