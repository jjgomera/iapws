#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Implemented multiparameter equation of state as a Helmholtz free energy
    * IAPWS-95 implementation
    * Heavy water formulation 2005
"""


from __future__ import division
from itertools import product
import warnings

from scipy import exp, log, ndarray
from scipy.optimize import fsolve

from .iapws97 import _TSat_P, IAPWS97
from ._iapws import M, Tc, Pc, rhoc, Tc_D2O, Pc_D2O, rhoc_D2O
from ._iapws import _Viscosity, _ThCond, _Dielectric, _Refractive, _Tension
from ._iapws import _D2O_Viscosity, _D2O_ThCond, _D2O_Tension
from ._utils import _fase, getphase, deriv_H


class MEoS(_fase):
    """
    General implementation of multiparameter equation of state. From this
    derived all child class specified per individual compounds

    Keyword Args
    ------------
    T : float
        Temperature [K]
    P : float
        Pressure [MPa]
    rho : float
        Density [kg/m³]
    v : float
        Specific volume [m³/kg]
    h : float
        Specific enthalpy [kJ/kg]
    s : float
        Specific entropy [kJ/kgK]
    u : float
        Specific internal energy [kJ/kg]
    x : float
        Vapor quality [-]

    l : float, optional
        Wavelength of light, for refractive index [nm]
    rho0 : float, optional
        Initial value of density, to improve iteration [kg/m³]
    T0 : float, optional
        Initial value of temperature, to improve iteration [K]
    x0 : Initial value of vapor quality, necessary in bad input pair definition
        where there are two valid solution (T-h, T-s)

    Notes
    -----
    * It needs two incoming properties of T, P, rho, h, s, u.
    * v as a alternate input parameter to rho
    * T-x, P-x, preferred input pair to specified a point in two phases region

    Returns
    -------
    The calculated instance has the following properties:
        * P: Pressure [MPa]
        * T: Temperature [K]
        * x: Vapor quality [-]
        * g: Specific Gibbs free energy [kJ/kg]
        * a: Specific Helmholtz free energy [kJ/kg]
        * v: Specific volume [m³/kg]
        * r: Density [kg/m³]
        * h: Specific enthalpy [kJ/kg]
        * u: Specific internal energy [kJ/kg]
        * s: Specific entropy [kJ/kg·K]
        * cp: Specific isobaric heat capacity [kJ/kg·K]
        * cv: Specific isochoric heat capacity [kJ/kg·K]
        * cp_cv: Heat capacity ratio, [-]
        * Z: Compression factor [-]
        * fi: Fugacity coefficient [-]
        * f: Fugacity [MPa]
        * gamma: Isoentropic exponent [-]

        * alfav: Isobaric cubic expansion coefficient [1/K]
        * kappa: Isothermal compressibility [1/MPa]
        * kappas: Adiabatic compresibility [1/MPa]
        * alfap: Relative pressure coefficient [1/K]
        * betap: Isothermal stress coefficient [kg/m³]
        * joule: Joule-Thomson coefficient [K/MPa]

        * betas: Isoentropic temperature-pressure coefficient [-]
        * Gruneisen: Gruneisen parameter [-]
        * virialB: Second virial coefficient [m³/kg]
        * virialC: Third virial coefficient [m⁶/kg²]
        * dpdT_rho: Derivatives, dp/dT at constant rho [MPa/K]
        * dpdrho_T: Derivatives, dp/drho at constant T [MPa·m³/kg]
        * drhodT_P: Derivatives, drho/dT at constant P [kg/m³·K]
        * drhodP_T: Derivatives, drho/dP at constant T [kg/m³·MPa]
        * dhdT_rho: Derivatives, dh/dT at constant rho [kJ/kg·K]
        * dhdP_T: Isothermal throttling coefficient [kJ/kg·MPa]
        * dhdT_P: Derivatives, dh/dT at constant P [kJ/kg·K]
        * dhdrho_T: Derivatives, dh/drho at constant T [kJ·m³/kg²]
        * dhdrho_P: Derivatives, dh/drho at constant P [kJ·m³/kg²]
        * dhdP_rho: Derivatives, dh/dP at constant rho [kJ/kg·MPa]
        * kt: Isothermal Expansion Coefficient [-]
        * ks: Adiabatic Compressibility [1/MPa]
        * Ks: Adiabatic bulk modulus [MPa]
        * Kt: Isothermal bulk modulus [MPa]

        * v0: Ideal specific volume [m³/kg]
        * rho0: Ideal gas density [kg/m³]
        * u0: Ideal specific internal energy [kJ/kg]
        * h0: Ideal specific enthalpy [kJ/kg]
        * s0: Ideal specific entropy [kJ/kg·K]
        * a0: Ideal specific Helmholtz free energy [kJ/kg]
        * g0: Ideal specific Gibbs free energy [kJ/kg]
        * cp0: Ideal specific isobaric heat capacity [kJ/kg·K]
        * cv0: Ideal specific isochoric heat capacity [kJ/kg·K]
        * w0: Ideal speed of sound [m/s]
        * gamma0: Ideal isoentropic exponent [-]

        * w: Speed of sound [m/s]
        * mu: Dynamic viscosity [Pa·s]
        * nu: Kinematic viscosity [m²/s]
        * k: Thermal conductivity [W/m·K]
        * alfa: Thermal diffusivity [m²/s]
        * sigma: Surface tension [N/m]
        * epsilon: Dielectric constant [-]
        * n: Refractive index [-]
        * Prandt: Prandtl number [-]
        * Pr: Reduced Pressure [-]
        * Tr: Reduced Temperature [-]
        * Hvap: Vaporization heat [kJ/kg]
        * Svap: Vaporization entropy [kJ/kg·K]

        * Z_rho: (Z-1) over the density [m³/kg]
        * IntP: Internal pressure [MPa]
        * invT: Negative reciprocal temperature [1/K]
        * hInput: Specific heat input [kJ/kg]
    """
    CP = None
    _Pv = None
    _rhoL = None
    _rhoG = None

    kwargs = {"T": 0.0,
              "P": 0.0,
              "rho": 0.0,
              "v": 0.0,
              "h": None,
              "s": None,
              "u": None,
              "x": None,
              "l": 0.5893,
              "rho0": None,
              "T0": None,
              "x0": 0.5}
    status = 0
    msg = "Undefined"

    def __init__(self, **kwargs):
        """Constructor, define common constant and initinialice kwargs"""
        self.R = self._constants["R"]/self._constants.get("M", self.M)
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
            try:
                self.status = 1
                self.calculo()
                self.msg = ""
            except RuntimeError as err:
                self.status = 0
                self.msg = err.args[0]
                raise(err)

            # Add msg for extrapolation state
            if self.name == "water" and 130 <= self.T < 273.15:
                self.msg = "Extrapolated state"
                self.status = 3
                warnings.warn("Using extrapolated values")
            elif self.name == "water" and 50 <= self.T < 130:
                self.msg = "Extrapolated state using Low-Temperature extension"
                self.status = 3
                warnings.warn("Using extrapolated values and Low-Temperature"
                              "extension")

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
        s = self.kwargs["s"]
        h = self.kwargs["h"]
        u = self.kwargs["u"]
        x = self.kwargs["x"]

        # Initial values
        T0 = self.kwargs["T0"]
        rho0 = self.kwargs["rho0"]

        if T0 or rho0:
            To = T0
            rhoo = rho0
        elif self.name == "air":
            To = 300
            rhoo = 1e-3
        else:
            try:
                st0 = IAPWS97(**self.kwargs)
            except NotImplementedError:
                To = 300
                rhoo = 900
            else:
                if st0.status:
                    To = st0.T
                    rhoo = st0.rho
                else:
                    To = 300
                    rhoo = 900

        self.R = self._constants["R"]/self._constants.get("M", self.M)

        propiedades = None

        if x is None:
            # Method with iteration necessary to get x
            if self._mode == "TP":
                try:
                    if self.name == "air":
                        raise ValueError
                    st0 = IAPWS97(**self.kwargs)
                    rhoo = st0.rho
                except NotImplementedError:
                    if rho0:
                        rhoo = rho0
                    elif T < self.Tc and P < self.Pc and \
                            self._Vapor_Pressure(T) < P:
                        rhoo = self._Liquid_Density(T)
                    elif T < self.Tc and P < self.Pc:
                        rhoo = self._Vapor_Density(T)
                    else:
                        rhoo = self.rhoc*3
                except ValueError:
                    rhoo = 1e-3
                rho = fsolve(
                    lambda rho: self._Helmholtz(rho, T)["P"]-P*1000, rhoo)[0]

            elif self._mode == "Th":
                def f(rho):
                    return self._Helmholtz(rho, T)["h"]-h

                if T >= self.Tc:
                    rhoo = self.rhoc
                    rho = fsolve(f, rhoo)[0]
                else:
                    x0 = self.kwargs["x0"]
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    hl = self._Helmholtz(rhol, T)["h"]
                    hv = self._Helmholtz(rhov, T)["h"]
                    if x0 not in (0, 1) and hl <= h <= hv:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        hv = vapor["h"]
                        hl = liquido["h"]
                        x = (h-hl)/(hv-hl)
                        rho = 1/(x/rhov+(1-x)/rhol)
                        P = Ps/1000
                    else:
                        if h > hv:
                            rhoo = rhov
                        else:
                            rhoo = rhol
                        rho = fsolve(f, rhoo)[0]

            elif self._mode == "Ts":
                def f(rho):
                    if rho < 0:
                        rho = 1e-20
                    return self._Helmholtz(rho, T)["s"]-s

                if T >= self.Tc:
                    rhoo = self.rhoc
                    rho = fsolve(f, rhoo)[0]
                else:
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    sl = self._Helmholtz(rhol, T)["s"]
                    sv = self._Helmholtz(rhov, T)["s"]
                    if sl <= s <= sv:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        sv = vapor["s"]
                        sl = liquido["s"]
                        x = (s-sl)/(sv-sl)
                        rho = 1/(x/rhov+(1-x)/rhol)
                        P = Ps/1000
                    else:
                        if s > sv:
                            rhoo = rhov
                        else:
                            rhoo = rhol
                        rho = fsolve(f, rhoo)[0]

            elif self._mode == "Tu":
                def f(rho):
                    prop = self._Helmholtz(rho, T)
                    return prop["h"]-prop["P"]*prop["v"]-u

                if T >= self.Tc:
                    rhoo = self.rhoc
                    rho = fsolve(f, rhoo)[0]
                else:
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    uv = vapor["h"]-vapor["P"]*vapor["v"]
                    ul = liquido["h"]-liquido["P"]*liquido["v"]
                    if ul <= u <= uv:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        uv = vapor["h"]-vapor["P"]*vapor["v"]
                        ul = liquido["h"]-liquido["P"]*liquido["v"]
                        x = (u-ul)/(uv-ul)
                        rho = 1/(x/rhov-(1-x)/rhol)
                        P = Ps/1000
                    else:
                        if u > uv:
                            rhoo = rhov
                        else:
                            rhoo = rhol
                        rho = fsolve(f, rhoo)[0]

            elif self._mode == "Prho":
                T = fsolve(
                    lambda T: self._Helmholtz(rho, T)["P"]-P*1000, To)[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:

                    def f(parr):
                        T, rhol, rhog = parr
                        deltaL = rhol/self.rhoc
                        deltaG = rhog/self.rhoc
                        liquido = self._Helmholtz(rhol, T)
                        vapor = self._Helmholtz(rhog, T)
                        Jl = rhol*(1+deltaL*liquido["fird"])
                        Jv = rhog*(1+deltaG*vapor["fird"])
                        K = liquido["fir"]-vapor["fir"]
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(
                            liquido["fir"]-vapor["fir"]+log(rhol/rhog))
                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                Ps - P*1000)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = sol[0]
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido["fir"]-vapor["fir"]+log(rhoL/rhoG))/1000

            elif self._mode == "Ph":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["P"]-P*1000, par["h"]-h
                rho, T = fsolve(funcion, [rhoo, To])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == rhoo or rhov <= rho <= rhol:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return Ps-P*1000, vapor["h"]*x+liquido["h"]*(1-x)-h
                    rho, T = fsolve(funcion, [2., 500.])
                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    hv = vapor["h"]
                    hl = liquido["h"]
                    x = (h-hl)/(hv-hl)

            elif self._mode == "Ps":
                try:
                    x0 = st0.x
                except NameError:
                    x0 = None

                if x0 is None or x0 == 0 or x0 == 1:
                    def funcion(parr):
                        par = self._Helmholtz(parr[0], parr[1])
                        return par["P"]-P*1000, par["s"]-s
                    rho, T = fsolve(funcion, [rhoo, To])

                else:
                    def funcion(parr):
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return Ps-P*1000, vapor["s"]*x+liquido["s"]*(1-x)-s
                    rho, T = fsolve(funcion, [2., 500.])
                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    sv = vapor["s"]
                    sl = liquido["s"]
                    x = (s-sl)/(sv-sl)

            elif self._mode == "Pu":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-par["P"]*par["v"]-u, par["P"]-P*1000
                sol = fsolve(funcion, [rhoo, To], full_output=True)
                rho, T = sol[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == rhoo or sol[2] != 1:
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
                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    uv = vapor["h"]-Ps/rhov
                    ul = liquido["h"]-Ps/rhol
                    x = (u-ul)/(uv-ul)

            elif self._mode == "rhoh":
                T = fsolve(lambda T: self._Helmholtz(rho, T)["h"]-h, To)[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:
                    def f(parr):
                        T, rhol, rhog = parr
                        deltaL = rhol/self.rhoc
                        deltaG = rhog/self.rhoc
                        liquido = self._Helmholtz(rhol, T)
                        vapor = self._Helmholtz(rhog, T)
                        Jl = rhol*(1+deltaL*liquido["fird"])
                        Jv = rhog*(1+deltaG*vapor["fird"])
                        K = liquido["fir"]-vapor["fir"]
                        x = (1./rho-1/rhol)/(1/rhog-1/rhol)
                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                liquido["h"]*(1-x)+vapor["h"]*x - h)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = sol[0]
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido["fir"]-vapor["fir"]+log(rhoL/rhoG))/1000

            elif self._mode == "rhos":
                T = fsolve(lambda T: self._Helmholtz(rho, T)["s"]-s, To)[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:
                    def f(parr):
                        T, rhol, rhog = parr
                        deltaL = rhol/self.rhoc
                        deltaG = rhog/self.rhoc
                        liquido = self._Helmholtz(rhol, T)
                        vapor = self._Helmholtz(rhog, T)
                        Jl = rhol*(1+deltaL*liquido["fird"])
                        Jv = rhog*(1+deltaG*vapor["fird"])
                        K = liquido["fir"]-vapor["fir"]
                        x = (1./rho-1/rhol)/(1/rhog-1/rhol)
                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                liquido["s"]*(1-x)+vapor["s"]*x - s)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = sol[0]
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido["fir"]-vapor["fir"]+log(rhoL/rhoG))/1000

            elif self._mode == "rhou":
                def funcion(T):
                    par = self._Helmholtz(rho, T)
                    return par["h"]-par["P"]/rho-u
                T = fsolve(funcion, To)[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:
                    def f(parr):
                        T, rhol, rhog = parr
                        deltaL = rhol/self.rhoc
                        deltaG = rhog/self.rhoc
                        liquido = self._Helmholtz(rhol, T)
                        vapor = self._Helmholtz(rhog, T)
                        Jl = rhol*(1+deltaL*liquido["fird"])
                        Jv = rhog*(1+deltaG*vapor["fird"])
                        K = liquido["fir"]-vapor["fir"]
                        x = (1./rho-1/rhol)/(1/rhog-1/rhol)
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(
                            liquido["fir"]-vapor["fir"]+log(rhol/rhog))
                        vu = vapor["h"]-Ps/rhog
                        lu = liquido["h"]-Ps/rhol
                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                lu*(1-x)+vu*x - u)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = sol[0]
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido["fir"]-vapor["fir"]+log(rhoL/rhoG))/1000

            elif self._mode == "hs":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-h, par["s"]-s
                rho, T = fsolve(funcion, [rhoo, To])
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
                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    sv = vapor["s"]
                    sl = liquido["s"]
                    x = (s-sl)/(sv-sl)
                    P = Ps/1000

            elif self._mode == "hu":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-par["P"]*par["v"]-u, par["h"]-h
                sol = fsolve(funcion, [rhoo, To], full_output=True)
                rho, T = sol[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if sol[2] != 1 or rhov <= rho <= rhol:
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

                    To = [500, 700, 300, 900]
                    if self.kwargs["T0"]:
                        To.insert(0, self.kwargs["T0"])
                    rhov = self._Vapor_Density(self.Tt)
                    rhol = self._Liquid_Density(self.Tt)
                    ro = [1, 1e-3, rhov, rhol]
                    if self.kwargs["rho0"]:
                        ro.insert(0, self.kwargs["rho0"])

                    for r, t in product(ro, To):
                        sol = fsolve(funcion, [r, t], full_output=True)
                        rho, T = sol[0]
                        if sol[2] == 1 and sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    hv = vapor["h"]
                    hl = liquido["h"]
                    x = (h-hl)/(hv-hl)
                    P = Ps/1000

            elif self._mode == "su":
                def funcion(parr):
                    par = self._Helmholtz(parr[0], parr[1])
                    return par["h"]-par["P"]*par["v"]-u, par["s"]-s
                sol = fsolve(funcion, [rhoo, To], full_output=True)
                rho, T = sol[0]
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if sol[2] != 1 or rhov <= rho <= rhol:
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

                    To = [500, 700, 300, 900]
                    if self.kwargs["T0"]:
                        To.insert(0, self.kwargs["T0"])
                    rhov = self._Vapor_Density(self.Tt)
                    rhol = self._Liquid_Density(self.Tt)
                    ro = [1, 1e-3, rhov, rhol]
                    if self.kwargs["rho0"]:
                        ro.insert(0, self.kwargs["rho0"])

                    for r, t in product(ro, To):
                        sol = fsolve(funcion, [r, t], full_output=True)
                        rho, T = sol[0]
                        if sol[2] == 1 and sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    sv = vapor["s"]
                    sl = liquido["s"]
                    x = (s-sl)/(sv-sl)
                    P = Ps/1000

            elif self._mode == "Trho":
                if T < self.Tc:
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    if rhol > rho > rhov:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1/rho-1/rhol)/(1/rhov-1/rhol)
                        rho = 1/(x/rhov-(1-x)/rhol)
                        P = Ps/1000

            rho = float(rho)
            T = float(T)
            propiedades = self._Helmholtz(rho, T)

            if T > self.Tc:
                x = 1
            elif x is None:
                x = 0

            if not P:
                P = propiedades["P"]/1000.

        elif self._mode == "Tx":
            # Check input T in saturation range
            if self.Tt > T or self.Tc < T or x > 1 or x < 0:
                raise NotImplementedError("Incoming out of bound")

            rhol, rhov, Ps = self._saturation(T)
            vapor = self._Helmholtz(rhov, T)
            liquido = self._Helmholtz(rhol, T)
            if x == 0:
                propiedades = liquido
            elif x == 1:
                propiedades = vapor
            P = Ps/1000.

        elif self._mode == "Px":
            # Check input P in saturation range
            if self.Pc < P or x > 1 or x < 0:
                raise NotImplementedError("Incoming out of bound")

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

            if T0:
                To = T0
            elif self.name == "water":
                To = _TSat_P(P)
            else:
                To = (self.Tc+self.Tt)/2
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
            self.sigma = self._surface(T)
        else:
            self.sigma = None

        if 0 < x < 1:
            self.virialB = vapor["B"]/self.rhoc
            self.virialC = vapor["C"]/self.rhoc**2
            self.Hvap = vapor["h"]-liquido["h"]
            self.Svap = vapor["s"]-liquido["s"]
        else:
            self.virialB = propiedades["B"]/self.rhoc
            self.virialC = propiedades["C"]/self.rhoc**2
            self.Hvap = None
            self.Svap = None

        self.invT = -1/self.T

        # Ideal properties
        cp0 = self._prop0(self.rho, self.T)
        self.v0 = self.R*self.T/self.P/1000
        self.rho0 = 1./self.v0
        self.h0 = cp0.h
        self.u0 = self.h0-self.P*self.v0
        self.s0 = cp0.s
        self.a0 = self.u0-self.T*self.s0
        self.g0 = self.h0-self.T*self.s0
        self.cp0 = cp0.cp
        self.cv0 = cp0.cv
        self.cp0_cv = self.cp0/self.cv0
        cp0.v = self.v0
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
        fase.fi = exp(estado["fir"]+estado["delta"]*estado["fird"] -
                      log(1+estado["delta"]*estado["fird"]))
        fase.f = fase.fi*self.P
        fase.cv = estado["cv"]

        fase.rhoM = fase.rho/self.M
        fase.hM = fase.h*self.M
        fase.sM = fase.s*self.M
        fase.uM = fase.u*self.M
        fase.aM = fase.a*self.M
        fase.gM = fase.g*self.M

        fase.alfap = estado["alfap"]
        fase.betap = estado["betap"]

        fase.cp = self.derivative("h", "T", "P", fase)
        fase.cp_cv = fase.cp/fase.cv
        fase.w = (self.derivative("P", "rho", "s", fase)*1000)**0.5
        fase.cvM = fase.cv*self.M
        fase.cpM = fase.cp*self.M

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
        fase.dhdrho_T = self.derivative("h", "rho", "T", fase)
        fase.dhdrho_P = self.derivative("h", "rho", "P", fase)
        fase.dpdT_rho = self.derivative("P", "T", "rho", fase)*1e-3
        fase.dpdrho_T = self.derivative("P", "rho", "T", fase)*1e-3
        fase.drhodP_T = self.derivative("rho", "P", "T", fase)*1e3
        fase.drhodT_P = self.derivative("rho", "T", "P", fase)

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
            try:
                fase.n = _Refractive(fase.rho, self.T, self.kwargs["l"])
            except NotImplementedError:
                fase.n = None
        else:
            fase.epsilon = None
            fase.n = None

    def derivative(self, z, x, y, fase):
        """Wrapper derivative for custom derived properties
        where x, y, z can be: P, T, v, u, h, s, g, a"""
        return deriv_H(self, z, x, y, fase)

    def _saturation(self, T):
        """Saturation calculation for two phase search"""
        if T > self.Tc:
            T = self.Tc
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

    def _Helmholtz(self, rho, T):
        """Calculated properties, table 3 pag 10"""
        if isinstance(rho, ndarray):
            rho = rho[0]
        if isinstance(T, ndarray):
            T = T[0]
        if rho < 0:
            rho = 1e-20
        if T < 50:
            T = 50
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        delta = rho/rhoc
        tau = Tc/T
        ideal = self._phi0(tau, delta)
        fio = ideal["fio"]
        fiot = ideal["fiot"]
        fiott = ideal["fiott"]

        res = self._phir(tau, delta)
        fir = res["fir"]
        firt = res["firt"]
        firtt = res["firtt"]
        fird = res["fird"]
        firdd = res["firdd"]
        firdt = res["firdt"]

        propiedades = {}
        propiedades["fir"] = fir
        propiedades["fird"] = fird
        propiedades["firdd"] = firdd
        propiedades["delta"] = delta

        propiedades["T"] = T
        propiedades["P"] = (1+delta*fird)*self.R*T*rho
        propiedades["v"] = 1./rho
        propiedades["h"] = self.R*T*(1+tau*(fiot+firt)+delta*fird)
        propiedades["s"] = self.R*(tau*(fiot+firt)-fio-fir)
        propiedades["cv"] = -self.R*tau**2*(fiott+firtt)
        propiedades["alfap"] = (1-delta*tau*firdt/(1+delta*fird))/T
        propiedades["betap"] = rho*(
            1+(delta*fird+delta**2*firdd)/(1+delta*fird))
        propiedades["B"] = res["B"]
        propiedades["C"] = res["C"]
#        dbt=-phi11/rho/t
#        propiedades["cps"] = propiedades["cv"] Add cps from Argon pag.27

        return propiedades

    def _prop0(self, rho, T):
        """Ideal gas properties"""
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        delta = rho/rhoc
        tau = Tc/T
        ideal = self._phi0(tau, delta)
        fio = ideal["fio"]
        fiot = ideal["fiot"]
        fiott = ideal["fiott"]

        propiedades = _fase()
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

        if "ao_exp2" in Fi0:
            for n, g, C in zip(Fi0["ao_exp2"], Fi0["titao2"], Fi0["sum2"]):
                fio += n*log(C+exp(g*tau))
                fiot += n*g/(C*exp(-g*tau)+1)
                fiott += C*n*g**2*exp(-g*tau)/(C*exp(-g*tau)+1)**2

        prop = {}
        prop["fio"] = fio
        prop["fiot"] = fiot
        prop["fiott"] = fiott
        prop["fiod"] = fiod
        prop["fiodd"] = fiodd
        prop["fiodt"] = fiodt
        return prop

    def _phir(self, tau, delta):
        delta_0 = 1e-200

        fir = fird = firdd = firt = firtt = firdt = B = C = 0

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

            fir += nr4[i]*Delta**b[i]*delta*F
            fird += nr4[i]*(Delta**b[i]*(F+delta*Fd)+DeltaBd*delta*F)
            firdd += nr4[i]*(Delta**b[i]*(2*Fd+delta*Fdd)+2*DeltaBd*(
                F+delta*Fd)+DeltaBdd*delta*F)
            firt += nr4[i]*delta*(DeltaBt*F+Delta**b[i]*Ft)
            firtt += nr4[i]*delta*(DeltaBtt*F+2*DeltaBt*Ft+Delta**b[i]*Ftt)
            firdt += nr4[i]*(Delta**b[i]*(Ft+delta*Fdt)+delta*DeltaBd*Ft +
                             DeltaBt*(F+delta*Fd)+DeltaBdt*delta*F)

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

        prop = {}
        prop["fir"] = fir
        prop["firt"] = firt
        prop["firtt"] = firtt
        prop["fird"] = fird
        prop["firdd"] = firdd
        prop["firdt"] = firdt
        prop["B"] = B
        prop["C"] = C
        return prop

    def _derivDimensional(self, rho, T):
        """Return the dimensional form or Helmholtz free energy derivatives
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 7,
        http://www.iapws.org/relguide/SeaAir.html
        """
        if not rho:
            prop = {}
            prop["fir"] = 0
            prop["firt"] = 0
            prop["fird"] = 0
            prop["firtt"] = 0
            prop["firdt"] = 0
            prop["firdd"] = 0
            return prop

        R = self._constants.get("R")/self._constants.get("M", self.M)
        rhoc = self._constants.get("rhoref", self.rhoc)
        Tc = self._constants.get("Tref", self.Tc)
        delta = rho/rhoc
        tau = Tc/T

        ideal = self._phi0(tau, delta)
        fio = ideal["fio"]
        fiot = ideal["fiot"]
        fiott = ideal["fiott"]
        fiod = ideal["fiod"]
        fiodd = ideal["fiodd"]

        res = self._phir(tau, delta)
        fir = res["fir"]
        firt = res["firt"]
        firtt = res["firtt"]
        fird = res["fird"]
        firdd = res["firdd"]
        firdt = res["firdt"]

        prop = {}
        prop["fir"] = R*T*(fio+fir)
        prop["firt"] = R*(fio+fir-(fiot+firt)*tau)
        prop["fird"] = R*T/rhoc*(fiod+fird)
        prop["firtt"] = R*tau**2/T*(fiott+firtt)
        prop["firdt"] = R/rhoc*(fiod+fird-firdt*tau)
        prop["firdd"] = R*T/rhoc**2*(fiodd+firdd)
        return prop

    def _surface(self, T):
        """Generic equation for the surface tension

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        sigma : float
            Surface tension [N/m]

        Notes
        -----
        Need a _surf dict in the derived class with the parameters keys:
            sigma: coefficient
            exp: exponent
        """
        tau = 1-T/self.Tc
        sigma = 0
        for n, t in zip(self._surf["sigma"], self._surf["exp"]):
            sigma += n*tau**t
        return sigma

    @classmethod
    def _Vapor_Pressure(cls, T):
        """Auxiliary equation for the vapour pressure

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        Pv : float
            Vapour pressure [Pa]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.1
        """
        Tita = 1-T/cls.Tc
        suma = 0
        for n, x in zip(cls._Pv["ao"], cls._Pv["exp"]):
            suma += n*Tita**x
        Pr = exp(cls.Tc/T*suma)
        Pv = Pr*cls.Pc
        return Pv

    @classmethod
    def _Liquid_Density(cls, T):
        """Auxiliary equation for the density or saturated liquid

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        rho : float
            Saturated liquid density [kg/m³]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.2
        """
        eq = cls._rhoL["eq"]
        Tita = 1-T/cls.Tc
        if eq == 2:
            Tita = Tita**(1./3)
        suma = 0
        for n, x in zip(cls._rhoL["ao"], cls._rhoL["exp"]):
            suma += n*Tita**x
        Pr = suma+1
        rho = Pr*cls.rhoc
        return rho

    @classmethod
    def _Vapor_Density(cls, T):
        """Auxiliary equation for the density or saturated vapor

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        rho : float
            Saturated vapor density [kg/m³]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.3
        """
        eq = cls._rhoG["eq"]
        Tita = 1-T/cls.Tc
        if eq == 4:
            Tita = Tita**(1./3)
        suma = 0
        for n, x in zip(cls._rhoG["ao"], cls._rhoG["exp"]):
            suma += n*Tita**x
        Pr = exp(suma)
        rho = Pr*cls.rhoc
        return rho

    @classmethod
    def _dPdT_sat(cls, T):
        """Auxiliary equation for the dP/dT along saturation line

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        dPdT : float
            dPdT [MPa/K]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, derived from Eq.1
        """
        Tita = 1-T/cls.Tc
        suma1 = 0
        suma2 = 0
        for n, x in zip(cls._Pv["ao"], cls._Pv["exp"]):
            suma1 -= n*x*Tita**(x-1)/cls.Tc
            suma2 += n*Tita**x
        Pr = (cls.Tc*suma1/T-cls.Tc/T**2*suma2)*exp(cls.Tc/T*suma2)
        dPdT = Pr*cls.Pc
        return dPdT


class IAPWS95(MEoS):
    """Implementation of IAPWS Formulation 1995 for ordinary water substance,
    (revised release of 2016), see MEoS __doc__

    Examples
    --------
    >>> water=IAPWS95(T=300, rho=996.5560)
    >>> water.P, water.cv, water.w, water.s
    0.0992418350 4.13018112 1501.51914 0.393062643

    >>> water=IAPWS95(T=500, rho=0.435)
    >>> water.P, water.cv, water.w, water.s
    0.0999679423 1.50817541 548.31425 7.944882714

    >>> water=IAPWS95(T=900., P=700)
    >>> water.rho, water.cv, water.w, water.s
    870.7690 2.66422350 2019.33608 4.17223802

    >>> water=IAPWS95(T=300., P=0.1)
    >>> water.P, water.rho, water.h, water.s, water.cp, water.w, water.virialB
    0.10000 996.56 112.65 0.39306 4.1806 1501.5 -0.066682

    >>> water=IAPWS95(T=500., P=0.1)
    >>> water.P, water.rho, water.h, water.s, water.cp, water.w, water.virialB
    0.10000 0.43514 2928.6 7.9447 1.9813 548.31 -0.0094137

    >>> water=IAPWS95(T=450., x=0.5)
    >>> water.T, water.P, water.rho, water.h, water.s, water.virialB
    450.00 0.93220 9.5723 1761.8 4.3589 -0.013028

    >>> water=IAPWS95(P=1.5, rho=1000.)
    >>> water.T, water.rho, water.h, water.s, water.cp, water.w, water.virialB
    286.44 1000.0 57.253 0.19931 4.1855 1462.1 -0.085566

    >>> water=IAPWS95(h=3000, s=8.)
    >>> water.T, water.P, water.h, water.s, water.cp, water.w, water.virialB
    536.24 0.11970 3000.0 8.0000 1.9984 567.04 -0.0076606

    >>> water=IAPWS95(h=150, s=0.4)
    >>> water.T, water.P, water.rho, water.h, water.s, water.cp, water.w
    301.27 35.50549 1011.48 150.00 0.40000 4.0932 1564.1

    >>> water=IAPWS95(T=450., rho=300)
    >>> water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB
    450.00 0.93220 300.00 770.82 2.1568 0.010693 -0.013028

    >>> water=IAPWS95(rho=300., P=0.1)
    >>> water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB
    372.76 0.10000 300.00 420.56 1.3110 0.0013528 -0.025144

    >>> water=IAPWS95(h=1500., P=0.1)
    >>> water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB
    372.76 0.10000 1.2303 1500.0 4.2068 0.47952 -0.025144

    >>> water=IAPWS95(s=5., P=3.5)
    >>> water.T, water.P, water.rho, water.h, water.s, water.x, water.virialB
    515.71 3.5000 25.912 2222.8 5.0000 0.66921 -0.0085877

    >>> water=IAPWS95(T=500., u=900)
    >>> water.P, water.rho, water.u, water.h, water.s, water.cp, water.w
    108.21 903.62 900.00 1019.8 2.4271 4.1751 1576.0

    >>> water=IAPWS95(P=0.3, u=1550.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    406.67 0.30000 3.3029 1550.0 1640.8 4.3260 0.49893

    >>> water=IAPWS95(rho=300, h=1000.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    494.92 2.3991 300.00 992.00 1000.0 2.6315 0.026071

    >>> water=IAPWS95(rho=30, s=8.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.cp
    1562.42 21.671 30.000 4628.5 5350.9 8.0000 2.7190

    >>> water=IAPWS95(rho=30, s=4.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    495.00 2.4029 30.000 1597.3 1677.4 4.0000 0.39218

    >>> water=IAPWS95(rho=300, u=1000.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    496.44 2.4691 300.000 1000.0 1008.2 2.6476 0.02680

    >>> water=IAPWS95(s=3., h=1000.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    345.73 0.034850 0.73526 952.60 1000.0 3.0000 0.29920

    >>> water=IAPWS95(u=995., h=1000.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    501.89 2.7329 546.58 995.00 1000.0 2.6298 0.00866

    >>> water=IAPWS95(u=1000., s=3.)
    >>> water.T, water.P, water.rho, water.u, water.h, water.s, water.x
    371.24 0.094712 1.99072 1000.00 1047.6 3.0000 0.28144

    References
    ----------
    IAPWS, Revised Release on the IAPWS Formulation 1995 for the Thermodynamic
    Properties of Ordinary Water Substance for General and Scientific Use,
    September 2016, http://www.iapws.org/relguide/IAPWS-95.html
    IAPWS, Revised Supplementary Release on Saturation Properties of Ordinary
    Water Substance September 1992, http://www.iapws.org/relguide/Supp-sat.html
    IAPWS, Guideline on a Low-Temperature Extension of the IAPWS-95 Formulation
    for Water Vapor, http://www.iapws.org/relguide/LowT.html
    IAPWS, Revised Advisory Note No. 3: Thermodynamic Derivatives from IAPWS
    Formulations, http://www.iapws.org/relguide/Advise3.pdf
    """
    name = "water"
    CASNumber = "7732-18-5"
    formula = "H2O"
    synonym = "R-718"
    Tc = Tc
    rhoc = rhoc
    Pc = Pc
    M = M
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

    _Pv = {
        "ao": [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719,
               1.80122502],
        "exp": [1, 1.5, 3, 3.5, 4, 7.5]}
    _rhoL = {
        "eq": 2,
        "ao": [1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352,
               -6.74694450e5],
        "exp": [1, 2, 5, 16, 43, 110]}
    _rhoG = {
        "eq": 4,
        "ao": [-2.0315024, -2.6830294, -5.38626492, -17.2991605, -44.7586581,
               -63.9201063],
        "exp": [1, 2, 4, 9, 18.5, 35.5]}

    def _phi0(self, tau, delta):
        """Low temperature extension of the IAPWS-95"""
        prop = MEoS._phi0(self, tau, delta)

        T = self.Tc/tau
        if 50 <= T < 130:
            fex, fext, fextt = self._phiex(T)
            prop["fio"] += fex
            prop["fiot"] += fext
            prop["fiott"] += fextt
        return prop

    def _phiex(self, T):
        """Low temperature extension"""
        tau = self.Tc/T
        E = 0.278296458178592
        ep = self.Tc/130
        fex = E*(-1/2/tau-3/ep**2*(tau+ep)*log(tau/ep)-9/2/ep+9*tau/2/ep**2 +
                 tau**2/2/ep**3)
        fext = E*(1/2/tau**2-3/tau/ep-3/ep**2*log(tau/ep)+3/2/ep**2+tau/ep**3)
        fextt = E*(-1/tau+1/ep)**3
        return fex, fext, fextt

    @classmethod
    def _alfa_sat(cls, T):
        """Auxiliary equation for the alfa coefficient for calculate the
        enthalpy along the saturation line

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        alfa : float
            alfa coefficient [kJ/kg]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.4
        """
        di = [-1135.905627715, -5.65134998e-8, 2690.66631, 127.287297,
              -135.003439, 0.981825814]
        expi = [0, -19, 1, 4.5, 5, 54.5]
        Tita = T/cls.Tc
        alfa = 0
        for d, x in zip(di, expi):
            alfa += d*Tita**x
        return alfa

    @classmethod
    def _phi_sat(cls, T):
        """Auxiliary equation for the phi coefficient for calculate the
        entropy along the saturation line

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        phi : float
            phi coefficient [kJ/kgK]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.5
        """
        di = [2319.5246, -5.65134998e-8*19/20, 2690.66631, 127.287297*9/7,
              -135.003439*5/4, 0.981825814*109/107]
        expi = [0, -20, None, 3.5, 4, 53.5]
        Tita = T/cls.Tc
        suma = 0
        for d, x in zip(di, expi):
            if x is None:
                suma += d*log(Tita)
            else:
                suma += d*Tita**x
        phi = suma/cls.Tc
        return phi

    @classmethod
    def _Liquid_Enthalpy(cls, T):
        """Auxiliary equation for the specific enthalpy for saturated liquid

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        h : float
            Saturated liquid enthalpy [kJ/kg]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.6
        """
        alfa = cls._alfa_sat(T)
        rho = cls._Liquid_Density(T)
        dpdT = cls._dPdT_sat(T)
        h = alfa+T/rho*dpdT*1000
        return h

    @classmethod
    def _Vapor_Enthalpy(cls, T):
        """Auxiliary equation for the specific enthalpy for saturated vapor

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        h : float
            Saturated vapor enthalpy [kJ/kg]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.7
        """
        alfa = cls._alfa_sat(T)
        rho = cls._Vapor_Density(T)
        dpdT = cls._dPdT_sat(T)
        h = alfa+T/rho*dpdT*1000
        return h

    @classmethod
    def _Liquid_Entropy(cls, T):
        """Auxiliary equation for the specific entropy for saturated liquid

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        s : float
            Saturated liquid entropy [kJ/kgK]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.8
        """
        phi = cls._phi_sat(T)
        rho = cls._Liquid_Density(T)
        dpdT = cls._dPdT_sat(T)
        s = phi+dpdT/rho*1000
        return s

    @classmethod
    def _Vapor_Entropy(cls, T):
        """Auxiliary equation for the specific entropy for saturated vapor

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        s : float
            Saturated liquid entropy [kJ/kgK]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.9
        """
        phi = cls._phi_sat(T)
        rho = cls._Vapor_Density(T)
        dpdT = cls._dPdT_sat(T)
        s = phi+dpdT/rho*1000
        return s

    def _visco(self, rho, T, fase):
        ref = IAPWS95()
        st = ref._Helmholtz(rho, 1.5*Tc)
        delta = rho/rhoc
        drho = 1e3/self.R/1.5/Tc/(1+2*delta*st["fird"]+delta**2*st["firdd"])
        return _Viscosity(rho, T, fase, drho)

    def _thermo(self, rho, T, fase):
        ref = IAPWS95()
        st = ref._Helmholtz(rho, 1.5*Tc)
        delta = rho/rhoc
        drho = 1e3/self.R/1.5/Tc/(1+2*delta*st["fird"]+delta**2*st["firdd"])
        return _ThCond(rho, T, fase, drho)

    def _surface(self, T):
        s = _Tension(T)
        return s


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


class IAPWS95_Px(IAPWS95):
    """Derivated class for direct P and v input"""
    def __init__(self, P, x):
        IAPWS95.__init__(self, P=P, x=x)


class IAPWS95_Tx(IAPWS95):
    """Derivated class for direct T and x input"""
    def __init__(self, T, x):
        IAPWS95.__init__(self, T=T, x=x)


class D2O(MEoS):
    """Implementation of IAPWS Formulation 1984 for heavy water substance,
    (revised release of 2005), see MEoS __doc__

    Examples
    --------
    >>> hwater=D2O(T=300, rho=996.5560)
    >>> hwater.P, hwater.Liquid.cv, hwater.Liquid.w
    0.0030675947 4.21191157 5332.04871

    References
    ----------
    IAPWS, Revised Release on the IAPS Formulation 1984 for the Thermodynamic
    Properties of Heavy Water Substance,
    http://www.iapws.org/relguide/D2O-2005.pdf
    IAPWS, Revised Advisory Note No. 3: Thermodynamic Derivatives from IAPWS
    Formulations, http://www.iapws.org/relguide/Advise3.pdf
    """
    name = "heavy water"
    CASNumber = "7789-20-0"
    formula = "D2O"
    synonym = "deuterium oxide"
    Tc = Tc_D2O
    rhoc = rhoc_D2O
    Pc = Pc_D2O
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

    _Pv = {
        "ao": [-0.80236e1, 0.23957e1, -0.42639e2, 0.99569e2, -0.62135e2],
        "exp": [1.0, 1.5, 2.75, 3.0, 3.2]}
    _rhoL = {
        "eq": 1,
        "ao": [0.26406e1, 0.97090e1, -0.18058e2, 0.87202e1, -0.74487e1],
        "exp": [0.3678, 1.9, 2.2, 2.63, 7.3]}
    _rhoG = {
        "eq": 3,
        "ao": [-0.37651e1, -0.38673e2, 0.73024e2, -0.13251e3, 0.75235e2,
               -0.70412e2],
        "exp": [0.409, 1.766, 2.24, 3.04, 3.42, 6.9]}

    def _visco(self, rho, T, fase):
        return _D2O_Viscosity(rho, T)

    def _thermo(self, rho, T, fase):
        return _D2O_ThCond(rho, T)

    def _surface(self, T):
        s = _D2O_Tension(T)
        return s
