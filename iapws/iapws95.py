#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Implemented multiparameter equation of state as a Helmholtz free energy:

    * :class:`MEoS`: Base class of multiparameter equation of state
    * :class:`IAPWS95`: 2016 revision of 1995 formulation for ordinaty water
    * :class:`D2O`: 2017 formulation for heavy water.
"""

from __future__ import division
import os
import platform
import warnings
from math import exp, log
from typing import Tuple, Dict, Optional, List, Callable, Any

from scipy.optimize import fsolve

from .iapws97 import _TSat_P, IAPWS97
from ._iapws import _global_M, Tc, Pc, rhoc, Tc_D2O, Pc_D2O, rhoc_D2O
from ._iapws import _Viscosity, _ThCond, _Dielectric, _Refractive, _Tension
from ._iapws import _D2O_Viscosity, _D2O_ThCond, _D2O_Tension
from ._utils import _fase, getphase, deriv_H
from .helmholtz import ResidualContribution, HelmholtzDerivatives


class MEoSProperties(object):
    """The properties required to fill a MEoS phase."""

    def __init__(self, rho: float, P: float, h: float, s: float, cv:
                 float, alfap: float, betap: float, fir: float, fird:
                 float, firdd: float, delta: float) -> None:
        self.rho = rho
        self.P = P
        self.h = h
        self.s = s
        self.cv = cv
        self.alfap = alfap
        self.betap = betap
        self.fir = fir
        self.fird = fird
        self.firdd = firdd
        self.delta = delta


class MEoS(_fase):
    r"""
    General implementation of multiparameter equation of state. From this
    derived all child class specified per individual compounds

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
    rho : float
        Density, [kg/m³]
    v : float
        Specific volume, [m³/kg]
    h : float
        Specific enthalpy, [kJ/kg]
    s : float
        Specific entropy, [kJ/kgK]
    u : float
        Specific internal energy, [kJ/kg]
    x : float
        Vapor quality, [-]

    l : float, optional
        Wavelength of light, for refractive index, [nm]
    rho0 : float, optional
        Initial value of density, to improve iteration, [kg/m³]
    T0 : float, optional
        Initial value of temperature, to improve iteration, [K]
    x0 : Initial value of vapor quality, necessary in bad input pair definition
        where there are two valid solution (T-h, T-s)

    Notes
    -----
    * It needs two incoming properties of T, P, rho, h, s, u.
    * v as a alternate input parameter to rho
    * T-x, P-x, preferred input pair to specified a point in two phases region

    The calculated instance has the following properties:

        * P: Pressure, [MPa]
        * T: Temperature, [K]
        * x: Vapor quality, [-]
        * g: Specific Gibbs free energy, [kJ/kg]
        * a: Specific Helmholtz free energy, [kJ/kg]
        * v: Specific volume, [m³/kg]
        * r: Density, [kg/m³]
        * h: Specific enthalpy, [kJ/kg]
        * u: Specific internal energy, [kJ/kg]
        * s: Specific entropy, [kJ/kg·K]
        * cp: Specific isobaric heat capacity, [kJ/kg·K]
        * cv: Specific isochoric heat capacity, [kJ/kg·K]
        * cp_cv: Heat capacity ratio, [-]
        * Z: Compression factor, [-]
        * fi: Fugacity coefficient, [-]
        * f: Fugacity, [MPa]
        * gamma: Isoentropic exponent, [-]

        * alfav: Isobaric cubic expansion coefficient, [1/K]
        * kappa: Isothermal compressibility, [1/MPa]
        * kappas: Adiabatic compresibility, [1/MPa]
        * alfap: Relative pressure coefficient, [1/K]
        * betap: Isothermal stress coefficient, [kg/m³]
        * joule: Joule-Thomson coefficient, [K/MPa]

        * betas: Isoentropic temperature-pressure coefficient, [-]
        * Gruneisen: Gruneisen parameter, [-]
        * virialB: Second virial coefficient, [m³/kg]
        * virialC: Third virial coefficient, [m⁶/kg²]
        * dpdT_rho: Derivatives, dp/dT at constant rho, [MPa/K]
        * dpdrho_T: Derivatives, dp/drho at constant T, [MPa·m³/kg]
        * drhodT_P: Derivatives, drho/dT at constant P, [kg/m³·K]
        * drhodP_T: Derivatives, drho/dP at constant T, [kg/m³·MPa]
        * dhdT_rho: Derivatives, dh/dT at constant rho, [kJ/kg·K]
        * dhdP_T: Isothermal throttling coefficient, [kJ/kg·MPa]
        * dhdT_P: Derivatives, dh/dT at constant P, [kJ/kg·K]
        * dhdrho_T: Derivatives, dh/drho at constant T, [kJ·m³/kg²]
        * dhdrho_P: Derivatives, dh/drho at constant P, [kJ·m³/kg²]
        * dhdP_rho: Derivatives, dh/dP at constant rho, [kJ/kg·MPa]
        * kt: Isothermal Expansion Coefficient, [-]
        * ks: Adiabatic Compressibility, [1/MPa]
        * Ks: Adiabatic bulk modulus, [MPa]
        * Kt: Isothermal bulk modulus, [MPa]

        * v0: Ideal specific volume, [m³/kg]
        * rho0: Ideal gas density, [kg/m³]
        * u0: Ideal specific internal energy, [kJ/kg]
        * h0: Ideal specific enthalpy, [kJ/kg]
        * s0: Ideal specific entropy, [kJ/kg·K]
        * a0: Ideal specific Helmholtz free energy, [kJ/kg]
        * g0: Ideal specific Gibbs free energy, [kJ/kg]
        * cp0: Ideal specific isobaric heat capacity, [kJ/kg·K]
        * cv0: Ideal specific isochoric heat capacity, [kJ/kg·K]
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

        * Z_rho: :math:`(Z-1)/\rho`, [m³/kg]
        * IntP: Internal pressure, [MPa]
        * invT: Negative reciprocal temperature, [1/K]
        * hInput: Specific heat input, [kJ/kg]
    """

    CP = None
    # The lists vary from 4 to 6 terms, which is why they're not
    # tuples of explicit length.  The equation numbers are used for
    # switching implementations based on class.
    _Pv_ao: List[float]
    _Pv_exp: List[float]
    _rhoG_eq: int
    _rhoG_ao: List[float]
    _rhoG_exp: List[float]
    _rhoL_eq: int
    _rhoL_ao: List[float]
    _rhoL_exp: List[float]
    # Defined in derived classes NH3 and Air.
    _surf: Dict[str, List[float]]
    Fi0: Dict[str, List[float]]
    # Defined by derived classes
    _constant_rhoc: float
    _constant_Tref: float

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

    # These are used in MeOS but are NOT actually defined by the
    # class.  They have to be defined by one of the subclasses which
    # is a little messy.  By statically typing them here, we at least
    # let mypy know that they're supposed to be set...
    rhoc: float
    Tc: float
    Pc: float
    Tt: float
    _constant_R: float
    _constants: Dict[str, List[float]]
    residual: ResidualContribution
    M: float
    # The name of the substance.
    # Set in NH3, Air, IAPWS95, D2O, and IAPWS97.
    name: str

    def __init__(self, **kwargs):
        """Constructor, define common constant and initinialice kwargs"""
        super().__init__()

        # These class variables must be defined by the subclass.
        assert(isinstance(self._Pv_ao, list))
        assert(isinstance(self._Pv_exp, list))
        assert(isinstance(self._rhoL_eq, int))
        assert(isinstance(self._rhoL_ao, list))
        assert(isinstance(self._rhoL_exp, list))
        assert(isinstance(self._rhoG_eq, int))
        assert(isinstance(self._rhoG_ao, list))
        assert(isinstance(self._rhoG_exp, list))
        assert(isinstance(self.M, float))

        self.sigma: Optional[float] = None
        self.Hvap: Optional[float] = None
        self.Svap: Optional[float] = None

        self.R = self.__class__._constant_R/self.M
        self.Zc = self.Pc/self._constant_rhoc/self.R/self._constant_Tref
        self.kwargs = MEoS.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs) -> None:
        """Make instance callable to can add input parameter one to one"""
        # Alternative rho input
        if "rhom" in kwargs:
            kwargs["rho"] = kwargs["rhom"]*self.M
            del kwargs["rhom"]
        elif kwargs.get("v", 0):
            kwargs["rho"] = 1./kwargs["v"]
            del kwargs["v"]
        elif kwargs.get("vm", 0):
            kwargs["rho"] = self.M/kwargs["vm"]
            del kwargs["vm"]
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
    def calculable(self) -> bool:
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

    def calculo(self) -> None:
        """Calculate procedure"""
        T: float = self.kwargs["T"]  # type: ignore
        rho: float = self.kwargs["rho"]  # type: ignore
        P: float = self.kwargs["P"]  # type: ignore
        s: float = self.kwargs["s"]  # type: ignore
        h: float = self.kwargs["h"]  # type: ignore
        u: float = self.kwargs["u"]  # type: ignore
        x: float = self.kwargs["x"]  # type: ignore

        # Initial values
        T0 = self.kwargs["T0"]
        rho0 = self.kwargs["rho0"]

        if T0 or rho0:
            if T0 is not None:
                To = float(T0)
            if rho0 is not None:
                rhoo = float(rho0)
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

        self.R = self._constant_R/self.M
        rhoc = self._constant_rhoc
        Tc = self._constant_Tref

        if self._mode not in ("Tx", "Px"):
            # Method with iteration necessary to get x
            if self._mode == "TP":
                try:
                    if self.name != "water":
                        raise NotImplementedError
                    st0 = IAPWS97(**self.kwargs)
                    rhoo = st0.rho
                except NotImplementedError:
                    if rho0:
                        rhoo = rho0
                    elif T < self._constant_Tref and P < self.Pc and \
                            self._Vapor_Pressure(T) < P:
                        rhoo = self._Liquid_Density(T)
                    elif T < self._constant_Tref and P < self.Pc:
                        rhoo = self._Vapor_Density(T)
                    else:
                        rhoo = self._constant_rhoc*3

                def rho_func(rho: float) -> float:
                    delta = rho/rhoc
                    tau = Tc/T

                    fird = self.residual.phird(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    return Po-P*1000

                rho = float(fsolve(rho_func, rhoo)[0])

                # Calculate quality
                if T > self._constant_Tref:
                    x = 1
                else:
                    Ps = self._Vapor_Pressure(T)
                    if Ps*0.95 < P < Ps*1.05:
                        rhol, rhov, Ps = self._saturation(T)
                        Ps *= 1e-3

                    if Ps > P:
                        x = 1
                    else:
                        x = 0

            elif self._mode == "Th":
                tau = Tc/T
                ideal = self._phi0(tau, 1)
                fiot = ideal.fit

                def rho_func(rho: float) -> float:
                    delta = rho/rhoc
                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    ho = self.R*T*(1+tau*(fiot+firt)+delta*fird)
                    return ho-h

                if T >= self._constant_Tref:
                    rhoo = self._constant_rhoc
                    rho = float(fsolve(rho_func, rhoo)[0])
                else:
                    x0 = self.kwargs["x0"]
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    deltaL = rhol/rhoc
                    deltaG = rhov/rhoc

                    firdL = self.residual.phird(tau, deltaL)
                    firtL = self.residual.phirt(tau, deltaL)
                    firdG = self.residual.phird(tau, deltaG)
                    firtG = self.residual.phirt(tau, deltaG)
                    hl = self.R*T*(1+tau*(fiot+firtL)+deltaL*firdL)
                    hv = self.R*T*(1+tau*(fiot+firtG)+deltaG*firdG)
                    if x0 not in (0, 1) and hl <= h <= hv:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        hv = vapor.h
                        hl = liquido.h
                        x = (h-hl)/(hv-hl)
                        rho = 1/(x/rhov+(1-x)/rhol)
                        P = Ps/1000
                    else:
                        if h > hv:
                            rhoo = rhov
                        else:
                            rhoo = rhol
                        rho = float(fsolve(rho_func, rhoo)[0])

            elif self._mode == "Ts":
                tau = Tc/T

                def rho_func(rho: float) -> float:
                    if rho < 0:
                        rho = 1e-20
                    delta = rho/rhoc

                    ideal = self._phi0(tau, delta)
                    fir = self.residual.phir(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    so = self.R*(tau*(ideal.fit+firt)-ideal.fi-fir)
                    return so-s

                if T >= self._constant_Tref:
                    rhoo = self._constant_rhoc
                    rho = float(fsolve(rho_func, rhoo)[0])
                else:
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    deltaL = rhol/rhoc
                    deltaG = rhov/rhoc

                    idealL = self._phi0(tau, deltaL)
                    idealG = self._phi0(tau, deltaG)

                    firL = self.residual.phir(tau, deltaL)
                    firtL = self.residual.phirt(tau, deltaL)
                    sl = self.R*(tau*(idealL.fit+firtL)-idealL.fi-firL)
                    firG = self.residual.phir(tau, deltaG)
                    firtG = self.residual.phirt(tau, deltaG)
                    sv = self.R*(tau*(idealL.fit+firtG)-idealG.fi-firG)

                    if sl <= s <= sv:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        sv = vapor.s
                        sl = liquido.s
                        x = (s-sl)/(sv-sl)
                        rho = 1/(x/rhov+(1-x)/rhol)
                        P = Ps/1000
                    else:
                        if s > sv:
                            rhoo = rhov
                        else:
                            rhoo = rhol
                        rho = float(fsolve(rho_func, rhoo)[0])

            elif self._mode == "Tu":
                tau = Tc/T
                ideal = self._phi0(tau, 1)

                def rho_func(rho: float) -> float:
                    delta = rho/rhoc

                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)

                    return ho-Po/rho-u

                if T >= self._constant_Tref:
                    rhoo = self._constant_rhoc
                    rho = float(fsolve(rho_func, rhoo)[0])
                else:
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    deltaL = rhol/rhoc
                    deltaG = rhov/rhoc

                    firdL = self.residual.phird(tau, deltaL)
                    firtL = self.residual.phirt(tau, deltaL)
                    firdG = self.residual.phird(tau, deltaG)
                    firtG = self.residual.phirt(tau, deltaG)
                    PoL = (1+deltaL*firdL)*self.R*T*rhol
                    PoG = (1+deltaG*firdG)*self.R*T*rhov
                    hoL = self.R*T*(1+tau*(ideal.fit+firtL)+deltaL*firdL)
                    hoG = self.R*T*(1+tau*(ideal.fit+firtG)+deltaG*firdG)

                    uv = hoG-PoG/rhov
                    ul = hoL-PoL/rhol
                    if ul <= u <= uv:
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        uv = vapor.h-vapor.P/rhov
                        ul = liquido.h-liquido.P/rhol
                        x = (u-ul)/(uv-ul)
                        rho = 1/(x/rhov-(1-x)/rhol)
                        P = Ps/1000
                    else:
                        if u > uv:
                            rhoo = rhov
                        else:
                            rhoo = rhol
                        rho = float(fsolve(rho_func, rhoo)[0])

            elif self._mode == "Prho":
                delta = rho/rhoc

                def t_func(T: float) -> float:
                    tau = Tc/T

                    fird = self.residual.phird(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    return Po-P*1000

                T = float(fsolve(t_func, To)[0])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:

                    def f3(parr: Tuple[float, float, float]) -> Tuple[float, float, float]:
                        T, rhol, rhog = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        try:
                            logrho = log(rhol/rhog)
                        except ValueError:
                            logrho = float('nan')
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(K+logrho)
                        return (Jl-Jv, Jl*(1/rhog-1/rhol)-logrho-K, Ps - P*1000)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f3, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = tuple(map(float, sol[0]))
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "Ph":
                def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                    rho, T = parr
                    delta = rho/rhoc
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)
                    return Po-P*1000, ho-h

                rho, T = tuple(map(float, fsolve(f2, [rhoo, To])))
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == rhoo or rhov <= rho <= rhol:

                    def f4(parr: Tuple[float, float, float, float]) -> Tuple[
                            float, float, float, float]:
                        T, rhol, rhog, x = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        ideal = self._phi0(tau, deltaL)

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(ideal.fit+firtL)+deltaL*firdL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(ideal.fit+firtG)+deltaG*firdG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(K+log(rhol/rhog))

                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                hoL*(1-x)+hoG*x - h,
                                Ps - P*1000)

                    for to in [To, 300, 400, 500, 600]:
                        rLo = self._Liquid_Density(to)
                        rGo = self._Vapor_Density(to)
                        sol = fsolve(f4, [to, rLo, rGo, 0.5], full_output=True)
                        T, rhoL, rhoG, x = tuple(map(float, sol[0]))
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "Ps":
                try:
                    x0 = st0.x
                except NameError:
                    x0 = None

                if x0 is None or x0 == 0 or x0 == 1:
                    def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                        rho, T = parr
                        delta = rho/rhoc
                        tau = Tc/T

                        ideal = self._phi0(tau, delta)
                        fird = self.residual.phird(tau, delta)
                        fir = self.residual.phir(tau, delta)
                        firt = self.residual.phirt(tau, delta)
                        Po = (1+delta*fird)*self.R*T*rho
                        so = self.R*(tau*(ideal.fit+firt)-ideal.fi-fir)
                        return Po-P*1000, so-s

                    rho, T = tuple(map(float, fsolve(f2, [rhoo, To])))

                else:
                    def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                        rho, T = parr
                        rhol, rhov, Ps = self._saturation(T)
                        vapor = self._Helmholtz(rhov, T)
                        liquido = self._Helmholtz(rhol, T)
                        x = (1./rho-1/rhol)/(1/rhov-1/rhol)
                        return Ps-P*1000, vapor.s*x+liquido.s*(1-x)-s
                    rho, T = tuple(map(float, fsolve(f2, [2., 500.])))
                    rhol, rhov, Ps = self._saturation(T)
                    vapor = self._Helmholtz(rhov, T)
                    liquido = self._Helmholtz(rhol, T)
                    sv = vapor.s
                    sl = liquido.s
                    x = (s-sl)/(sv-sl)

            elif self._mode == "Pu":
                def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                    rho, T = parr
                    delta = rho/rhoc
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)
                    return ho-Po/rho-u, Po-P*1000

                sol = fsolve(f2, [rhoo, To], full_output=True)
                rho, T = tuple(map(float, sol[0]))
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rho == rhoo or sol[2] != 1:

                    def f4(parr: Tuple[float, float, float, float]) -> Tuple[
                            float, float, float, float]:
                        T, rhol, rhog, x = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        ideal = self._phi0(tau, deltaL)

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(ideal.fit+firtL)+deltaL*firdL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(ideal.fit+firtG)+deltaG*firdG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        try:
                            logrho = log(rhol/rhog)
                        except ValueError:
                            logrho = float('nan')
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(K+logrho)
                        vu = hoG-Ps/rhog
                        lu = hoL-Ps/rhol
                        return (Jl-Jv, Jl*(1/rhog-1/rhol)-logrho-K,
                                lu*(1-x)+vu*x - u, Ps - P*1000)

                    for to in [To, 300, 400, 500, 600]:
                        rLo = self._Liquid_Density(to)
                        rGo = self._Vapor_Density(to)
                        sol = fsolve(f4, [to, rLo, rGo, 0.5], full_output=True)
                        T, rhoL, rhoG, x = tuple(map(float, sol[0]))
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "rhoh":
                delta = rho/rhoc

                def t_func(T: float) -> float:
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)
                    return ho-h

                T = float(fsolve(t_func, To)[0])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:
                    def f3(parr: Tuple[float, float, float]) -> Tuple[float, float, float]:
                        T, rhol, rhog = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        ideal = self._phi0(tau, deltaL)
                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(ideal.fit+firtL)+deltaL*firdL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(ideal.fit+firtG)+deltaG*firdG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        x = (1./rho-1/rhol)/(1/rhog-1/rhol)
                        try:
                            logrho = log(rhol/rhog)
                        except ValueError:
                            logrho = float('nan')
                        return (Jl-Jv, Jl*(1/rhog-1/rhol)-logrho-K,
                                hoL*(1-x)+hoG*x - h)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f3, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = tuple(map(float, sol[0]))
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "rhos":
                delta = rho/rhoc

                def t_func(T: float) -> float:
                    tau = Tc/T
                    ideal = self._phi0(tau, delta)
                    fir = self.residual.phir(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    so = self.R*(tau*(ideal.fit+firt)-ideal.fi-fir)
                    return so-s

                T = float(fsolve(t_func, To)[0])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:

                    def f3(parr: Tuple[float, float, float]) -> Tuple[float, float, float]:
                        T, rhol, rhog = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        idealL = self._phi0(tau, deltaL)
                        idealG = self._phi0(tau, deltaG)

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        soL = self.R*(tau*(idealL.fit+firtL)-idealL.fi-firL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        soG = self.R*(tau*(idealL.fit+firtG)-idealG.fi-firG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        x = (1./rho-1/rhol)/(1/rhog-1/rhol)
                        try:
                            logrho = log(rhol/rhog)
                        except ValueError:
                            logrho = float('nan')
                        return (Jl-Jv, Jl*(1/rhog-1/rhol)-logrho-K,
                                soL*(1-x)+soG*x - s)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f3, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = tuple(map(float, sol[0]))
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "rhou":
                delta = rho/rhoc

                def t_func(T: float) -> float:
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)
                    return ho-Po/rho-u

                T = float(fsolve(t_func, To)[0])
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if T == To or rhov <= rho <= rhol:
                    def f3(parr: Tuple[float, float, float]) -> Tuple[float, float, float]:
                        T, rhol, rhog = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        ideal = self._phi0(tau, deltaL)
                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(ideal.fit+firtL)+deltaL*firdL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(ideal.fit+firtG)+deltaG*firdG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        x = (1./rho-1/rhol)/(1/rhog-1/rhol)
                        try:
                            logrho = log(rhol/rhog)
                        except ValueError:
                            logrho = float('nan')
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(K+logrho)
                        vu = hoG-Ps/rhog
                        lu = hoL-Ps/rhol
                        return (Jl-Jv, Jl*(1/rhog-1/rhol)-logrho-K,
                                lu*(1-x)+vu*x - u)

                    for to in [To, 300, 400, 500, 600]:
                        rhoLo = self._Liquid_Density(to)
                        rhoGo = self._Vapor_Density(to)
                        sol = fsolve(f3, [to, rhoLo, rhoGo], full_output=True)
                        T, rhoL, rhoG = tuple(map(float, sol[0]))
                        x = (1./rho-1/rhoL)/(1/rhoG-1/rhoL)
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "hs":
                def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                    rho, T = parr
                    delta = rho/rhoc
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    fir = self.residual.phir(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)
                    so = self.R*(tau*(ideal.fit+firt)-ideal.fi-fir)
                    return ho-h, so-s

                rho, T = tuple(map(float, fsolve(f2, [rhoo, To])))
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if rhov <= rho <= rhol:

                    def f4(parr: Tuple[float, float, float, float]) -> Tuple[
                            float, float, float, float]:
                        T, rhol, rhog, x = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        idealL = self._phi0(tau, deltaL)
                        idealG = self._phi0(tau, deltaG)

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(idealL.fit+firtL)+deltaL*firdL)
                        soL = self.R*(tau*(idealL.fit+firtL)-idealL.fi-firL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(idealL.fit+firtG)+deltaG*firdG)
                        soG = self.R*(tau*(idealL.fit+firtG)-idealG.fi-firG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG
                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                hoL*(1-x)+hoG*x - h,
                                soL*(1-x)+soG*x - s)

                    for to in [To, 300, 400, 500, 600]:
                        rLo = self._Liquid_Density(to)
                        rGo = self._Vapor_Density(to)
                        sol = fsolve(f4, [to, rLo, rGo, 0.5], full_output=True)
                        T, rhoL, rhoG, x = tuple(map(float, sol[0]))
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "hu":
                def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                    rho, T = parr
                    delta = rho/rhoc
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    Po = (1+delta*fird)*self.R*T*rho
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)

                    return ho-Po/rho-u, ho-h

                sol = fsolve(f2, [rhoo, To], full_output=True)
                rho, T = tuple(map(float, sol[0]))
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if sol[2] != 1 or rhov <= rho <= rhol:

                    def f4(parr: Tuple[float, float, float, float]) -> Tuple[
                            float, float, float, float]:
                        T, rhol, rhog, x = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        ideal = self._phi0(tau, deltaL)

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(ideal.fit+firtL)+deltaL*firdL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(ideal.fit+firtG)+deltaG*firdG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG

                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(K+log(rhol/rhog))
                        vu = hoG-Ps/rhog
                        lu = hoL-Ps/rhol

                        return (Jl-Jv,
                                Jl*(1/rhog-1/rhol)-log(rhol/rhog)-K,
                                hoL*(1-x)+hoG*x - h,
                                lu*(1-x)+vu*x - u)

                    for to in [To, 300, 400, 500, 600]:
                        rLo = self._Liquid_Density(to)
                        rGo = self._Vapor_Density(to)
                        sol = fsolve(f4, [to, rLo, rGo, 0.5], full_output=True)
                        T, rhoL, rhoG, x = tuple(map(float, sol[0]))
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "su":
                def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
                    rho, T = parr
                    delta = rho/rhoc
                    tau = Tc/T

                    ideal = self._phi0(tau, delta)
                    fird = self.residual.phird(tau, delta)
                    fir = self.residual.phir(tau, delta)
                    firt = self.residual.phirt(tau, delta)
                    ho = self.R*T*(1+tau*(ideal.fit+firt)+delta*fird)
                    so = self.R*(tau*(ideal.fit+firt)-ideal.fi-fir)
                    Po = (1+delta*fird)*self.R*T*rho
                    return ho-Po/rho-u, so-s

                sol = fsolve(f2, [rhoo, To], full_output=True)
                rho, T = tuple(map(float, sol[0]))
                rhol = self._Liquid_Density(T)
                rhov = self._Vapor_Density(T)
                if sol[2] != 1 or rhov <= rho <= rhol:

                    def f4(parr: Tuple[float, float, float, float]) -> Tuple[
                            float, float, float, float]:
                        T, rhol, rhog, x = parr
                        tau = Tc/T
                        deltaL = rhol/self._constant_rhoc
                        deltaG = rhog/self._constant_rhoc

                        idealL = self._phi0(tau, deltaL)
                        idealG = self._phi0(tau, deltaG)

                        firL = self.residual.phir(tau, deltaL)
                        firdL = self.residual.phird(tau, deltaL)
                        firtL = self.residual.phirt(tau, deltaL)
                        hoL = self.R*T*(1+tau*(idealL.fit+firtL)+deltaL*firdL)
                        soL = self.R*(tau*(idealL.fit+firtL)-idealL.fi-firL)
                        firG = self.residual.phir(tau, deltaG)
                        firdG = self.residual.phird(tau, deltaG)
                        firtG = self.residual.phirt(tau, deltaG)
                        hoG = self.R*T*(1+tau*(idealL.fit+firtG)+deltaG*firdG)
                        soG = self.R*(tau*(idealL.fit+firtG)-idealG.fi-firG)

                        Jl = rhol*(1+deltaL*firdL)
                        Jv = rhog*(1+deltaG*firdG)
                        K = firL-firG

                        try:
                            logrho = log(rhol/rhog)
                        except ValueError:
                            logrho = float('nan')
                        Ps = self.R*T*rhol*rhog/(rhol-rhog)*(K+logrho)
                        vu = hoG-Ps/rhog
                        lu = hoL-Ps/rhol

                        return (Jl-Jv, Jl*(1/rhog-1/rhol)-logrho-K,
                                soL*(1-x)+soG*x - s, lu*(1-x)+vu*x - u)

                    for to in [To, 300, 400, 500, 600]:
                        rLo = self._Liquid_Density(to)
                        rGo = self._Vapor_Density(to)
                        sol = fsolve(f4, [to, rLo, rGo, 0.5], full_output=True)
                        T, rhoL, rhoG, x = tuple(map(float, sol[0]))
                        if sol[2] == 1 and 0 <= x <= 1 and \
                                sum(abs(sol[1]["fvec"])) < 1e-5:
                            break

                    if sum(abs(sol[1]["fvec"])) > 1e-5:
                        raise(RuntimeError(sol[3]))

                    liquido = self._Helmholtz(rhoL, T)
                    vapor = self._Helmholtz(rhoG, T)
                    P = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(
                        liquido.fir-vapor.fir+log(rhoL/rhoG))/1000

            elif self._mode == "Trho":
                if T < self._constant_Tref:
                    rhov = self._Vapor_Density(T)
                    rhol = self._Liquid_Density(T)
                    if rhol > rho > rhov:
                        rhol, rhov, Ps = self._saturation(T)
                        if rhol > rho > rhov:
                            vapor = self._Helmholtz(rhov, T)
                            liquido = self._Helmholtz(rhol, T)
                            x = (1/rho-1/rhol)/(1/rhov-1/rhol)
                            P = Ps/1000

            rho = float(rho)
            T = float(T)
            propiedades = self._Helmholtz(rho, T)

            if T > self._constant_Tref:
                x = 1
            elif x is None:
                x = 0

            if not P:
                P = propiedades.P/1000.

        elif self._mode == "Tx":
            # Check input T in saturation range
            if self.Tt > T or self._constant_Tref < T or x > 1 or x < 0:
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
            def t_func(T: float) -> float:
                rhol = self._Liquid_Density(T)
                rhog = self._Vapor_Density(T)

                deltaL = rhol/self._constant_rhoc
                deltaG = rhog/self._constant_rhoc

                tau = Tc/T

                firL = self.residual.phir(tau, deltaL)
                firG = self.residual.phir(tau, deltaG)
                Ps = self.R*T*rhol*rhog/(rhol-rhog)*(
                    firL-firG+log(deltaL/deltaG))
                return Ps/1000-P

            if T0:
                To = T0
            elif self.name == "water":
                To = _TSat_P(P)
            else:
                To = (self._constant_Tref+self.Tt)/2
            T = float(fsolve(t_func, To)[0])
            rhol, rhov, Ps = self._saturation(T)
            vapor = self._Helmholtz(rhov, T)
            liquido = self._Helmholtz(rhol, T)
            if x == 0:
                propiedades = liquido
            elif x == 1:
                propiedades = vapor

        self.T = T
        self.Tr = T/self._constant_Tref
        self.P = P
        self.Pr = self.P/self.Pc
        self.x = x
        if self._mode in ["Tx", "Px"] or 0 < x < 1:
            region = 4
        else:
            region = 0
        self.phase = getphase(self._constant_Tref, self.Pc, self.T, self.P, self.x, region)

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
        if self._mode in ("Px", "Tx") or (x < 1 and self.Tt <= T <= self._constant_Tref):
            self.sigma = self._surface(T)
        else:
            self.sigma = None

        vir = self.residual._virial(T, self._constant_Tref)
        self.virialB = vir["B"]/self._constant_rhoc
        self.virialC = vir["C"]/self._constant_rhoc**2

        if 0 < x < 1:
            self.Hvap = vapor.h-liquido.h
            self.Svap = vapor.s-liquido.s
        else:
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

    # Derived classes must implement _visco to call fill()
    def _visco(self, rho: float, T: float, fase: Optional[_fase] = None) -> float:
        raise NotImplementedError

    # Derived classes must implement _thermo to call fill()
    def _thermo(self, rho: float, T: float, fase: Optional[_fase] = None) -> float:
        raise NotImplementedError

    def fill(self, fase: _fase, estado: MEoSProperties) -> None:
        """Fill phase properties"""
        fase.rho = estado.rho
        fase.v = 1/fase.rho

        assert(isinstance(self.M, float))
        assert(isinstance(self.R, float))

        fase.h = estado.h
        fase.s = estado.s
        fase.u = fase.h-self.P*1000*fase.v
        fase.a = fase.u-self.T*fase.s
        fase.g = fase.h-self.T*fase.s

        fase.Z = self.P*fase.v/self.T/self.R*1e3
        fase.fi = exp(estado.fir+estado.delta*estado.fird
                      - log(1+estado.delta*estado.fird))
        fase.f = fase.fi*self.P
        fase.cv = estado.cv

        fase.rhoM = fase.rho/self.M
        fase.hM = fase.h*self.M
        fase.sM = fase.s*self.M
        fase.uM = fase.u*self.M
        fase.aM = fase.a*self.M
        fase.gM = fase.g*self.M

        fase.alfap = estado.alfap
        fase.betap = estado.betap

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
        fase.epsilon = None
        fase.n = None
        if self.name == "water":
            try:
                fase.epsilon = _Dielectric(fase.rho, self.T)
            except NotImplementedError:
                fase.epsilon = None

            if self.kwargs["l"] is not None:
                try:
                    fase.n = _Refractive(fase.rho, self.T, self.kwargs["l"])
                except NotImplementedError:
                    fase.n = None

    def derivative(self, z: str, x: str, y: str, fase: _fase) -> float:
        """
        Wrapper derivative for custom derived properties
        where x, y, z can be: P, T, v, rho, u, h, s, g, a
        """
        return deriv_H(self, z, x, y, fase)

    def _saturation(self, T: float) -> Tuple[float, float, float]:
        """Saturation calculation for two phase search"""
        rhoc = self._constant_rhoc
        Tc = self._constant_Tref

        if T > Tc:
            T = Tc
        tau = Tc/T

        rhoLo = self._Liquid_Density(T)
        rhoGo = self._Vapor_Density(T)

        def f2(parr: Tuple[float, float]) -> Tuple[float, float]:
            rhol, rhog = parr
            deltaL = rhol/rhoc
            deltaG = rhog/rhoc
            phirL = self.residual.phir(tau, deltaL)
            phirG = self.residual.phir(tau, deltaG)
            phirdL = self.residual.phird(tau, deltaL)
            phirdG = self.residual.phird(tau, deltaG)
            Jl = deltaL*(1+deltaL*phirdL)
            Jv = deltaG*(1+deltaG*phirdG)
            Kl = deltaL*phirdL+phirL+log(deltaL)
            Kv = deltaG*phirdG+phirG+log(deltaG)
            return Kv-Kl, Jv-Jl

        rhoL, rhoG = tuple(map(float, fsolve(f2, [rhoLo, rhoGo])))
        if rhoL == rhoG:
            Ps = self.Pc
        else:
            deltaL = rhoL/self._constant_rhoc
            deltaG = rhoG/self._constant_rhoc
            firL = self.residual.phir(tau, deltaL)
            firG = self.residual.phir(tau, deltaG)

            Ps = self.R*T*rhoL*rhoG/(rhoL-rhoG)*(firL-firG+log(deltaL/deltaG))
        return rhoL, rhoG, Ps

    def _Helmholtz(self, rho: float, T: float) -> MEoSProperties:
        """Calculated properties from helmholtz free energy and derivatives

        Parameters
        ----------
        rho : float
            Density, [kg/m³]
        T : float
            Temperature, [K]

        Returns
        -------
        prop : dict
            Dictionary with calculated properties:
                * fir: [-]
                * fird: ∂fir/∂δ|τ
                * firdd: ∂²fir/∂δ²|τ
                * delta: Reducen density rho/rhoc, [-]
                * P: Pressure, [kPa]
                * h: Enthalpy, [kJ/kg]
                * s: Entropy, [kJ/kgK]
                * cv: Isochoric specific heat, [kJ/kgK]
                * alfav: Thermal expansion coefficient, [1/K]
                * betap: Isothermal compressibility, [1/kPa]

        References
        ----------
        IAPWS, Revised Release on the IAPWS Formulation 1995 for the
        Thermodynamic Properties of Ordinary Water Substance for General and
        Scientific Use, September 2016, Table 3
        http://www.iapws.org/relguide/IAPWS-95.html
        """
        if rho < 0:
            rho = 1e-20
        if T < 50:
            T = 50
        rhoc = self._constant_rhoc
        Tc = self._constant_Tref
        delta = rho/rhoc
        tau = Tc/T
        ideal = self._phi0(tau, delta)

        res = self.residual.helmholtz(tau, delta)

        P = (1+delta*res.fid)*self.R*T*rho
        h = self.R*T*(1+tau*(ideal.fit+res.fit)+delta*res.fid)
        s = self.R*(tau*(ideal.fit+res.fit)-ideal.fi-res.fi)
        cv = -self.R*tau**2*(ideal.fitt+res.fitt)
        alfap = (1-delta*tau*res.fidt/(1+delta*res.fid))/T
        betap = rho*(1+(delta*res.fid+delta**2*res.fidd)/(1+delta*res.fid))
        return MEoSProperties(rho, P, h, s, cv, alfap, betap,
                              res.fi, res.fid, res.fidd, delta)

    def _prop0(self, rho: float, T: float) -> _fase:
        """Ideal gas properties"""
        rhoc = self._constant_rhoc
        Tc = self._constant_Tref

        delta = rho/rhoc
        tau = Tc/T
        ideal = self._phi0(tau, delta)

        propiedades = _fase()
        propiedades.h = self.R*T*(1+tau*ideal.fit)
        propiedades.s = self.R*(tau*ideal.fit-ideal.fi)
        propiedades.cv = -self.R*tau**2*ideal.fitt
        propiedades.cp = self.R*(-tau**2*ideal.fitt+1)
        propiedades.alfap = 1.0/T
        propiedades.betap = rho
        return propiedades

    def _phi0(self, tau: float, delta: float) -> HelmholtzDerivatives:
        """Ideal gas Helmholtz free energy and derivatives

        Parameters
        ----------
        tau : float
            Inverse reduced temperature Tc/T, [-]
        delta : float
            Reduced density rho/rhoc, [-]

        Returns
        -------
        prop : dictionary with ideal adimensional helmholtz energy and deriv
            fio, [-]
            fiot: ∂fio/∂τ|δ
            fiod: ∂fio/∂δ|τ
            fiott: ∂²fio/∂τ²|δ
            fiodt: ∂²fio/∂τ∂δ
            fiodd: ∂²fio/∂δ²|τ

        References
        ----------
        IAPWS, Revised Release on the IAPWS Formulation 1995 for the
        Thermodynamic Properties of Ordinary Water Substance for General and
        Scientific Use, September 2016, Table 4
        http://www.iapws.org/relguide/IAPWS-95.html
        """
        Fi0 = self.Fi0

        try:
            dlog = log(delta)
        except ValueError:
            dlog = float('nan')

        try:
            tlog = log(tau)
        except ValueError:
            tlog = float('nan')

        fio = Fi0["ao_log"][0]*dlog+Fi0["ao_log"][1]*tlog
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
            try:
                expt = exp(-t*tau)
            except OverflowError:
                expt = float('inf')
            try:
                logterm = log(1-expt)
            except ValueError:
                logterm = float('nan')
            fio += n*logterm
            fiot += n*t*((1-expt)**-1-1)
            fiott -= n*t**2*expt*(1-expt)**-2

        # Extension to especial terms of air
        if "ao_exp2" in Fi0:
            for n, g, C in zip(Fi0["ao_exp2"], Fi0["titao2"], Fi0["sum2"]):
                fio += n*log(C+exp(g*tau))
                fiot += n*g/(C*exp(-g*tau)+1)
                fiott += C*n*g**2*exp(-g*tau)/(C*exp(-g*tau)+1)**2

        return HelmholtzDerivatives(fio, fiot, fiod, fiott, fiodd, fiodt)

    def _derivDimensional(self, rho: float, T: float) -> HelmholtzDerivatives:
        """Calcule the dimensional form or Helmholtz free energy derivatives

        Parameters
        ----------
        rho : float
            Density, [kg/m³]
        T : float
            Temperature, [K]

        Returns
        -------
        prop : dict
            Dictionary with residual helmholtz energy and derivatives:

                * fir, [kJ/kg]
                * firt: ∂fir/∂T|ρ, [kJ/kgK]
                * fird: ∂fir/∂ρ|T, [kJ/m³kg²]
                * firtt: ∂²fir/∂T²|ρ, [kJ/kgK²]
                * firdt: ∂²fir/∂T∂ρ, [kJ/m³kg²K]
                * firdd: ∂²fir/∂ρ²|T, [kJ/m⁶kg]

        References
        ----------
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 7,
        http://www.iapws.org/relguide/SeaAir.html
        """
        if not rho:
            return HelmholtzDerivatives(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        rhoc = self._constant_rhoc
        Tc = self._constant_Tref
        delta = rho/rhoc
        tau = Tc/T

        ideal = self._phi0(tau, delta)

        res = self.residual.helmholtz(tau, delta)

        R = self.R

        fir = R*T*(ideal.fi+res.fi)
        firt = R*(ideal.fi+res.fi-(ideal.fit+res.fit)*tau)
        fird = R*T/rhoc*(ideal.fid+res.fid)
        firtt = R*tau**2/T*(ideal.fitt+res.fitt)
        firdt = R/rhoc*(ideal.fid+res.fid-res.fidt*tau)
        firdd = R*T/rhoc**2*(ideal.fidd+res.fidd)
        return HelmholtzDerivatives(fir, firt, fird, firtt, firdd, firdt)

    def _surface(self, T: float) -> float:
        """Generic equation for the surface tension

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        σ : float
            Surface tension, [N/m]

        Notes
        -----
        Need a _surf dict in the derived class with the parameters keys:
            sigma: coefficient
            exp: exponent
        """
        tau = 1-T/self._constant_Tref
        sigma = 0.0
        for n, t in zip(self._surf["sigma"], self._surf["exp"]):
            sigma += n*tau**t
        return sigma

    @classmethod
    def _Vapor_Pressure(cls, T: float) -> float:
        """Auxiliary equation for the vapour pressure

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        Pv : float
            Vapour pressure, [Pa]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.1
        """
        Tita = 1-T/cls.Tc
        suma = 0.0
        for n, x in zip(cls._Pv_ao, cls._Pv_exp):
            suma += n*Tita**x
        Pr = exp(cls.Tc/T*suma)
        Pv = Pr*cls.Pc
        return Pv

    @classmethod
    def _Liquid_Density(cls, T: float) -> float:
        """Auxiliary equation for the density of saturated liquid

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        rho : float
            Saturated liquid density, [kg/m³]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.2
        """
        # Convert numpy.Float64 back into Python floats.
        T = float(T)
        Tita = 1-T/cls.Tc
        if cls._rhoL_eq == 2:
            Tita = Tita**(1./3)
        suma = 0.0
        for n, x in zip(cls._rhoL_ao, cls._rhoL_exp):
            suma += n*Tita**x
        Pr = suma+1
        rho = Pr*cls.rhoc
        return rho

    @classmethod
    def _Vapor_Density(cls, T: float) -> float:
        """Auxiliary equation for the density of saturated vapor

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        rho : float
            Saturated vapor density, [kg/m³]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, Eq.3
        """
        # Convert numpy.Float64 back into Python floats.
        T = float(T)
        Tita = 1-T/cls.Tc
        if cls._rhoG_eq == 4:
            Tita = Tita**(1./3)
        suma = 0.0
        for n, x in zip(cls._rhoG_ao, cls._rhoG_exp):
            suma += n*Tita**x
            if type(suma) == complex:
                suma = float('nan')

        Pr = exp(suma)
        rho = Pr*cls.rhoc
        return rho

    @classmethod
    def _dPdT_sat(cls, T: float) -> float:
        """Auxiliary equation for the dP/dT along saturation line

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        dPdT : float
            dPdT, [MPa/K]

        References
        ----------
        IAPWS, Revised Supplementary Release on Saturation Properties of
        Ordinary Water Substance September 1992,
        http://www.iapws.org/relguide/Supp-sat.html, derived from Eq.1
        """
        Tita = 1-T/cls.Tc
        suma1 = 0.0
        suma2 = 0.0
        for n, x in zip(cls._Pv_ao, cls._Pv_exp):
            suma1 -= n*x*Tita**(x-1)/cls.Tc
            suma2 += n*Tita**x
        Pr = (cls.Tc*suma1/T-cls.Tc/T**2*suma2)*exp(cls.Tc/T*suma2)
        dPdT = Pr*cls.Pc
        return dPdT


def mainClassDoc() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Function decorator used to automatic adiction of base class MEoS in
    subclass __doc__
    """
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # __doc__ is only writable in python3.
        # The doc build must be done with python3 so this snippet do the work
        py_version = platform.python_version()
        if py_version[0] == "3":
            if isinstance(f.__doc__, str):
                doc = f.__doc__.split(os.linesep)
            try:
                ind = doc.index("")
            except ValueError:
                ind = 1

            doc1 = os.linesep.join(doc[:ind])
            doc3 = os.linesep.join(doc[ind:])
            if isinstance(MEoS.__doc__, str):
                doc2 = os.linesep.join(MEoS.__doc__.split(os.linesep)[3:])

            f.__doc__ = doc1 + os.linesep + os.linesep + \
                doc2 + os.linesep + os.linesep + doc3
        return f

    return decorator


@mainClassDoc()
class IAPWS95(MEoS):
    """Implementation of IAPWS Formulation 1995 for ordinary water substance,
    (revised release of 2016), for internal procedures, see MEoS base class

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
    M = _global_M
    Tt = 273.16
    Tb = 373.1243
    f_acent = 0.3443
    momentoDipolar = 1.855

    Fi0 = {"ao_log": [1.0, 3.00632],
           "pow": [0.0, 1.0],
           "ao_pow": [-8.3204464837497, 6.6832105275932],
           "ao_exp": [0.012436, 0.97315, 1.2795, 0.96956, 0.24873],
           "titao": [1.28728967, 3.53734222, 7.74073708, 9.24437796,
                     27.5075105]}

    _constant_R = 8.314371357587
    _constant_rhoc = rhoc
    _constant_Tref = Tc
    residual = ResidualContribution(
        nr1=[0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
             0.31802509345418, -0.26145533859358, -0.78199751687981e-2,
             0.88089493102134e-2],
        d1=[1, 1, 1, 2, 2, 3, 4],
        t1=[-0.5, 0.875, 1, 0.5, 0.75, 0.375, 1],

        nr2=[-0.66856572307965, 0.20433810950965, -0.66212605039687e-4,
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
        c2=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 6, 6,
            6, 6],
        d2=[1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10, 11, 13, 15, 1, 2, 2, 2, 3,
            4, 4, 4, 5, 6, 6, 7, 9, 9, 9, 9, 9, 10, 10, 12, 3, 4, 4, 5, 14,
            3, 6, 6, 6],
        t2=[4, 6, 12, 1, 5, 4, 2, 13, 9, 3, 4, 11, 4, 13, 1, 7, 1, 9, 10,
            10, 3, 7, 10, 10, 6, 10, 10, 1, 2, 3, 4, 8, 6, 9, 8, 16, 22, 23,
            23, 10, 50, 44, 46, 50],
        gamma2=[1]*44,

        nr3=[-0.31306260323435e2, 0.31546140237781e2, -0.25213154341695e4],
        d3=[3]*3,
        t3=[0, 1, 4],
        alfa3=[20]*3,
        beta3=[150, 150, 250],
        gamma3=[1.21, 1.21, 1.25],
        epsilon3=[1.]*3,

        nr4=[-0.14874640856724, 0.31806110878444],
        a4=[3.5, 3.5],
        b4=[0.85, 0.95],
        B=[0.2, 0.2],
        C=[28, 32],
        D=[700, 800],
        A=[0.32, .32],
        beta4=[0.3, 0.3],
    )

    _Pv_ao = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719,
              1.80122502]
    _Pv_exp = [1.0, 1.5, 3.0, 3.5, 4.0, 7.5]
    _rhoL_eq = 2
    _rhoL_ao = [1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352,
                -6.74694450e5]
    _rhoL_exp = [1.0, 2.0, 5.0, 16.0, 43.0, 110]
    _rhoG_eq = 4
    _rhoG_ao = [-2.0315024, -2.6830294, -5.38626492, -17.2991605, -44.7586581,
                -63.9201063]
    _rhoG_exp = [1.0, 2.0, 4.0, 9.0, 18.5, 35.5]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _phi0(self, tau: float, delta: float) -> HelmholtzDerivatives:
        """Low temperature extension of the IAPWS-95"""
        prop = MEoS._phi0(self, tau, delta)

        T = self._constant_Tref/tau
        if 50 <= T < 130:
            fex, fext, fextt = self._phiex(T)
            prop.fi += fex
            prop.fit += fext
            prop.fitt += fextt
        return prop

    def _phiex(self, T: float) -> Tuple[float, float, float]:
        """Low temperature extension"""
        tau = self._constant_Tref/T
        E = 0.278296458178592
        ep = self._constant_Tref/130
        fex = E*(-1/2/tau-3/ep**2*(tau+ep)*log(tau/ep)-9/2/ep+9*tau/2/ep**2
                 + tau**2/2/ep**3)
        fext = E*(1/2/tau**2-3/tau/ep-3/ep**2*log(tau/ep)+3/2/ep**2+tau/ep**3)
        fextt = E*(-1/tau+1/ep)**3
        return fex, fext, fextt

    @classmethod
    def _alfa_sat(cls, T: float) -> float:
        """Auxiliary equation for the alfa coefficient for calculate the
        enthalpy along the saturation line

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        alfa : float
            alfa coefficient, [kJ/kg]

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
        alfa = 0.0
        for d, x in zip(di, expi):
            alfa += d*Tita**x
        return alfa

    @classmethod
    def _phi_sat(cls, T: float) -> float:
        """Auxiliary equation for the phi coefficient for calculate the
        entropy along the saturation line

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        phi : float
            phi coefficient, [kJ/kgK]

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
        suma = 0.0
        for d, x in zip(di, expi):
            if x is None:
                suma += d*log(Tita)
            else:
                suma += d*Tita**x
        phi = suma/cls.Tc
        return phi

    @classmethod
    def _Liquid_Enthalpy(cls, T: float) -> float:
        """Auxiliary equation for the specific enthalpy for saturated liquid

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        h : float
            Saturated liquid enthalpy, [kJ/kg]

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
    def _Vapor_Enthalpy(cls, T: float) -> float:
        """Auxiliary equation for the specific enthalpy for saturated vapor

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        h : float
            Saturated vapor enthalpy, [kJ/kg]

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
    def _Liquid_Entropy(cls, T: float) -> float:
        """Auxiliary equation for the specific entropy for saturated liquid

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        s : float
            Saturated liquid entropy, [kJ/kgK]

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
    def _Vapor_Entropy(cls, T: float) -> float:
        """Auxiliary equation for the specific entropy for saturated vapor

        Parameters
        ----------
        T : float
            Temperature, [K]

        Returns
        -------
        s : float
            Saturated liquid entropy, [kJ/kgK]

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

    def _visco(self, rho: float, T: float, fase: Optional[_fase] = None) -> float:
        ref = IAPWS95()
        st = ref._Helmholtz(rho, 1.5*Tc)
        delta = rho/rhoc
        drho = 1e3/self.R/1.5/Tc/(1+2*delta*st.fird+delta**2*st.firdd)
        return _Viscosity(rho, T, fase, drho)

    def _thermo(self, rho: float, T: float, fase: Optional[_fase] = None) -> float:
        ref = IAPWS95()
        st = ref._Helmholtz(rho, 1.5*Tc)
        delta = rho/rhoc
        drho = 1e3/self.R/1.5/Tc/(1+2*delta*st.fird+delta**2*st.firdd)
        return _ThCond(rho, T, fase, drho)

    def _surface(self, T: float) -> float:
        s = _Tension(T)
        return s


class IAPWS95_PT(IAPWS95):
    """Derivated class for direct P and T input"""

    def __init__(self, P: float, T: float):
        IAPWS95.__init__(self, T=T, P=P)


class IAPWS95_Ph(IAPWS95):
    """Derivated class for direct P and h input"""

    def __init__(self, P: float, h: float):
        IAPWS95.__init__(self, P=P, h=h)


class IAPWS95_Ps(IAPWS95):
    """Derivated class for direct P and s input"""

    def __init__(self, P: float, s: float):
        IAPWS95.__init__(self, P=P, s=s)


class IAPWS95_Px(IAPWS95):
    """Derivated class for direct P and v input"""

    def __init__(self, P: float, x):
        IAPWS95.__init__(self, P=P, x=x)


class IAPWS95_Tx(IAPWS95):
    """Derivated class for direct T and x input"""

    def __init__(self, T: float, x):
        IAPWS95.__init__(self, T=T, x=x)


@mainClassDoc()
class D2O(MEoS):
    """Implementation of IAPWS Formulation for heavy water substance,
    for internal procedures, see MEoS base class

    Examples
    --------
    >>> hwater=D2O(T=300, rho=996.5560)
    >>> hwater.P, hwater.Liquid.cv, hwater.Liquid.w
    0.0030675947 4.21191157 5332.04871

    References
    ----------
    IAPWS, Release on the IAPWS Formulation 2017 for the Thermodynamic
    Properties of Heavy Water, http://www.iapws.org/relguide/Heavy-2017.pdf
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

    Fi0 = {"ao_log": [1.0, 3.0],
           "pow": [0.0, 1.0],
           "ao_pow": [-8.670994022646, 6.96033578458778],
           "ao_exp": [0.010633, 0.99787, 2.1483, 0.3549],
           "titao": [308.0/Tc, 1695.0/Tc, 3949.0/Tc, 10317.0/Tc],
           "ao_hyp": [], "hyp": []}

    _constant_R = 8.3144598
    _constant_rhoc = rhoc_D2O
    _constant_Tref = Tc_D2O
    residual = ResidualContribution(
        nr1=[0.122082060e-1, 0.296956870e1, -0.379004540e1, 0.941089600,
             -0.922466250, -0.139604190e-1],
        d1=[4, 1, 1, 2, 2, 3],
        t1=[1.0000, 0.6555, 0.9369, 0.5610, 0.7017, 1.0672],
        nr2=[-0.125203570, -0.555391500e1, -0.493009740e1, -0.359470240e-1,
             -0.936172870e1, -0.691835150],
        c2=[1, 2, 2, 1, 2, 2],
        d2=[1, 1, 3, 2, 2, 1],
        t2=[3.9515, 4.6000, 5.1590, 0.2000, 5.4644, 2.3660],
        gamma2=[1]*6,

        nr3=[-0.456110600e-1, -0.224513300e1, 0.860006070e1, -0.248410420e1,
             0.164476900e2, 0.270393360e1, 0.375637470e2, -0.177607760e1,
             0.220924640e1, 0.519652000e1, 0.421097400, -0.391921100],
        t3=[3.4553, 1.4150, 1.5745, 3.4540, 3.8106, 4.8950, 1.4300, 1.5870,
            3.7900, 2.6200, 1.9000, 4.3200],
        d3=[1, 3, 1, 3, 1, 1, 2, 2, 2, 1, 1, 1],
        alfa3=[0.6014, 1.4723, 1.5305, 2.4297, 1.3086, 1.3528, 3.4456,
               1.2645, 2.5547, 1.2148, 18.738, 18.677],
        beta3=[0.4200, 2.4318, 1.2888, 8.2710, 0.3673, 0.9504, 7.8318,
               3.3281, 7.1753, 0.9465, 1177.0, 1167.0],
        epsilon3=[1.8663, 0.2895, 0.5803, 0.2236, 0.6815, 0.9495, 1.1158,
                  0.1607, 0.4144, 0.9683, 0.9488, 0.9487],
        gamma3=[1.5414, 1.3794, 1.7385, 1.3045, 2.7242, 3.5321, 2.4552,
                0.8319, 1.3500, 2.5617, 1.0491, 1.0486],
        nr4=[], a4=[], b4=[], A=[], B=[], C=[], D=[], beta4=[],
    )

    _Pv_ao = [-0.80236e1, 0.23957e1, -0.42639e2, 0.99569e2, -0.62135e2]
    _Pv_exp = [1.0, 1.5, 2.75, 3.0, 3.2]
    _rhoL_eq = 1
    _rhoL_ao = [0.26406e1, 0.97090e1, -0.18058e2, 0.87202e1, -0.74487e1]
    _rhoL_exp = [0.3678, 1.9, 2.2, 2.63, 7.3]
    _rhoG_eq = 3
    _rhoG_ao = [-0.37651e1, -0.38673e2, 0.73024e2, -0.13251e3, 0.75235e2,
                -0.70412e2]
    _rhoG_exp = [0.409, 1.766, 2.24, 3.04, 3.42, 6.9]

    # fase is unused
    def _visco(self, rho: float, T: float, fase: Optional[_fase] = None) -> float:
        return _D2O_Viscosity(rho, T)

    # fase is unused
    def _thermo(self, rho: float, T: float, fase: Optional[_fase] = None) -> float:
        return _D2O_ThCond(rho, T)

    def _surface(self, T: float) -> float:
        s = _D2O_Tension(T)
        return s
