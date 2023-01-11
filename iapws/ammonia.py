#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
# pylint: disable=too-many-locals, too-many-statements

"""
Module with Ammonia-water mixture properties and related properties. The module
include:

    * :class:`NH3`: Multiparameter equation of state for ammonia
    * :class:`H2ONH3`:  Thermodynamic properties of ammonia-water mixtures
    * :class:`Ttr`: Triple point of ammonia-water mixtures
"""


from __future__ import division
from math import exp, log, pi
import warnings

from scipy.constants import Boltzmann
from .iapws95 import MEoS, IAPWS95, mainClassDoc


@mainClassDoc()
class NH3(MEoS):
    """Multiparameter equation of state for ammonia
    for internal procedures, see MEoS base class

    References
    ----------
    Baehr, H.D., Tillner-Roth, R.; Thermodynamic Properties of Environmentally
    Acceptable Refrigerants: Equations of State and Tables for Ammonia, R22,
    R134a, R152a, and R123. Springer-Verlag, Berlin, 1994.
    http://doi.org/10.1007/978-3-642-79400-1
    """

    name = "ammonia"
    CASNumber = "7664-41-7"
    formula = "NH3"
    synonym = "R-717"
    rhoc = 225.
    Tc = 405.40
    Pc = 11.333   # MPa
    M = 17.03026  # g/mol
    Tt = 195.495
    Tb = 239.823
    f_acent = 0.25601
    momentoDipolar = 1.470

    Fi0 = {"ao_log": [1, -1],
           "pow": [0, 1, 1./3, -1.5, -1.75],
           "ao_pow": [-15.81502, 4.255726, 11.47434, -1.296211, 0.5706757],
           "ao_exp": [], "titao": [],
           "ao_hyp": [], "hyp": []}

    _constants = {
        "R": 8.314471,

        "nr1": [-0.1858814e01, 0.4554431e-1, 0.7238548, 0.1229470e-1,
                0.2141882e-10],
        "d1": [1, 2, 1, 4, 15],
        "t1": [1.5, -0.5, 0.5, 1., 3.],

        "nr2": [-0.1430020e-1, 0.3441324, -0.2873571, 0.2352589e-4,
                -0.3497111e-1, 0.1831117e-2, 0.2397852e-1, -0.4085375e-1,
                0.2379275, -0.3548972e-1, -0.1823729, 0.2281556e-1,
                -0.6663444e-2, -0.8847486e-2, 0.2272635e-2, -0.5588655e-3],
        "d2": [3, 3, 1, 8, 2, 8, 1, 1, 2, 3, 2, 4, 3, 1, 2, 4],
        "t2": [0, 3, 4, 4, 5, 5, 3, 6, 8, 8, 10, 10, 5, 7.5, 15, 30],
        "c2": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        "gamma2": [1]*16}

    _melting = {"eq": 1, "Tref": Tt, "Pref": 1000,
                "Tmin": Tt, "Tmax": 700.0,
                "a1": [], "exp1": [], "a2": [], "exp2": [],
                "a3": [0.2533125e4], "exp3": [1]}

    _surf = {"sigma": [0.1028, -0.09453], "exp": [1.211, 5.585]}
    _Pv = {
        "eq": 5,
        "ao": [-0.70993e1, -0.24330e1, 0.87591e1, -0.64091e1, -0.21185e1],
        "exp": [1., 1.5, 1.7, 1.95, 4.2]}
    _rhoL = {
        "eq": 1,
        "ao": [0.34488e2, -0.12849e3, 0.17382e3, -0.10699e3, 0.30339e2],
        "exp": [0.58, 0.75, 0.9, 1.1, 1.3]}
    _rhoG = {
        "eq": 3,
        "ao": [-.38435, -4.0846, -6.6634, -0.31881e2, 0.21306e3, -0.24648e3],
        "exp": [0.218, 0.55, 1.5, 3.7, 5.5, 5.8]}

    def _visco(self, rho, T, fase=None):
        """Equation for the Viscosity

        Parameters
        ----------
        rho : float
            Density [kg/m³]
        T : float
            Temperature [K]

        Returns
        -------
        mu : float
            Viscosity [Pa·s]

        References
        ----------
        Fenghour, A., Wakeham, W.A., Vesovic, V., Watson, J.T.R., Millat, J.,
        and Vogel, E., The viscosity of ammonia, J. Phys. Chem. Ref. Data 24,
        1649 (1995). doi:10.1063/1.555961
        """
        ek = 386
        sigma = 0.2957

        rho = rho/self.M
        T_ = T/ek

        # Eq 4
        a = [4.99318220, -0.61122364, 0.0, 0.18535124, -0.11160946]
        omega = exp(sum(ai*log(T_)**i for i, ai in enumerate(a)))

        # Eq 2, Zero-Density Limit
        muo = 2.1357*(T*self.M)**0.5/sigma**2/omega

        # Eq 8, Viscosity virial coefficient
        cv = [-0.17999496e1, 0.46692621e2, -0.53460794e3, 0.33604074e4,
              -0.13019164e5, 0.33414230e5, -0.58711743e5, 0.71426686e5,
              -0.59834012e5, 0.33652741e5, -0.1202735e5, 0.24348205e4,
              -0.20807957e3]
        Bn = 0.6022137*sigma**3*sum(c*T_**(-i/2) for i, c in enumerate(cv))
        # Eq 7
        mub = Bn*muo*rho

        # Eq 10
        dij = [2.19664285e-1, -0.83651107e-1, 0.17366936e-2, -0.64250359e-2,
               1.67668649e-4, -1.49710093e-4, 0.77012274e-4]
        ji = [2, 4, 0, 1, 2, 3, 4]
        ii = [2, 2, 3, 3, 4, 4, 4]
        mur = sum(d/T_**j*rho**i for d, j, i in zip(dij, ji, ii))

        # Eq 1
        mu = muo + mub + mur
        return mu*1e-6

    def _thermo(self, rho, T, fase):
        """Equation for the thermal conductivity

        Parameters
        ----------
        rho : float
            Density, [kg/m³]
        T : float
            Temperature, [K]
        fase: dict
            phase properties

        Returns
        -------
        k : float
            Thermal conductivity [W/mK]

        References
        ----------
        Tufeu, R., Ivanov, D.Y., Garrabos, Y., and Le Neindre, B., Thermal
        conductivity of ammonia in a large temperature and pressure range
        including the critical region, Ber. Bunsenges. Phys. Chem., 88:422-427,
        1984. doi:10.1002/bbpc.19840880421
        """
        # The paper use a diferent rhoc value to the EoS
        rhoc = 235

        if rho == rhoc and T == self.Tc:
            warnings.warn("Thermal conductiviy undefined in critical point")
            return None

        # Eq 6
        no = [0.3589e-1, -0.1750e-3, 0.4551e-6, 0.1685e-9, -0.4828e-12]
        Lo = sum(n*T**i for i, n in enumerate(no))

        # Eq 7
        nb = [0.16207e-3, 0.12038e-5, -0.23139e-8, 0.32749e-11]
        L_ = sum(n*rho**(i+1) for i, n in enumerate(nb))

        # Critical enchancement
        t = abs(T-405.4)/405.4
        dPT = 1e5*(2.18-0.12/exp(17.8*t))
        nb = 1e-5*(2.6+1.6*t)

        DL = 1.2*Boltzmann*T**2/6/pi/nb/(1.34e-10/t**0.63*(1+t**0.5)) * \
            dPT**2 * 0.423e-8/t**1.24*(1+t**0.5/0.7)

        # Add correction for entire range of temperature, Eq 10
        DL *= exp(-36*t**2)

        X = 0.61*rhoc+16.5*log(t)
        if rho > 0.6*rhoc:
            # Eq 11
            DL *= X**2/(X**2+(rho-0.96*rhoc)**2)
        else:
            # Eq 14
            DL = X**2/(X**2+(0.6*rhoc-0.96*rhoc)**2)
            DL *= rho**2/(0.6*rhoc)**2

        # Eq 5
        k = Lo+L_+DL
        return k


class H2ONH3(object):
    """Ammonia-water mixtures."""

    # TODO: Add equilibrium routine

    def _prop(self, rho, T, x):
        """Thermodynamic properties of ammonia-water mixtures

        Parameters
        ----------
        T : float
            Temperature [K]
        rho : float
            Density [kg/m³]
        x : float
            Mole fraction of ammonia in mixture [mol/mol]

        Returns
        -------
        prop : dict
            Dictionary with thermodynamic properties of ammonia-water mixtures:

                * M: Mixture molecular mass, [g/mol]
                * P: Pressure, [MPa]
                * u: Specific internal energy, [kJ/kg]
                * s: Specific entropy, [kJ/kgK]
                * h: Specific enthalpy, [kJ/kg]
                * a: Specific Helmholtz energy, [kJ/kg]
                * g: Specific gibbs energy, [kJ/kg]
                * cv: Specific isochoric heat capacity, [kJ/kgK]
                * cp: Specific isobaric heat capacity, [kJ/kgK]
                * w: Speed of sound, [m/s]
                * fugH2O: Fugacity of water, [-]
                * fugNH3: Fugacity of ammonia, [-]

        References
        ----------
        IAPWS, Guideline on the IAPWS Formulation 2001 for the Thermodynamic
        Properties of Ammonia-Water Mixtures,
        http://www.iapws.org/relguide/nh3h2o.pdf, Table 4
        """
        # FIXME: The values are good, bad difer by 1%, a error I can´t find
        # In Pressure happen and only use fird

        M = (1-x)*IAPWS95.M + x*NH3.M
        R = 8.314471/M

        phio = self._phi0(rho, T, x)
        fio = phio["fio"]
        tau0 = phio["tau"]
        fiot = phio["fiot"]
        fiott = phio["fiott"]

        phir = self._phir(rho, T, x)
        fir = phir["fir"]
        tau = phir["tau"]
        delta = phir["delta"]
        firt = phir["firt"]
        firtt = phir["firtt"]
        fird = phir["fird"]
        firdd = phir["firdd"]
        firdt = phir["firdt"]
        F = phir["F"]

        prop = {}
        Z = 1 + delta*fird
        prop["M"] = M
        prop["P"] = Z*R*T*rho/1000
        prop["u"] = R*T*(tau0*fiot + tau*firt)
        prop["s"] = R*(tau0*fiot + tau*firt - fio - fir)
        prop["h"] = R*T*(1+delta*fird+tau0*fiot+tau*firt)
        prop["g"] = prop["h"]-T*prop["s"]
        prop["a"] = prop["u"]-T*prop["s"]
        cvR = -tau0**2*fiott - tau**2*firtt
        prop["cv"] = R*cvR
        prop["cp"] = R*(cvR+(1+delta*fird-delta*tau*firdt)**2
                        / (1+2*delta*fird+delta**2*firdd))
        prop["w"] = (R*T*1000*(1+2*delta*fird+delta**2*firdd
                               + (1+delta*fird-delta*tau*firdt)**2 / cvR))**0.5
        prop["fugH2O"] = Z*exp(fir+delta*fird-x*F)
        prop["fugNH3"] = Z*exp(fir+delta*fird+(1-x)*F)
        return prop

    @staticmethod
    def _phi0(rho, T, x):
        """Ideal gas Helmholtz energy of binary mixtures and derivatives

        Parameters
        ----------
        rho : float
            Density, [kg/m³]
        T : float
            Temperature, [K]
        x : float
            Mole fraction of ammonia in mixture, [mol/mol]

        Returns
        -------
        prop : dict
            Dictionary with ideal adimensional helmholtz energy and
            derivatives:

                * tau: the adimensional temperature variable, [-]
                * delta: the adimensional density variable, [-]
                * fio,[-]
                * fiot: [∂fio/∂τ]δ  [-]
                * fiod: [∂fio/∂δ]τ  [-]
                * fiott: [∂²fio/∂τ²]δ  [-]
                * fiodt: [∂²fio/∂τ∂δ]  [-]
                * fiodd: [∂²fio/∂δ²]τ  [-]

        References
        ----------
        IAPWS, Guideline on the IAPWS Formulation 2001 for the Thermodynamic
        Properties of Ammonia-Water Mixtures,
        http://www.iapws.org/relguide/nh3h2o.pdf, Eq 2
        """
        # Define reducing parameters for mixture model
        M = (1-x)*IAPWS95.M + x*NH3.M
        tau = 500/T
        delta = rho/15/M

        # Table 2
        Fi0 = {
            "log_water": 3.006320,
            "ao_water": [-7.720435, 8.649358],
            "pow_water": [0, 1],
            "ao_exp": [0.012436, 0.97315, 1.279500, 0.969560, 0.248730],
            "titao": [1.666, 4.578, 10.018, 11.964, 35.600],
            "log_nh3": -1.0,
            "ao_nh3": [-16.444285, 4.036946, 10.69955, -1.775436, 0.82374034],
            "pow_nh3": [0, 1, 1/3, -3/2, -7/4]}

        fiod = 1/delta
        fiodd = -1/delta**2
        fiodt = 0
        fiow = fiotw = fiottw = 0
        fioa = fiota = fiotta = 0

        # Water section
        if x < 1:
            fiow = Fi0["log_water"]*log(tau) + log(1-x)
            fiotw = Fi0["log_water"]/tau
            fiottw = -Fi0["log_water"]/tau**2
            for n, t in zip(Fi0["ao_water"], Fi0["pow_water"]):
                fiow += n*tau**t
                if t != 0:
                    fiotw += t*n*tau**(t-1)
                if t not in [0, 1]:
                    fiottw += n*t*(t-1)*tau**(t-2)
            for n, t in zip(Fi0["ao_exp"], Fi0["titao"]):
                fiow += n*log(1-exp(-tau*t))
                fiotw += n*t*((1-exp(-t*tau))**-1-1)
                fiottw -= n*t**2*exp(-t*tau)*(1-exp(-t*tau))**-2

        # ammonia section
        if x > 0:
            fioa = Fi0["log_nh3"]*log(tau) + log(x)
            fiota = Fi0["log_nh3"]/tau
            fiotta = -Fi0["log_nh3"]/tau**2
            for n, t in zip(Fi0["ao_nh3"], Fi0["pow_nh3"]):
                fioa += n*tau**t
                if t != 0:
                    fiota += t*n*tau**(t-1)
                if t not in [0, 1]:
                    fiotta += n*t*(t-1)*tau**(t-2)

        prop = {}
        prop["tau"] = tau
        prop["delta"] = delta
        prop["fio"] = log(delta) + (1-x)*fiow + x*fioa
        prop["fiot"] = (1-x)*fiotw + x*fiota
        prop["fiott"] = (1-x)*fiottw + x*fiotta
        prop["fiod"] = fiod
        prop["fiodd"] = fiodd
        prop["fiodt"] = fiodt
        return prop

    def _phir(self, rho, T, x):
        """Residual contribution to the free Helmholtz energy

        Parameters
        ----------
        rho : float
            Density, [kg/m³]
        T : float
            Temperature, [K]
        x : float
            Mole fraction of ammonia in mixture, [mol/mol]

        Returns
        -------
        prop : dict
            dictionary with residual adimensional helmholtz energy and
            derivatives:

                * tau: the adimensional temperature variable, [-]
                * delta: the adimensional density variable, [-]
                * fir, [-]
                * firt: [∂fir/∂τ]δ,x  [-]
                * fird: [∂fir/∂δ]τ,x  [-]
                * firtt: [∂²fir/∂τ²]δ,x  [-]
                * firdt: [∂²fir/∂τ∂δ]x  [-]
                * firdd: [∂²fir/∂δ²]τ,x  [-]
                * firx: [∂fir/∂x]τ,δ  [-]
                * F: Function for fugacity calculation, [-]

        References
        ----------
        IAPWS, Guideline on the IAPWS Formulation 2001 for the Thermodynamic
        Properties of Ammonia-Water Mixtures,
        http://www.iapws.org/relguide/nh3h2o.pdf, Eq 3
        """
        # Temperature reducing value, Eq 4
        Tc12 = 0.9648407/2*(IAPWS95.Tc+NH3.Tc)
        Tn = (1-x)**2*IAPWS95.Tc + x**2*NH3.Tc + 2*x*(1-x**1.125455)*Tc12

        # sympy.diff("(1-x)^2*Tc1 + x^2*Tc2 + 2*x*(1-x^a)*Tc12", "x")
        # Out[]: Tc1*(2*x - 2) - 2*Tc12*a*x**a + 2*Tc12*(1 - x**a) + 2*Tc2*x
        dTnx = -2*IAPWS95.Tc*(1-x) + 2*x*NH3.Tc + 2*Tc12*(1-x**1.125455) - \
            2*Tc12*1.12455*x**1.12455

        # Density reducing value, Eq 5
        b = 0.8978069
        rhoc1m = IAPWS95.rhoc/(IAPWS95.M/1000)
        rhoc2m = NH3.rhoc/(NH3.M/1000)
        M = (1-x)*IAPWS95.M + x*NH3.M

        rhoc12 = 1/(1.2395117/2*(1/rhoc1m + 1/rhoc2m))
        rhonm = 1/((1-x)**2/rhoc1m + x**2/rhoc2m
                   + 2*x*(1-x**b)/rhoc12)
        rhon = rhonm*M/1000

        # sympy.diff("1/((1-x)^2/rhoc1 + x^2/rhoc2 + 2*x*(1-x^b)/rhoc12)", "x")
        # Out[]: (2*b*x**b/rhoc12 - 2*x/rhoc2 - 2*(1 - x**b)/rhoc12 -
                # (2*x - 2)/rhoc1)/(x**2/rhoc2 + 2*x*(1 - x**b)/rhoc12 +
                # (1 - x)**2/rhoc1)**2
        drhonx = M/1000*(2*b*x**b/rhoc12 - 2*(1-x**b)/rhoc12
                         - 2*x/rhoc2m + 2*(1-x)/rhoc1m)/(
                         2*x*(1-x**b)/rhoc12 + x**2/rhoc2m
                         + (1-x)**2/rhoc1m)**2

        tau = Tn/T
        delta = rho/rhon

        water = IAPWS95()
        phi1 = water._phir(tau, delta)

        ammonia = NH3()
        phi2 = ammonia._phir(tau, delta)

        Dphi = self._Dphir(tau, delta, x)

        prop = {}
        prop["tau"] = tau
        prop["delta"] = delta
        prop["fir"] = (1-x)*phi1["fir"] + x*phi2["fir"] + Dphi["fir"]
        prop["firt"] = (1-x)*phi1["firt"] + x*phi2["firt"] + Dphi["firt"]
        prop["firtt"] = (1-x)*phi1["firtt"] + x*phi2["firtt"] + Dphi["firtt"]
        prop["fird"] = (1-x)*phi1["fird"] + x*phi2["fird"] + Dphi["fird"]
        prop["firdd"] = (1-x)*phi1["firdd"] + x*phi2["firdd"] + Dphi["firdd"]
        prop["firdt"] = (1-x)*phi1["firdt"] + x*phi2["firdt"] + Dphi["firdt"]
        prop["firx"] = -phi1["fir"] + phi2["fir"] + Dphi["firx"]
        prop["F"] = prop["firx"] - delta/rhon*drhonx*prop["fird"] + \
            tau/Tn*dTnx*prop["firt"]
        return prop

    @staticmethod
    def _Dphir(tau, delta, x):
        """Departure function to the residual contribution to the free
        Helmholtz energy

        Parameters
        ----------
        tau : float
            Adimensional temperature, [-]
        delta : float
            Adimensional density, [-]
        x : float
            Mole fraction of ammonia in mixture, [mol/mol]

        Returns
        -------
        prop : dict
            Dictionary with departure contribution to the residual adimensional
            helmholtz energy and derivatives:

                * fir  [-]
                * firt: [∂Δfir/∂τ]δ,x  [-]
                * fird: [∂Δfir/∂δ]τ,x  [-]
                * firtt: [∂²Δfir/∂τ²]δ,x  [-]
                * firdt: [∂²Δfir/∂τ∂δ]x  [-]
                * firdd: [∂²Δfir/∂δ²]τ,x  [-]
                * firx: [∂Δfir/∂x]τ,δ  [-]

        References
        ----------
        IAPWS, Guideline on the IAPWS Formulation 2001 for the Thermodynamic
        Properties of Ammonia-Water Mixtures,
        http://www.iapws.org/relguide/nh3h2o.pdf, Eq 8
        """
        fx = x*(1-x**0.5248379)
        dfx = 1-1.5248379*x**0.5248379

        # Polinomial terms
        n = -1.855822e-2
        t = 1.5
        d = 4
        fir = n*delta**d*tau**t
        fird = n*d*delta**(d-1)*tau**t
        firdd = n*d*(d-1)*delta**(d-2)*tau**t
        firt = n*t*delta**d*tau**(t-1)
        firtt = n*t*(t-1)*delta**d*tau**(t-2)
        firdt = n*t*d*delta**(d-1)*tau**(t-1)
        firx = dfx*n*delta**d*tau**t

        # Exponential terms
        nr2 = [5.258010e-2, 3.552874e-10, 5.451379e-6, -5.998546e-13,
               -3.687808e-6]
        t2 = [0.5, 6.5, 1.75, 15, 6]
        d2 = [5, 15, 12, 12, 15]
        c2 = [1, 1, 1, 1, 2]
        for n, d, t, c in zip(nr2, d2, t2, c2):
            fir += n*delta**d*tau**t*exp(-delta**c)
            fird += n*exp(-delta**c)*delta**(d-1)*tau**t*(d-c*delta**c)
            firdd += n*exp(-delta**c)*delta**(d-2)*tau**t * \
                ((d-c*delta**c)*(d-1-c*delta**c)-c**2*delta**c)
            firt += n*t*delta**d*tau**(t-1)*exp(-delta**c)
            firtt += n*t*(t-1)*delta**d*tau**(t-2)*exp(-delta**c)
            firdt += n*t*delta**(d-1)*tau**(t-1)*(d-c*delta**c)*exp(
                -delta**c)
            firx += dfx*n*delta**d*tau**t*exp(-delta**c)

        # Exponential terms with composition
        nr3 = [0.2586192, -1.368072e-8, 1.226146e-2, -7.181443e-2, 9.970849e-2,
               1.0584086e-3, -0.1963687]
        t3 = [-1, 4, 3.5, 0, -1, 8, 7.5]
        d3 = [4, 15, 4, 5, 6, 10, 6]
        c3 = [1, 1, 1, 1, 2, 2, 2]
        for n, d, t, c in zip(nr3, d3, t3, c3):
            fir += x*n*delta**d*tau**t*exp(-delta**c)
            fird += x*n*exp(-delta**c)*delta**(d-1)*tau**t*(d-c*delta**c)
            firdd += x*n*exp(-delta**c)*delta**(d-2)*tau**t * \
                ((d-c*delta**c)*(d-1-c*delta**c)-c**2*delta**c)
            firt += x*n*t*delta**d*tau**(t-1)*exp(-delta**c)
            firtt += x*n*t*(t-1)*delta**d*tau**(t-2)*exp(-delta**c)
            firdt += x*n*t*delta**(d-1)*tau**(t-1)*(d-c*delta**c)*exp(
                -delta**c)
            firx += x*dfx*n*delta**d*tau**t*exp(-delta**c)

        n = -0.7777897
        t = 4
        d = 2
        c = 2
        fir += x**2*n*delta**d*tau**t*exp(-delta**c)
        fird += x**2*n*exp(-delta**c)*delta**(d-1)*tau**t*(d-c*delta**c)
        firdd += x**2*n*exp(-delta**c)*delta**(d-2)*tau**t * \
            ((d-c*delta**c)*(d-1-c*delta**c)-c**2*delta**c)
        firt += x**2*n*t*delta**d*tau**(t-1)*exp(-delta**c)
        firtt += x**2*n*t*(t-1)*delta**d*tau**(t-2)*exp(-delta**c)
        firdt += x**2*n*t*delta**(d-1)*tau**(t-1)*(d-c*delta**c)*exp(
            -delta**c)
        firx += x**2*dfx*n*delta**d*tau**t*exp(-delta**c)

        prop = {}
        prop["fir"] = fir*fx
        prop["firt"] = firt*fx
        prop["firtt"] = firtt*fx
        prop["fird"] = fird*fx
        prop["firdd"] = firdd*fx
        prop["firdt"] = firdt*fx
        prop["firx"] = firx
        return prop


def Ttr(x):
    """Equation for the triple point of ammonia-water mixture

    Parameters
    ----------
    x : float
        Mole fraction of ammonia in mixture, [mol/mol]

    Returns
    -------
    Ttr : float
        Triple point temperature, [K]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 0 ≤ x ≤ 1

    References
    ----------
    IAPWS, Guideline on the IAPWS Formulation 2001 for the Thermodynamic
    Properties of Ammonia-Water Mixtures,
    http://www.iapws.org/relguide/nh3h2o.pdf, Eq 9
    """
    if 0 <= x <= 0.33367:
        Tr = 273.16*(1-0.3439823*x-1.3274271*x**2-274.973*x**3)
    elif 0.33367 < x <= 0.58396:
        Tr = 193.549*(1-4.987368*(x-0.5)**2)
    elif 0.58396 < x <= 0.81473:
        Tr = 194.38*(1-4.886151*(x-2/3)**2+10.37298*(x-2/3)**3)
    elif 0.81473 < x <= 1:
        Tr = 195.495*(1-0.323998*(1-x)-15.87560*(1-x)**4)
    else:
        raise NotImplementedError("Incoming out of bound")
    return Tr
