#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Guideline on the IAPWS Formulation 2001 for the Thermodynamic Properties of
Ammonia-Water Mistures
"""


from __future__ import division
from math import exp, log, pi
import warnings

from scipy.constants import Boltzmann as kb
from .iapws95 import MEoS


class NH3(MEoS):
    """Multiparameter equation of state for ammonia"""
    name = "ammonia"
    CASNumber = "7664-41-7"
    formula = "NH3"
    synonym = "R-717"
    rhoc = 225.
    Tc = 405.40
    Pc = 11333.0  # kPa
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
        # FIXME: Dont work
        ek = 386
        sigma = 0.2957

        rho = rho/self.M
        T_ = T/ek

        # Eq 4
        a = [4.99318220, -0.61122364, 0.0, 0.18535124, -0.11160946]
        omega = exp(sum([ai*log(T_)**i for i, ai in enumerate(a)]))

        # Eq 2, Zero-Density Limit
        muo = 0.021357*(T*self.M)**0.5/sigma**2/omega

        # Eq 8, Viscosity virial coefficient
        cv = [-0.17999496e1, 0.46692621e2, -0.53460794e3, 0.33604074e4,
              -0.13019164e5, 0.33414230e5, -0.58711743e5, 0.71426686e5,
              -0.59834012e5, 0.33652741e5, -0.1202735e5, 0.24348205e4,
              -0.20807957e3]
        Bn = 0.6022137*sigma**3*sum([c*T_**(-i/2) for i, c in enumerate(cv)])
        # Eq 7
        mub = Bn*muo*rho

        # Eq 10
        dij = [2.19664285e-1, -0.83651107e-1, 0.17366936e-2, -0.64250359e-2,
               1.67668649e-4, -1.49710093e-4, 0.77012274e-4]
        ji = [2, 4, 0, 1, 2, 3, 4]
        ii = [2, 2, 3, 3, 4, 4, 4]
        mur = sum([d/T_**j*rho**i for d, j, i in zip(dij, ji, ii)])

        # Eq 1
        mu = muo + mub + mur
        return mu*1e-6

    def _thermo(self, rho, T, fase):
        """Equation for the thermal conductivity

        Parameters
        ----------
        rho : float
            Density [kg/m³]
        T : float
            Temperature [K]
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
        Lo = sum([n*T**i for i, n in enumerate(no)])

        # Eq 7
        nb = [0.16207e-3, 0.12038e-5, -0.23139e-8, 0.32749e-11]
        L_ = sum([n*rho**(i+1) for i, n in enumerate(nb)])

        # Critical enchancement
        t = abs(T-405.4)/405.4
        dPT = 1e5*(2.18-0.12/exp(17.8*t))
        nb = 1e-5*(2.6+1.6*t)

        DL = 1.2*kb*T**2/6/pi/nb/(1.34e-10/t**0.63*(1+t**0.5))*dPT**2 * \
            0.423e-8/t**1.24*(1+t**0.5/0.7)

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

