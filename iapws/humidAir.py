#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Guideline on the IAPWS Formulation 2001 for the Thermodynamic Properties of
Ammonia-Water Mistures
"""


from __future__ import division
from math import exp, log
import warnings

from ._iapws import M as Mw
from .iapws95 import MEoS, IAPWS95


Ma = 28.96546  # g/mol
R = 8.314472  # J/molK


def _virial(T):
    """Virial equations for humid air

    Parameters
    ----------
    T : float
        Temperature [K]

    Returns
    -------
    prop : float
        dictionary with critical coefficient
        Baa: Second virial coefficient of dry air [m³/mol]
        Baw: Second air-water cross virial coefficient [m³/mol]
        Bww: Second virial coefficient of water [m³/mol]
        Caaa: Third virial coefficient of dry air [m⁶/mol]
        Caaw: Third air-water cross virial coefficient [m⁶/mol]
        Caww: Third air-water cross virial coefficient [m⁶/mol]
        Cwww: Third virial coefficient of dry air [m⁶/mol]
        Bawt: dBaw/dT [m³/molK]
        Bawtt: d²Baw/dT² [m³/molK²]
        Caawt: dCaaw/dT [m⁶/molK]
        Caawtt: d²Caaw/dT² [m⁶/molK²]
        Cawwt: dCaww/dT [m⁶/molK]
        Cawwtt: d²Caww/dT² [m⁶/molK²]

    Raises
    ------
    Warning : If T isn't in range of validity
        * Baa: 60 ≤ T ≤ 2000
        * Baw: 130 ≤ T ≤ 2000
        * Bww: 130 ≤ T ≤ 1273
        * Caaa: 60 ≤ T ≤ 2000
        * Caaw: 193 ≤ T ≤ 493
        * Caww: 173 ≤ T ≤ 473
        * Cwww: 130 ≤ T ≤ 1273

    Examples
    --------
    >>> _virial(200)["Baa"]
    -3.92722567e-5

    References
    ----------
    IAPWS, Guideline on a Virial Equation for the Fugacity of H2O in Humid Air,
    http://www.iapws.org/relguide/VirialFugacity.html
    IAPWS, Guideline on an Equation of State for Humid Air in Contact with
    Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
    Thermodynamic Properties of Seawater, Table 10,
    http://www.iapws.org/relguide/SeaAir.html
    """
    # Check input parameters
    if T < 60 or T > 2000:
        warnings.warn("Baa out of validity range")
    if T < 130 or T > 2000:
        warnings.warn("Baw out of validity range")
    if T < 130 or T > 1273:
        warnings.warn("Bww out of validity range")
    if T < 60 or T > 2000:
        warnings.warn("Caaa out of validity range")
    if T < 193 or T > 493:
        warnings.warn("Caaw out of validity range")
    if T < 173 or T > 473:
        warnings.warn("Caww out of validity range")
    if T < 130 or T > 1273:
        warnings.warn("Cwww out of validity range")

    T_ = T/100
    tau = IAPWS95.Tc/T

    # Table 1
    # Reorganizated to easy use in equations
    tb = [-0.5, 0.875, 1, 4, 6, 12, 7]
    nb = [0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1,
          -0.66856572307965, 0.20433810950965, -0.66212605039687e-4,
          -0.10793600908932]
    bc = [0.5, 0.75, 1, 5, 1, 9, 10]
    nc = [0.31802509345418, -0.26145533859358, -0.19232721156002,
          -0.25709043003438, 0.17611491008752e-1, 0.22132295167546,
          -0.40247669763528]
    bc2 = [4, 6, 12]
    nc2 = [-0.66856572307965, 0.20433810950965, -0.66212605039687e-4]

    # Table 2
    ai = [3.5, 3.5]
    bi = [0.85, 0.95]
    Bi = [0.2, 0.2]
    ni = [-0.14874640856724, 0.31806110878444]
    Ci = [28, 32]
    Di = [700, 800]
    Ai = [0.32, 0.32]
    betai = [0.3, 0.3]

    # Eq 5
    sum1 = sum([n*tau**t for n, t in zip(nb, tb)])
    sum2 = 0
    for n, b, B, A, C, D in zip(ni, bi, Bi, Ai, Ci, Di):
        sum2 += n*((A+1-tau)**2+B)**b*exp(-C-D*(tau-1)**2)
    Bww = Mw/IAPWS95.rhoc*(sum1+sum2)

    # Eq 6
    sum1 = sum([n*tau**t for n, t in zip(nc, bc)])
    sum2 = sum([n*tau**t for n, t in zip(nc2, bc2)])
    sum3 = 0
    for a, b, B, n, C, D, A, beta in zip(ai, bi, Bi, ni, Ci, Di, Ai, betai):
        Tita = A+1-tau
        sum3 += n*(C*(Tita**2+B)-b*(A*Tita/beta+B*a))*(Tita**2+B)**(b-1) * \
            exp(-C-D*(tau-1)**2)
    Cwww = 2*(Mw/IAPWS95.rhoc)**2*(sum1-sum2+2*sum3)

    # Table 3
    ai = [0.482737e-3, 0.105678e-2, -0.656394e-2, 0.294442e-1, -0.319317e-1]
    bi = [-10.728876, 34.7802, -38.3383, 33.406]
    ci = [66.5687, -238.834, -176.755]
    di = [-0.237, -1.048, -3.183]

    Baw = 1e-6*sum([c*T_**d for c, d in zip(ci, di)])                  # Eq 7
    Caaw = 1e-6*sum([a/T_**i for i, a in enumerate(ai)])               # Eq 8
    Caww = -1e-6*exp(sum([b/T_**i for i, b in enumerate(bi)]))         # Eq 9

    # Eq T56
    Bawt = 1e-6*T_/T*sum([c*d*T_**(d-1) for c, d in zip(ci, di)])
    # Eq T57
    Bawtt = 1e-6*T_**2/T**2*sum(
        [c*d*(d-1)*T_**(d-2) for c, d in zip(ci, di)])
    # Eq T59
    Caawt = -1e-6*T_/T*sum([i*a*T_**(-i-1) for i, a in enumerate(ai)])
    # Eq T60
    Caawtt = 1e-6*T_**2/T**2*sum(
        [i*(i+1)*a*T_**(-i-2) for i, a in enumerate(ai)])
    # Eq T62
    Cawwt = 1e-6*T_/T*sum([i*b*T_**(-i-1) for i, b in enumerate(bi)]) * \
        exp(sum([b/T_**i for i, b in enumerate(bi)]))
    # Eq T63
    Cawwtt = -1e-6*T_**2/T**2*((
        sum([i*(i+1)*b*T_**(-i-2) for i, b in enumerate(bi)]) +
        sum([i*b*T_**(-i-1) for i, b in enumerate(bi)])**2) *
        exp(sum([b/T_**i for i, b in enumerate(bi)])))

    # Table 4
    # Reorganizated to easy use in equations
    ji = [0, 0.33, 1.01, 1.6, 3.6, 3.5]
    ni = [0.118160747229, 0.713116392079, -0.161824192067e1, -0.101365037912,
          -0.146629609713, 0.148287891978e-1]
    tau = 132.6312/T

    Baa = 1/10.4477*sum([n*tau**j for j, n in zip(ji, ni)])          # Eq 10
    Caaa = 2/10.4477**2*(0.714140178971e-1+0.101365037912*tau**1.6)  # Eq 11

    prop = {}
    prop["Baa"] = Baa/1000
    prop["Baw"] = Baw
    prop["Bww"] = Bww/1000
    prop["Caaa"] = Caaa/1e6
    prop["Caaw"] = Caaw
    prop["Caww"] = Caww
    prop["Cwww"] = Cwww/1e6
    prop["Bawt"] = Bawt
    prop["Bawtt"] = Bawtt
    prop["Caawt"] = Caawt
    prop["Caawtt"] = Caawtt
    prop["Cawwt"] = Cawwt
    prop["Cawwtt"] = Cawwtt
    return prop


def _fugacity(T, P, x):
    """Fugacity equation for humid air

    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [MPa]
    x : float
        Mole fraction of water-vapor [-]

    Returns
    -------
    fv : float
        fugacity coefficient [MPa]

    Raises
    ------
    NotImplementedError : If input isn't in range of validity
        * 193 ≤ T ≤ 473
        * 0 ≤ P ≤ 5
        * 0 ≤ x ≤ 1
        Really the xmax is the xsaturation but isn't implemented

    Examples
    --------
    >>> _fugacity(300, 1, 0.1)
    0.0884061686

    References
    ----------
    IAPWS, Guideline on a Virial Equation for the Fugacity of H2O in Humid Air,
    http://www.iapws.org/relguide/VirialFugacity.html
    """
    # Check input parameters
    if T < 193 or T > 473 or P < 0 or P > 5 or x < 0 or x > 1:
        raise(NotImplementedError("Input not in range of validity"))

    R = 8.314462  # J/molK

    # Virial coefficients
    vir = _virial(T)

    # Eq 3
    beta = x*(2-x)*vir["Bww"]+(1-x)**2*(2*vir["Baw"]-vir["Baa"])

    # Eq 4
    gamma = x**2*(3-2*x)*vir["Cwww"] + \
        (1-x)**2*(6*x*vir["Caww"]+3*(1-2*x)*vir["Caaw"]-2*(1-x)*vir["Caaa"]) +\
        (x**2*vir["Bww"]+2*x*(1-x)*vir["Baw"]+(1-x)**2*vir["Baa"]) * \
        (x*(3*x-4)*vir["Bww"]+2*(1-x)*(3*x-2)*vir["Baw"]+3*(1-x)**2*vir["Baa"])

    # Eq 2
    fv = x*P*exp(beta*P*1e6/R/T+0.5*gamma*(P*1e6/R/T)**2)
    return fv


class MEoSBlend(MEoS):
    """Special meos class to implement pseudocomponent blend and defining its
    ancillary dew and bubble point"""
    @classmethod
    def _dewP(cls, T):
        """Using ancillary equation return the pressure of dew point"""
        c = cls._blend["dew"]
        Tj = cls._blend["Tj"]
        Pj = cls._blend["Pj"]
        Tita = 1-T/Tj

        suma = 0
        for i, n in zip(c["i"], c["n"]):
            suma += n*Tita**(i/2.)
        P = Pj*exp(Tj/T*suma)
        return P

    @classmethod
    def _bubbleP(cls, T):
        """Using ancillary equation return the pressure of bubble point"""
        c = cls._blend["bubble"]
        Tj = cls._blend["Tj"]
        Pj = cls._blend["Pj"]
        Tita = 1-T/Tj

        suma = 0
        for i, n in zip(c["i"], c["n"]):
            suma += n*Tita**(i/2.)
        P = Pj*exp(Tj/T*suma)
        return P


class Air(MEoSBlend):
    """Multiparameter equation of state for Air as pseudocomponent"""
    name = "air"
    CASNumber = "1"
    formula = "N2+Ar+O2"
    synonym = "R-729"
    rhoc = 10.4477*Ma
    Tc = 132.6306
    Pc = 3786.0  # kPa
    M = Ma
    Tt = 59.75
    Tb = 78.903
    f_acent = 0.0335
    momentoDipolar = 0.0

    Fi0 = {"ao_log": [1, 2.490888032],
           "pow": [-3, -2, -1, 0, 1, 1.5],
           "ao_pow": [0.6057194e-7, -0.210274769e-4, -0.158860716e-3,
                      9.7450251743948, 10.0986147428912, -0.19536342e-3],
           "ao_exp": [0.791309509, 0.212236768],
           "titao": [25.36365, 16.90741],
           "ao_exp2": [-0.197938904],
           "titao2": [87.31279],
           "sum2": [2./3]
           }

    _constants = {
        "R": 8.31451,
        "Tref": 132.6312, "rhoref": 10.4477*Ma,

        "nr1": [0.118160747229, 0.713116392079, -0.161824192067e1,
                0.714140178971e-1, -0.865421396646e-1, 0.134211176704,
                0.112626704218e-1, -0.420533228842e-1, 0.349008431982e-1,
                0.164957183186e-3],
        "d1": [1, 1, 1, 2, 3, 3, 4, 4, 4, 6],
        "t1": [0, 0.33, 1.01, 0, 0, 0.15, 0, 0.2, 0.35, 1.35],

        "nr2": [-0.101365037912, -0.173813690970, -0.472103183731e-1,
                -0.122523554253e-1, -0.146629609713, -0.316055879821e-1,
                0.233594806142e-3, 0.148287891978e-1, -0.938782884667e-2],
        "d2": [1, 3, 5, 6, 1, 3, 11, 1, 3],
        "t2": [1.6, 0.8, 0.95, 1.25, 3.6, 6, 3.25, 3.5, 15],
        "c2": [1, 1, 1, 1, 2, 2, 2, 3, 3],
        "gamma2": [1]*9}

    _blend = {
        "Tj": 132.6312, "Pj": 3.78502,
        "dew": {"i": [1, 2, 5, 8],
                "n": [-0.1567266, -5.539635, 0.7567212, -3.514322]},
        "bubble": {"i": [1, 2, 3, 4, 5, 6],
                   "n": [0.2260724, -7.080499, 5.700283, -12.44017, 17.81926,
                         -10.81364]}}

    _melting = {"eq": 1, "Tref": Tb, "Pref": 5.265,
                "Tmin": 59.75, "Tmax": 2000.0,
                "a1": [1, 0.354935e5, -0.354935e5],
                "exp1": [0, 0.178963e1, 0],
                "a2": [], "exp2": [], "a3": [], "exp3": []}
    _rhoG = {
        "eq": 3,
        "ao": [-0.20466e1, -0.4752e1, -0.13259e2, -0.47652e2],
        "exp": [0.41, 1, 2.8, 6.5]}
    _Pv = {
        "ao": [-0.1567266, -0.5539635e1, 0.7567212, -0.3514322e1],
        "exp": [0.5, 1, 2.5, 4]}

    def _surface(self, T):
        """Equation for the surface tension"""
        tau = 1-T/self.Tc
        tension = 0
        sigmai = [0.03046]
        ni = [1.28]
        for sigma, n in zip(sigmai, ni):
            tension += sigma*tau**n
        return tension

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
        """
        Tc = 132.6312
        rhoc = 10.4477*cls.M
        Ni = [44.3413, -240.073, 285.139, -88.3366]
        ti = [0.65, 0.85, 0.95, 1.1]
        Tita = 1-T/Tc
        suma = 1
        for n, t in zip(Ni, ti):
            suma += n*Tita**t
        suma -= 0.892181*log(T/Tc)
        rho = suma*rhoc
        return rho

    @staticmethod
    def _visco(rho, T, fase=None):
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
        Lemmon, E.W. and Jacobsen, R.T., Viscosity and Thermal Conductivity
        Equations for Nitrogen, Oxygen, Argon, and Air, Int. J. Thermophys.,
        25:21-69, 2004. doi:10.1023/B:IJOT.0000022327.04529.f3
        """
        ek = 103.3
        sigma = 0.36
        M = 28.9586
        rhoc = 10.4477*M
        tau = 132.6312/T
        delta = rho/rhoc

        b = [0.431, -0.4623, 0.08406, 0.005341, -0.00331]
        T_ = log(T/ek)
        suma = 0
        for i, bi in enumerate(b):
            suma += bi*T_**i
        omega = exp(suma)

        # Eq 2
        muo = 0.0266958*(M*T)**0.5/(sigma**2*omega)

        n_poly = [10.72, 1.122, 0.002019, -8.876, -0.02916]
        t_poly = [.2, .05, 2.4, .6, 3.6]
        d_poly = [1, 4, 9, 1, 8]
        l_poly = [0, 0, 0, 1, 1]
        g_poly = [0, 0, 0, 1, 1]

        # Eq 3
        mur = 0
        for n, t, d, l, g in zip(n_poly, t_poly, d_poly, l_poly, g_poly):
            mur += n*tau**t*delta**d*exp(-g*delta**l)

        # Eq 1
        mu = muo+mur
        return mu*1e-6

    def _thermo(self, rho, T, fase=None):
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
        Lemmon, E.W. and Jacobsen, R.T., Viscosity and Thermal Conductivity
        Equations for Nitrogen, Oxygen, Argon, and Air, Int. J. Thermophys.,
        25:21-69, 2004. doi:10.1023/B:IJOT.0000022327.04529.f3
        """
        ek = 103.3
        sigma = 0.36
        M = 28.9586
        rhoc = 10.4477*M
        tau = 132.6312/T
        delta = rho/rhoc

        b = [0.431, -0.4623, 0.08406, 0.005341, -0.00331]
        T_ = log(T/ek)
        suma = 0
        for i, bi in enumerate(b):
            suma += bi*T_**i
        omega = exp(suma)

        # Eq 2
        muo = 0.0266958*(M*T)**0.5/(sigma**2*omega)

        # Eq 5
        N = [1.308, 1.405, -1.036]
        t = [-1.1, -0.3]
        lo = N[0]*muo+N[1]*tau**t[0]+N[2]*tau**t[1]

        n_poly = [8.743, 14.76, -16.62, 3.793, -6.142, -0.3778]
        t_poly = [0.1, 0, 0.5, 2.7, 0.3, 1.3]
        d_poly = [1, 2, 3, 7, 7, 11]
        g_poly = [0, 0, 1, 1, 1, 1]
        l_poly = [0, 0, 2, 2, 2, 2]

        # Eq 6
        lr = 0
        for n, t, d, l, g in zip(n_poly, t_poly, d_poly, l_poly, g_poly):
            lr += n*tau**t*delta**d*exp(-g*delta**l)

        lc = 0
        # TODO: Critical enchancement
        # if fase:
        #     qd = 0.31e-9
        #     Gamma = 0.055
        #     Xio = 0.11e-9
        #     Tref = 265.262
        #     k = 1.380658e-23  # J/K

        #     # Eq 11
        #     X = self.Pc*rho/rhoc**2*fase.drhodP_T
        #     refst = self._Helmholtz(rho, Tref)
        #     drhodP_T = 1/refst["dpdrho"]
        #     Xref = 3.78502*rho/rhoc**2*drhodP_T

        #     # Eq 10
        #     bracket = X-Xref*Tref/T
        #     if bracket > 0:
        #         Xi = Xio*(bracket/Gamma)**(0.63/1.2415)

        #         Xq = Xi/qd
        #         # Eq 8
        #         Omega = 2/pi*((fase.cp-fase.cv)/fase.cp*atan(Xq) +
        #                       fase.cv/fase.cp*(Xq))
        #         # Eq 9
        #         Omega0 = 2/pi*(1-exp(-1/(1/Xq+Xq**2/3*rhoc**2/rho**2)))

        #         # Eq 7
        #         lc = rho*fase.cp*k*1.01*T/6/pi/Xi/fase.mu*(Omega-Omega0)
        #     else:
        #         lc = 0

        # Eq 4
        k = lo+lr+lc

        return k*1e-3


class HumidAir(object):

    def _fav(self, T, rho, A):
        """Table 6, pag 10"""
        water = IAPWS95()
        rhov = (1-A)*rho
        fv = water._derivDimensional(rhov, T)

        air = Air()
        rhoa = A*rho
        fa = air._derivDimensional(rhoa, T)

        fmix = self._fmix(T, rho, A)

        f = (1-A)*fv["fir"] + A*fa["fir"] + fmix["fir"]
        fA = -fv["fir"]-rhov*fv["fird"]+fa["fir"]+rhoa*fa["fird"]+fmix["fira"]
        ft = (1-A)*fv["firt"]+A*fa["firt"]+fmix["firt"]
        fd = (1-A)**2*fv["fird"]+A**2*fa["fird"]+fmix["fird"]
        faa = rho*(2*fv["fird"]+rhov*fv["firdd"] +
                   2*fa["fird"]+rhoa*fa["firdd"])+fmix["firaa"]
        fat = -fv["firt"]-rhov*fv["firdt"]+fa["firt"]+rhoa*fa["firdt"] + \
            fmix["firat"]
        fad = -(1-A)*(2*fv["fird"]+rhov*fv["firdd"]) + \
            A*(2*fa["fird"]+rhoa*fa["firdd"])+fmix["firad"]
        ftt = (1-A)*fv["firtt"]+A*fa["firtt"]+fmix["firtt"]
        ftd = (1-A)**2*fv["firdt"]+A**2*fa["firdt"]+fmix["firdt"]
        fdd = (1-A)**3*fv["firdd"]+A**3*fa["firdd"]+fmix["firdd"]

        prop = {}
        prop["fir"] = f
        prop["fira"] = fA
        prop["firt"] = ft
        prop["fird"] = fd
        prop["firaa"] = faa
        prop["firat"] = fat
        prop["firad"] = fad
        prop["firtt"] = ftt
        prop["firdt"] = ftd
        prop["firdd"] = fdd
        return prop

    def _fmix(self, T, rho, A):
        """Specific Helmholtz energy of air-water interaction

        Parameters
        ----------
        T : float
            Temperature [K]
        rho : float
            Density [kg/m³]
        A : float
            Mass fraction of dry air in humid air [kg/kg]

        Returns
        -------
        prop : float
            dictionary with helmholtz energy and derivatives
            fir
            fira: [∂fmix/∂A]T,ρ
            firt: [∂fmix/∂T]A,ρ
            fird: [∂fmix/∂ρ]A,T
            firaa: [∂²fmix/∂A²]T,ρ
            firat: [∂²fmix/∂A∂T]ρ
            firad: [∂²fmix/∂A∂ρ]T
            firtt: [∂²fmix/∂T²]A,ρ
            firdt: [∂²fmix/∂T∂ρ]A
            firdd: [∂²fmix/∂ρ²]A,T

        References
        ----------
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 10,
        http://www.iapws.org/relguide/SeaAir.html
        """
        Ma = Air.M/1000
        Mw = IAPWS95.M/1000
        vir = _virial(T)
        Baw = vir["Baw"]
        Bawt = vir["Bawt"]
        Bawtt = vir["Bawtt"]
        Caaw = vir["Caaw"]
        Caawt = vir["Caawt"]
        Caawtt = vir["Caawtt"]
        Caww = vir["Caww"]
        Cawwt = vir["Cawwt"]
        Cawwtt = vir["Cawwtt"]

        # Eq T45
        f = 2*A*(1-A)*rho*R*T/Ma/Mw*(Baw+3*rho/4*(A/Ma*Caaw+(1-A)/Mw*Caww))
        # Eq T46
        fa = 2*rho*R*T/Ma/Mw*((1-2*A)*Baw+3*rho/4*(
            A*(2-3*A)/Ma*Caaw+(1-A)*(1-3*A)/Mw*Caww))
        # Eq T47
        ft = 2*A*(1-A)*rho*R/Ma/Mw*(
            Baw+T*Bawt+3*rho/4*(A/Ma*(Caaw+T*Caawt)+(1-A)/Mw*(Caww+T*Cawwt)))
        # Eq T48
        fd = A*(1-A)*R*T/Ma/Mw*(2*Baw+3*rho*(A/Ma*Caaw+(1-A)/Mw*Caww))
        # Eq T49
        faa = rho*R*T/Ma/Mw*(-4*Baw+3*rho*((1-3*A)/Ma*Caaw-(2-3*A)/Mw*Caww))
        # Eq T50
        fat = 2*rho*R/Ma/Mw*(1-2*A)*(Baw+T*Bawt)+3*rho**2*R/2/Ma/Mw*(
            A*(2-3*A)/Ma*(Caaw+T*Caawt)+(1-A)*(1-3*A)/Mw*(Caww+T*Cawwt))
        # Eq T51
        fad = 2*R*T/Ma/Mw*((1-2*A)*Baw+3/2*rho*(
            A*(2-3*A)/Ma*Caaw+(1-A)*(1-3*A)/Mw*Caww))
        # Eq T52
        ftt = 2*A*(1-A)*rho*R/Ma/Mw*(2*Bawt+T*Bawtt+3*rho/4*(
            A/Ma*(2*Caawt+T*Caawtt)+(1-A)/Mw*(2*Cawwt+T*Cawwtt)))
        # Eq T53
        ftd = 2*A*(1-A)*R/Ma/Mw*(Baw+T*Bawt+3*rho/2*(
            A/Ma*(Caaw+T*Caawt)+(1-A)/Mw*(Caww+T*Cawwt)))
        # Eq T54
        fdd = 3*A*(1-A)*R*T/Ma/Mw*(A/Ma*Caaw+(1-A)/Mw*Caww)

        prop = {}
        prop["fir"] = f/1000
        prop["fira"] = fa/1000
        prop["firt"] = ft/1000
        prop["fird"] = fd/1000
        prop["firaa"] = faa/1000
        prop["firat"] = fat/1000
        prop["firad"] = fad/1000
        prop["firtt"] = ftt/1000
        prop["firdt"] = ftd/1000
        prop["firdd"] = fdd/1000
        return prop
