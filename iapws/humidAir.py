#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
# pylint: disable=too-many-lines, too-many-locals, too-many-statements
# pylint: disable=too-many-instance-attributes, too-many-boolean-expressions

"""
Module with Air-water mixture properties and related properties. The module
include:

    * :func:`_virial`: Virial equations for humid air
    * :func:`_fugacity`: Fugacity equation for humid air
    * :class:`MEoSBlend`: Special MEoS subclass to implement pseudocomponent
      blend with ancillary dew and bubble point
    * :class:`Air`: Multiparameter equation of state for Air as pseudocomponent
    * :class:`HumidAir`: Humid air mixture with complete functionality
"""


from __future__ import division
from math import exp, log, pi, atan
import warnings

from scipy.optimize import fsolve

from ._iapws import M as MW
from ._iapws import _Ice
from ._utils import deriv_G
from .iapws95 import MEoS, IAPWS95, mainClassDoc


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
    prop : dict
        Dictionary with critical coefficient:

            * Baa: Second virial coefficient of dry air, [m³/mol]
            * Baw: Second air-water cross virial coefficient, [m³/mol]
            * Bww: Second virial coefficient of water, [m³/mol]
            * Caaa: Third virial coefficient of dry air, [m⁶/mol]
            * Caaw: Third air-water cross virial coefficient, [m⁶/mol]
            * Caww: Third air-water cross virial coefficient, [m⁶/mol]
            * Cwww: Third virial coefficient of dry air, [m⁶/mol]
            * Bawt: dBaw/dT, [m³/molK]
            * Bawtt: d²Baw/dT², [m³/molK²]
            * Caawt: dCaaw/dT, [m⁶/molK]
            * Caawtt: d²Caaw/dT², [m⁶/molK²]
            * Cawwt: dCaww/dT, [m⁶/molK]
            * Cawwtt: d²Caww/dT², [m⁶/molK²]

    Notes
    -----
    Raise :class:`Warning` if T isn't in range of validity:

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

    # Virial coefficient for water
    # The paper use the specific formulation of virial, here using the general
    # formulation from helmholtz mEoS
    wt = IAPWS95()
    vir = wt._virial(T)
    Bww = vir["B"]/wt.rhoc*wt.M
    Cwww = vir["C"]/wt.rhoc**2*wt.M**2

    # Table 3
    ai = [0.482737e-3, 0.105678e-2, -0.656394e-2, 0.294442e-1, -0.319317e-1]
    bi = [-10.728876, 34.7802, -38.3383, 33.406]
    ci = [66.5687, -238.834, -176.755]
    di = [-0.237, -1.048, -3.183]

    Baw = 1e-6*sum(c*T_**d for c, d in zip(ci, di))                  # Eq 7
    Caaw = 1e-6*sum(a/T_**i for i, a in enumerate(ai))               # Eq 8
    Caww = -1e-6*exp(sum(b/T_**i for i, b in enumerate(bi)))         # Eq 9

    # Eq T56
    Bawt = 1e-6*T_/T*sum(c*d*T_**(d-1) for c, d in zip(ci, di))
    # Eq T57
    Bawtt = 1e-6*T_**2/T**2*sum(
        c*d*(d-1)*T_**(d-2) for c, d in zip(ci, di))
    # Eq T59
    Caawt = -1e-6*T_/T*sum(i*a*T_**(-i-1) for i, a in enumerate(ai))
    # Eq T60
    Caawtt = 1e-6*T_**2/T**2*sum(
        i*(i+1)*a*T_**(-i-2) for i, a in enumerate(ai))
    # Eq T62
    Cawwt = 1e-6*T_/T*sum(i*b*T_**(-i-1) for i, b in enumerate(bi)) * \
        exp(sum(b/T_**i for i, b in enumerate(bi)))
    # Eq T63
    Cawwtt = -1e-6*T_**2/T**2*((
        sum(i*(i+1)*b*T_**(-i-2) for i, b in enumerate(bi))
        + sum(i*b*T_**(-i-1) for i, b in enumerate(bi))**2)
        * exp(sum(b/T_**i for i, b in enumerate(bi))))

    # Virial coefficient for air, using too the general virial procedure
    air = Air()
    vir = air._virial(T)
    Baa = vir["B"]/air.rhoc*air.M
    Caaa = vir["C"]/air.rhoc**2*air.M**2

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
        Temperature, [K]
    P : float
        Pressure, [MPa]
    x : float
        Mole fraction of water-vapor, [-]

    Returns
    -------
    fv : float
        fugacity coefficient, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in range of validity:

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
        raise NotImplementedError("Input not in range of validity")

    Rg = 8.314462  # J/molK

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
    fv = x*P*exp(beta*P*1e6/Rg/T+0.5*gamma*(P*1e6/Rg/T)**2)
    return fv


class MEoSBlend(MEoS):
    """
    Special meos class to implement pseudocomponent blend and defining its
    ancillary dew and bubble point
    """

    _blend = {}

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


@mainClassDoc()
class Air(MEoSBlend):
    """Multiparameter equation of state for Air as pseudocomponent
    for internal procedures, see MEoS base class

    References
    ----------
    Lemmon, E.W., Jacobsen, R.T, Penoncello, S.G., Friend, D.G.; Thermodynamic
    Properties of Air and Mixtures of Nitrogen, Argon, and Oxygen From 60 to
    2000 K at Pressures to 2000 MPa. J. Phys. Chem. Ref. Data 29, 331 (2000).
    http://dx.doi.org/10.1063/1.1285884
    """

    name = "air"
    CASNumber = "1"
    formula = "N2+Ar+O2"
    synonym = "R-729"
    rhoc = 10.4477*Ma
    Tc = 132.6306
    Pc = 3.7860  # MPa
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
           "sum2": [2./3],
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
    _surf = {"sigma": [0.03046], "exp": [1.28]}
    _rhoG = {
        "eq": 3,
        "ao": [-0.20466e1, -0.4752e1, -0.13259e2, -0.47652e2],
        "exp": [0.41, 1, 2.8, 6.5]}
    _Pv = {
        "ao": [-0.1567266, -0.5539635e1, 0.7567212, -0.3514322e1],
        "exp": [0.5, 1, 2.5, 4]}

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
            Density, [kg/m³]
        T : float
            Temperature, [K]

        Returns
        -------
        μ : float
            Viscosity, [Pa·s]

        References
        ----------
        Lemmon, E.W., Jacobsen, R.T. Viscosity and Thermal Conductivity
        Equations for Nitrogen, Oxygen, Argon, and Air. Int. J. Thermophys. 25
        (1) (2004) 21-69. http://dx.doi.org/10.1023/B:IJOT.0000022327.04529.f3
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
            Density, [kg/m³]
        T : float
            Temperature, [K]
        fase: dict
            phase properties

        Returns
        -------
        k : float
            Thermal conductivity, [W/mK]

        References
        ----------
        Lemmon, E.W., Jacobsen, R.T. Viscosity and Thermal Conductivity
        Equations for Nitrogen, Oxygen, Argon, and Air. Int. J. Thermophys. 25
        (1) (2004) 21-69. http://dx.doi.org/10.1023/B:IJOT.0000022327.04529.f3
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
        # FIXME: Tiny desviation in the test in paper, 0.06% at critical point
        if fase:
            qd = 0.31
            Gamma = 0.055
            Xio = 0.11
            Tref = 265.262
            k = 1.380658e-23  # J/K

            # Eq 11
            X = self.Pc*rho/rhoc**2*fase.drhodP_T

            ref = Air()
            st = ref._Helmholtz(rho, Tref)
            drho = 1e3/self.R/Tref/(1+2*delta*st["fird"]+delta**2*st["firdd"])

            Xref = self.Pc*rho/rhoc**2*drho

            # Eq 10
            bracket = X-Xref*Tref/T
            if bracket > 0:
                Xi = Xio*(bracket/Gamma)**(0.63/1.2415)

                Xq = Xi/qd
                # Eq 8
                Omega = 2/pi*((fase.cp-fase.cv)/fase.cp*atan(Xq)
                              + fase.cv/fase.cp*(Xq))
                # Eq 9
                Omega0 = 2/pi*(1-exp(-1/(1/Xq+Xq**2/3*rhoc**2/rho**2)))

                # Eq 7
                lc = rho*fase.cp*k*1.01*T/6/pi/Xi/fase.mu*(Omega-Omega0)*1e15
            else:
                lc = 0

        # Eq 4
        k = lo+lr+lc

        return k*1e-3


class HumidAir(object):
    """
    Humid air class with complete functionality

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
    A : float
        Mass fraction of dry air in humid air, [kg/kg]
    xa : float
        Mole fraction of dry air in humid air, [-]
    W : float
        Mass fraction of water in humid air, [kg/kg]
    xw : float
        Mole fraction of water in humid air, [-]
    HR : float
        Humidity ratio, Mass fraction of water in dry air, [kg/kg]

    Notes
    -----
    * It needs two incoming properties of T, P, rho.
    * v as a alternate input parameter to rho
    * For composition need one of A, xa, W, xw, HR.

    The calculated instance has the following properties:

        * P: Pressure, [MPa]
        * T: Temperature, [K]
        * g: Specific Gibbs free energy, [kJ/kg]
        * a: Specific Helmholtz free energy, [kJ/kg]
        * v: Specific volume, [m³/kg]
        * rho: Density, [kg/m³]
        * h: Specific enthalpy, [kJ/kg]
        * u: Specific internal energy, [kJ/kg]
        * s: Specific entropy, [kJ/kg·K]
        * cp: Specific isobaric heat capacity, [kJ/kg·K]
        * w: Speed of sound, [m/s]

        * alfav: Isobaric cubic expansion coefficient, [1/K]
        * betas: Isoentropic temperature-pressure coefficient, [-]
        * xkappa: Isothermal Expansion Coefficient, [-]
        * ks: Adiabatic Compressibility, [1/MPa]

        * A: Mass fraction of dry air in humid air, [kg/kg]
        * W: Mass fraction of water in humid air, [kg/kg]
        * xa: Mole fraction of dry air, [-]
        * xw: Mole fraction of water, [-]
        * Pv: Partial pressure of water, [MPa]
        * xa_sat: Mole fraction of dry air at saturation state, [-]
        * mu: Relative chemical potential, [kJ/kg]
        * muw: Chemical potential of water, [kJ/kg]
        * M: Molar mass of humid air, [g/mol]
        * HR: Humidity ratio, Mass fraction of water in dry air, [kg/kg]
        * RH: Relative humidity, [-]
    """

    kwargs = {"T": 0.0,
              "P": 0.0,
              "rho": 0.0,
              "v": 0.0,
              "A": None,
              "xa": None,
              "W": None,
              "xw": None,
              "HR": None}

    status = 0
    msg = "Undefined"
    _mode = None
    _composition = None

    T = None
    rho = None
    v = None
    P = None
    s = None
    cp = None
    h = None
    g = None
    u = None
    alfav = None
    betas = None
    xkappa = None
    ks= None
    w = None

    A = None
    W = None
    mu = None
    muw = None
    M = None
    HR = None
    xa = None
    xw = None
    Pv = None

    xa_sat = None
    RH = None

    def __init__(self, **kwargs):
        """Constructor, define common constant and initinialice kwargs"""
        self.kwargs = HumidAir.kwargs.copy()
        self.__call__(**kwargs)

    def __call__(self, **kwargs):
        """Make instance callable to can add input parameter one to one"""
        # Check alernate input parameters
        if kwargs.get("v", None) is not None:
            kwargs["rho"] = 1./kwargs["v"]
            del kwargs["v"]
        if kwargs.get("W", None) is not None:
            kwargs["A"] = 1-kwargs["W"]
            del kwargs["W"]
        if kwargs.get("xw", None) is not None:
            kwargs["xa"] = 1-kwargs["xw"]
            del kwargs["xw"]
        if kwargs.get("HR", None) is not None:
            kwargs["A"] = 1/(1+kwargs["HR"])
            del kwargs["HR"]

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
        elif self.kwargs["P"] and self.kwargs["rho"]:
            self._mode = "Prho"

        # Composition definition
        self._composition = ""
        if self.kwargs["A"] is not None:
            self._composition = "A"
        elif self.kwargs["xa"] is not None:
            self._composition = "xa"

        return bool(self._mode) and bool(self._composition)

    def calculo(self):
        """Calculate procedure"""
        T = self.kwargs["T"]
        rho = self.kwargs["rho"]
        P = self.kwargs["P"]

        # Composition alternate definition
        if self._composition == "A":
            A = self.kwargs["A"]
        elif self._composition == "xa":
            xa = self.kwargs["xa"]
            A = xa/(1-(1-xa)*(1-MW/Ma))

        # Thermodynamic definition
        if self._mode == "TP":
            def f(rho):
                fav = self._fav(T, rho, A)
                return rho**2*fav["fird"]/1000-P
            rho = fsolve(f, 1)[0]
        elif self._mode == "Prho":
            def f(T):
                fav = self._fav(T, rho, A)
                return rho**2*fav["fird"]/1000-P
            T = fsolve(f, 300)[0]

        # General calculation procedure
        fav = self._fav(T, rho, A)

        # Common thermodynamic properties
        prop = self._prop(T, rho, fav)
        self.T = T
        self.rho = rho
        self.v = 1/rho
        self.P = prop["P"]
        self.s = prop["s"]
        self.cp = prop["cp"]
        self.h = prop["h"]
        self.g = prop["g"]
        self.u = self.h-self.P*1000*self.v
        self.alfav = prop["alfav"]
        self.betas = prop["betas"]
        self.xkappa = prop["xkappa"]
        self.ks = prop["ks"]
        self.w = prop["w"]

        # Coligative properties
        coligative = self._coligative(rho, A, fav)
        self.A = A
        self.W = 1-A
        self.mu = coligative["mu"]
        self.muw = coligative["muw"]
        self.M = coligative["M"]
        self.HR = coligative["HR"]
        self.xa = coligative["xa"]
        self.xw = coligative["xw"]
        self.Pv = (1-self.xa)*self.P

        # Saturation related properties
        A_sat = self._eq(self.T, self.P)
        if A_sat:
            self.xa_sat = A_sat*MW/Ma/(1-A_sat*(1-MW/Ma))
            self.RH = (1-self.xa)/(1-self.xa_sat)
        else:
            self.xa_sat = None
            self.RH = None
            self.msg = "Saturation state don't converge"
            self.status = 3

    def derivative(self, z, x, y):
        """
        Wrapper derivative for custom derived properties
        where x, y, z can be: P, T, v, rho, u, h, s, g, a
        """
        return deriv_G(self, z, x, y, self)

    def _eq(self, T, P):
        """Procedure for calculate the composition in saturation state

        Parameters
        ----------
        T : float
            Temperature [K]
        P : float
            Pressure [MPa]

        Returns
        -------
        Asat : float
            Saturation mass fraction of dry air in humid air [kg/kg]
        """
        # ao initial value of air mass fraction for iteration
        if T <= 273.16:
            ice = _Ice(T, P)
            gw = ice["g"]
            ao = 0.99999
        elif T <= 373.16:
            water = IAPWS95(T=T, P=P)
            gw = water.g
            ao = 0.99
        else:
            raise NotImplementedError("Incoming out of bound")

        def f(parr):
            rho, a = parr
            if a > 1:
                a = 1
            fa = self._fav(T, rho, a)
            muw = fa["fir"]+rho*fa["fird"]-a*fa["fira"]
            return gw-muw, rho**2*fa["fird"]/1000-P

        air = Air(T=T, P=P)
        rinput = fsolve(f, [air.rho, ao], full_output=True)
        Asat = rinput[0][1]
        if rinput[2] == 1 and 0 <= Asat <= 1:
            return Asat

        warnings.warn("Convergence failed")
        print(rinput)

    @staticmethod
    def _prop(T, rho, fav):
        """Thermodynamic properties of humid air

        Parameters
        ----------
        T : float
            Temperature, [K]
        rho : float
            Density, [kg/m³]
        fav : dict
            dictionary with helmholtz energy and derivatives

        Returns
        -------
        prop : dict
            Dictionary with thermodynamic properties of humid air:

                * P: Pressure, [MPa]
                * s: Specific entropy, [kJ/kgK]
                * cp: Specific isobaric heat capacity, [kJ/kgK]
                * h: Specific enthalpy, [kJ/kg]
                * g: Specific gibbs energy, [kJ/kg]
                * alfav: Thermal expansion coefficient, [1/K]
                * betas: Isentropic T-P coefficient, [K/MPa]
                * xkappa: Isothermal compressibility, [1/MPa]
                * ks: Isentropic compressibility, [1/MPa]
                * w: Speed of sound, [m/s]

        References
        ----------
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 5,
        http://www.iapws.org/relguide/SeaAir.html
        """
        prop = {}
        prop["P"] = rho**2*fav["fird"]/1000                             # Eq T1
        prop["s"] = -fav["firt"]                                        # Eq T2
        prop["cp"] = -T*fav["firtt"]+T*rho*fav["firdt"]**2/(            # Eq T3
            2*fav["fird"]+rho*fav["firdd"])
        prop["h"] = fav["fir"]-T*fav["firt"]+rho*fav["fird"]            # Eq T4
        prop["g"] = fav["fir"]+rho*fav["fird"]                          # Eq T5
        prop["alfav"] = fav["firdt"]/(2*fav["fird"]+rho*fav["firdd"])   # Eq T6
        prop["betas"] = 1000*fav["firdt"]/rho/(                         # Eq T7
            rho*fav["firdt"]**2-fav["firtt"]*(2*fav["fird"]+rho*fav["firdd"]))
        prop["xkappa"] = 1e3/(rho**2*(2*fav["fird"]+rho*fav["firdd"]))  # Eq T8
        prop["ks"] = 1000*fav["firtt"]/rho**2/(                         # Eq T9
            fav["firtt"]*(2*fav["fird"]+rho*fav["firdd"])-rho*fav["firdt"]**2)
        prop["w"] = (rho**2*1000*(fav["firtt"]*fav["firdd"]-fav["firdt"]**2)
                     / fav["firtt"]+2*rho*fav["fird"]*1000)**0.5       # Eq T10
        return prop

    @staticmethod
    def _coligative(rho, A, fav):
        """Miscelaneous properties of humid air

        Parameters
        ----------
        rho : float
            Density, [kg/m³]
        A : float
            Mass fraction of dry air in humid air, [kg/kg]
        fav : dict
            dictionary with helmholtz energy and derivatives

        Returns
        -------
        prop : dict
            Dictionary with calculated properties:

                * mu: Relative chemical potential, [kJ/kg]
                * muw: Chemical potential of water, [kJ/kg]
                * M: Molar mass of humid air, [g/mol]
                * HR: Humidity ratio, [-]
                * xa: Mole fraction of dry air, [-]
                * xw: Mole fraction of water, [-]

        References
        ----------
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 12,
        http://www.iapws.org/relguide/SeaAir.html
        """
        prop = {}
        prop["mu"] = fav["fira"]
        prop["muw"] = fav["fir"]+rho*fav["fird"]-A*fav["fira"]
        prop["M"] = 1/((1-A)/MW+A/Ma)
        prop["HR"] = 1/A-1
        prop["xa"] = A*MW/Ma/(1-A*(1-MW/Ma))
        prop["xw"] = 1-prop["xa"]
        return prop

    def _fav(self, T, rho, A):
        r"""Specific Helmholtz energy of humid air and derivatives

        Parameters
        ----------
        T : float
            Temperature, [K]
        rho : float
            Density, [kg/m³]
        A : float
            Mass fraction of dry air in humid air, [kg/kg]

        Returns
        -------
        prop : dict
          Dictionary with helmholtz energy and derivatives:

            * fir, [kJ/kg]
            * fira: :math:`\left.\frac{\partial f_{av}}{\partial A}\right|_{T,\rho}`, [kJ/kg]
            * firt: :math:`\left.\frac{\partial f_{av}}{\partial T}\right|_{A,\rho}`, [kJ/kgK]
            * fird: :math:`\left.\frac{\partial f_{av}}{\partial \rho}\right|_{A,T}`, [kJ/m³kg²]
            * firaa: :math:`\left.\frac{\partial^2 f_{av}}{\partial A^2}\right|_{T, \rho}`, [kJ/kg]
            * firat: :math:`\left.\frac{\partial^2 f_{av}}{\partial A \partial T}\right|_{\rho}`, [kJ/kgK]
            * firad: :math:`\left.\frac{\partial^2 f_{av}}{\partial A \partial \rho}\right|_T`, [kJ/m³kg²]
            * firtt: :math:`\left.\frac{\partial^2 f_{av}}{\partial T^2}\right|_{A, \rho}`, [kJ/kgK²]
            * firdt: :math:`\left.\frac{\partial^2 f_{av}}{\partial \rho \partial T}\right|_A`, [kJ/m³kg²K]
            * firdd: :math:`\left.\frac{\partial^2 f_{av}}{\partial \rho^2}\right|_{A, T}`, [kJ/m⁶kg³]

        References
        ----------
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 6,
        http://www.iapws.org/relguide/SeaAir.html
        """  # noqa
        water = IAPWS95()
        rhov = (1-A)*rho
        fv = water._derivDimensional(rhov, T)

        air = Air()
        rhoa = A*rho
        fa = air._derivDimensional(rhoa, T)

        fmix = self._fmix(T, rho, A)

        prop = {}
        # Eq T11
        prop["fir"] = (1-A)*fv["fir"] + A*fa["fir"] + fmix["fir"]
        # Eq T12
        prop["fira"] = -fv["fir"]-rhov*fv["fird"]+fa["fir"] + \
            rhoa*fa["fird"]+fmix["fira"]
        # Eq T13
        prop["firt"] = (1-A)*fv["firt"]+A*fa["firt"]+fmix["firt"]
        # Eq T14
        prop["fird"] = (1-A)**2*fv["fird"]+A**2*fa["fird"]+fmix["fird"]
        # Eq T15
        prop["firaa"] = rho*(2*fv["fird"]+rhov*fv["firdd"]
                             + 2*fa["fird"]+rhoa*fa["firdd"])+fmix["firaa"]
        # Eq T16
        prop["firat"] = -fv["firt"]-rhov*fv["firdt"]+fa["firt"] + \
            rhoa*fa["firdt"]+fmix["firat"]
        # Eq T17
        prop["firad"] = -(1-A)*(2*fv["fird"]+rhov*fv["firdd"]) + \
            A*(2*fa["fird"]+rhoa*fa["firdd"])+fmix["firad"]
        # Eq T18
        prop["firtt"] = (1-A)*fv["firtt"]+A*fa["firtt"]+fmix["firtt"]
        # Eq T19
        prop["firdt"] = (1-A)**2*fv["firdt"]+A**2*fa["firdt"]+fmix["firdt"]
        # Eq T20
        prop["firdd"] = (1-A)**3*fv["firdd"]+A**3*fa["firdd"]+fmix["firdd"]
        return prop

    @staticmethod
    def _fmix(T, rho, A):
        r"""Specific Helmholtz energy of air-water interaction

        Parameters
        ----------
        T : float
            Temperature, [K]
        rho : float
            Density, [kg/m³]
        A : float
            Mass fraction of dry air in humid air, [kg/kg]

        Returns
        -------
        prop : dict
          Dictionary with helmholtz energy and derivatives:

            * fir, [kJ/kg]
            * fira: :math:`\left.\frac{\partial f_{mix}}{\partial A}\right|_{T,\rho}`, [kJ/kg]
            * firt: :math:`\left.\frac{\partial f_{mix}}{\partial T}\right|_{A,\rho}`, [kJ/kgK]
            * fird: :math:`\left.\frac{\partial f_{mix}}{\partial \rho}\right|_{A,T}`, [kJ/m³kg²]
            * firaa: :math:`\left.\frac{\partial^2 f_{mix}}{\partial A^2}\right|_{T, \rho}`, [kJ/kg]
            * firat: :math:`\left.\frac{\partial^2 f_{mix}}{\partial A \partial T}\right|_{\rho}`, [kJ/kgK]
            * firad: :math:`\left.\frac{\partial^2 f_{mix}}{\partial A \partial \rho}\right|_T`, [kJ/m³kg²]
            * firtt: :math:`\left.\frac{\partial^2 f_{mix}}{\partial T^2}\right|_{A, \rho}`, [kJ/kgK²]
            * firdt: :math:`\left.\frac{\partial^2 f_{mix}}{\partial \rho \partial T}\right|_A`, [kJ/m³kg²K]
            * firdd: :math:`\left.\frac{\partial^2 f_{mix}}{\partial \rho^2}\right|_{A, T}`, [kJ/m⁶kg³]

        References
        ----------
        IAPWS, Guideline on an Equation of State for Humid Air in Contact with
        Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the
        Thermodynamic Properties of Seawater, Table 10,
        http://www.iapws.org/relguide/SeaAir.html
        """
        ma = Air.M/1000
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
        f = 2*A*(1-A)*rho*R*T/ma/Mw*(Baw+3*rho/4*(A/ma*Caaw+(1-A)/Mw*Caww))
        # Eq T46
        fa = 2*rho*R*T/ma/Mw*((1-2*A)*Baw+3*rho/4*(
            A*(2-3*A)/ma*Caaw+(1-A)*(1-3*A)/Mw*Caww))
        # Eq T47
        ft = 2*A*(1-A)*rho*R/ma/Mw*(
            Baw+T*Bawt+3*rho/4*(A/ma*(Caaw+T*Caawt)+(1-A)/Mw*(Caww+T*Cawwt)))
        # Eq T48
        fd = A*(1-A)*R*T/ma/Mw*(2*Baw+3*rho*(A/ma*Caaw+(1-A)/Mw*Caww))
        # Eq T49
        faa = rho*R*T/ma/Mw*(-4*Baw+3*rho*((1-3*A)/ma*Caaw-(2-3*A)/Mw*Caww))
        # Eq T50
        fat = 2*rho*R/ma/Mw*(1-2*A)*(Baw+T*Bawt)+3*rho**2*R/2/ma/Mw*(
            A*(2-3*A)/ma*(Caaw+T*Caawt)+(1-A)*(1-3*A)/Mw*(Caww+T*Cawwt))
        # Eq T51
        fad = 2*R*T/ma/Mw*((1-2*A)*Baw+3/2*rho*(
            A*(2-3*A)/ma*Caaw+(1-A)*(1-3*A)/Mw*Caww))
        # Eq T52
        ftt = 2*A*(1-A)*rho*R/ma/Mw*(2*Bawt+T*Bawtt+3*rho/4*(
            A/ma*(2*Caawt+T*Caawtt)+(1-A)/Mw*(2*Cawwt+T*Cawwtt)))
        # Eq T53
        ftd = 2*A*(1-A)*R/ma/Mw*(Baw+T*Bawt+3*rho/2*(
            A/ma*(Caaw+T*Caawt)+(1-A)/Mw*(Caww+T*Cawwt)))
        # Eq T54
        fdd = 3*A*(1-A)*R*T/ma/Mw*(A/ma*Caaw+(1-A)/Mw*Caww)

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
