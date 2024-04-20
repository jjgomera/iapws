#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
# pylint: disable=too-many-lines, too-many-locals, too-many-statements
# pylint: disable=too-many-branches, too-many-boolean-expressions

"""
Miscelaneous IAPWS standards. This module include:

    * :func:`_Ice`: Ice Ih state equation
    * :func:`_Liquid`: Properties of liquid water at 0.1 MPa
    * :func:`_Supercooled`: Thermodynamic properties of supercooled water
    * :func:`_Sublimation_Pressure`: Sublimation pressure correlation
    * :func:`_Melting_Pressure`: Melting pressure correlation
    * :func:`_Viscosity`: Viscosity correlation
    * :func:`_ThCond`: Themal conductivity correlation
    * :func:`_Tension`: Surface tension correlation
    * :func:`_Dielectric`: Dielectric constant correlation
    * :func:`_Refractive`: Refractive index correlation
    * :func:`_Kw`: Ionization constant correlation for ordinary water
    * :func:`_Conductivity`: Electrolytic conductivity correlation
    * :func:`_D2O_Viscosity`: Viscosity correlation for heavy water
    * :func:`_D2O_ThCond`: Thermal conductivity correlation for heavy water
    * :func:`_D2O_Tension`: Surface tension correlation for heavy water
    * :func:`_D2O_Sublimation_Pressure`: Sublimation Pressure correlation
      for heavy water
    * :func:`_D2O_Melting_Pressure`: Melting Pressure correlation for heavy
      water
    * :func:`_Henry`: Henry constant for liquid-gas equilibrium
    * :func:`_Kvalue`: Vapor-liquid distribution constant
"""

from __future__ import division

from cmath import log as log_c
from math import log, exp, tan, atan, acos, sin, pi, log10
import warnings

from scipy.optimize import newton


# Constants
M = 18.015268     # g/mol
R = 0.461526      # kJ/kg·K

# Table 1 from Release on the Values of Temperature, Pressure and Density of
# Ordinary and Heavy Water Substances at their Respective Critical Points
Tc = 647.096      # K
Pc = 22.064       # MPa
rhoc = 322.       # kg/m³
Tc_D2O = 643.847  # K
Pc_D2O = 21.6618   # MPa
rhoc_D2O = 355.9999698294    # kg/m³

Tt = 273.16       # K
Pt = 611.657e-6   # MPa
Tb = 373.1243     # K
f_acent = 0.3443

# IAPWS, Guideline on the Use of Fundamental Physical Constants and Basic
# Constants of Water, http://www.iapws.org/relguide/fundam.pdf
Dipole = 1.85498  # Debye


# IAPWS-06 for Ice
def _Ice(T, P):
    """Basic state equation for Ice Ih

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties of ice. The available properties are:

            * rho: Density, [kg/m³]
            * h: Specific enthalpy, [kJ/kg]
            * u: Specific internal energy, [kJ/kg]
            * a: Specific Helmholtz energy, [kJ/kg]
            * g: Specific Gibbs energy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * alfav: Cubic expansion coefficient, [1/K]
            * beta: Pressure coefficient, [MPa/K]
            * xkappa: Isothermal compressibility, [1/MPa]
            * ks: Isentropic compressibility, [1/MPa]
            * gt: [∂g/∂T]P
            * gtt: [∂²g/∂T²]P
            * gp: [∂g/∂P]T
            * gpp: [∂²g/∂P²]T
            * gtp: [∂²g/∂T∂P]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * T ≤ 273.16
        * P ≤ 208.566
        * State below the melting and sublimation lines

    Examples
    --------
    >>> st1 = _Ice(100, 100)
    >>> st1["rho"], st1["h"], st1["s"]
    941.678203297 -483.491635676 -2.61195122589

    >>> st2 = _Ice(273.152519,0.101325)
    >>> st2["a"], st2["u"], st2["cp"]
    -0.00918701567 -333.465403393 2.09671391024

    >>> st3 = _Ice(273.16,611.657e-6)
    >>> st3["alfav"], st3["beta"], st3["xkappa"], st3["ks"]
    0.000159863102566 1.35714764659 1.17793449348e-04 1.14161597779e-04

    References
    ----------
    IAPWS, Revised Release on the Equation of State 2006 for H2O Ice Ih
    September 2009, http://iapws.org/relguide/Ice-2009.html
    """
    # Check input in range of validity
    if T > 273.16:
        # No Ice Ih stable
        warnings.warn("Metastable ice")
    elif P > 208.566:
        # Ice Ih limit upper pressure
        raise NotImplementedError("Incoming out of bound")
    elif P < Pt:
        Psub = _Sublimation_Pressure(T)
        if Psub > P:
            # Zone Gas
            warnings.warn("Metastable ice in vapor region")
    elif T > 251.165:
        Pmel = _Melting_Pressure(T)
        if Pmel < P:
            # Zone Liquid
            warnings.warn("Metastable ice in liquid region")

    Tr = T/Tt
    Pr = P/Pt
    P0 = 101325e-6/Pt
    s0 = -0.332733756492168e4*1e-3  # Express in kJ/kgK

    gok = [-0.632020233335886e6, 0.655022213658955, -0.189369929326131e-7,
           0.339746123271053e-14, -0.556464869058991e-21]
    r2k = [complex(-0.725974574329220e2, -0.781008427112870e2)*1e-3,
           complex(-0.557107698030123e-4, 0.464578634580806e-4)*1e-3,
           complex(0.234801409215913e-10, -0.285651142904972e-10)*1e-3]
    t1 = complex(0.368017112855051e-1, 0.510878114959572e-1)
    t2 = complex(0.337315741065416, 0.335449415919309)
    r1 = complex(0.447050716285388e2, 0.656876847463481e2)*1e-3

    go = gop = gopp = 0
    for k in range(5):
        go += gok[k]*1e-3*(Pr-P0)**k
    for k in range(1, 5):
        gop += gok[k]*1e-3*k/Pt*(Pr-P0)**(k-1)
    for k in range(2, 5):
        gopp += gok[k]*1e-3*k*(k-1)/Pt**2*(Pr-P0)**(k-2)
    r2 = r2p = 0
    for k in range(3):
        r2 += r2k[k]*(Pr-P0)**k
    for k in range(1, 3):
        r2p += r2k[k]*k/Pt*(Pr-P0)**(k-1)
    r2pp = r2k[2]*2/Pt**2

    c = r1*((t1-Tr)*log_c(t1-Tr)+(t1+Tr)*log_c(t1+Tr)-2*t1*log_c(
        t1)-Tr**2/t1)+r2*((t2-Tr)*log_c(t2-Tr)+(t2+Tr)*log_c(
            t2+Tr)-2*t2*log_c(t2)-Tr**2/t2)
    ct = r1*(-log_c(t1-Tr)+log_c(t1+Tr)-2*Tr/t1)+r2*(
        -log_c(t2-Tr)+log_c(t2+Tr)-2*Tr/t2)
    ctt = r1*(1/(t1-Tr)+1/(t1+Tr)-2/t1) + r2*(1/(t2-Tr)+1/(t2+Tr)-2/t2)
    cp = r2p*((t2-Tr)*log_c(t2-Tr)+(t2+Tr)*log_c(
        t2+Tr)-2*t2*log_c(t2)-Tr**2/t2)
    ctp = r2p*(-log_c(t2-Tr)+log_c(t2+Tr)-2*Tr/t2)
    cpp = r2pp*((t2-Tr)*log_c(t2-Tr)+(t2+Tr)*log_c(
        t2+Tr)-2*t2*log_c(t2)-Tr**2/t2)

    g = go-s0*Tt*Tr+Tt*c.real
    gt = -s0+ct.real
    gp = gop+Tt*cp.real
    gtt = ctt.real/Tt
    gtp = ctp.real
    gpp = gopp+Tt*cpp.real

    propiedades = {}
    propiedades["gt"] = gt
    propiedades["gp"] = gp
    propiedades["gtt"] = gtt
    propiedades["gpp"] = gpp
    propiedades["gtp"] = gtp
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = gp/1000
    propiedades["rho"] = 1000./gp
    propiedades["h"] = g-T*gt
    propiedades["s"] = -gt
    propiedades["cp"] = -T*gtt
    propiedades["u"] = g-T*gt-P*gp
    propiedades["g"] = g
    propiedades["a"] = g-P*gp
    propiedades["alfav"] = gtp/gp
    propiedades["beta"] = -gtp/gpp
    propiedades["xkappa"] = -gpp/gp
    propiedades["ks"] = (gtp**2-gtt*gpp)/gp/gtt
    return propiedades


# IAPWS-08 for Liquid water at 0.1 MPa
def _Liquid(T, P=0.1):
    """Supplementary release on properties of liquid water at 0.1 MPa

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]
        Although this relation is for P=0.1MPa, can be extrapoled at pressure
        0.3 MPa

    Returns
    -------
    prop : dict
        Dict with calculated properties of water. The available properties are:

            * h: Specific enthalpy, [kJ/kg]
            * u: Specific internal energy, [kJ/kg]
            * a: Specific Helmholtz energy, [kJ/kg]
            * g: Specific Gibbs energy, [kJ/kg]
            * s: Specific entropy, [kJ/kgK]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isochoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s²]
            * rho: Density, [kg/m³]
            * v: Specific volume, [m³/kg]
            * vt: [∂v/∂T]P, [m³/kgK]
            * vtt: [∂²v/∂T²]P, [m³/kgK²]
            * vp: [∂v/∂P]T, [m³/kg/MPa]
            * vtp: [∂²v/∂T∂P], [m³/kg/MPa]
            * alfav: Cubic expansion coefficient, [1/K]
            * xkappa : Isothermal compressibility, [1/MPa]
            * ks: Isentropic compressibility, [1/MPa]
            * mu: Viscosity, [Pas]
            * k: Thermal conductivity, [W/mK]
            * epsilon: Dielectric constant, [-]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 253.15 ≤ T ≤ 383.15
        * 0.1 ≤ P ≤ 0.3

    Examples
    --------
    >>> st1 = _Liquid(260)
    >>> st1["rho"], st1["h"], st1["s"]
    997.0683602710492 -55.86223174460868 -0.20998554842619535

    References
    ----------
    IAPWS, Revised Supplementary Release on Properties of Liquid Water at 0.1
    MPa, http://www.iapws.org/relguide/LiquidWater.html
    """
    # Check input in range of validity
    if T <= 253.15 or T >= 383.15 or P < 0.1 or P > 0.3:
        raise NotImplementedError("Incoming out of bound")
    if P != 0.1:
        # Raise a warning if the P value is extrapolated
        warnings.warn("Using extrapolated values")

    Rg = 0.46151805   # kJ/kgK
    Po = 0.1
    Tr = 10
    tau = T/Tr
    alfa = Tr/(593-T)
    beta = Tr/(T-232)

    a = [None, -1.661470539e5, 2.708781640e6, -1.557191544e8, None,
         1.93763157e-2, 6.74458446e3, -2.22521604e5, 1.00231247e8,
         -1.63552118e9, 8.32299658e9, -7.5245878e-6, -1.3767418e-2,
         1.0627293e1, -2.0457795e2, 1.2037414e3]
    b = [None, -8.237426256e-1, 1.908956353, -2.017597384, 8.546361348e-1,
         5.78545292e-3, -1.53195665E-2, 3.11337859e-2, -4.23546241e-2,
         3.38713507e-2, -1.19946761e-2, -3.1091470e-6, 2.8964919e-5,
         -1.3112763e-4, 3.0410453e-4, -3.9034594e-4, 2.3403117e-4,
         -4.8510101e-5]
    c = [None, -2.452093414e2, 3.869269598e1, -8.983025854]
    n = [None, 4, 5, 7, None, None, 4, 5, 7, 8, 9, 1, 3, 5, 6, 7]
    m = [None, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 3, 4, 5, 6, 7, 9]

    suma1 = sum(a[i]*alfa**n[i] for i in range(1, 4))
    suma2 = sum(b[i]*beta**m[i] for i in range(1, 5))
    go = Rg*Tr*(c[1]+c[2]*tau+c[3]*tau*log(tau)+suma1+suma2)

    suma1 = sum(a[i]*alfa**n[i] for i in range(6, 11))
    suma2 = sum(b[i]*beta**m[i] for i in range(5, 11))
    vo = Rg*Tr/Po/1000*(a[5]+suma1+suma2)

    suma1 = sum(a[i]*alfa**n[i] for i in range(11, 16))
    suma2 = sum(b[i]*beta**m[i] for i in range(11, 18))
    vpo = Rg*Tr/Po**2/1000*(suma1+suma2)

    suma1 = sum(n[i]*a[i]*alfa**(n[i]+1) for i in range(1, 4))
    suma2 = sum(m[i]*b[i]*beta**(m[i]+1) for i in range(1, 5))
    so = -Rg*(c[2]+c[3]*(1+log(tau))+suma1-suma2)

    suma1 = sum(n[i]*(n[i]+1)*a[i]*alfa**(n[i]+2) for i in range(1, 4))
    suma2 = sum(m[i]*(m[i]+1)*b[i]*beta**(m[i]+2) for i in range(1, 5))
    cpo = -Rg*(c[3]+tau*suma1+tau*suma2)

    suma1 = sum(n[i]*a[i]*alfa**(n[i]+1) for i in range(6, 11))
    suma2 = sum(m[i]*b[i]*beta**(m[i]+1) for i in range(5, 11))
    vto = Rg/Po/1000*(suma1-suma2)

    # This properties are only neccessary for computing thermodynamic
    # properties at pressures different from 0.1 MPa
    suma1 = sum(n[i]*(n[i]+1)*a[i]*alfa**(n[i]+2) for i in range(6, 11))
    suma2 = sum(m[i]*(m[i]+1)*b[i]*beta**(m[i]+2) for i in range(5, 11))
    vtto = Rg/Tr/Po/1000*(suma1+suma2)

    suma1 = sum(n[i]*a[i]*alfa**(n[i]+1) for i in range(11, 16))
    suma2 = sum(m[i]*b[i]*beta**(m[i]+1) for i in range(11, 18))
    vpto = Rg/Po**2/1000*(suma1-suma2)

    if P != 0.1:
        go += vo*(P-0.1)
        so -= vto*(P-0.1)
        cpo -= T*vtto*(P-0.1)
        vo -= vpo*(P-0.1)
        vto += vpto*(P-0.1)
        vppo = 3.24e-10*Rg*Tr/0.1**3
        vpo += vppo*(P-0.1)

    h = go+T*so
    u = h-P*vo
    a = go-P*vo
    cv = cpo+T*vto**2/vpo
    xkappa = -vpo/vo
    alfa = vto/vo
    ks = -(T*vto**2/cpo+vpo)/vo
    w = (-vo**2*1e9/(vpo*1e3+T*vto**2*1e6/cpo))**0.5

    propiedades = {}
    propiedades["g"] = go
    propiedades["T"] = T
    propiedades["P"] = P
    propiedades["v"] = vo
    propiedades["vt"] = vto
    propiedades["vp"] = vpo
    propiedades["vpt"] = vpto
    propiedades["vtt"] = vtto
    propiedades["rho"] = 1/vo
    propiedades["h"] = h
    propiedades["s"] = so
    propiedades["cp"] = cpo
    propiedades["cv"] = cv
    propiedades["u"] = u
    propiedades["a"] = a
    propiedades["xkappa"] = xkappa
    propiedades["alfav"] = vto/vo
    propiedades["ks"] = ks
    propiedades["w"] = w

    # Viscosity correlation, Eq 7
    a = [None, 280.68, 511.45, 61.131, 0.45903]
    b = [None, -1.9, -7.7, -19.6, -40]
    T_ = T/300
    mu = sum(a[i]*T_**b[i] for i in range(1, 5))/1e6
    propiedades["mu"] = mu

    # Thermal conductivity correlation, Eq 8
    c = [None, 1.6630, -1.7781, 1.1567, -0.432115]
    d = [None, -1.15, -3.4, -6.0, -7.6]
    k = sum(c[i]*T_**d[i] for i in range(1, 5))
    propiedades["k"] = k

    # Dielectric constant correlation, Eq 9
    e = [None, -43.7527, 299.504, -399.364, 221.327]
    f = [None, -0.05, -1.47, -2.11, -2.31]
    epsilon = sum(e[i]*T_**f[i] for i in range(1, 5))
    propiedades["epsilon"] = epsilon

    return propiedades


# IAPWS-15 for supercooled liquid water
def _Supercooled(T, P):
    """Guideline on thermodynamic properties of supercooled water

    Parameters
    ----------
    T : float
        Temperature, [K]
    P : float
        Pressure, [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties of water. The available properties are:

            * L: Ordering field, [-]
            * x: Mole fraction of low-density structure, [-]
            * rho: Density, [kg/m³]
            * s: Specific entropy, [kJ/kgK]
            * h: Specific enthalpy, [kJ/kg]
            * u: Specific internal energy, [kJ/kg]
            * a: Specific Helmholtz energy, [kJ/kg]
            * g: Specific Gibbs energy, [kJ/kg]
            * alfap: Thermal expansion coefficient, [1/K]
            * xkappa : Isothermal compressibility, [1/MPa]
            * cp: Specific isobaric heat capacity, [kJ/kgK]
            * cv: Specific isochoric heat capacity, [kJ/kgK]
            * w: Speed of sound, [m/s²]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * Tm ≤ T ≤ 300
        * 0 < P ≤ 1000

    The minimum temperature in range of validity is the melting temperature, it
    depend of pressure

    Raise :class:`RuntimeError` if solution isn't founded

    Examples
    --------
    >>> liq = _supercooled(235.15, 0.101325)
    >>> liq["rho"], liq["cp"], liq["w"]
    968.09999 5.997563 1134.5855

    References
    ----------
    iapws, guideline on thermodynamic properties of supercooled water,
    http://iapws.org/relguide/Supercooled.html
    """
    # Check input in range of validity
    if P < 198.9:
        Tita = T/235.15
        Ph = 0.1+228.27*(1-Tita**6.243)+15.724*(1-Tita**79.81)
        if P < Ph or T > 300:
            raise NotImplementedError("Incoming out of bound")
    else:
        Th = 172.82+0.03718*P+3.403e-5*P**2-1.573e-8*P**3
        if T < Th or T > 300 or P > 1000:
            raise NotImplementedError("Incoming out of bound")

    # Parameters, Table 1
    Tll = 228.2
    rho0 = 1081.6482
    Rg = 0.461523087
    pi0 = 300e3/rho0/Rg/Tll
    omega0 = 0.5212269
    L0 = 0.76317954
    k0 = 0.072158686
    k1 = -0.31569232
    k2 = 5.2992608

    # Reducing parameters, Eq 2
    tau = T/Tll-1
    p = P*1000/rho0/Rg/Tll
    tau_ = tau+1
    p_ = p+pi0

    # Eq 3
    ci = [-8.1570681381655, 1.2875032, 7.0901673598012, -3.2779161e-2,
          7.3703949e-1, -2.1628622e-1, -5.1782479, 4.2293517e-4, 2.3592109e-2,
          4.3773754, -2.9967770e-3, -9.6558018e-1, 3.7595286, 1.2632441,
          2.8542697e-1, -8.5994947e-1, -3.2916153e-1, 9.0019616e-2,
          8.1149726e-2, -3.2788213]
    ai = [0, 0, 1, -0.2555, 1.5762, 1.6400, 3.6385, -0.3828, 1.6219, 4.3287,
          3.4763, 5.1556, -0.3593, 5.0361, 2.9786, 6.2373, 4.0460, 5.3558,
          9.0157, 1.2194]
    bi = [0, 1, 0, 2.1051, 1.1422, 0.9510, 0, 3.6402, 2.0760, -0.0016, 2.2769,
          0.0008, 0.3706, -0.3975, 2.9730, -0.3180, 2.9805, 2.9265, 0.4456,
          0.1298]
    di = [0, 0, 0, -0.0016, 0.6894, 0.0130, 0.0002, 0.0435, 0.0500, 0.0004,
          0.0528, 0.0147, 0.8584, 0.9924, 1.0041, 1.0961, 1.0228, 1.0303,
          1.6180, 0.5213]
    phir = phirt = phirp = phirtt = phirtp = phirpp = 0
    for c, a, b, d in zip(ci, ai, bi, di):
        phir += c*tau_**a*p_**b*exp(-d*p_)
        phirt += c*a*tau_**(a-1)*p_**b*exp(-d*p_)
        phirp += c*tau_**a*p_**(b-1)*(b-d*p_)*exp(-d*p_)
        phirtt += c*a*(a-1)*tau_**(a-2)*p_**b*exp(-d*p_)
        phirtp += c*a*tau_**(a-1)*p_**(b-1)*(b-d*p_)*exp(-d*p_)
        phirpp += c*tau_**a*p_**(b-2)*((d*p_-b)**2-b)*exp(-d*p_)

    # Eq 5
    K1 = ((1+k0*k2+k1*(p-k2*tau))**2-4*k0*k1*k2*(p-k2*tau))**0.5
    K2 = (1+k2**2)**0.5

    # Eq 6
    omega = 2+omega0*p

    # Eq 4
    L = L0*K2/2/k1/k2*(1+k0*k2+k1*(p+k2*tau)-K1)

    # Define interval of solution, Table 4
    if omega < 10/9*(log(19)-L):
        xmin = 0.049
        xmax = 0.5
    elif 10/9*(log(19)-L) <= omega < 50/49*(log(99)-L):
        xmin = 0.0099
        xmax = 0.051
    else:
        xmin = 0.99*exp(-50/49*L-omega)
        xmax = min(1.1*exp(-L-omega), 0.0101)

    # Eq 8
    def f(x):
        "Function for iterative calculation"
        if x < xmin:
            x = xmin
        if x > xmax:
            x = xmax
        return L+log(x/(1-x))+omega*(1-2*x)

    x = None
    for xo in (xmin, xmax, (xmin+xmax)/2):
        try:
            x, sol = newton(f, xo, full_output=True)
        except RuntimeError:
            pass
        else:
            if sol.converged:
                break

    # Exit when solution don't found
    if not x:
        raise RuntimeError("Solution don't found")

    # Eq 12
    fi = 2*x-1
    Xi = 1/(2/(1-fi**2)-omega)

    # Derivatives, Table 3
    Lt = L0*K2/2*(1+(1-k0*k2+k1*(p-k2*tau))/K1)
    Lp = L0*K2*(K1+k0*k2-k1*p+k1*k2*tau-1)/2/k2/K1
    Ltt = -2*L0*K2*k0*k1*k2**2/K1**3
    Ltp = 2*L0*K2*k0*k1*k2/K1**3
    Lpp = -2*L0*K2*k0*k1/K1**3

    prop = {}
    prop["L"] = L
    prop["x"] = x

    # Eq 13
    prop["rho"] = rho0/((tau+1)/2*(omega0/2*(1-fi**2)+Lp*(fi+1))+phirp)

    # Eq 1
    prop["g"] = phir+(tau+1)*(x*L+x*log(x)+(1-x)*log(1-x)+omega*x*(1-x))

    # Eq 14
    prop["s"] = -Rg*((tau+1)/2*Lt*(fi+1)
                    + (x*L+x*log(x)+(1-x)*log(1-x)+omega*x*(1-x))+phirt)

    # Basic derived state properties
    prop["h"] = prop["g"]+T*prop["s"]
    prop["u"] = prop["h"]+P/prop["rho"]
    prop["a"] = prop["u"]-T*prop["s"]

    # Eq 15
    prop["xkappa"] = prop["rho"]/rho0**2/Rg*1000/Tll*(
        (tau+1)/2*(Xi*(Lp-omega0*fi)**2-(fi+1)*Lpp)-phirpp)
    prop["alfap"] = prop["rho"]/rho0/Tll*(
        Ltp/2*(tau+1)*(fi+1) + (omega0*(1-fi**2)/2+Lp*(fi+1))/2
        - (tau+1)*Lt/2*Xi*(Lp-omega0*fi) + phirtp)
    prop["cp"] = -Rg*(tau+1)*(Lt*(fi+1)+(tau+1)/2*(Ltt*(fi+1)-Lt**2*Xi)+phirtt)

    # Eq 16
    prop["cv"] = prop["cp"]-T*prop["alfap"]**2/prop["rho"]/prop["xkappa"]*1e3

    # Eq 17
    prop["w"] = (prop["rho"]*prop["xkappa"]*1e-6*prop["cv"]/prop["cp"])**-0.5
    return prop


def _Sublimation_Pressure(T):
    """Sublimation Pressure correlation

    Parameters
    ----------
    T : float
        Temperature, [K]

    Returns
    -------
    P : float
        Pressure at sublimation line, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 50 ≤ T ≤ 273.16

    Examples
    --------
    >>> _Sublimation_Pressure(230)
    8.947352740189152e-06

    References
    ----------
    IAPWS, Revised Release on the Pressure along the Melting and Sublimation
    Curves of Ordinary Water Substance, http://iapws.org/relguide/MeltSub.html.
    """
    if 50 <= T <= 273.16:
        Tita = T/Tt
        suma = 0
        a = [-0.212144006e2, 0.273203819e2, -0.61059813e1]
        expo = [0.333333333e-2, 1.20666667, 1.70333333]
        for ai, expi in zip(a, expo):
            suma += ai*Tita**expi
        return exp(suma/Tita)*Pt

    raise NotImplementedError("Incoming out of bound")


def _Melting_Pressure(T, ice="Ih"):
    """Melting Pressure correlation

    Parameters
    ----------
    T : float
        Temperature, [K]
    ice: string
        Type of ice: Ih, III, V, VI, VII.
        Below 273.15 is a mandatory input, the ice Ih is the default value.
        Above 273.15, the ice type is unnecesary.

    Returns
    -------
    P : float
        Pressure at sublimation line, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 251.165 ≤ T ≤ 715

    Examples
    --------
    >>> _Melting_Pressure(260)
    8.947352740189152e-06
    >>> _Melting_Pressure(254, "III")
    268.6846466336108

    References
    ----------
    IAPWS, Revised Release on the Pressure along the Melting and Sublimation
    Curves of Ordinary Water Substance, http://iapws.org/relguide/MeltSub.html.
    """
    if ice == "Ih" and 251.165 <= T <= 273.16:
        # Ice Ih
        Tref = Tt
        Pref = Pt
        Tita = T/Tref
        a = [0.119539337e7, 0.808183159e5, 0.33382686e4]
        expo = [3., 0.2575e2, 0.10375e3]
        suma = 1
        for ai, expi in zip(a, expo):
            suma += ai*(1-Tita**expi)
        P = suma*Pref
    elif ice == "III" and 251.165 < T <= 256.164:
        # Ice III
        Tref = 251.165
        Pref = 208.566
        Tita = T/Tref
        P = Pref*(1-0.299948*(1-Tita**60.))
    elif (ice == "V" and 256.164 < T <= 273.15) or 273.15 < T <= 273.31:
        # Ice V
        Tref = 256.164
        Pref = 350.100
        Tita = T/Tref
        P = Pref*(1-1.18721*(1-Tita**8.))
    elif 273.31 < T <= 355:
        # Ice VI
        Tref = 273.31
        Pref = 632.400
        Tita = T/Tref
        P = Pref*(1-1.07476*(1-Tita**4.6))
    elif 355. < T <= 715:
        # Ice VII
        Tref = 355
        Pref = 2216.000
        Tita = T/Tref
        P = Pref*exp(1.73683*(1-1./Tita)-0.544606e-1*(1-Tita**5)
                     + 0.806106e-7*(1-Tita**22))
    else:
        raise NotImplementedError("Incoming out of bound")
    return P


# Transport properties
def _Viscosity(rho, T, fase=None, drho=None):
    """Equation for the Viscosity

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]
    fase: dict, optional for calculate critical enhancement
        phase properties
    drho: float, optional for calculate critical enhancement
        [∂ρ/∂P]T at reference state,

    Returns
    -------
    μ : float
        Viscosity, [Pa·s]

    Examples
    --------
    >>> _Viscosity(998, 298.15)
    0.0008897351001498108
    >>> _Viscosity(600, 873.15)
    7.743019522728247e-05

    References
    ----------
    IAPWS, Release on the IAPWS Formulation 2008 for the Viscosity of Ordinary
    Water Substance, http://www.iapws.org/relguide/viscosity.html
    """
    Tr = T/Tc
    Dr = rho/rhoc

    # Eq 11
    H = [1.67752, 2.20462, 0.6366564, -0.241605]
    mu0 = 100*Tr**0.5/sum(Hi/Tr**i for i, Hi in enumerate(H))

    # Eq 12
    li = [0, 1, 2, 3, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 0, 1, 0, 3, 4, 3, 5]
    lj = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6]
    Hij = [0.520094, 0.850895e-1, -0.108374e1, -0.289555, 0.222531, 0.999115,
           0.188797e1, 0.126613e1, 0.120573, -0.281378, -0.906851, -0.772479,
           -0.489837, -0.257040, 0.161913, 0.257399, -0.325372e-1, 0.698452e-1,
           0.872102e-2, -0.435673e-2, -0.593264e-3]
    mu1 = exp(Dr*sum((1/Tr-1)**i*h*(Dr-1)**j for i, j, h in zip(li, lj, Hij)))

    # Critical enhancement
    if rho and fase and drho:
        qc = 1/1.9
        qd = 1/1.1

        # Eq 21
        DeltaX = Pc*Dr**2*(fase.drhodP_T/rho-drho/rho*1.5/Tr)
        if DeltaX < 0:
            DeltaX = 0

        # Eq 20
        X = 0.13*(DeltaX/0.06)**(0.63/1.239)

        if X <= 0.3817016416:
            # Eq 15
            Y = qc/5*X*(qd*X)**5*(1-qc*X+(qc*X)**2-765./504*(qd*X)**2)

        else:
            Fid = acos((1+qd**2*X**2)**-0.5)                            # Eq 17
            w = abs((qc*X-1)/(qc*X+1))**0.5*tan(Fid/2)                  # Eq 19

            # Eq 18
            if qc*X > 1:
                Lw = log((1+w)/(1-w))
            else:
                Lw = 2*atan(abs(w))

            # Eq 16
            Y = sin(3*Fid)/12-sin(2*Fid)/4/qc/X+(1-5/4*(qc*X)**2)/(
                qc*X)**2*sin(Fid)-((1-3/2*(qc*X)**2)*Fid-abs((
                    qc*X)**2-1)**1.5*Lw)/(qc*X)**3

        # Eq 14
        mu2 = exp(0.068*Y)
    else:
        mu2 = 1

    # Eq 10
    mu = mu0*mu1*mu2
    return mu*1e-6


def _ThCond(rho, T, fase=None, drho=None):
    """Equation for the thermal conductivity

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]
    fase: dict, optional for calculate critical enhancement
        phase properties
    drho: float, optional for calculate critical enhancement
        [∂ρ/∂P]T at reference state,

    Returns
    -------
    k : float
        Thermal conductivity, [W/mK]

    Examples
    --------
    >>> _ThCond(998, 298.15)
    0.6077128675880629
    >>> _ThCond(0, 873.15)
    0.07910346589648833

    References
    ----------
    IAPWS, Release on the IAPWS Formulation 2011 for the Thermal Conductivity
    of Ordinary Water Substance, http://www.iapws.org/relguide/ThCond.html
    """
    d = rho/rhoc
    Tr = T/Tc

    # Eq 16
    no = [2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4]
    k0 = Tr**0.5/sum(n/Tr**i for i, n in enumerate(no))

    # Eq 17
    li = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4,
          4, 4, 4, 4, 4]
    lj = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0,
          1, 2, 3, 4, 5]
    nij = [1.60397357, -0.646013523, 0.111443906, 0.102997357, -0.0504123634,
           0.00609859258, 2.33771842, -2.78843778, 1.53616167, -0.463045512,
           0.0832827019, -0.00719201245, 2.19650529, -4.54580785, 3.55777244,
           -1.40944978, 0.275418278, -0.0205938816, -1.21051378, 1.60812989,
           -0.621178141, 0.0716373224, -2.7203370, 4.57586331, -3.18369245,
           1.1168348, -0.19268305, 0.012913842]
    k1 = exp(d*sum((1/Tr-1)**i*n*(d-1)**j for i, j, n in zip(li, lj, nij)))

    # Critical enhancement
    if rho and fase:
        Rg = 0.46151805

        if not drho:
            # Industrial formulation
            # Eq 25
            if d <= 0.310559006:
                ai = [6.53786807199516, -5.61149954923348, 3.39624167361325,
                      -2.27492629730878, 10.2631854662709, 1.97815050331519]
            elif d <= 0.776397516:
                ai = [6.52717759281799, -6.30816983387575, 8.08379285492595,
                      -9.82240510197603, 12.1358413791395, -5.54349664571295]
            elif d <= 1.242236025:
                ai = [5.35500529896124, -3.96415689925446, 8.91990208918795,
                      -12.0338729505790, 9.19494865194302, -2.16866274479712]
            elif d <= 1.863354037:
                ai = [1.55225959906681, 0.464621290821181, 8.93237374861479,
                      -11.0321960061126, 6.16780999933360, -0.965458722086812]
            else:
                ai = [1.11999926419994, 0.595748562571649, 9.88952565078920,
                      -10.3255051147040, 4.66861294457414, -0.503243546373828]
            drho = 1/sum(a*d**i for i, a in enumerate(ai))*rhoc/Pc

        DeltaX = d*(Pc/rhoc*fase.drhodP_T-Pc/rhoc*drho*1.5/Tr)
        if DeltaX < 0:
            DeltaX = 0

        X = 0.13*(DeltaX/0.06)**(0.63/1.239)                            # Eq 22
        y = X/0.4                                                       # Eq 20

        # Eq 19
        if y < 1.2e-7:
            Z = 0
        else:
            Z = 2/pi/y*(((1-1/fase.cp_cv)*atan(y)+y/fase.cp_cv)-(
                1-exp(-1/(1/y+y**2/3/d**2))))

        # Eq 18
        k2 = 177.8514*d*fase.cp/Rg*Tr/fase.mu*1e-6*Z

    else:
        # No critical enhancement
        k2 = 0

    # Eq 10
    k = k0*k1+k2
    return 1e-3*k


def _Tension(T):
    """Equation for the surface tension

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
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 248.15 ≤ T ≤ 647
        * Estrapolate to -25ºC in supercooled liquid metastable state

    Examples
    --------
    >>> _Tension(300)
    0.0716859625
    >>> _Tension(450)
    0.0428914992

    References
    ----------
    IAPWS, Revised Release on Surface Tension of Ordinary Water Substance
    June 2014, http://www.iapws.org/relguide/Surf-H2O.html
    """
    if 248.15 <= T <= Tc:
        tau = 1-T/Tc
        sigma = 235.8 * tau**1.256 * (1-0.625*tau)

        # The equation give surface tension in mN/m², converted to N/m²
        return 1e-3*sigma

    raise NotImplementedError("Incoming out of bound")


def _Dielectric(rho, T):
    """Equation for the Dielectric constant

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]

    Returns
    -------
    epsilon : float
        Dielectric constant, [-]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 238 ≤ T ≤ 1200

    Examples
    --------
    >>> _Dielectric(999.242866, 298.15)
    78.5907250
    >>> _Dielectric(26.0569558, 873.15)
    1.12620970

    References
    ----------
    IAPWS, Release on the Static Dielectric Constant of Ordinary Water
    Substance for Temperatures from 238 K to 873 K and Pressures up to 1000
    MPa, http://www.iapws.org/relguide/Dielec.html
    """
    # Check input parameters
    if T < 238 or T > 1200:
        raise NotImplementedError("Incoming out of bound")

    k = 1.380658e-23
    Na = 6.0221367e23
    alfa = 1.636e-40
    epsilon0 = 8.854187817e-12
    mu = 6.138e-30

    d = rho/rhoc
    Tr = Tc/T
    li = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10]
    lj = [0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10]
    ni = [0.978224486826, -0.957771379375, 0.237511794148, 0.714692244396,
          -0.298217036956, -0.108863472196, 0.949327488264e-1,
          -.980469816509e-2, 0.165167634970e-4, 0.937359795772e-4,
          -0.12317921872e-9, 0.196096504426e-2]

    g = 1+ni[11]*d/(Tc/228/Tr-1)**1.2
    for n, i, j in zip(ni, li, lj):
        g += n * d**i * Tr**j
    A = Na*mu**2*rho*g/M*1000/epsilon0/k/T
    B = Na*alfa*rho/3/M*1000/epsilon0
    e = (1+A+5*B+(9+2*A+18*B+A**2+10*A*B+9*B**2)**0.5)/4/(1-B)
    return e


def _Refractive(rho, T, lr=0.5893):
    """Equation for the refractive index

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]
    lr : float, optional
        Light Wavelength, [μm]

    Returns
    -------
    n : float
        Refractive index, [-]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 0 ≤ ρ ≤ 1060
        * 261.15 ≤ T ≤ 773.15
        * 0.2 ≤ λ ≤ 1.1

    Examples
    --------
    >>> _Refractive(997.047435, 298.15, 0.2265)
    1.39277824
    >>> _Refractive(30.4758534, 773.15, 0.5893)
    1.00949307

    References
    ----------
    IAPWS, Release on the Refractive Index of Ordinary Water Substance as a
    Function of Wavelength, Temperature and Pressure,
    http://www.iapws.org/relguide/rindex.pdf
    """
    # Check input parameters
    if rho < 0 or rho > 1060 or \
            T < 261.15 or T > 773.15 or \
            lr < 0.2 or lr > 1.1:
        raise NotImplementedError("Incoming out of bound")

    Lir = 5.432937
    Luv = 0.229202
    d = rho/1000.
    Tr = T/273.15
    L = lr/0.589
    a = [0.244257733, 0.974634476e-2, -0.373234996e-2, 0.268678472e-3,
         0.158920570e-2, 0.245934259e-2, 0.900704920, -0.166626219e-1]
    A = d*(a[0]+a[1]*d+a[2]*Tr+a[3]*L**2*Tr+a[4]/L**2+a[5]/(L**2-Luv**2)+a[6]/(
        L**2-Lir**2)+a[7]*d**2)
    return ((2*A+1)/(1-A))**0.5


def _Kw(rho, T):
    """Equation for the ionization constant of ordinary water

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]

    Returns
    -------
    pKw : float
        Ionization constant in -log10(kw), [-]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 0 ≤ ρ ≤ 1250
        * 273.15 ≤ T ≤ 1073.15

    Examples
    --------
    >>> _Kw(1000, 300)
    13.906565

    References
    ----------
    IAPWS, Release on the Ionization Constant of H2O,
    http://www.iapws.org/relguide/Ionization.pdf
    """
    # Check input parameters
    if rho < 0 or rho > 1250 or T < 273.15 or T > 1073.15:
        raise NotImplementedError("Incoming out of bound")

    # The internal method of calculation use rho in g/cm³
    d = rho/1000.

    # Water molecular weight different
    Mw = 18.015268

    gamma = [6.1415e-1, 4.825133e4, -6.770793e4, 1.01021e7]
    pKg = 0
    for i, g in enumerate(gamma):
        pKg += g/T**i

    Q = d*exp(-0.864671+8659.19/T-22786.2/T**2*d**(2./3))
    pKw = -12*(log10(1+Q)-Q/(Q+1)*d*(0.642044-56.8534/T-0.375754*d)) + \
        pKg+2*log10(Mw/1000)
    return pKw


def _Conductivity(rho, T):
    """Equation for the electrolytic conductivity of liquid and dense
    supercrítical water

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]

    Returns
    -------
    K : float
        Electrolytic conductivity, [S/m]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 600 ≤ ρ ≤ 1200
        * 273.15 ≤ T ≤ 1073.15

    Examples
    --------
    >>> _Conductivity(1000, 373.15)
    1.13

    References
    ----------
    IAPWS, Electrolytic Conductivity (Specific Conductance) of Liquid and Dense
    Supercritical Water from 0°C to 800°C and Pressures up to 1000 MPa,
    http://www.iapws.org/relguide/conduct.pdf
    """
    # density in g/l
    rho_ = rho/1000

    # This guideline predates the current standard on the ionization constant,
    # therefore the standard accepted at that time must be used in order to
    # obtain the values of the tables for testing.
    # Marshall, W.L., Franck, E.U.
    # Ion product of water substance, 0-1000ºC, 1-10,000 bars New International
    # Formulation and its background
    # J. Phys. Chem. Ref. Data 10(2) (1981) 295-304
    # doi: 10.1063/1.555643

    # Eq 4
    kw = 10**(-4.098 - 3245.2/T + 2.2362e5/T**2 - 3.984e7/T**3 +
              (13.957 - 1262.3/T + 8.5641e5/T**2)*log10(rho_))

    # kw = 10**-_Kw(rho, T)

    A = [1850., 1410., 2.16417e-6, 1.81609e-7, -1.75297e-9, 7.20708e-12]
    B = [16., 11.6, 3.26e-4, -2.3e-6, 1.1e-8]
    t = T-273.15

    Loo = A[0]-1/(1/A[1] + A[2]*t + A[3]*t**2 + A[4]*t**3 + A[5]*t**4)   # Eq 5
    rho_h = B[0]-1/(1/B[1] + B[2]*t + B[3]*t**2 + B[4]*t**3)             # Eq 6

    # Eq 4
    L_o = (rho_h-rho_)*Loo/rho_h

    # Eq 1
    k = 1e-3*L_o*kw**0.5*rho_
    return k*1e2


# Heavy water transport properties
def _D2O_Viscosity(rho, T, fase=None, drho=None):
    """Equation for the Viscosity of heavy water

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]
    fase: dict, optional for calculate critical enhancement
        phase properties
    drho: float, optional for calculate critical enhancement
        [∂ρ/∂P]T at reference state,

    Returns
    -------
    μ : float
        Viscosity, [Pa·s]

    Examples
    --------
    >>> _D2O_Viscosity(998, 298.15)
    0.0008897351001498108
    >>> _D2O_Viscosity(600, 873.15)
    7.743019522728247e-05

    References
    ----------
    IAPWS, Release on the IAPWS Formulation 2020 for the Viscosity of Heavy
    Water, http://iapws.org/relguide/D2Ovisc.pdf
    """
    Tr = T/Tc_D2O
    rhor = rho/356

    # Eq 11, viscosity in the dilute-gas limit
    no = 0.889754+61.22217*Tr-44.8866*Tr**2+111.5812*Tr**3+3.547412*Tr**4
    do = 0.79637+2.38127*Tr-0.33463*Tr**2+2.669*Tr**3+0.000211366*Tr**4
    mu0 = Tr**0.5 * no/do

    # Eq 12
    hi = [0, 2, 3, 4, 5, 6, 0, 1, 3, 4, 6, 0, 1, 5, 0, 1, 2, 5, 6, 0, 2, 3, 5,
          2, 2]
    hj = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
          5, 6]
    Hij = [0.510953, -0.558947, -2.718820, 0.480990, 2.404510, -1.824320,
           0.275847, 0.762957, 1.760340, 0.0819086, 1.417750, -0.228148,
           -0.321497, -2.302500, 0.0661035, 0.0449393, 1.466670, 0.938984,
           -0.108354, -0.00481265, -1.545710, -0.0570938, -0.0753783, 0.553080,
           -0.0650201]

    arr = [(1/Tr-1)**i*(rhor-1)**j*hij for i, j, hij in zip(hi, hj, Hij)]
    mu1 = exp(rhor*sum(arr))

    # Critical enhancement
    if rho and fase and drho:
        qc = 1/1.9
        qd = 1/0.4

        # Eq 21
        DeltaX = Pc_D2O*rhor**2*(fase.drhodP_T/rho-drho/rho*1.5/Tr)
        if DeltaX < 0:
            DeltaX = 0

        # Eq 20
        X = 0.13*(DeltaX/0.06)**(0.63/1.239)

        if X <= 0.03021806692:
            # Eq 15
            Y = qc/5*X*(qd*X)**5*(1-qc*X+(qc*X)**2-765/504*(qd*X)**2)

        else:
            Fid = acos((1+qd**2*X**2)**-0.5)                            # Eq 17
            w = abs((qc*X-1)/(qc*X+1))**0.5*tan(Fid/2)                  # Eq 19

            # Eq 18
            if qc*X > 1:
                Lw = log((1+w)/(1-w))
            else:
                Lw = 2*atan(abs(w))

            # Eq 16
            Y = sin(3*Fid)/12-sin(2*Fid)/4/qc/X+(1-5/4*(qc*X)**2)/(
                qc*X)**2*sin(Fid)-((1-3/2*(qc*X)**2)*Fid-abs((
                    qc*X)**2-1)**1.5*Lw)/(qc*X)**3

        # Eq 14
        mu2 = exp(0.068*Y)
    else:
        mu2 = 1

    return mu0*mu1*mu2*1e-6


def _D2O_ThCond(rho, T, fase=None, drho=None):
    """Equation for the thermal conductivity of heavy water

    Parameters
    ----------
    rho : float
        Density, [kg/m³]
    T : float
        Temperature, [K]
    fase: dict, optional for calculate critical enhancement
        phase properties
    drho: float, optional for calculate critical enhancement
        [∂ρ/∂P]T at reference state,

    Returns
    -------
    k : float
        Thermal conductivity, [W/mK]

    Examples
    --------
    >>> _D2O_ThCond(998, 298.15)
    0.6077128675880629
    >>> _D2O_ThCond(0, 873.15)
    0.07910346589648833

    References
    ----------
    IAPWS, Release on the IAPWS Formulation 2021 for the Thermal Conductivity
    of Heavy Water, http://iapws.org/relguide/D2OThCond.pdf
    """
    d = rho/356
    Tr = T/643.847

    # Eq 16
    no = [1, 3.3620798, -1.0191198, 2.8518117]
    do = [0.10779213, -0.034637234, 0.036603464, 0.0091018912]
    k0 = Tr**0.5 * sum(Li*Tr**i for i, Li in enumerate(no)) / \
        sum(Li*Tr**i for i, Li in enumerate(do))

    # Eq 17
    li = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
          3, 4, 4, 4, 4, 4, 4]
    lj = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4,
          5, 0, 1, 2, 3, 4, 5]
    nij = [1.50933576, -0.65831078, 0.111174263, 0.140185152, -0.0656227722,
           0.00785155213, 2.8414715, -2.9826577, 1.34357932, -0.599233641,
           0.28116337, -0.0533292833, 4.86095723, -6.19784468, 2.20941867,
           0.224691518, -0.322191265, 0.0596204654, 2.06156007, -3.48612456,
           1.47962309, 0.625101458, -0.56123225, 0.0974446139, -2.06105687,
           0.416240028, 2.92524513, -2.81703583, 1.00551476, -0.127884416]

    k1 = exp(d*sum((1/Tr-1)**i * n*(d-1)**j for i, j, n in zip(li, lj, nij)))

    # Critical enhancement
    if rho and fase and drho:
        Rg = 0.415151994
        DeltaX = d*(Pc_D2O/rhoc_D2O*fase.drhodP_T-Pc_D2O/rhoc_D2O*drho*1.5/Tr)
        if DeltaX < 0:
            DeltaX = 0

        X = 0.13*(DeltaX/0.06)**(0.63/1.239)                            # Eq 22
        y = X/0.36                                                      # Eq 20

        # Eq 19
        if y < 1.2e-7:
            Z = 0
        else:
            Z = 2/pi/y*(((1-1/fase.cp_cv)*atan(y)+y/fase.cp_cv)-(
                1-exp(-1/(1/y+y**2/3/d**2))))

        # Eq 18
        k2 = 175.987*d*fase.cp/Rg*Tr/fase.mu*1e-6*Z

    else:
        # No critical enhancement
        k2 = 0

    # Eq 15
    k = k0*k1+k2
    return 1e-3*k


def _D2O_Tension(T):
    """Equation for the surface tension of heavy water

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
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 269.65 ≤ T ≤ 643.847

    Examples
    --------
    >>> _D2O_Tension(298.15)
    0.07186
    >>> _D2O_Tension(573.15)
    0.01399

    References
    ----------
    IAPWS, Release on Surface Tension of Heavy Water Substance,
    http://www.iapws.org/relguide/surfd2o.pdf
    """
    Tr = T/643.847
    if 269.65 <= T < 643.847:
        return 1e-3*(238*(1-Tr)**1.25*(1-0.639*(1-Tr)))

    raise NotImplementedError("Incoming out of bound")


def _D2O_Sublimation_Pressure(T):
    """Sublimation Pressure correlation for heavy water

    Parameters
    ----------
    T : float
        Temperature, [K]

    Returns
    -------
    P : float
        Pressure at sublimation line, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 210 ≤ T ≤ 276.969

    Examples
    --------
    >>> _Sublimation_Pressure(245)
    3.27390934e-5

    References
    ----------
    IAPWS, Revised Release on the IAPWS Formulation 2017 for the Thermodynamic
    Properties of Heavy Water, http://www.iapws.org/relguide/Heavy.html.
    """
    if 210 <= T <= 276.969:
        Tita = T/276.969
        suma = 0
        ai = [-0.1314226e2, 0.3212969e2]
        ti = [-1.73, -1.42]
        for a, t in zip(ai, ti):
            suma += a*(1-Tita**t)
        return exp(suma)*0.00066159

    raise NotImplementedError("Incoming out of bound")


def _D2O_Melting_Pressure(T, ice="Ih"):
    """Melting Pressure correlation for heavy water

    Parameters
    ----------
    T : float
        Temperature, [K]
    ice: string
        Type of ice: Ih, III, V, VI, VII.
        Below 276.969 is a mandatory input, the ice Ih is the default value.
        Above 276.969, the ice type is unnecesary.

    Returns
    -------
    P : float
        Pressure at melting line, [MPa]

    Notes
    -----
    Raise :class:`NotImplementedError` if input isn't in limit:

        * 254.415 ≤ T ≤ 315

    Examples
    --------
    >>> _D2O__Melting_Pressure(260)
    8.947352740189152e-06
    >>> _D2O__Melting_Pressure(254, "III")
    268.6846466336108

    References
    ----------
    IAPWS, Revised Release on the Pressure along the Melting and Sublimation
    Curves of Ordinary Water Substance, http://iapws.org/relguide/MeltSub.html.
    """
    if ice == "Ih" and 254.415 <= T <= 276.969:
        # Ice Ih, Eq 9
        Tita = T/276.969
        ai = [-0.30153e5, 0.692503e6]
        ti = [5.5, 8.2]
        suma = 1
        for a, t in zip(ai, ti):
            suma += a*(1-Tita**t)
        P = suma*0.00066159
    elif ice == "III" and 254.415 < T <= 258.661:
        # Ice III, Eq 10
        Tita = T/254.415
        P = 222.41*(1-0.802871*(1-Tita**33))
    elif ice == "V" and 258.661 < T <= 275.748:
        # Ice V, Eq 11
        Tita = T/258.661
        P = 352.19*(1-1.280388*(1-Tita**7.6))
    elif (ice == "VI" and 275.748 < T <= 276.969) or 276.969 < T <= 315:
        # Ice VI
        Tita = T/275.748
        P = 634.53*(1-1.276026*(1-Tita**4))
    else:
        raise NotImplementedError("Incoming out of bound")
    return P


def _Henry(T, gas, liquid="H2O"):
    """Equation for the calculation of Henry's constant

    Parameters
    ----------
    T : float
        Temperature, [K]
    gas : string
        Name of gas to calculate solubility
    liquid : string
        Name of liquid solvent, can be H20 (default) or D2O

    Returns
    -------
    kw : float
        Henry's constant, [MPa]

    Notes
    -----
    The gas availables for H2O solvent are He, Ne, Ar, Kr, Xe, H2, N2, O2, CO,
    CO2, H2S, CH4, C2H6, SF6
    For D2O as solvent He, Ne, Ar, Kr, Xe, D2, CH4

    Raise :class:`NotImplementedError` if input gas or liquid are unsupported

    Examples
    --------
    >>> _Henry(500, "He")
    1.1973
    >>> _Henry(300, "D2", "D2O")
    1.6594

    References
    ----------
    IAPWS, Guideline on the Henry's Constant and Vapor-Liquid Distribution
    Constant for Gases in H2O and D2O at High Temperatures,
    http://www.iapws.org/relguide/HenGuide.html
    """
    if liquid == "D2O":
        gas += "(D2O)"

    limit = {
        "He": (273.21, 553.18),
        "Ne": (273.20, 543.36),
        "Ar": (273.19, 568.36),
        "Kr": (273.19, 525.56),
        "Xe": (273.22, 574.85),
        "H2": (273.15, 636.09),
        "N2": (278.12, 636.46),
        "O2": (274.15, 616.52),
        "CO": (278.15, 588.67),
        "CO2": (274.19, 642.66),
        "H2S": (273.15, 533.09),
        "CH4": (275.46, 633.11),
        "C2H6": (275.44, 473.46),
        "SF6": (283.14, 505.55),
        "He(D2O)": (288.15, 553.18),
        "Ne(D2O)": (288.18, 549.96),
        "Ar(D2O)": (288.30, 583.76),
        "Kr(D2O)": (288.19, 523.06),
        "Xe(D2O)": (295.39, 574.85),
        "D2(D2O)": (288.17, 581.00),
        "CH4(D2O)": (288.16, 517.46)}

    # Check input parameters
    if liquid not in ("D2O", "H2O"):
        raise NotImplementedError("Solvent liquid unsupported")
    if gas not in limit:
        raise NotImplementedError("Gas unsupported")

    Tmin, Tmax = limit[gas]
    if T < Tmin or T > Tmax:
        warnings.warn("Temperature out of data of correlation")

    if liquid == "D2O":
        Tc_ = Tc_D2O
        Pc_ = 21.671
    else:
        Tc_ = Tc
        Pc_ = Pc

    Tr = T/Tc_
    tau = 1-Tr

    # Eq 4
    if liquid == "H2O":
        ai = [-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719,
              1.80122502]
        bi = [1, 1.5, 3, 3.5, 4, 7.5]
    else:
        ai = [-7.896657, 24.73308, -27.81128, 9.355913, -9.220083]
        bi = [1, 1.89, 2, 3, 3.6]
    ps = Pc_*exp(1/Tr*sum(a*tau**b for a, b in zip(ai, bi)))

    # Select values from Table 2
    par = {
        "He": (-3.52839, 7.12983, 4.47770),
        "Ne": (-3.18301, 5.31448, 5.43774),
        "Ar": (-8.40954, 4.29587, 10.52779),
        "Kr": (-8.97358, 3.61508, 11.29963),
        "Xe": (-14.21635, 4.00041, 15.60999),
        "H2": (-4.73284, 6.08954, 6.06066),
        "N2": (-9.67578, 4.72162, 11.70585),
        "O2": (-9.44833, 4.43822, 11.42005),
        "CO": (-10.52862, 5.13259, 12.01421),
        "CO2": (-8.55445, 4.01195, 9.52345),
        "H2S": (-4.51499, 5.23538, 4.42126),
        "CH4": (-10.44708, 4.66491, 12.12986),
        "C2H6": (-19.67563, 4.51222, 20.62567),
        "SF6": (-16.56118, 2.15289, 20.35440),
        "He(D2O)": (-0.72643, 7.02134, 2.04433),
        "Ne(D2O)": (-0.91999, 5.65327, 3.17247),
        "Ar(D2O)": (-7.17725, 4.48177, 9.31509),
        "Kr(D2O)": (-8.47059, 3.91580, 10.69433),
        "Xe(D2O)": (-14.46485, 4.42330, 15.60919),
        "D2(D2O)": (-5.33843, 6.15723, 6.53046),
        "CH4(D2O)": (-10.01915, 4.73368, 11.75711)}
    A, B, C = par[gas]

    # Eq 3
    kh = ps*exp(A/Tr+B*tau**0.355/Tr+C*Tr**-0.41*exp(tau))
    return kh


def _Kvalue(T, gas, liquid="H2O"):
    """Equation for the vapor-liquid distribution constant

    Parameters
    ----------
    T : float
        Temperature, [K]
    gas : string
        Name of gas to calculate solubility
    liquid : string
        Name of liquid solvent, can be H20 (default) or D2O

    Returns
    -------
    kd : float
        Vapor-liquid distribution constant, [-]

    Notes
    -----
    The gas availables for H2O solvent are He, Ne, Ar, Kr, Xe, H2, N2, O2, CO,
    CO2, H2S, CH4, C2H6, SF6

    For D2O as solvent He, Ne, Ar, Kr, Xe, D2, CH4

    Raise :class:`NotImplementedError` if input gas or liquid are unsupported

    Examples
    --------
    >>> _Kvalue(600, "He")
    3.8019
    >>> _Kvalue(300, "D2", "D2O")
    14.3520

    References
    ----------
    IAPWS, Guideline on the Henry's Constant and Vapor-Liquid Distribution
    Constant for Gases in H2O and D2O at High Temperatures,
    http://www.iapws.org/relguide/HenGuide.html
    """
    if liquid == "D2O":
        gas += "(D2O)"

    limit = {
        "He": (273.21, 553.18),
        "Ne": (273.20, 543.36),
        "Ar": (273.19, 568.36),
        "Kr": (273.19, 525.56),
        "Xe": (273.22, 574.85),
        "H2": (273.15, 636.09),
        "N2": (278.12, 636.46),
        "O2": (274.15, 616.52),
        "CO": (278.15, 588.67),
        "CO2": (274.19, 642.66),
        "H2S": (273.15, 533.09),
        "CH4": (275.46, 633.11),
        "C2H6": (275.44, 473.46),
        "SF6": (283.14, 505.55),
        "He(D2O)": (288.15, 553.18),
        "Ne(D2O)": (288.18, 549.96),
        "Ar(D2O)": (288.30, 583.76),
        "Kr(D2O)": (288.19, 523.06),
        "Xe(D2O)": (295.39, 574.85),
        "D2(D2O)": (288.17, 581.00),
        "CH4(D2O)": (288.16, 517.46)}

    # Check input parameters
    if liquid not in ("D2O", "H2O"):
        raise NotImplementedError("Solvent liquid unsupported")
    if gas not in limit:
        raise NotImplementedError("Gas unsupported")

    Tmin, Tmax = limit[gas]
    if T < Tmin or T > Tmax:
        warnings.warn("Temperature out of data of correlation")

    if liquid == "D2O":
        Tc_ = Tc_D2O
    else:
        Tc_ = Tc

    Tr = T/Tc_
    tau = 1-Tr

    # Eq 6
    if liquid == "H2O":
        ci = [1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352,
              -6.7469445e5]
        di = [1/3, 2/3, 5/3, 16/3, 43/3, 110/3]
        q = -0.023767
    else:
        ci = [2.7072, 0.58662, -1.3069, -45.663]
        di = [0.374, 1.45, 2.6, 12.3]
        q = -0.024552
    f = sum(c*tau**d for c, d in zip(ci, di))

    # Select values from Table 2
    par = {"He": (2267.4082, -2.9616, -3.2604, 7.8819),
           "Ne": (2507.3022, -38.6955, 110.3992, -71.9096),
           "Ar": (2310.5463, -46.7034, 160.4066, -118.3043),
           "Kr": (2276.9722, -61.1494, 214.0117, -159.0407),
           "Xe": (2022.8375, 16.7913, -61.2401, 41.9236),
           "H2": (2286.4159, 11.3397, -70.7279, 63.0631),
           "N2": (2388.8777, -14.9593, 42.0179, -29.4396),
           "O2": (2305.0674, -11.3240, 25.3224, -15.6449),
           "CO": (2346.2291, -57.6317, 204.5324, -152.6377),
           "CO2": (1672.9376, 28.1751, -112.4619, 85.3807),
           "H2S": (1319.1205, 14.1571, -46.8361, 33.2266),
           "CH4": (2215.6977, -0.1089, -6.6240, 4.6789),
           "C2H6": (2143.8121, 6.8859, -12.6084, 0),
           "SF6": (2871.7265, -66.7556, 229.7191, -172.7400),
           "He(D2O)": (2293.2474, -54.7707, 194.2924, -142.1257),
           "Ne(D2O)": (2439.6677, -93.4934, 330.7783, -243.0100),
           "Ar(D2O)": (2269.2352, -53.6321, 191.8421, -143.7659),
           "Kr(D2O)": (2250.3857, -42.0835, 140.7656, -102.7592),
           "Xe(D2O)": (2038.3656, 68.1228, -271.3390, 207.7984),
           "D2(D2O)": (2141.3214, -1.9696, 1.6136, 0),
           "CH4(D2O)": (2216.0181, -40.7666, 152.5778, -117.7430)}
    E, F, G, H = par[gas]

    # Eq 5
    kd = exp(q*F+E/T*f+(F+G*tau**(2./3)+H*tau)*exp((273.15-T)/100))
    return kd
