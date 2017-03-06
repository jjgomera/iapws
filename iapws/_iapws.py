#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscelaneous IAPWS standards
"""

from __future__ import division

from cmath import log as log_c
from math import log, exp, tan, atan, acos, sin, pi, log10
import warnings


# Constants
Rm = 8.31451      # kJ/kmol·K
M = 18.015257     # kg/kmol
R = 0.461526      # kJ/kg·K

# Table 1 from Release on the Values of Temperature, Pressure and Density of
# Ordinary and Heavy Water Substances at their Respective Critical Points
Tc = 647.096      # K
Pc = 22.064       # MPa
rhoc = 322.       # kg/m³
Tc_D2O = 643.847  # K
Pc_D2O = 21.671   # MPa
rhoc_D2O = 356    # kg/m³

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
        Temperature [K]
    P : float
        Pressure [MPa]

    Returns
    -------
    prop : dict
        Dict with calculated properties of ice. The available properties are:

            * rho: Density [kg/m³]
            * h: Specific enthalpy [kJ/kg]
            * u: Specific internal energy [kJ/kg]
            * a: Specific Helmholtz energy [kJ/kg]
            * g: Specific Gibbs energy [kJ/kg]
            * s: Specific entropy [kJ/kgK]
            * cp: Specific isobaric heat capacity [kJ/kgK]
            * alfav: Cubic expansion coefficient [1/K]
            * beta: Pressure coefficient [MPa/K]
            * kt: Isothermal compressibility [1/MPa]
            * ks: Isentropic compressibility [1/MPa]
            * gt: [∂g/∂T]P
            * gtt: [∂²g/∂T²]P
            * gp: [∂g/∂P]T
            * gpp: [∂²g/∂P²]T
            * gtp: [∂²g/∂T∂P]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
    >>> st3["alfav"], st3["beta"], st3["kt"], st3["ks"]
    0.000159863102566 1.35714764659 1.17793449348e-04 1.14161597779e-04

    References
    ----------
    IAPWS, Revised Release on the Equation of State 2006 for H2O Ice Ih
    September 2009, http://iapws.org/relguide/Ice-2009.html
    """
    # Check input in range of validity
    if T > 273.16:
        # No Ice Ih stable
        raise NotImplementedError("Incoming out of bound")
    elif P > 208.566:
        # Ice Ih limit upper pressure
        raise NotImplementedError("Incoming out of bound")
    elif P < Pt:
        Psub = _Sublimation_Pressure(T)
        if Psub > P:
            # Zone Gas
            raise NotImplementedError("Incoming out of bound")
    elif 251.165 < T:
        Pmel = _Melting_Pressure(T)
        if Pmel < P:
            # Zone Liquid
            raise NotImplementedError("Incoming out of bound")

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
    propiedades["kt"] = -gpp/gp
    propiedades["ks"] = (gtp**2-gtt*gpp)/gp/gtt
    return propiedades


# IAPWS-08 for Liquid wagter at 0.1 MPa
def _Liquid(T, P=0.1):
    """Supplementary release on properties of liquid water at 0.1 MPa

    Parameters
    ----------
    T : float
        Temperature [K]
    P : float
        Pressure [MPa]
    optional, although this relation is for P=0.1MPa, can be extrapoled at
    pressure 0.3MPa

    Returns
    -------
    prop : dict
        Dict with calculated properties of water. The available properties are:

            * h: Specific enthalpy [kJ/kg]
            * u: Specific internal energy [kJ/kg]
            * a: Specific Helmholtz energy [kJ/kg]
            * g: Specific Gibbs energy [kJ/kg]
            * s: Specific entropy [kJ/kgK]
            * cp: Specific isobaric heat capacity [kJ/kgK]
            * cv: Specific isochoric heat capacity [kJ/kgK]
            * w: Speed of sound [m/s²]
            * rho: Density [kg/m³]
            * v: Specific volume [m³/kg]
            * vt: [∂v/∂T]P [m³/kgK]
            * vtt: [∂²v/∂T²]P [m³/kgK²]
            * vp: [∂v/∂P]T [m³/kg/MPa]
            * vtp: [∂²v/∂T∂P] [m³/kg/MPa]
            * alfav: Cubic expansion coefficient [1/K]
            * kt: Isothermal compressibility [1/MPa]
            * ks: Isentropic compressibility [1/MPa]
            * mu: Viscosity [mPas]
            * k: Thermal conductivity [W/mK]
            * epsilon: Dielectric constant [-]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
    elif P != 0.1:
        # Raise a warning if the P value is extrapolated
        warnings.warn("Using extrapolated values")

    R = 0.46151805   # kJ/kgK
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

    suma1 = sum([a[i]*alfa**n[i] for i in range(1, 4)])
    suma2 = sum([b[i]*beta**m[i] for i in range(1, 5)])
    go = R*Tr*(c[1]+c[2]*tau+c[3]*tau*log(tau)+suma1+suma2)

    suma1 = sum([a[i]*alfa**n[i] for i in range(6, 11)])
    suma2 = sum([b[i]*beta**m[i] for i in range(5, 11)])
    vo = R*Tr/Po/1000*(a[5]+suma1+suma2)

    suma1 = sum([a[i]*alfa**n[i] for i in range(11, 16)])
    suma2 = sum([b[i]*beta**m[i] for i in range(11, 18)])
    vpo = R*Tr/Po**2/1000*(suma1+suma2)

    suma1 = sum([n[i]*a[i]*alfa**(n[i]+1) for i in range(1, 4)])
    suma2 = sum([m[i]*b[i]*beta**(m[i]+1) for i in range(1, 5)])
    so = -R*(c[2]+c[3]*(1+log(tau))+suma1-suma2)

    suma1 = sum([n[i]*(n[i]+1)*a[i]*alfa**(n[i]+2) for i in range(1, 4)])
    suma2 = sum([m[i]*(m[i]+1)*b[i]*beta**(m[i]+2) for i in range(1, 5)])
    cpo = -R*(c[3]+tau*suma1+tau*suma2)

    suma1 = sum([n[i]*a[i]*alfa**(n[i]+1) for i in range(6, 11)])
    suma2 = sum([m[i]*b[i]*beta**(m[i]+1) for i in range(5, 11)])
    vto = R/Po/1000*(suma1-suma2)

    # This properties are only neccessary for computing thermodynamic
    # properties at pressures different from 0.1 MPa
    suma1 = sum([n[i]*(n[i]+1)*a[i]*alfa**(n[i]+2) for i in range(6, 11)])
    suma2 = sum([m[i]*(m[i]+1)*b[i]*beta**(m[i]+2) for i in range(5, 11)])
    vtto = R/Tr/Po/1000*(suma1+suma2)

    suma1 = sum([n[i]*a[i]*alfa**(n[i]+1) for i in range(11, 16)])
    suma2 = sum([m[i]*b[i]*beta**(m[i]+1) for i in range(11, 18)])
    vpto = R/Po**2/1000*(suma1-suma2)

    if P != 0.1:
        go += vo*(P-0.1)
        so -= vto*(P-0.1)
        cpo -= T*vtto*(P-0.1)
        vo -= vpo*(P-0.1)
        vto += vpto*(P-0.1)
        vppo = 3.24e-10*R*Tr/0.1**3
        vpo += vppo*(P-0.1)

    h = go+T*so
    u = h-P*vo
    a = go-P*vo
    cv = cpo+T*vto**2/vpo
    kt = -vpo/vo
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
    propiedades["kt"] = kt
    propiedades["alfav"] = vto/vo
    propiedades["ks"] = ks
    propiedades["w"] = w

    # Viscosity correlation, Eq 7
    a = [None, 280.68, 511.45, 61.131, 0.45903]
    b = [None, -1.9, -7.7, -19.6, -40]
    T_ = T/300
    mu = sum([a[i]*T_**b[i] for i in range(1, 5)])/1e6
    propiedades["mu"] = mu

    # Thermal conductivity correlation, Eq 8
    c = [None, 1.6630, -1.7781, 1.1567, -0.432115]
    d = [None, -1.15, -3.4, -6.0, -7.6]
    k = sum([c[i]*T_**d[i] for i in range(1, 5)])
    propiedades["k"] = k

    # Dielectric constant correlation, Eq 9
    e = [None, -43.7527, 299.504, -399.364, 221.327]
    f = [None, -0.05, -1.47, -2.11, -2.31]
    epsilon = sum([e[i]*T_**f[i] for i in range(1, 5)])
    propiedades["epsilon"] = epsilon

    return propiedades


def _Sublimation_Pressure(T):
    """Sublimation Pressure correlation

    Parameters
    ----------
    T : float
        Temperature [K]

    Returns
    -------
    P : float
        Pressure at sublimation line [MPa]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
    else:
        raise NotImplementedError("Incoming out of bound")


def _Melting_Pressure(T, ice="Ih"):
    """Melting Pressure correlation

    Parameters
    ----------
    T : float
        Temperature [K]
    ice: string
        Type of ice: Ih, III, V, VI, VII.
        Below 273.15 is a mandatory input, the ice Ih is the default value.
        Above 273.15, the ice type is unnecesary.

    Returns
    -------
    P : float
        Pressure at sublimation line [MPa]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
        P = Pref*exp(1.73683*(1-1./Tita)-0.544606e-1*(1-Tita**5) +
                     0.806106e-7*(1-Tita**22))
    else:
        raise NotImplementedError("Incoming out of bound")
    return P


# Transport properties
def _Viscosity(rho, T, fase=None, drho=None):
    """Equation for the Viscosity

    Parameters
    ----------
    rho : float
        Density [kg/m³]
    T : float
        Temperature [K]
    fase: dict
        phase properties
    drho: float
        [∂ρ/∂P]T at reference state,
        optional for calculate critical enhancement

    Returns
    -------
    mu : float
        Viscosity [Pa·s]

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

    no = [1.67752, 2.20462, 0.6366564, -0.241605]
    suma = 0
    for i in range(4):
        suma += no[i]/Tr**i
    fi0 = 100*Tr**0.5/suma

    I = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6]
    J = [0, 1, 2, 3, 0, 1, 2, 3, 5, 0, 1, 2, 3, 4, 0, 1, 0, 3, 4, 3, 5]
    nr = [0.520094, 0.850895e-1, -0.108374e1, -0.289555, 0.222531, 0.999115,
          0.188797e1, 0.126613e1, 0.120573, -0.281378, -0.906851, -0.772479,
          -0.489837, -0.257040, 0.161913, 0.257399, -0.325372e-1, 0.698452e-1,
          0.872102e-2, -0.435673e-2, -0.593264e-3]
    suma = 0
    for i in range(21):
        suma += nr[i]*(Dr-1)**I[i]*(1/Tr-1)**J[i]
    fi1 = exp(Dr*suma)
    if fase and drho:
        qc = 1/1.9
        qd = 1/1.1

        DeltaX = Pc*Dr**2*(fase.drhodP_T/rho-drho/rho*1.5/Tr)
        if DeltaX < 0:
            DeltaX = 0
        X = 0.13*(DeltaX/0.06)**(0.63/1.239)
        if X <= 0.3817016416:
            Y = qc/5*X*(qd*X)**5*(1-qc*X+(qc*X)**2-765./504*(qd*X)**2)
        else:
            Fid = acos((1+qd**2*X**2)**-0.5)
            w = abs((qc*X-1)/(qc*X+1))**0.5*tan(Fid/2)
            if qc*X > 1:
                Lw = log((1+w)/(1-w))
            else:
                Lw = 2*atan(abs(w))
            Y = sin(3*Fid)/12-sin(2*Fid)/4/qc/X+(1-5/4*(qc*X)**2)/(
                qc*X)**2*sin(Fid)-((1-3/2*(qc*X)**2)*Fid-abs((
                    qc*X)**2-1)**1.5*Lw)/(qc*X)**3
        fi2 = exp(0.068*Y)
    else:
        fi2 = 1
    return fi0*fi1*fi2*1e-6


def _ThCond(rho, T, fase=None, drho=None):
    """Equation for the thermal conductivity

    Parameters
    ----------
    rho : float
        Density [kg/m³]
    T : float
        Temperature [K]
    fase: dict
        phase properties
    drho: float
        [∂ρ/∂P]T at reference state,
        optional for calculate critical enhancement

    Returns
    -------
    k : float
        Thermal conductivity [W/mK]

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
    d = rho/322.
    Tr = T/647.096

    no = [2.443221e-3, 1.323095e-2, 6.770357e-3, -3.454586e-3, 4.096266e-4]
    suma = 0
    for i in range(5):
        suma += no[i]/Tr**i
    L0 = Tr**0.5/suma

    nij = [
        [1.60397357, -0.646013523, 0.111443906, 0.102997357, -0.0504123634,
         0.00609859258],
        [2.33771842, -2.78843778, 1.53616167, -0.463045512, 0.0832827019,
         -0.00719201245],
        [2.19650529, -4.54580785, 3.55777244, -1.40944978, 0.275418278,
         -0.0205938816],
        [-1.21051378, 1.60812989, -0.621178141, 0.0716373224, 0, 0],
        [-2.7203370, 4.57586331, -3.18369245, 1.1168348, -0.19268305,
         0.012913842]]
    suma = 0
    for i in range(len(nij)):
        suma2 = 0
        for j in range(len(nij[i])):
            suma2 += nij[i][j]*(d-1)**j
        suma += (1/Tr-1)**i*suma2
    L1 = exp(d*suma)

    L2 = 0
    if fase and drho:
        R = 0.46151805

        DeltaX = Pc*d**2*(fase.drhodP_T/rho-drho/rho*1.5/Tr)
        if DeltaX < 0:
            DeltaX = 0
        X = 0.13*(DeltaX/0.06)**(0.63/1.239)
        y = X/0.4
        if y < 1.2e-7:
            Z = 0
        else:
            Z = 2/pi/y*(((1-1/fase.cp_cv)*atan(y)+y/fase.cp_cv)-(
                1-exp(-1/(1/y+y**2/3/d**2))))
        L2 = 177.8514*d*fase.cp/R*Tr/fase.mu*1e-6*Z
    return 1e-3*(L0*L1+L2)


def _Tension(T):
    """Equation for the surface tension

    Parameters
    ----------
    T : float
        Temperature [K]

    Returns
    -------
    sigma : float
        Surface tension [N/m]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
        Tr = T/Tc
        return 1e-3*(235.8*(1-Tr)**1.256*(1-0.625*(1-Tr)))
    else:
        raise NotImplementedError("Incoming out of bound")


def _Dielectric(rho, T):
    """Equation for the Dielectric constant

    Parameters
    ----------
    rho : float
        Density [kg/m³]
    T : float
        Temperature [K]

    Returns
    -------
    epsilon : float
        Dielectric constant [-]

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
    k = 1.380658e-23
    Na = 6.0221367e23
    alfa = 1.636e-40
    epsilon0 = 8.854187817e-12
    mu = 6.138e-30
    M = 0.018015268

    d = rho/rhoc
    Tr = Tc/T
    I = [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10, None]
    J = [0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10, None]
    n = [0.978224486826, -0.957771379375, 0.237511794148, 0.714692244396,
         -0.298217036956, -0.108863472196, .949327488264e-1, -.980469816509e-2,
         .165167634970e-4, .937359795772e-4, -.12317921872e-9,
         .196096504426e-2]

    g = 1+n[11]*d/(Tc/228/Tr-1)**1.2
    for i in range(11):
        g += n[i]*d**I[i]*Tr**J[i]
    A = Na*mu**2*rho*g/M/epsilon0/k/T
    B = Na*alfa*rho/3/M/epsilon0
    e = (1+A+5*B+(9+2*A+18*B+A**2+10*A*B+9*B**2)**0.5)/4/(1-B)
    return e


def _Refractive(rho, T, l=0.5893):
    """Equation for the refractive index

    Parameters
    ----------
    rho : float
        Density [kg/m³]
    T : float
        Temperature [K]
    l : float, optional
        Light Wavelength [μm]

    Returns
    -------
    n : float
        Refractive index [-]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
    if rho < 0 or rho > 1060 or T < 261.15 or T > 773.15 or l < 0.2 or l > 1.1:
        raise NotImplementedError("Incoming out of bound")

    Lir = 5.432937
    Luv = 0.229202
    d = rho/1000.
    Tr = T/273.15
    L = l/0.589
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
        Density [kg/m³]
    T : float
        Temperature [K]

    Returns
    -------
    pKw : float
        Ionization constant in -log10(kw) [-]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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


# Heavy water transport properties
def _D2O_Viscosity(rho, T):
    """Equation for the Viscosity of heavy water

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

    Examples
    --------
    >>> _D2O_Viscosity(998, 298.15)
    0.0008897351001498108
    >>> _D2O_Viscosity(600, 873.15)
    7.743019522728247e-05

    References
    ----------
    IAPWS, Revised Release on Viscosity and Thermal Conductivity of Heavy
    Water Substance, http://www.iapws.org/relguide/TransD2O-2007.pdf
    """
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


def _D2O_ThCond(rho, T):
    """Equation for the thermal conductivity of heavy water

    Parameters
    ----------
    rho : float
        Density [kg/m³]
    T : float
        Temperature [K]

    Returns
    -------
    k : float
        Thermal conductivity [W/mK]

    Examples
    --------
    >>> _D2O_ThCond(998, 298.15)
    0.6077128675880629
    >>> _D2O_ThCond(0, 873.15)
    0.07910346589648833

    References
    ----------
    IAPWS, Revised Release on Viscosity and Thermal Conductivity of Heavy
    Water Substance, http://www.iapws.org/relguide/TransD2O-2007.pdf
    """
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


def _D2O_Tension(T):
    """Equation for the surface tension of heavy water

    Parameters
    ----------
    T : float
        Temperature [K]

    Returns
    -------
    sigma : float
        Surface tension [N/m]

    Raises
    ------
    NotImplementedError : If input isn't in limit
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
    else:
        raise NotImplementedError("Incoming out of bound")


def getphase(Tc, Pc, T, P, x, region):
    """Return fluid phase string name

    Parameters
    ----------
    Tc : float
        Critical temperature [K]
    Pc : float
        Critical pressure [MPa]
    T : float
        Temperature [K]
    P : float
        Pressure [MPa]
    x : float
        Quality [-]
    region: integer
        Region number, specific to IAPWS97 region definition

    Returns
    -------
    phase : string
        Phase name
    """
    # Avoid round problem
    P = round(P, 8)
    T = round(T, 8)
    if P > Pc and T > Tc:
        phase = "Supercritical fluid"
    elif T > Tc:
        phase = "Gas"
    elif P > Pc:
        phase = "Compressible liquid"
    elif P == Pc and T == Tc:
        phase = "Critical point"
    elif region == 4 and x == 1:
        phase = "Saturated vapor"
    elif region == 4 and x == 0:
        phase = "Saturated liquid"
    elif region == 4:
        phase = "Two phases"
    elif x == 1:
        phase = "Vapour"
    elif x == 0:
        phase = "Liquid"
    return phase


class _fase(object):
    """Class to implement a null phase"""
    v = None
    rho = None

    h = None
    s = None
    u = None
    a = None
    g = None

    cp = None
    cv = None
    cp_cv = None
    w = None
    Z = None
    fi = None
    f = None

    mu = None
    k = None
    nu = None
    Prandt = None
    epsilon = None
    alfa = None
    n = None

    alfap = None
    betap = None
    joule = None
    Gruneisen = None
    alfav = None
    kappa = None
    betas = None
    gamma = None
    Kt = None
    kt = None
    Ks = None
    ks = None
    dpdT_rho = None
    dpdrho_T = None
    drhodT_P = None
    drhodP_T = None
    dhdT_rho = None
    dhdT_P = None
    dhdrho_T = None
    dhdrho_P = None
    dhdP_T = None
    dhdP_rho = None

    Z_rho = None
    IntP = None
    hInput = None
