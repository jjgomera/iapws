#!/usr/bin/python
# -*- coding: utf-8 -*-

###############################################################################
# Several IAPWS standards
###############################################################################

from __future__ import division

from math import log, exp, tan, atan, acos, sin, pi
from cmath import log as log_c


# Constants
Rm = 8.31451      # kJ/kmol·K
M = 18.015257     # kg/kmol
R = 0.461526      # kJ/kg·K
Tc = 647.096      # K
Pc = 22.064       # MPa
rhoc = 322.       # kg/m³
Tt = 273.16       # K
Pt = 611.657e-6   # MPa
Tb = 373.1243     # K
Dipole = 1.855    # Debye
f_acent = 0.3443


# IAPWS-06 for Ice
def _Ice(T, P):
    """Basic equation for Ice Ih

    >>> "%.9f" % _Ice(100,100)["rho"]
    '941.678203297'
    >>> "%.9f" % _Ice(100,100)["h"]
    '-483.491635676'
    >>> "%.11f" % _Ice(100,100)["s"]
    '-2.61195122589'
    >>> "%.11f" % _Ice(273.152519,0.101325)["a"]
    '-0.00918701567'
    >>> "%.9f" % _Ice(273.152519,0.101325)["u"]
    '-333.465403393'
    >>> "%.11f" % _Ice(273.152519,0.101325)["cp"]
    '2.09671391024'
    >>> "%.15f" % _Ice(273.16,611.657e-6)["alfav"]
    '0.000159863102566'
    >>> "%.11f" % _Ice(273.16,611.657e-6)["beta"]
    '1.35714764659'
    >>> "%.11e" % _Ice(273.16,611.657e-6)["kt"]
    '1.17793449348e-04'
    >>> "%.11e" % _Ice(273.16,611.657e-6)["ks"]
    '1.14161597779e-04'
    """
    # Check input in range of validity
    if P < Pt:
        Psub = _Sublimation_Pressure(T)
        if Psub > P:
            # Zone Gas
            raise NotImplementedError("Incoming out of bound")
    elif P > 208.566:
        # Ice Ih limit upper pressure
        raise NotImplementedError("Incoming out of bound")
    else:
        Pmel = _Melting_Pressure(T, P)
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


def _Sublimation_Pressure(T):
    """Sublimation Pressure correlation"""
    Tita = T/Tt
    suma = 0
    a = [-0.212144006e2, 0.273203819e2, -0.61059813e1]
    expo = [0.333333333e-2, 1.20666667, 1.70333333]
    for ai, expi in zip(a, expo):
        suma += ai*Tita**expi
    return exp(suma/Tita)*Pt


def _Melting_Pressure(T, P):
    """Melting Pressure correlation"""
    if P < 208.566 and 251.165 <= T <= 273.16:
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
    elif 208.566 < P < 350.1 and 251.165 < T <= 256.164:
        # Ice III
        Tref = 251.165
        Pref = 208.566
        Tita = T/Tref
        P = Pref*(1-0.299948*(1-Tita**60.))
    elif 350.1 < P < 632.4 and 256.164 < T <= 273.31:
        # Ice V
        Tref = 256.164
        Pref = 350.100
        Tita = T/Tref
        P = Pref*(1-1.18721*(1-Tita**8.))
    elif 632.4 < P < 2216 and 273.31 < T <= 355:
        # Ice VI
        Tref = 273.31
        Pref = 632.400
        Tita = T/Tref
        P = Pref*(1-1.07476*(1-Tita**4.6))
    elif 2216 < P and 355. < T <= 715:
        # Ice VII
        Tref = 355
        Pref = 2216.000
        Tita = T/Tref
        P = Pref*exp(1.73683*(1-1./Tita)-0.544606e-1*(1-Tita**5) +
                     0.806106e-7*(1-Tita**22))

    return P


# Transport properties
def _Viscosity(rho, T, fase=None, drho=None):
    """Equation for the Viscosity

    >>> "%.12f" % _Viscosity(997.047435,298.15)
    '0.000890022551'
    >>> "%.13f" % _Viscosity(54.9921814,873.15)
    '0.0000339743835'
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

    >>> "%.9f" % _ThCond(997.047435,298.15)
    '0.606515826'
    >>> "%.10f" % _ThCond(26.0569558,873.15)
    '0.0870480934'
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

    >>> "%.10f" % _Tension(300)
    '0.0716859625'
    >>> "%.10f" % _Tension(450)
    '0.0428914992'
    """
    Tr = T/Tc
    if 273.15 <= T < Tc:
        return 1e-3*(235.8*(1-Tr)**1.256*(1-0.625*(1-Tr)))
    else:
        return 0


def _Dielectric(rho, T):
    """Equation for the Dielectric constant

    >>> "%.7f" % _Dielectric(999.242866, 298.15)
    '78.5907250'
    >>> "%.8f" % _Dielectric(26.0569558, 873.15)
    '1.12620970'
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

    >>> "%.8f" % _Refractive(997.047435, 298.15, 0.2265)
    '1.39277824'
    >>> "%.8f" % _Refractive(30.4758534, 773.15, 0.5893)
    '1.00949307'
    """
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


def getphase(Tc, Pc, T, P, x, region):
    """Return fluid phase"""
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
    else:
        phase = "Unknown"
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
