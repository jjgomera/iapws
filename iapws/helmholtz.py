#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Calculations related to free Helmholtz Energy."""

from typing import Dict, List
from math import exp


class HelmholtzDerivatives(object):
    """Helmholtz free energy and derivatives."""

    def __init__(self, fi: float, fit: float, fid: float,
                 fitt: float, fidd: float, fidt: float) -> None:
        self.fi = fi
        self.fit = fit
        self.fid = fid
        self.fitt = fitt
        self.fidd = fidd
        self.fidt = fidt


class ResidualContribution(object):
    """Residual contribution to the adimensional free Helmholtz energy"""

    def __init__(self, nr1: List[float], d1: List[float], t1: List[float],
                 nr2: List[float], d2: List[float], t2: List[float],
                 c2: List[int], nr3: List[float], d3: List[int], t3: List[float],
                 alfa3: List[float], epsilon3: List[float], beta3: List[float],
                 gamma3: List[float], nr4: List[float], a4: List[float],
                 b4: List[float], A: List[float], B: List[float], C: List[int],
                 D: List[int], beta4: List[float]) -> None:
        # Polinomial terms
        self.nr1 = nr1
        self.d1 = d1
        self.t1 = t1
        # Exponential terms
        self.nr2 = nr2
        self.d2 = d2
        self.t2 = t2
        self.c2 = c2
        # Gaussian terms (optional, may be empty lists)
        self.nr3 = nr3
        self.d3 = d3
        self.t3 = t3
        self.alfa3 = alfa3
        self.epsilon3 = epsilon3
        self.beta3 = beta3
        self.gamma3 = gamma3
        # Non analitic terms (optional, may be empty lists)
        self.nr4 = nr4
        self.a4 = a4
        self.b4 = b4
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.beta4 = beta4

    def phir(self, tau: float, delta: float) -> float:
        """Residual contribution to the adimensional free Helmholtz energy

        Parameters
        ----------
        tau : float
            Inverse reduced temperature Tc/T, [-]
        delta : float
            Reduced density rho/rhoc, [-]
        coef : dict
            Dictionary with equation of state parameters

        Returns
        -------
        fir : float
            Adimensional free Helmholtz energy

        References
        ----------
        IAPWS, Revised Release on the IAPWS Formulation 1995 for the
        Thermodynamic Properties of Ordinary Water Substance for General and
        Scientific Use, September 2016, Table 5
        http://www.iapws.org/relguide/IAPWS-95.html
        """
        # Ensure parameters are really floats.
        tau = float(tau)
        delta = float(delta)

        fir = 0.0
        # Polinomial terms
        for n, d, t in zip(self.nr1, self.d1, self.t1):
            fir += n*delta**d*tau**t

        # Exponential terms
        for n, d, t, c in zip(self.nr2, self.d2, self.t2, self.c2):
            fir += n*delta**d*tau**t*exp(-delta**c)

        # Gaussian terms
        for n, d, t, a, e, b, g in zip(self.nr3, self.d3, self.t3, self.alfa3,
                                       self.epsilon3, self.beta3, self.gamma3):
            fir += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)

        # Non analitic terms
        for n, a, b, A, B, C, D, bt in zip(self.nr4, self.a4, self.b4, self.A,
                                           self.B, self.C, self.D, self.beta4):
            Tita = (1-tau)+A*((delta-1)**2)**(0.5/bt)
            F = exp(-C*(delta-1)**2-D*(tau-1)**2)
            Delta = Tita**2+B*((delta-1)**2)**a
            fir += n*Delta**b*delta*F

        return fir

    def phird(self, tau: float, delta: float) -> float:
        r"""Residual contribution to the adimensional free Helmholtz energy, delta
        derivative

        Parameters
        ----------
        tau : float
            Inverse reduced temperature Tc/T, [-]
        delta : float
            Reduced density rho/rhoc, [-]
        coef : dict
            Dictionary with equation of state parameters

        Returns
        -------
        fird : float
            .. math::
              \left.\frac{\partial \phi^r_{\delta}}{\partial \delta}\right|_{\tau}

        References
        ----------
        IAPWS, Revised Release on the IAPWS Formulation 1995 for the
        Thermodynamic Properties of Ordinary Water Substance for General and
        Scientific Use, September 2016, Table 5
        http://www.iapws.org/relguide/IAPWS-95.html
        """
        # Ensure parameters are really floats.
        tau = float(tau)
        delta = float(delta)

        fird = 0.0
        # Polinomial terms
        for n, d, t in zip(self.nr1, self.d1, self.t1):
            fird += n*d*delta**(d-1)*tau**t

        # Exponential terms
        for n, d, t, c in zip(self.nr2, self.d2, self.t2, self.c2):
            try:
                expt = exp(-delta**c)
            except OverflowError:
                fird = float('nan')
                break
            fird += n*expt*delta**(d-1)*tau**t*(d-c*delta**c)

        # Gaussian terms
        for n, d, t, a, e, b, g in zip(self.nr3, self.d3, self.t3, self.alfa3,
                                       self.epsilon3, self.beta3, self.gamma3):
            expt = exp(-a*(delta-e)**2-b*(tau-g)**2)
            fird += n*delta**d*tau**t*expt*(d/delta-2*a*(delta-e))

        # Non analitic terms
        for n, a, b, A, B, C, D, bt in zip(self.nr4, self.a4, self.b4, self.A,
                                           self.B, self.C, self.D, self.beta4):
            Tita = (1-tau)+A*((delta-1)**2)**(0.5/bt)
            F = exp(-C*(delta-1)**2-D*(tau-1)**2)
            Fd = -2*C*F*(delta-1)

            Delta = Tita**2+B*((delta-1)**2)**a
            Deltad = (delta-1)*(A*Tita*2/bt*((delta-1)**2)**(0.5/bt-1)
                                + 2*B*a*((delta-1)**2)**(a-1))
            DeltaBd = b*Delta**(b-1)*Deltad

            fird += n*(Delta**b*(F+delta*Fd)+DeltaBd*delta*F)

        return fird

    def phirt(self, tau: float, delta: float) -> float:
        r"""Residual contribution to the adimensional free Helmholtz energy, tau
        derivative

        Parameters
        ----------
        tau : float
            Inverse reduced temperature Tc/T, [-]
        delta : float
            Reduced density rho/rhoc, [-]
        coef : dict
            Dictionary with equation of state parameters

        Returns
        -------
        firt : float
            .. math::
                \left.\frac{\partial \phi^r_{\tau}}{\partial \tau}\right|_{\delta}

        References
        ----------
        IAPWS, Revised Release on the IAPWS Formulation 1995 for the
        Thermodynamic Properties of Ordinary Water Substance for General and
        Scientific Use, September 2016, Table 5
        http://www.iapws.org/relguide/IAPWS-95.html
        """
        # Ensure parameters are really floats.
        tau = float(tau)
        delta = float(delta)

        firt = 0.0
        # Polinomial terms
        for n, d, t in zip(self.nr1, self.d1, self.t1):
            firt += n*t*delta**d*tau**(t-1)

        # Exponential terms
        for n, d, t, c in zip(self.nr2, self.d2, self.t2, self.c2):
            firt += n*t*delta**d*tau**(t-1)*exp(-delta**c)

        # Gaussian terms
        for n, d, t, a, e, b, g in zip(self.nr3, self.d3, self.t3, self.alfa3,
                                       self.epsilon3, self.beta3, self.gamma3):
            firt += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                t/tau-2*b*(tau-g))

        # Non analitic terms
        for n, a, b, A, B, C, D, bt in zip(self.nr4, self.a4, self.b4, self.A,
                                           self.B, self.C, self.D, self.beta4):
            Tita = (1-tau)+A*((delta-1)**2)**(0.5/bt)
            F = exp(-C*(delta-1)**2-D*(tau-1)**2)
            Ft = -2*D*F*(tau-1)
            Delta = Tita**2+B*((delta-1)**2)**a
            DeltaBt = -2*Tita*b*Delta**(b-1)
            firt += n*delta*(DeltaBt*F+Delta**b*Ft)

        return firt

    def helmholtz(self, tau: float, delta: float) -> HelmholtzDerivatives:
        """Residual contribution to the free Helmholtz energy

        Parameters
        ----------
        tau : float
            Inverse reduced temperature Tc/T, [-]
        delta : float
            Reduced density rho/rhoc, [-]

        Returns
        -------
        HelmholtzDerivatives class, with residual adimensional helmholtz
          energy and deriv:
            * fir
            * firt: ∂fir/∂τ|δ,x
            * fird: ∂fir/∂δ|τ,x
            * firtt: ∂²fir/∂τ²|δ,x
            * firdt: ∂²fir/∂τ∂δ|x
            * firdd: ∂²fir/∂δ²|τ,x

        References
        ----------
        IAPWS, Revised Release on the IAPWS Formulation 1995 for the
        Thermodynamic Properties of Ordinary Water Substance for General and
        Scientific Use, September 2016, Table 5
        http://www.iapws.org/relguide/IAPWS-95.html
        """
        # Ensure parameters are really floats.
        tau = float(tau)
        delta = float(delta)

        fir = fird = firdd = firt = firtt = firdt = 0.0
        # Polinomial terms
        for n, d, t in zip(self.nr1, self.d1, self.t1):
            fir += n*delta**d*tau**t
            fird += n*d*delta**(d-1)*tau**t
            firdd += n*d*(d-1)*delta**(d-2)*tau**t
            firt += n*t*delta**d*tau**(t-1)
            firtt += n*t*(t-1)*delta**d*tau**(t-2)
            firdt += n*t*d*delta**(d-1)*tau**(t-1)

        # Exponential terms
        failed = False
        for n, d, t, c in zip(self.nr2, self.d2, self.t2, self.c2):
            try:
                expdc = exp(-delta**c)
            except OverflowError:
                failed = True
                expdc = float('inf')
            fir += n*delta**d*tau**t*expdc
            fird += n*expdc*delta**(d-1)*tau**t*(d-c*delta**c)
            firdd += n*expdc*delta**(d-2)*tau**t * \
                ((d-c*delta**c)*(d-1-c*delta**c)-c**2*delta**c)
            firt += n*t*delta**d*tau**(t-1)*expdc
            firtt += n*t*(t-1)*delta**d*tau**(t-2)*expdc
            firdt += n*t*delta**(d-1)*tau**(t-1)*(d-c*delta**c)*expdc
        if failed:
            fir = float('nan')

        # Gaussian terms
        for n, d, t, a, e, b, g in zip(self.nr3, self.d3, self.t3, self.alfa3,
                                       self.epsilon3, self.beta3, self.gamma3):
            fir += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)
            fird += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                d/delta-2*a*(delta-e))
            firdd += n*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                -2*a*delta**d + 4*a**2*delta**d*(delta-e)**2
                - 4*d*a*delta**(d-1)*(delta-e) + d*(d-1)*delta**(d-2))
            firt += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                t/tau-2*b*(tau-g))
            firtt += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                (t/tau-2*b*(tau-g))**2-t/tau**2-2*b)
            firdt += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                t/tau-2*b*(tau-g))*(d/delta-2*a*(delta-e))

        # Non analitic terms
        for n, a, b, A, B, C, D, bt in zip(self.nr4, self.a4, self.b4, self.A,
                                           self.B, self.C, self.D, self.beta4):
            Tita = (1-tau)+A*((delta-1)**2)**(0.5/bt)
            F = exp(-C*(delta-1)**2-D*(tau-1)**2)
            Fd = -2*C*F*(delta-1)
            Fdd = 2*C*F*(2*C*(delta-1)**2-1)
            Ft = -2*D*F*(tau-1)
            Ftt = 2*D*F*(2*D*(tau-1)**2-1)
            Fdt = 4*C*D*F*(delta-1)*(tau-1)

            Delta = Tita**2+B*((delta-1)**2)**a
            Deltad = (delta-1)*(A*Tita*2/bt*((delta-1)**2)**(0.5/bt-1)
                                + 2*B*a*((delta-1)**2)**(a-1))
            if delta == 1:
                Deltadd = 0
            else:
                Deltadd = Deltad/(delta-1)+(delta-1)**2*(
                    4*B*a*(a-1)*((delta-1)**2)**(a-2)
                    + 2*A**2/bt**2*(((delta-1)**2)**(0.5/bt-1))**2
                    + A*Tita*4/bt*(0.5/bt-1)*((delta-1)**2)**(0.5/bt-2))

            DeltaBd = b*Delta**(b-1)*Deltad
            DeltaBdd = b*(Delta**(b-1)*Deltadd+(b-1)*Delta**(b-2)*Deltad**2)
            DeltaBt = -2*Tita*b*Delta**(b-1)
            DeltaBtt = 2*b*Delta**(b-1)+4*Tita**2*b*(b-1)*Delta**(b-2)
            DeltaBdt = -A*b*2/bt*Delta**(b-1)*(delta-1)*((delta-1)**2)**(
                0.5/bt-1)-2*Tita*b*(b-1)*Delta**(b-2)*Deltad

            fir += n*Delta**b*delta*F
            fird += n*(Delta**b*(F+delta*Fd)+DeltaBd*delta*F)
            firdd += n*(Delta**b*(2*Fd+delta*Fdd) + 2*DeltaBd*(F+delta*Fd)
                        + DeltaBdd*delta*F)
            firt += n*delta*(DeltaBt*F+Delta**b*Ft)
            firtt += n*delta*(DeltaBtt*F+2*DeltaBt*Ft+Delta**b*Ftt)
            firdt += n*(Delta**b*(Ft+delta*Fdt)+delta*DeltaBd*Ft
                        + DeltaBt*(F+delta*Fd)+DeltaBdt*delta*F)

        return HelmholtzDerivatives(fir, firt, fird, firtt, firdd, firdt)

    def _virial(self, T: float, Tc: float) -> Dict[str, float]:
        """Virial coefficient

        Parameters
        ----------
        T : float
            Temperature [K]

        Returns
        -------
        prop : dict
            Dictionary with residual adimensional helmholtz energy:
                * B: ∂fir/∂δ|δ->0
                * C: ∂²fir/∂δ²|δ->0
        """
        tau = Tc/T
        B = C = 0.0
        delta = 1e-200

        # Polinomial terms
        for n, d, t in zip(self.nr1, self.d1, self.t1):
            B += n*d*delta**(d-1)*tau**t
            C += n*d*(d-1)*delta**(d-2)*tau**t

        # Exponential terms
        for n, d, t, c in zip(self.nr2, self.d2, self.t2, self.c2):
            B += n*exp(-delta**c)*delta**(d-1)*tau**t*(d-c*delta**c)
            C += n*exp(-delta**c)*(delta**(d-2)*tau**t*(
                (d-c*delta**c)*(d-1-c*delta**c)-c**2*delta**c))

        # Gaussian terms
        for n, d, t, a, e, b, g in zip(self.nr3, self.d3, self.t3, self.alfa3,
                                       self.epsilon3, self.beta3, self.gamma3):
            B += n*delta**d*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                d/delta-2*a*(delta-e))
            C += n*tau**t*exp(-a*(delta-e)**2-b*(tau-g)**2)*(
                -2*a*delta**d+4*a**2*delta**d*(
                    delta-e)**2-4*d*a*delta**2*(
                        delta-e)+d*2*delta)

        # Non analitic terms
        for n, a, b, A, B_, C_, D, bt in zip(self.nr4, self.a4, self.b4, self.A,
                                             self.B, self.C, self.D, self.beta4):
            Tita = (1-tau)+A*((delta-1)**2)**(0.5/bt)
            Delta = Tita**2+B_*((delta-1)**2)**a
            Deltad = (delta-1)*(A*Tita*2/bt*((delta-1)**2)**(
                0.5/bt-1)+2*B_*a*((delta-1)**2)**(a-1))
            Deltadd = Deltad/(delta-1) + (delta-1)**2*(
                4*B_*a*(a-1)*((delta-1)**2)**(a-2)
                + 2*A**2/bt**2*(((delta-1)**2)**(0.5/bt-1))**2
                + A*Tita*4/bt*(0.5/bt-1)*((delta-1)**2)**(0.5/bt-2))
            DeltaBd = b*Delta**(b-1)*Deltad
            DeltaBdd = b*(Delta**(b-1)*Deltadd+(b-1)*Delta**(b-2)*Deltad**2)
            F = exp(-C_*(delta-1)**2-D*(tau-1)**2)
            Fd = -2*C_*F*(delta-1)
            Fdd = 2*C_*F*(2*C_*(delta-1)**2-1)

            B += n*(Delta**b*(F+delta*Fd)+DeltaBd*delta*F)
            C += n*(Delta**b*(2*Fd+delta*Fdd)+2*DeltaBd*(F+delta*Fd)
                    + DeltaBdd*delta*F)

        prop = {}
        prop["B"] = B
        prop["C"] = C
        return prop
