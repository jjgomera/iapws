#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Test IAPWS mopdule."""

from math import log
import sys
import unittest

from iapws.iapws97 import (IAPWS97, IAPWS97_Tx, IAPWS97_Px, IAPWS97_Ph,
                           IAPWS97_Ps, IAPWS97_PT)
from iapws.iapws97 import (_Region1, _Region2, _Region3, _Region5,
                           _Backward1_T_Ph, _Backward1_T_Ps, _Backward1_P_hs,
                           _Backward2_T_Ph, _Backward2_T_Ps, _Backward2_P_hs,
                           _h_3ab, _Backward3_T_Ph, _Backward3_v_Ph,
                           _Backward3_T_Ps, _Backward3_v_Ps, _PSat_h, _PSat_s,
                           _Backward3_P_hs, _h1_s, _h3a_s, _h2ab_s, _h2c3b_s,
                           _PSat_T, _TSat_P, _h13_s, _t_hs, _Backward4_T_hs,
                           _tab_P, _top_P, _twx_P, _tef_P, _txx_P, _hab_s,
                           _Backward3_v_PT, _P23_T, _t_P, _P_2bc, _hbc_P)
from iapws.iapws95 import (IAPWS95, IAPWS95_PT, IAPWS95_Tx, IAPWS95_Ph,
                           IAPWS95_Px, IAPWS95_Ps, D2O)
from iapws.iapws08 import (SeaWater, _ThCond_SeaWater, _Tension_SeaWater,
                           _solNa2SO4, _critNaCl, _Tb, _Tf, _Triple,
                           _OsmoticPressure)
from iapws._iapws import (_Ice, _Sublimation_Pressure, _Melting_Pressure,
                          _Viscosity, _ThCond, _Tension, _Kw, _Liquid,
                          _D2O_Viscosity, _D2O_ThCond, _D2O_Tension,
                          _D2O_Sublimation_Pressure, _D2O_Melting_Pressure,
                          _Conductivity, _Henry, _Kvalue, _Supercooled)
from iapws.humidAir import _virial, _fugacity, Air, HumidAir
from iapws.ammonia import NH3, H2ONH3, Ttr


# Python version detect for new capacities of unittest
major = sys.version_info[0]
minor = sys.version_info[1]


# Test
class Test(unittest.TestCase):
    """
    Global unittest for module
    Run for python2 and python3 before to release distribution
    """

    def test_Helmholtz(self) -> None:
        """Table 6 from IAPWS95, pag 14"""
        T = 500
        rho = 838.025
        fluid = IAPWS95()
        delta = rho/fluid.rhoc
        tau = fluid.Tc/T

        ideal = fluid._phi0(tau, delta)
        self.assertEqual(round(ideal["fio"], 8), 2.04797733)
        self.assertEqual(round(ideal["fiod"], 9), 0.384236747)
        self.assertEqual(round(ideal["fiodd"], 9), -0.147637878)
        self.assertEqual(round(ideal["fiot"], 8), 9.04611106)
        self.assertEqual(round(ideal["fiott"], 8), -1.93249185)
        self.assertEqual(round(ideal["fiodt"], 8), 0.0)

        res = fluid._phir(tau, delta)
        self.assertEqual(round(res["fir"], 8), -3.42693206)
        self.assertEqual(round(res["fird"], 9), -0.364366650)
        self.assertEqual(round(res["firdd"], 9), 0.856063701)
        self.assertEqual(round(res["firt"], 8), -5.81403435)
        self.assertEqual(round(res["firtt"], 8), -2.23440737)
        self.assertEqual(round(res["firdt"], 8), -1.12176915)

        # Revised release of 2018
        # Virial coefficient in Table 3
        vir = fluid._virial(600)
        self.assertEqual(round(vir["B"]/fluid.rhoc, 11), -0.555366808e-2)
        self.assertEqual(round(vir["C"]/fluid.rhoc**2, 14), -0.669015050e-5)

    def test_phase(self) -> None:
        """Table 7 from IAPWS95, pag 14"""
        state = IAPWS95(rho=996.556, T=300)
        # See footnote for imprecise P value in last significant figures
        self.assertEqual(round(state.P*1e3, 7), 99.241835)
        self.assertEqual(round(state.cv, 8), 4.13018112)
        self.assertEqual(round(state.w, 5), 1501.51914)
        self.assertEqual(round(state.s, 9), 0.393062643)

        state = IAPWS95(rho=1005.308, T=300)
        self.assertEqual(round(state.P*1e3, 4), 20002.2515)
        self.assertEqual(round(state.cv, 8), 4.06798347)
        self.assertEqual(round(state.w, 5), 1534.92501)
        self.assertEqual(round(state.s, 9), 0.387405401)

        state = IAPWS95(rho=1188.202, T=300)
        self.assertEqual(round(state.P*1e3, 3), 700004.704)
        self.assertEqual(round(state.cv, 8), 3.46135580)
        self.assertEqual(round(state.w, 5), 2443.57992)
        self.assertEqual(round(state.s, 9), 0.132609616)

        state = IAPWS95(rho=0.435, T=500)
        self.assertEqual(round(state.P*1e3, 7), 99.9679423)
        self.assertEqual(round(state.cv, 8), 1.50817541)
        self.assertEqual(round(state.w, 6), 548.314253)
        self.assertEqual(round(state.s, 8), 7.94488271)

        state = IAPWS95(rho=4.532, T=500)
        self.assertEqual(round(state.P*1e3, 6), 999.938125)
        self.assertEqual(round(state.cv, 8), 1.66991025)
        self.assertEqual(round(state.w, 6), 535.739001)
        self.assertEqual(round(state.s, 8), 6.82502725)

        state = IAPWS95(rho=838.025, T=500)
        self.assertEqual(round(state.P*1e3, 4), 10000.3858)
        self.assertEqual(round(state.cv, 8), 3.22106219)
        self.assertEqual(round(state.w, 5), 1271.28441)
        self.assertEqual(round(state.s, 8), 2.56690919)

        state = IAPWS95(rho=1084.564, T=500)
        self.assertEqual(round(state.P*1e3, 3), 700000.405)
        self.assertEqual(round(state.cv, 8), 3.07437693)
        self.assertEqual(round(state.w, 5), 2412.00877)
        self.assertEqual(round(state.s, 8), 2.03237509)

        state = IAPWS95(rho=358., T=647)
        self.assertEqual(round(state.P*1e3, 4), 22038.4756)
        self.assertEqual(round(state.cv, 8), 6.18315728)
        self.assertEqual(round(state.w, 6), 252.145078)
        self.assertEqual(round(state.s, 8), 4.32092307)

        state = IAPWS95(rho=0.241, T=900)
        self.assertEqual(round(state.P*1e3, 6), 100.062559)
        self.assertEqual(round(state.cv, 8), 1.75890657)
        self.assertEqual(round(state.w, 6), 724.027147)
        self.assertEqual(round(state.s, 8), 9.16653194)

        state = IAPWS95(rho=52.615, T=900)
        self.assertEqual(round(state.P*1e3, 3), 20000.069)
        self.assertEqual(round(state.cv, 8), 1.93510526)
        self.assertEqual(round(state.w, 6), 698.445674)
        self.assertEqual(round(state.s, 8), 6.59070225)

        state = IAPWS95(rho=870.769, T=900)
        self.assertEqual(round(state.P*1e3, 3), 700000.006)
        self.assertEqual(round(state.cv, 8), 2.66422350)
        self.assertEqual(round(state.w, 5), 2019.33608)
        self.assertEqual(round(state.s, 8), 4.17223802)

    def test_saturation(self) -> None:
        """Table 8 from IAPWS95, pag 14"""
        fluid = IAPWS95()

        rhol, rhov, Ps = fluid._saturation(275)
        liquid = fluid._Helmholtz(rhol, 275)
        vapor = fluid._Helmholtz(rhov, 275)
        self.assertEqual(round(Ps, 9), 0.698451167)
        self.assertEqual(round(rhol, 6), 999.887406)
        self.assertEqual(round(rhov, 11), 0.00550664919)
        self.assertEqual(round(liquid["h"], 8), 7.75972202)
        self.assertEqual(round(vapor["h"], 5), 2504.28995)
        self.assertEqual(round(liquid["s"], 10), 0.0283094670)
        self.assertEqual(round(vapor["s"], 8), 9.10660121)

        rhol, rhov, Ps = fluid._saturation(450)
        liquid = fluid._Helmholtz(rhol, 450)
        vapor = fluid._Helmholtz(rhov, 450)
        self.assertEqual(round(Ps, 6), 932.203564)
        self.assertEqual(round(rhol, 6), 890.341250)
        self.assertEqual(round(rhov, 8), 4.81200360)
        self.assertEqual(round(liquid["h"], 6), 749.161585)
        self.assertEqual(round(vapor["h"], 5), 2774.41078)
        self.assertEqual(round(liquid["s"], 8), 2.10865845)
        self.assertEqual(round(vapor["s"], 8), 6.60921221)

        rhol, rhov, Ps = fluid._saturation(625)
        liquid = fluid._Helmholtz(rhol, 625)
        vapor = fluid._Helmholtz(rhov, 625)
        self.assertEqual(round(Ps, 4), 16908.2693)
        self.assertEqual(round(rhol, 6), 567.090385)
        self.assertEqual(round(rhov, 6), 118.290280)
        self.assertEqual(round(liquid["h"], 5), 1686.26976)
        self.assertEqual(round(vapor["h"], 5), 2550.71625)
        self.assertEqual(round(liquid["s"], 8), 3.80194683)
        self.assertEqual(round(vapor["s"], 8), 5.18506121)

    def test_LowT(self) -> None:
        """Table 3, pag 5"""
        fluid = IAPWS95()
        fex, fext, fextt = fluid._phiex(50)
        self.assertEqual(round(fex, 11), 0.00381124912)
        self.assertEqual(round(fext, 11), 0.00172505502)
        self.assertEqual(round(fextt, 12), 0.000525861643)
        self.assertEqual(round(fluid._prop0(1, 50).cp/fluid.R, 8), 3.91824190)

        fex, fext, fextt = fluid._phiex(100)
        self.assertEqual("%0.8e" % fex, "3.98019838e-06")
        self.assertEqual(round(fext, 13), 0.120506637e-4)
        self.assertEqual(round(fextt, 13), 0.277309851e-4)
        self.assertEqual(round(fluid._prop0(1, 100).cp/fluid.R, 8), 4.00536708)

        fluid = IAPWS95(T=120, P=0.1)

    def test_Melting(self) -> None:
        """Table 3, pag 7"""
        self.assertRaises(NotImplementedError, _Sublimation_Pressure, 49)
        self.assertRaises(NotImplementedError, _Sublimation_Pressure, 274)
        self.assertEqual(round(_Sublimation_Pressure(230), 11), 8.94735e-6)
        self.assertRaises(NotImplementedError, _Melting_Pressure, 250)
        self.assertEqual(round(_Melting_Pressure(260, "Ih"), 3), 138.268)
        self.assertEqual(round(_Melting_Pressure(254, "III"), 3), 268.685)
        self.assertEqual(round(_Melting_Pressure(265, "V"), 3), 479.640)
        self.assertEqual(round(_Melting_Pressure(320, "VI"), 2), 1356.76)
        self.assertEqual(round(_Melting_Pressure(550, "VII"), 2), 6308.71)

    def test_Viscosity_1(self) -> None:
        """Table 4, pag 8"""
        self.assertEqual(round(_Viscosity(998, 298.15)*1e6, 6), 889.735100)
        self.assertEqual(round(_Viscosity(1200, 298.15)*1e6, 6), 1437.649467)
        self.assertEqual(round(_Viscosity(1000, 373.15)*1e6, 6), 307.883622)
        self.assertEqual(round(_Viscosity(1, 433.15)*1e6, 6), 14.538324)
        self.assertEqual(round(_Viscosity(1000, 433.15)*1e6, 6), 217.685358)
        self.assertEqual(round(_Viscosity(1, 873.15)*1e6, 6), 32.619287)
        self.assertEqual(round(_Viscosity(100, 873.15)*1e6, 6), 35.802262)
        self.assertEqual(round(_Viscosity(600, 873.15)*1e6, 6), 77.430195)
        self.assertEqual(round(_Viscosity(1, 1173.15)*1e6, 6), 44.217245)
        self.assertEqual(round(_Viscosity(100, 1173.15)*1e6, 6), 47.640433)
        self.assertEqual(round(_Viscosity(400, 1173.15)*1e6, 6), 64.154608)

        # Table 5, pag 9
        fluid = IAPWS95(rho=122, T=647.35)
        self.assertEqual(round(fluid.mu*1e6, 6), 25.520677)
        fluid = IAPWS95(rho=222, T=647.35)
        self.assertEqual(round(fluid.mu*1e6, 6), 31.337589)
        fluid = IAPWS95(rho=272, T=647.35)
        self.assertEqual(round(fluid.mu*1e6, 6), 36.228143)
        fluid = IAPWS95(rho=322, T=647.35)
        self.assertEqual(round(fluid.mu*1e6, 6), 42.961579)
        fluid = IAPWS95(rho=372, T=647.35)
        self.assertEqual(round(fluid.mu*1e6, 6), 45.688204)
        fluid = IAPWS95(rho=422, T=647.35)
        self.assertEqual(round(fluid.mu*1e6, 6), 49.436256)

    def test_ThCond(self) -> None:
        """Table 4, pag 10"""
        self.assertEqual(round(_ThCond(0, 298.15)*1000, 7), 18.4341883)
        self.assertEqual(round(_ThCond(998, 298.15)*1000, 6), 607.712868)
        self.assertEqual(round(_ThCond(1200, 298.15)*1000, 6), 799.038144)
        self.assertEqual(round(_ThCond(0, 873.15)*1000, 7), 79.1034659)

        # Table 5, pag 10
        fluid95 = IAPWS95(rho=1, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 7), 51.9298924)
        fluid95 = IAPWS95(rho=122, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 6), 130.922885)
        fluid95 = IAPWS95(rho=222, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 6), 367.787459)
        fluid95 = IAPWS95(rho=272, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 6), 757.959776)
        fluid95 = IAPWS95(rho=322, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 5), 1443.75556)
        fluid95 = IAPWS95(rho=372, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 6), 650.319402)
        fluid95 = IAPWS95(rho=422, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 6), 448.883487)
        fluid95 = IAPWS95(rho=750, T=647.35)
        self.assertEqual(round(fluid95.k*1000, 6), 600.961346)

        # Industrial formulation, Table 7, 8, 9
        fluid97 = IAPWS97(T=620, P=20)
        self.assertEqual(round(fluid97.k*1000, 6), 481.485195)
        fluid97 = IAPWS97(T=620, P=50)
        self.assertEqual(round(fluid97.k*1000, 6), 545.038940)
        fluid97 = IAPWS97(T=650, P=0.3)
        self.assertEqual(round(fluid97.k*1000, 7), 52.2311024)
        fluid97 = IAPWS97(T=800, P=50)
        self.assertEqual(round(fluid97.k*1000, 6), 177.709914)
        P = _Region3(T=647.35, rho=222)["P"]
        fluid97 = IAPWS97(T=647.35, P=P)
        self.assertEqual(round(fluid97.k*1000, 6), 366.879411)
        P = _Region3(T=647.35, rho=322)["P"]
        fluid97 = IAPWS97(T=647.35, P=P)
        self.assertEqual(round(fluid97.k*1000, 5), 1241.82415)

    def test_Tension(self) -> None:
        """Selected values from table 1"""
        self.assertRaises(NotImplementedError, _Tension, 230)
        self.assertEqual(round(_Tension(273.16)*1000, 2), 75.65)
        self.assertEqual(round(_Tension(283.15)*1000, 2), 74.22)
        self.assertEqual(round(_Tension(293.15)*1000, 2), 72.74)
        self.assertEqual(round(_Tension(303.15)*1000, 2), 71.19)
        self.assertEqual(round(_Tension(313.15)*1000, 2), 69.60)
        self.assertEqual(round(_Tension(323.15)*1000, 2), 67.94)
        self.assertEqual(round(_Tension(333.15)*1000, 2), 66.24)
        self.assertEqual(round(_Tension(343.15)*1000, 2), 64.48)
        self.assertEqual(round(_Tension(353.15)*1000, 2), 62.67)
        self.assertEqual(round(_Tension(363.15)*1000, 2), 60.82)
        self.assertEqual(round(_Tension(373.15)*1000, 2), 58.91)
        self.assertEqual(round(_Tension(393.15)*1000, 2), 54.97)
        self.assertEqual(round(_Tension(413.15)*1000, 2), 50.86)
        self.assertEqual(round(_Tension(433.15)*1000, 2), 46.59)
        self.assertEqual(round(_Tension(453.15)*1000, 2), 42.19)
        self.assertEqual(round(_Tension(473.15)*1000, 2), 37.67)
        self.assertEqual(round(_Tension(523.15)*1000, 2), 26.04)
        self.assertEqual(round(_Tension(573.15)*1000, 2), 14.36)
        self.assertEqual(round(_Tension(623.15)*1000, 2), 3.67)
        self.assertEqual(round(_Tension(643.15)*1000, 2), 0.39)

    def test_Dielect(self) -> None:
        """Table 4, pag 8"""
        fluid = IAPWS95(P=0.101325, T=240)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 104.34982)
        fluid = IAPWS95(P=0.101325, T=300)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 77.74735)
        fluid = IAPWS95(P=10, T=300)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 78.11269)
        fluid = IAPWS95(P=1000, T=300)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 103.69632)
        fluid = IAPWS95(P=10, T=650)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 1.26715)
        fluid = IAPWS95(P=100, T=650)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 17.71733)
        fluid = IAPWS95(P=500, T=650)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 26.62132)
        fluid = IAPWS95(P=10, T=870)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 1.12721)
        fluid = IAPWS95(P=100, T=870)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 4.98281)
        fluid = IAPWS95(P=500, T=870)
        assert(fluid.epsilon is not None)
        self.assertEqual(round(fluid.epsilon, 5), 15.09746)

    def test_Refractive(self) -> None:
        """Selected values from table 3, pag 6"""
        fluid = IAPWS95(P=0.1, T=273.15, l=0.2265)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 6), 1.394527)
        fluid = IAPWS95(P=10., T=273.15, l=0.2265)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 6), 1.396526)
        fluid = IAPWS95(P=1., T=373.15, l=0.2265)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 6), 1.375622)
        fluid = IAPWS95(P=100., T=373.15, l=0.2265)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 6), 1.391983)
        fluid = IAPWS95(P=0.1, T=473.15, l=0.589)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 7), 1.0001456)
        fluid = IAPWS95(P=1., T=773.15, l=0.589)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 7), 1.0008773)
        fluid = IAPWS95(P=10., T=273.15, l=1.01398)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 6), 1.327710)
        fluid = IAPWS95(P=100., T=473.15, l=1.01398)
        assert(fluid.n is not None)
        self.assertEqual(round(fluid.n, 6), 1.298369)

    def test_kw(self) -> None:
        """Table 3, pag 5"""
        self.assertRaises(NotImplementedError, _Kw, *(1000, 270))
        self.assertEqual(round(_Kw(1000, 300), 6), 13.906565)
        self.assertEqual(round(_Kw(70, 600), 6), 21.048874)
        self.assertEqual(round(_Kw(700, 600), 6), 11.203153)
        self.assertEqual(round(_Kw(200, 800), 6), 15.089765)
        self.assertEqual(round(_Kw(1200, 800), 6), 6.438330)

    def test_liquid(self) -> None:
        """Table 8, pag 11"""
        liq = _Liquid(260)
        self.assertEqual(round(liq["g"], 7), -1.2659892)
        self.assertEqual(round(liq["s"], 8), -.20998555)
        self.assertEqual(round(liq["cp"], 8), 4.30017472)
        self.assertEqual(round(liq["rho"], 6), 997.068360)
        self.assertEqual(round(liq["vt"]*1e7, 8), -3.86550941)
        self.assertEqual(round(liq["vtt"]*1e8, 8), 3.27442503)
        self.assertEqual(round(liq["vp"]*1e7, 8), -5.82096820)
        self.assertEqual(round(liq["vpt"]*1e9, 8), 7.80938294)
        self.assertEqual(round(liq["w"], 5), 1324.87258)
        self.assertEqual(round(liq["mu"]*1e6, 5), 3058.36075)
        self.assertEqual(round(liq["k"], 9), 0.515628010)
        self.assertEqual(round(liq["epsilon"], 6), 93.455835)

        liq = _Liquid(298.15)
        self.assertEqual(round(liq["g"], 7), -4.5617537)
        self.assertEqual(round(liq["s"], 8), 0.36720145)
        self.assertEqual(round(liq["cp"], 8), 4.18144618)
        self.assertEqual(round(liq["rho"], 6), 997.047013)
        self.assertEqual(round(liq["vt"]*1e7, 8), 2.58054178)
        self.assertEqual(round(liq["vtt"]*1e8, 8), 0.97202076)
        self.assertEqual(round(liq["vp"]*1e7, 8), -4.53803340)
        self.assertEqual(round(liq["vpt"]*1e9, 8), 1.00038567)
        self.assertEqual(round(liq["w"], 5), 1496.69922)
        self.assertEqual(round(liq["mu"]*1e6, 6), 889.996774)
        self.assertEqual(round(liq["k"], 9), 0.606502308)
        self.assertEqual(round(liq["epsilon"], 6), 78.375218)

        liq = _Liquid(375)
        self.assertEqual(round(liq["g"], 7), -71.0588021)
        self.assertEqual(round(liq["s"], 8), 1.32806616)
        self.assertEqual(round(liq["cp"], 8), 4.21774697)
        self.assertEqual(round(liq["rho"], 6), 957.009710)
        self.assertEqual(round(liq["vt"]*1e7, 8), 7.94706623)
        self.assertEqual(round(liq["vtt"]*1e8, 8), 0.62024104)
        self.assertEqual(round(liq["vp"]*1e7, 8), -5.15666528)
        self.assertEqual(round(liq["vpt"]*1e9, 8), -2.27073594)
        self.assertEqual(round(liq["w"], 5), 1541.46611)
        self.assertEqual(round(liq["mu"]*1e6, 6), 276.207245)
        self.assertEqual(round(liq["k"], 9), 0.677913788)
        self.assertEqual(round(liq["epsilon"], 6), 55.266199)

        if major == 3:
            self.assertWarns(Warning, _Liquid, *(375, 0.2))
        self.assertRaises(NotImplementedError, _Liquid, *(375, 0.4))

    def test_superCooled(self) -> None:
        """Table 5, pag 9"""
        liq = _Supercooled(273.15, 0.101325)
        self.assertEqual(round(liq["x"], 8), 0.09665472)
        self.assertEqual(round(liq["L"], 8), 0.62120474)
        self.assertEqual(round(liq["rho"], 5), 999.84229)
        self.assertEqual(round(liq["alfap"]*1e4, 6), -0.683042)
        self.assertEqual(round(liq["xkappa"]*1e4, 6), 5.088499)
        self.assertEqual(round(liq["cp"], 7), 4.2183002)
        self.assertEqual(round(liq["w"], 4), 1402.3886)

        liq = _Supercooled(235.15, 0.101325)
        self.assertEqual(round(liq["x"], 8), 0.25510286)
        self.assertEqual(round(liq["L"], 8), 0.09176368)
        self.assertEqual(round(liq["rho"], 5), 968.09999)
        self.assertEqual(round(liq["alfap"]*1e4, 5), -29.63382)
        self.assertEqual(round(liq["xkappa"]*1e4, 6), 11.580785)
        self.assertEqual(round(liq["cp"], 6), 5.997563)
        self.assertEqual(round(liq["w"], 4), 1134.5855)

        liq = _Supercooled(250, 200)
        self.assertEqual(round(liq["x"], 8), 0.03042927)
        self.assertEqual(round(liq["L"], 8), 0.72377081)
        self.assertEqual(round(liq["rho"], 5), 1090.45677)
        self.assertEqual(round(liq["alfap"]*1e4, 6), 3.267768)
        self.assertEqual(round(liq["xkappa"]*1e4, 6), 3.361311)
        self.assertEqual(round(liq["cp"], 7), 3.7083902)
        self.assertEqual(round(liq["w"], 4), 1668.2020)

        liq = _Supercooled(200, 400)
        self.assertEqual(round(liq["x"], 8), 0.00717008)
        self.assertEqual(round(liq["L"], 7), 1.1553965)
        self.assertEqual(round(liq["rho"], 5), 1185.02800)
        self.assertEqual(round(liq["alfap"]*1e4, 6), 6.716009)
        self.assertEqual(round(liq["xkappa"]*1e4, 6), 2.567237)
        self.assertEqual(round(liq["cp"], 7), 3.3385250)
        self.assertEqual(round(liq["w"], 4), 1899.3294)

        liq = _Supercooled(250, 400)
        self.assertEqual(round(liq["x"], 8), 0.00535884)
        self.assertEqual(round(liq["L"], 7), 1.4345145)
        self.assertEqual(round(liq["rho"], 4), 1151.7152)
        self.assertEqual(round(liq["alfap"]*1e4, 5), 4.92993)
        self.assertEqual(round(liq["xkappa"]*1e4, 6), 2.277029)
        self.assertEqual(round(liq["cp"], 7), 3.7572144)
        self.assertEqual(round(liq["w"], 4), 2015.8782)

        self.assertRaises(NotImplementedError, _Supercooled, *(200, 100))
        self.assertRaises(NotImplementedError, _Supercooled, *(180, 300))

    def test_auxiliarySaturation(self) -> None:
        """Table 1 pag 7"""
        fluid = IAPWS95()
        self.assertEqual(round(fluid._Vapor_Pressure(273.16), 9), 0.000611657)
        self.assertEqual(round(fluid._dPdT_sat(273.16), 12), 44.436693e-6)
        self.assertEqual(round(fluid._Liquid_Density(273.16), 3), 999.789)
        self.assertEqual(round(fluid._Vapor_Density(273.16), 8), 0.00485426)
        self.assertEqual(round(fluid._alfa_sat(273.16), 9), -0.011529101)
        self.assertEqual(round(fluid._Liquid_Enthalpy(273.16), 9), 0.000611786)
        self.assertEqual(round(fluid._Vapor_Enthalpy(273.16), 1), 2500.5)
        self.assertEqual(round(fluid._phi_sat(273.16), 5), -0.00004)
        self.assertEqual(round(fluid._Liquid_Entropy(273.16), 1), 0)
        self.assertEqual(round(fluid._Vapor_Entropy(273.16), 3), 9.154)

        self.assertEqual(round(fluid._Vapor_Pressure(373.1243), 6), 0.101325)
        self.assertEqual(round(fluid._dPdT_sat(373.1243), 6), 0.003616)
        self.assertEqual(round(fluid._Liquid_Density(373.1243), 3), 958.365)
        self.assertEqual(round(fluid._Vapor_Density(373.1243), 6), 0.597586)
        self.assertEqual(round(fluid._alfa_sat(373.1243), 2), 417.65)
        self.assertEqual(round(fluid._Liquid_Enthalpy(373.1243), 2), 419.05)
        self.assertEqual(round(fluid._Vapor_Enthalpy(373.1243), 1), 2675.7)
        self.assertEqual(round(fluid._phi_sat(373.1243), 3), 1.303)
        self.assertEqual(round(fluid._Liquid_Entropy(373.1243), 3), 1.307)
        self.assertEqual(round(fluid._Vapor_Entropy(373.1243), 3), 7.355)

        self.assertEqual(round(fluid._Vapor_Pressure(647.096), 3), 22.064)
        self.assertEqual(round(fluid._dPdT_sat(647.096), 3), 0.268)
        self.assertEqual(round(fluid._Liquid_Density(647.096), 3), 322)
        self.assertEqual(round(fluid._Vapor_Density(647.096), 8), 322)
        self.assertEqual(round(fluid._alfa_sat(647.096), 0), 1548)
        self.assertEqual(round(fluid._Liquid_Enthalpy(647.096), 1), 2086.6)
        self.assertEqual(round(fluid._Vapor_Enthalpy(647.096), 1), 2086.6)
        self.assertEqual(round(fluid._phi_sat(647.096), 3), 3.578)
        self.assertEqual(round(fluid._Liquid_Entropy(647.096), 3), 4.410)
        self.assertEqual(round(fluid._Vapor_Entropy(647.096), 3), 4.410)

    def test_IAPWS97_1(self) -> None:
        """Table 5, pag 9"""
        fluid = _Region1(300, 3)
        self.assertEqual(round(fluid["v"], 11), 0.00100215168)
        self.assertEqual(round(fluid["h"], 6), 115.331273)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 6), 112.324818)
        self.assertEqual(round(fluid["s"], 9), 0.392294792)
        self.assertEqual(round(fluid["cp"], 8), 4.17301218)
        self.assertEqual(round(fluid["w"], 5), 1507.73921)

        fluid = _Region1(300, 80)
        self.assertEqual(round(fluid["v"], 12), 0.000971180894)
        self.assertEqual(round(fluid["h"], 6), 184.142828)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 6), 106.448356)
        self.assertEqual(round(fluid["s"], 9), 0.368563852)
        self.assertEqual(round(fluid["cp"], 8), 4.01008987)
        self.assertEqual(round(fluid["w"], 5), 1634.69054)

        fluid = _Region1(500, 3)
        self.assertEqual(round(fluid["v"], 10), 0.0012024180)
        self.assertEqual(round(fluid["h"], 6), 975.542239)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 6), 971.934985)
        self.assertEqual(round(fluid["s"], 9), 2.58041912)
        self.assertEqual(round(fluid["cp"], 8), 4.65580682)
        self.assertEqual(round(fluid["w"], 5), 1240.71337)

        # _Backward1_T_Ph Table 7 pag 11
        self.assertEqual(round(_Backward1_T_Ph(3, 500), 6), 391.798509)
        self.assertEqual(round(_Backward1_T_Ph(80, 500), 6), 378.108626)
        self.assertEqual(round(_Backward1_T_Ph(80, 1500), 6), 611.041229)

        # _Backward1_T_Ps Table 9 pag 12
        self.assertEqual(round(_Backward1_T_Ps(3, 0.5), 6), 307.842258)
        self.assertEqual(round(_Backward1_T_Ps(80, 0.5), 6), 309.979785)
        self.assertEqual(round(_Backward1_T_Ps(80, 3), 6), 565.899909)

        # _Backward1_P_hs Table 3 pag 6 for supplementary p(h,s)
        self.assertEqual(round(_Backward1_P_hs(0.001, 0), 13), 0.0009800980612)
        self.assertEqual(round(_Backward1_P_hs(90, 0), 8), 91.92954727)
        self.assertEqual(round(_Backward1_P_hs(1500, 3.4), 8), 58.68294423)

    def test_IAPWS97_2(self) -> None:
        """Table 15, pag 17"""
        # Auxiliary equation for the boundary 2-3
        self.assertEqual(round(_P23_T(623.15), 7), 16.5291643)
        self.assertEqual(round(_t_P(16.5291643), 6), 623.15)

        # Auxiliary equation for the subregion2 boundary
        self.assertEqual(round(_P_2bc(3516.004323), 6), 100.0)
        self.assertEqual(round(_hbc_P(100), 6), 3516.004323)
        self.assertEqual(round(_hab_s(7), 6), 3376.437884)

        fluid = _Region2(300, 0.0035)
        self.assertEqual(round(fluid["v"], 7), 39.4913866)
        self.assertEqual(round(fluid["h"], 5), 2549.91145)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 2411.69160)
        self.assertEqual(round(fluid["s"], 8), 8.52238967)
        self.assertEqual(round(fluid["cp"], 8), 1.91300162)
        self.assertEqual(round(fluid["w"], 6), 427.920172)

        fluid = _Region2(700, 0.0035)
        self.assertEqual(round(fluid["v"], 7), 92.3015898)
        self.assertEqual(round(fluid["h"], 5), 3335.68375)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 3012.62819)
        self.assertEqual(round(fluid["s"], 7), 10.1749996)
        self.assertEqual(round(fluid["cp"], 8), 2.08141274)
        self.assertEqual(round(fluid["w"], 6), 644.289068)

        fluid = _Region2(700, 30)
        self.assertEqual(round(fluid["v"], 11), 0.00542946619)
        self.assertEqual(round(fluid["h"], 5), 2631.49474)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 2468.61076)
        self.assertEqual(round(fluid["s"], 8), 5.17540298)
        self.assertEqual(round(fluid["cp"], 7), 10.3505092)
        self.assertEqual(round(fluid["w"], 6), 480.386523)

        # Backward2_T_Ph Table 24 pag 25
        self.assertEqual(round(_Backward2_T_Ph(0.001, 3000), 6), 534.433241)
        self.assertEqual(round(_Backward2_T_Ph(3, 3000), 6), 575.373370)
        self.assertEqual(round(_Backward2_T_Ph(3, 4000), 5), 1010.77577)
        self.assertEqual(round(_Backward2_T_Ph(5, 3500), 6), 801.299102)
        self.assertEqual(round(_Backward2_T_Ph(5, 4000), 5), 1015.31583)
        self.assertEqual(round(_Backward2_T_Ph(25, 3500), 6), 875.279054)
        self.assertEqual(round(_Backward2_T_Ph(40, 2700), 6), 743.056411)
        self.assertEqual(round(_Backward2_T_Ph(60, 2700), 6), 791.137067)
        self.assertEqual(round(_Backward2_T_Ph(60, 3200), 6), 882.756860)

        # _Backward2_T_Ps Table 9 pag 12
        self.assertEqual(round(_Backward2_T_Ps(0.1, 7.5), 6), 399.517097)
        self.assertEqual(round(_Backward2_T_Ps(0.1, 8), 6), 514.127081)
        self.assertEqual(round(_Backward2_T_Ps(2.5, 8), 5), 1039.84917)
        self.assertEqual(round(_Backward2_T_Ps(8, 6), 6), 600.484040)
        self.assertEqual(round(_Backward2_T_Ps(8, 7.5), 5), 1064.95556)
        self.assertEqual(round(_Backward2_T_Ps(90, 6), 5), 1038.01126)
        self.assertEqual(round(_Backward2_T_Ps(20, 5.75), 6), 697.992849)
        self.assertEqual(round(_Backward2_T_Ps(80, 5.25), 6), 854.011484)
        self.assertEqual(round(_Backward2_T_Ps(80, 5.75), 6), 949.017998)

        # _Backward2_P_hs Table 9 pag 10 for supplementary p(h,s)
        self.assertEqual(round(_Backward2_P_hs(2800, 6.5), 9), 1.371012767)
        self.assertEqual(round(_Backward2_P_hs(2800, 9.5), 12), 0.001879743844)
        self.assertEqual(round(_Backward2_P_hs(4100, 9.5), 10), 0.1024788997)
        self.assertEqual(round(_Backward2_P_hs(2800, 6), 9), 4.793911442)
        self.assertEqual(round(_Backward2_P_hs(3600, 6), 8), 83.95519209)
        self.assertEqual(round(_Backward2_P_hs(3600, 7), 9), 7.527161441)
        self.assertEqual(round(_Backward2_P_hs(2800, 5.1), 8), 94.39202060)
        self.assertEqual(round(_Backward2_P_hs(2800, 5.8), 9), 8.414574124)
        self.assertEqual(round(_Backward2_P_hs(3400, 5.8), 8), 83.76903879)

    def test_IAPWS97_3(self) -> None:
        """Table 33, pag 49"""
        fluid = _Region3(500, 650)
        self.assertEqual(round(fluid["P"], 7), 25.5837018)
        self.assertEqual(round(fluid["h"], 5), 1863.43019)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 1812.26279)
        self.assertEqual(round(fluid["s"], 8), 4.05427273)
        self.assertEqual(round(fluid["cp"], 7), 13.8935717)
        self.assertEqual(round(fluid["w"], 6), 502.005554)

        fluid = _Region3(200, 650)
        self.assertEqual(round(fluid["P"], 7), 22.2930643)
        self.assertEqual(round(fluid["h"], 5), 2375.12401)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 2263.65868)
        self.assertEqual(round(fluid["s"], 8), 4.85438792)
        self.assertEqual(round(fluid["cp"], 7), 44.6579342)
        self.assertEqual(round(fluid["w"], 6), 383.444594)

        fluid = _Region3(500, 750)
        self.assertEqual(round(fluid["P"], 7), 78.3095639)
        self.assertEqual(round(fluid["h"], 5), 2258.68845)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 2102.06932)
        self.assertEqual(round(fluid["s"], 8), 4.46971906)
        self.assertEqual(round(fluid["cp"], 8), 6.34165359)
        self.assertEqual(round(fluid["w"], 6), 760.696041)

        # _h_3ab   pag 7
        self.assertEqual(round(_h_3ab(25), 6), 2095.936454)

    def test_IAPWS97_3_Sup03(self) -> None:
        """Test for supplementary 03 for region 3"""
        # _Backward3_T_Ph Table 5 pag 8
        self.assertEqual(round(_Backward3_T_Ph(20, 1700), 7), 629.3083892)
        self.assertEqual(round(_Backward3_T_Ph(50, 2000), 7), 690.5718338)
        self.assertEqual(round(_Backward3_T_Ph(100, 2100), 7), 733.6163014)
        self.assertEqual(round(_Backward3_T_Ph(20, 2500), 7), 641.8418053)
        self.assertEqual(round(_Backward3_T_Ph(50, 2400), 7), 735.1848618)
        self.assertEqual(round(_Backward3_T_Ph(100, 2700), 7), 842.0460876)

        # _Backward3_v_Ph Table 8 pag 10
        self.assertEqual(round(_Backward3_v_Ph(20, 1700), 12), 1.749903962e-3)
        self.assertEqual(round(_Backward3_v_Ph(50, 2000), 12), 1.908139035e-3)
        self.assertEqual(round(_Backward3_v_Ph(100, 2100), 12), 1.676229776e-3)
        self.assertEqual(round(_Backward3_v_Ph(20, 2500), 12), 6.670547043e-3)
        self.assertEqual(round(_Backward3_v_Ph(50, 2400), 12), 2.801244590e-3)
        self.assertEqual(round(_Backward3_v_Ph(100, 2700), 12), 2.404234998e-3)

        # _Backward3_T_Ps Table 12 pag 13
        self.assertEqual(round(_Backward3_T_Ps(20, 3.8), 7), 628.2959869)
        self.assertEqual(round(_Backward3_T_Ps(50, 3.6), 7), 629.7158726)
        self.assertEqual(round(_Backward3_T_Ps(100, 4.0), 7), 705.6880237)
        self.assertEqual(round(_Backward3_T_Ps(20, 5.0), 7), 640.1176443)
        self.assertEqual(round(_Backward3_T_Ps(50, 4.5), 7), 716.3687517)
        self.assertEqual(round(_Backward3_T_Ps(100, 5.0), 7), 847.4332825)

        # _Backward3_v_Ps Table 15 pag 15
        self.assertEqual(round(_Backward3_v_Ps(20, 3.8), 12), 1.733791463e-3)
        self.assertEqual(round(_Backward3_v_Ps(50, 3.6), 12), 1.469680170e-3)
        self.assertEqual(round(_Backward3_v_Ps(100, 4.0), 12), 1.555893131e-3)
        self.assertEqual(round(_Backward3_v_Ps(20, 5.0), 12), 6.262101987e-3)
        self.assertEqual(round(_Backward3_v_Ps(50, 4.5), 12), 2.332634294e-3)
        self.assertEqual(round(_Backward3_v_Ps(100, 5.0), 12), 2.449610757e-3)

        # _PSat_h Table 18 pag 18
        self.assertRaises(NotImplementedError, _PSat_h, 2.6)
        self.assertEqual(round(_PSat_h(1700), 8), 17.24175718)
        self.assertEqual(round(_PSat_h(2000), 8), 21.93442957)
        self.assertEqual(round(_PSat_h(2400), 8), 20.18090839)

        # _PSat_s Table 20 pag 19
        self.assertRaises(NotImplementedError, _PSat_s, 3.6)
        self.assertEqual(round(_PSat_s(3.8), 8), 16.87755057)
        self.assertEqual(round(_PSat_s(4.2), 8), 21.64451789)
        self.assertEqual(round(_PSat_s(5.2), 8), 16.68968482)

    def test_IAPWS97_3_Sup04(self) -> None:
        """Test for supplementary 04 for region 3"""
        # _Backward3_P_hs Table 5 pag 10
        self.assertEqual(round(_Backward3_P_hs(1700, 3.8), 8), 25.55703246)
        self.assertEqual(round(_Backward3_P_hs(2000, 4.2), 8), 45.40873468)
        self.assertEqual(round(_Backward3_P_hs(2100, 4.3), 8), 60.78123340)
        self.assertEqual(round(_Backward3_P_hs(2600, 5.1), 8), 34.34999263)
        self.assertEqual(round(_Backward3_P_hs(2400, 4.7), 8), 63.63924887)
        self.assertEqual(round(_Backward3_P_hs(2700, 5.0), 8), 88.39043281)

        # _h1_s _h3a_s Table 11 pag 17
        self.assertRaises(NotImplementedError, _h1_s, 4)
        self.assertEqual(round(_h1_s(1), 7), 308.5509647)
        self.assertEqual(round(_h1_s(2), 7), 700.6304472)
        self.assertEqual(round(_h1_s(3), 6), 1198.359754)
        self.assertRaises(NotImplementedError, _h3a_s, 4.5)
        self.assertEqual(round(_h3a_s(3.8), 6), 1685.025565)
        self.assertEqual(round(_h3a_s(4), 6), 1816.891476)
        self.assertEqual(round(_h3a_s(4.2), 6), 1949.352563)

        # _h2ab_s _h2c3b_s Table 18 pag 21
        self.assertRaises(NotImplementedError, _h2ab_s, 5)
        self.assertEqual(round(_h2ab_s(7), 6), 2723.729985)
        self.assertEqual(round(_h2ab_s(8), 6), 2599.047210)
        self.assertEqual(round(_h2ab_s(9), 6), 2511.861477)
        self.assertRaises(NotImplementedError, _h2c3b_s, 6)
        self.assertEqual(round(_h2c3b_s(5.5), 6), 2687.693850)
        self.assertEqual(round(_h2c3b_s(5.0), 6), 2451.623609)
        self.assertEqual(round(_h2c3b_s(4.5), 6), 2144.360448)

        # _h13_s Table 18 pag 21
        self.assertRaises(NotImplementedError, _h13_s, 3.3)
        self.assertEqual(round(_h13_s(3.7), 6), 1632.525047)
        self.assertEqual(round(_h13_s(3.6), 6), 1593.027214)
        self.assertEqual(round(_h13_s(3.5), 6), 1566.104611)

        # _t_hs Table 26 pag 26
        self.assertRaises(NotImplementedError, _t_hs, *(2600, 5))
        self.assertEqual(round(_t_hs(2600, 5.1), 7), 713.5259364)
        self.assertEqual(round(_t_hs(2700, 5.15), 7), 768.5345532)
        self.assertEqual(round(_t_hs(2800, 5.2), 7), 817.6202120)

        # _Backward4_T_hs Table 29 pag 31
        self.assertEqual(round(_Backward4_T_hs(1800, 5.3), 7), 346.8475498)
        self.assertEqual(round(_Backward4_T_hs(2400, 6.0), 7), 425.1373305)
        self.assertEqual(round(_Backward4_T_hs(2500, 5.5), 7), 522.5579013)

    def test_IAPWS97_3_Sup05(self) -> None:
        """Test for supplementary 05 for region 3 v=f(T,P)"""
        # T=f(P) limit Table 3 pag 11
        self.assertEqual(round(_tab_P(40), 7), 693.0341408)
        self.assertEqual(round(_txx_P(25, "cd"), 7), 649.3659208)
        self.assertEqual(round(_tef_P(40), 7), 713.9593992)
        self.assertEqual(round(_txx_P(23, "gh"), 7), 649.8873759)
        self.assertEqual(round(_txx_P(23, "ij"), 7), 651.5778091)
        self.assertEqual(round(_txx_P(23, "jk"), 7), 655.8338344)
        self.assertEqual(round(_txx_P(22.8, "mn"), 7), 649.6054133)
        self.assertEqual(round(_top_P(22.8), 7), 650.0106943)
        self.assertEqual(round(_txx_P(22, "qu"), 7), 645.6355027)
        self.assertEqual(round(_txx_P(22, "rx"), 7), 648.2622754)

        # _Backward3_v_PT Table 5 pag 13
        self.assertEqual(round(_Backward3_v_PT(50, 630), 12), 1.470853100e-3)
        self.assertEqual(round(_Backward3_v_PT(80, 670), 12), 1.503831359e-3)
        self.assertEqual(round(_Backward3_v_PT(50, 710), 12), 2.204728587e-3)
        self.assertEqual(round(_Backward3_v_PT(80, 750), 12), 1.973692940e-3)
        self.assertEqual(round(_Backward3_v_PT(20, 630), 12), 1.761696406e-3)
        self.assertEqual(round(_Backward3_v_PT(30, 650), 12), 1.819560617e-3)
        self.assertEqual(round(_Backward3_v_PT(26, 656), 12), 2.245587720e-3)
        self.assertEqual(round(_Backward3_v_PT(30, 670), 12), 2.506897702e-3)
        self.assertEqual(round(_Backward3_v_PT(26, 661), 12), 2.970225962e-3)
        self.assertEqual(round(_Backward3_v_PT(30, 675), 12), 3.004627086e-3)
        self.assertEqual(round(_Backward3_v_PT(26, 671), 12), 5.019029401e-3)
        self.assertEqual(round(_Backward3_v_PT(30, 690), 12), 4.656470142e-3)
        self.assertEqual(round(_Backward3_v_PT(23.6, 649), 12), 2.163198378e-3)
        self.assertEqual(round(_Backward3_v_PT(24, 650), 12), 2.166044161e-3)
        self.assertEqual(round(_Backward3_v_PT(23.6, 652), 12), 2.651081407e-3)
        self.assertEqual(round(_Backward3_v_PT(24, 654), 12), 2.967802335e-3)
        self.assertEqual(round(_Backward3_v_PT(23.6, 653), 12), 3.273916816e-3)
        self.assertEqual(round(_Backward3_v_PT(24, 655), 12), 3.550329864e-3)
        self.assertEqual(round(_Backward3_v_PT(23.5, 655), 12), 4.545001142e-3)
        self.assertEqual(round(_Backward3_v_PT(24, 660), 12), 5.100267704e-3)
        self.assertEqual(round(_Backward3_v_PT(23, 660), 12), 6.109525997e-3)
        self.assertEqual(round(_Backward3_v_PT(24, 670), 12), 6.427325645e-3)
        self.assertEqual(round(_Backward3_v_PT(22.6, 646), 12), 2.117860851e-3)
        self.assertEqual(round(_Backward3_v_PT(23, 646), 12), 2.062374674e-3)
        self.assertEqual(round(_Backward3_v_PT(22.6, 648.6), 12), 2.533063780e-3)
        self.assertEqual(round(_Backward3_v_PT(22.8, 649.3), 12), 2.572971781e-3)
        self.assertEqual(round(_Backward3_v_PT(22.6, 649.0), 12), 2.923432711e-3)
        self.assertEqual(round(_Backward3_v_PT(22.8, 649.7), 12), 2.913311494e-3)
        self.assertEqual(round(_Backward3_v_PT(22.6, 649.1), 12), 3.131208996e-3)
        self.assertEqual(round(_Backward3_v_PT(22.8, 649.9), 12), 3.221160278e-3)
        self.assertEqual(round(_Backward3_v_PT(22.6, 649.4), 12), 3.715596186e-3)
        self.assertEqual(round(_Backward3_v_PT(22.8, 650.2), 12), 3.664754790e-3)
        self.assertEqual(round(_Backward3_v_PT(21.1, 640), 12), 1.970999272e-3)
        self.assertEqual(round(_Backward3_v_PT(21.8, 643), 12), 2.043919161e-3)
        self.assertEqual(round(_Backward3_v_PT(21.1, 644), 12), 5.251009921e-3)
        self.assertEqual(round(_Backward3_v_PT(21.8, 648), 12), 5.256844741e-3)
        self.assertEqual(round(_Backward3_v_PT(19.1, 635), 12), 1.932829079e-3)
        self.assertEqual(round(_Backward3_v_PT(20, 638), 12), 1.985387227e-3)
        self.assertEqual(round(_Backward3_v_PT(17, 626), 12), 8.483262001e-3)
        self.assertEqual(round(_Backward3_v_PT(20, 640), 12), 6.227528101e-3)

        # T=f(P) limit Table 11 pag 19
        self.assertEqual(round(_txx_P(22.3, "uv"), 7), 647.7996121)
        self.assertEqual(round(_twx_P(22.3), 7), 648.2049480)

        # _Backward3_v_PT Table 13 pag 20
        self.assertEqual(round(_Backward3_v_PT(21.5, 644.6), 12), 2.268366647e-3)
        self.assertEqual(round(_Backward3_v_PT(22, 646.1), 12), 2.296350553e-3)
        self.assertEqual(round(_Backward3_v_PT(22.5, 648.6), 12), 2.832373260e-3)
        self.assertEqual(round(_Backward3_v_PT(22.3, 647.9), 12), 2.811424405e-3)
        self.assertEqual(round(_Backward3_v_PT(22.15, 647.5), 12), 3.694032281e-3)
        self.assertEqual(round(_Backward3_v_PT(22.3, 648.1), 12), 3.622226305e-3)
        self.assertEqual(round(_Backward3_v_PT(22.11, 648), 12), 4.528072649e-3)
        self.assertEqual(round(_Backward3_v_PT(22.3, 649), 12), 4.556905799e-3)
        self.assertEqual(round(_Backward3_v_PT(22, 646.84), 12), 2.698354719e-3)
        self.assertEqual(round(_Backward3_v_PT(22.064, 647.05), 12), 2.717655648e-3)
        self.assertEqual(round(_Backward3_v_PT(22, 646.89), 12), 3.798732962e-3)
        self.assertEqual(round(_Backward3_v_PT(22.064, 647.15), 11), 3.701940010e-3)

    def test_IAPWS97_4(self) -> None:
        """Saturation line"""
        # _PSat_T Table 35 pag 34
        self.assertRaises(NotImplementedError, _PSat_T, 270)
        self.assertEqual(round(_PSat_T(300), 11), 0.00353658941)
        self.assertEqual(round(_PSat_T(500), 8), 2.63889776)
        self.assertEqual(round(_PSat_T(600), 7), 12.3443146)

        # _TSat_P Table 36 pag 36
        self.assertRaises(NotImplementedError, _TSat_P, 30)
        self.assertEqual(round(_TSat_P(0.1), 6), 372.755919)
        self.assertEqual(round(_TSat_P(1), 6), 453.035632)
        self.assertEqual(round(_TSat_P(10), 6), 584.149488)

    def test_IAPWS97_5(self) -> None:
        """Table 42, pag 40"""
        fluid = _Region5(1500, 0.5)
        self.assertEqual(round(fluid["v"], 8), 1.38455090)
        self.assertEqual(round(fluid["h"], 5), 5219.76855)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 4527.49310)
        self.assertEqual(round(fluid["s"], 8), 9.65408875)
        self.assertEqual(round(fluid["cp"], 8), 2.61609445)
        self.assertEqual(round(fluid["w"], 6), 917.068690)

        fluid = _Region5(1500, 30)
        self.assertEqual(round(fluid["v"], 10), 0.0230761299)
        self.assertEqual(round(fluid["h"], 5), 5167.23514)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 4474.95124)
        self.assertEqual(round(fluid["s"], 8), 7.72970133)
        self.assertEqual(round(fluid["cp"], 8), 2.72724317)
        self.assertEqual(round(fluid["w"], 6), 928.548002)

        fluid = _Region5(2000, 30)
        self.assertEqual(round(fluid["v"], 10), 0.0311385219)
        self.assertEqual(round(fluid["h"], 5), 6571.22604)
        self.assertEqual(round(fluid["h"]-fluid["P"]*1000*fluid["v"], 5), 5637.07038)
        self.assertEqual(round(fluid["s"], 8), 8.53640523)
        self.assertEqual(round(fluid["cp"], 8), 2.88569882)
        self.assertEqual(round(fluid["w"], 5), 1067.36948)

    def test_IAPWS97_custom(self) -> None:
        """Cycle input parameter from selected point for IAPWS97"""
        # Region 1
        P = 50.0  # MPa
        T = 470   # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20.0  # MPa
        T = 370   # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Region 2
        P = 25.0  # MPa
        T = 700   # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 10.0  # MPa
        T = 700   # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20   # MPa
        T = 800  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 0.01  # MPa
        T = 1000  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 2.0   # MPa
        T = 1000  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Region 3
        P = 50.0  # MPa
        T = 700   # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20.0  # MPa
        s = 4     # kJ/kgK
        f_ps = IAPWS97(P=P, s=s)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_pt = IAPWS97(P=f_ph.P, T=f_ph.T)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.s-s, 6), 0)

        P = 19.0  # MPa
        f_px = IAPWS97(P=P, x=0)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 3), 0)
        self.assertEqual(round(f_tx.x, 6), 0)
        P = 19.0  # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 3), 0)
        self.assertEqual(round(f_tx.x, 6), 1)

        P = 21.0  # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 1)

        P = 21.5  # MPa
        f_px = IAPWS97(P=P, x=0)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 0)
        P = 21.5  # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 1)

        P = 22.02  # MPa
        f_px = IAPWS97(P=P, x=0)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 0)
        P = 22.02   # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 1)

        P = 24.0  # MPa
        T = 630   # K
        f_pt = IAPWS97(P=P, T=T)
        f_hs = IAPWS97(h=f_pt.h, s=f_pt.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Boundary 1-4
        T = 340  # K
        f_tx = IAPWS97(T=T, x=0)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_px = IAPWS97(P=f_ps.P, x=f_ps.x)
        f_hs = IAPWS97(h=f_px.h, s=f_px.s)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        T = 620  # K
        f_tx = IAPWS97(T=T, x=0)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_px = IAPWS97(P=f_ps.P, x=f_ps.x)
        f_hs = IAPWS97(h=f_px.h, s=f_px.s)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Boundary 2-4
        T = 340  # K
        f_tx = IAPWS97(T=T, x=1)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_px = IAPWS97(P=f_ps.P, x=f_ps.x)
        f_hs = IAPWS97(h=f_px.h, s=f_px.s)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        T = 340  # K
        f_tx = IAPWS97(T=T, x=0)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_px = IAPWS97(P=f_ps.P, x=f_ps.x)
        f_hs = IAPWS97(h=f_px.h, s=f_px.s)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Boundary 3-4
        T = 640  # K
        f_tx = IAPWS97(T=T, x=0)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        f_px = IAPWS97(P=f_hs.P, x=f_hs.x)
        self.assertEqual(round(f_px.T-T, 4), 0)

        # Region 4
        T = 325  # K
        f_tx = IAPWS97(T=T, x=0.5)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_px = IAPWS97(P=f_ps.P, x=f_ps.x)
        f_hs = IAPWS97(h=f_px.h, s=f_px.s)
        self.assertEqual(round(f_hs.T-T, 2), 0)

        T = 640  # K
        f_tx = IAPWS97(T=T, x=0.5)
        f_ph = IAPWS97(h=f_tx.h, P=f_tx.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_px = IAPWS97(P=f_ps.P, x=f_ps.x)
        f_hs = IAPWS97(h=f_px.h, s=f_px.s)
        self.assertEqual(round(f_hs.T-T, 0), 0)

        P = 17.0  # MPa
        h = 2000  # kJkg
        f_ph = IAPWS97(P=P, h=h)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        self.assertEqual(round(f_ph.P-P, 4), 0)
        self.assertEqual(round(f_ph.h-h, 4), 0)

        T = 274  # K
        f_tx = IAPWS97(x=.01, T=T)
        f_hs = IAPWS97(h=f_tx.h, s=f_tx.s)
        self.assertEqual(round(f_hs.x-0.01, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        T = 274  # K
        f_tx = IAPWS97(x=.99, T=T)
        f_hs = IAPWS97(h=f_tx.h, s=f_tx.s)
        self.assertEqual(round(f_hs.x-0.99, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Region 5
        P = 25.0  # MPa
        T = 1100  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 10.0  # MPa
        T = 1100  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20.0  # MPa
        T = 1100  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # P-T subregion 3
        Plist = [17.0, 21.0, 21.0, 21.0, 21.0, 22.0, 23.2, 23.2, 23.2, 23.2,
                 23.2, 23.2, 23.2, 23.0, 23.0, 22.2, 22.065, 22.065, 22.065,
                 22.065, 21.8, 22.0]
        Tlist = [625.0, 625.0, 640.0, 643.0, 645.0, 630.0, 640.0, 650.0, 651.0,
                 652.0, 653.0, 656.0, 660.0, 640.0, 652.0, 647.0, 646.0, 647.05,
                 647.1, 647.2, 647.0, 647.0]
        for p, t in zip(Plist, Tlist):
            f_pt = IAPWS97(P=float(p), T=t)
            f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
            f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
            f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
            self.assertEqual(round(f_hs.P-p, 6), 0)
            self.assertEqual(round(f_hs.T-t, 6), 0)

        # Other h-s region
        hlist = [2700, 2700, 1500, 2500, 2000, 2000, 3000, 2400, 2500, 2850, 2600]
        slist = [5.15, 5.87, 3.5, 5, 5.5, 7, 6, 5.1, 5.05, 5.25, 5.25]
        for H, S in zip(hlist, slist):
            f_hs = IAPWS97(h=H, s=S)
            f_pt = IAPWS97(P=f_hs.P, T=f_hs.T)
            self.assertEqual(round(f_hs.h-H, 6), 0)
            self.assertEqual(round(f_hs.s-S, 6), 0)

        # Critical point
        st = IAPWS97(T=647.096, x=0.9)
        st = IAPWS97(P=22.064, x=0.9)
        st = IAPWS97(T=647.096, P=22.064)
        st2 = IAPWS97(h=st.h, s=st.s)
        self.assertEqual(round(st2.T-st.T, 6), 0)
        self.assertEqual(round(st2.P-st.P, 6), 0)

        # Derived classes
        st = IAPWS97(T=300, x=0.9)
        st2 = IAPWS97_Tx(st.T, st.x)
        st3 = IAPWS97_Px(st2.P, st2.x)
        st4 = IAPWS97_Ps(st3.P, st3.s)
        st5 = IAPWS97_Ph(st4.P, st4.h)
        st6 = IAPWS97_PT(st5.P, st5.T)
        self.assertEqual(round(st6.T-300, 6), 0)

        # Exceptions
        self.assertRaises(NotImplementedError, IAPWS97, **{"T": 300, "x": 1.5})
        self.assertRaises(NotImplementedError, IAPWS97, **{"P": 1, "x": 1.5})
        self.assertRaises(NotImplementedError, IAPWS97, **{"P": 10, "T": 270})
        self.assertRaises(NotImplementedError, IAPWS97, **{"P": 105, "h": 400})
        self.assertRaises(NotImplementedError, IAPWS97, **{"P": 65, "s": 9})
        self.assertRaises(NotImplementedError, IAPWS97, **{"h": 700, "s": -1})

    def test_IAPWS95_custom1(self) -> None:
        """Cycle input parameter from selected point for IAPWS95"""
        P = 50.0  # MPa
        T = 470   # K
        f_pt = IAPWS95_PT(P, T)
        f_ph = IAPWS95_Ph(f_pt.P, f_pt.h)
        f_ps = IAPWS95_Ps(f_ph.P, f_ph.s)
        f_hs = IAPWS95(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 5), 0)
        self.assertEqual(round(f_hs.T-T, 5), 0)

    def test_IAPWS95_custom2(self) -> None:
        """Cycle input parameter from selected point for IAPWS95"""
        P = 2.0  # MPa
        f_px = IAPWS95_Px(P, 0.5)
        f_tx = IAPWS95_Tx(f_px.T, f_px.x)
        f_tv = IAPWS95(T=f_px.T, v=f_px.v)
        f_th = IAPWS95(T=f_tv.T, h=f_tv.h)
        f_ts = IAPWS95(T=f_th.T, s=f_th.s)
        f_tu = IAPWS95(T=f_ts.T, u=f_ts.u)
        f_ph = IAPWS95(P=f_tu.P, h=f_tu.h)
        f_ps = IAPWS95(P=f_ph.P, s=f_ph.s)
        f_pu = IAPWS95(P=f_ps.P, u=f_ps.u)
        f_hs = IAPWS95(h=f_pu.h, s=f_pu.s)
        f_hu = IAPWS95(h=f_hs.h, u=f_hs.u, T0=f_px.T, rho0=f_px.rho)
        f_su = IAPWS95(s=f_hu.s, u=f_hu.u)
        f_rhoh = IAPWS95(rho=f_su.rho, h=f_su.h)
        f_rhos = IAPWS95(rho=f_rhoh.rho, s=f_rhoh.s)
        f_rhou = IAPWS95(rho=f_rhos.rho, u=f_rhos.u)
        f_Prho = IAPWS95(rho=f_rhou.rho, P=f_rhou.P)
        self.assertEqual(round(f_Prho.P-P, 5), 0)
        self.assertEqual(round(f_Prho.x-0.5, 5), 0)

        P = 50.0  # MPa
        T = 770   # K
        f_pt = IAPWS95_PT(P, T)
        f_tv = IAPWS95(T=f_pt.T, v=f_pt.v)
        f_th = IAPWS95(T=f_tv.T, h=f_tv.h)
        f_ts = IAPWS95(T=f_th.T, s=f_th.s)
        f_tu = IAPWS95(T=f_ts.T, u=f_ts.u)
        f_ph = IAPWS95(P=f_tu.P, h=f_tu.h)
        f_ps = IAPWS95(P=f_ph.P, s=f_ph.s)
        f_pu = IAPWS95(P=f_ps.P, u=f_ps.u)
        f_hs = IAPWS95(h=f_pu.h, s=f_pu.s)
        f_hu = IAPWS95(h=f_hs.h, u=f_hs.u, T0=T, rho0=f_hs.rho)
        f_su = IAPWS95(s=f_hu.s, u=f_hu.u, T0=T, rho0=f_hu.rho)
        f_rhoh = IAPWS95(rho=f_su.rho, h=f_su.h)
        f_rhos = IAPWS95(rho=f_rhoh.rho, s=f_rhoh.s)
        f_rhou = IAPWS95(rho=f_rhos.rho, u=f_rhos.u)
        f_Prho = IAPWS95(rho=f_rhou.rho, P=f_rhou.P)
        self.assertEqual(round(f_Prho.P-P, 5), 0)
        self.assertEqual(round(f_Prho.T-T, 5), 0)

        P = 0.1  # MPa
        T = 300  # K
        f_pt = IAPWS95_PT(P, T)
        f_tv = IAPWS95(T=f_pt.T, v=f_pt.v)
        f_th = IAPWS95(T=f_tv.T, h=f_tv.h, x0=0)
        f_ts = IAPWS95(T=f_th.T, s=f_th.s)
        f_tu = IAPWS95(T=f_ts.T, u=f_ts.u)
        f_ph = IAPWS95(P=f_tu.P, h=f_tu.h)
        f_ps = IAPWS95(P=f_ph.P, s=f_ph.s)
        f_pu = IAPWS95(P=f_ps.P, u=f_ps.u)
        f_hs = IAPWS95(h=f_pu.h, s=f_pu.s)
        f_hu = IAPWS95(h=f_hs.h, u=f_hs.u)
        f_su = IAPWS95(s=f_hu.s, u=f_hu.u, T0=T, rho0=f_hu.rho)
        f_rhoh = IAPWS95(rho=f_su.rho, h=f_su.h)
        f_rhos = IAPWS95(rho=f_rhoh.rho, s=f_rhoh.s)
        f_rhou = IAPWS95(rho=f_rhos.rho, u=f_rhos.u)
        f_Prho = IAPWS95(rho=f_rhou.rho, P=f_rhou.P)
        self.assertEqual(round(f_Prho.P-P, 5), 0)
        self.assertEqual(round(f_Prho.T-T, 5), 0)

        P = 0.1  # MPa
        T = 500  # K
        f_pt = IAPWS95_PT(P, T)
        f_tv = IAPWS95(T=f_pt.T, v=f_pt.v)
        f_th = IAPWS95(T=f_tv.T, h=f_tv.h, x0=1)
        f_ts = IAPWS95(T=f_th.T, s=f_th.s)
        f_tu = IAPWS95(T=f_ts.T, u=f_ts.u)
        f_ph = IAPWS95(P=f_tu.P, h=f_tu.h)
        f_ps = IAPWS95(P=f_ph.P, s=f_ph.s)
        f_hu = IAPWS95(P=f_ps.P, u=f_ps.u, T0=T, rho0=f_pt.rho)
        f_su = IAPWS95(s=f_hu.s, u=f_hu.u, T0=T, rho0=f_hu.rho)
        f_rhoh = IAPWS95(rho=f_su.rho, h=f_su.h)
        f_rhos = IAPWS95(rho=f_rhoh.rho, s=f_rhoh.s)
        f_rhou = IAPWS95(rho=f_rhos.rho, u=f_rhos.u)
        f_Prho = IAPWS95(rho=f_rhou.rho, P=f_rhou.P)
        self.assertEqual(round(f_Prho.P-P, 5), 0)
        self.assertEqual(round(f_Prho.T-T, 5), 0)

        P = 2.0  # MPa
        f_px = IAPWS95_Px(P, 0)
        f_tx = IAPWS95_Tx(f_px.T, f_px.x)
        self.assertEqual(round(f_tx.P-P, 5), 0)
        f_px = IAPWS95_Px(P, 1)
        f_tx = IAPWS95_Tx(f_px.T, f_px.x)
        self.assertEqual(round(f_tx.P-P, 5), 0)

        P = 0.1  # MPa
        T = 300  # K
        d2o_pt = D2O(P=P, T=T)
        self.assertEqual(round(d2o_pt.P-P, 5), 0)
        self.assertEqual(round(d2o_pt.T-T, 5), 0)

        self.assertRaises(NotImplementedError, IAPWS95, **{"T": 700, "x": 0})
        self.assertRaises(NotImplementedError, IAPWS95, **{"P": 25, "x": 1})

    def test_D2O(self) -> None:
        """Tables 6-8, page 12-13."""
        # Table 6, pag 12"""
        fluid = D2O()

        delta = 46.26*fluid.M/fluid.rhoc
        tau = fluid.Tc/500
        ideal = fluid._phi0(tau, delta)
        self.assertEqual(round(ideal["fio"], 8), 1.96352717)
        self.assertEqual(round(ideal["fiod"], 9), 0.384253134)
        self.assertEqual(round(ideal["fiodd"], 9), -0.147650471)
        self.assertEqual(round(ideal["fiot"], 8), 9.39259413)
        self.assertEqual(round(ideal["fiott"], 8), -2.09517144)
        self.assertEqual(round(ideal["fiodt"], 8), 0)
        res = fluid._phir(tau, delta)
        self.assertEqual(round(res["fir"], 8), -3.42291092)
        self.assertEqual(round(res["fird"], 9), -0.367562780)
        self.assertEqual(round(res["firdd"], 9), 0.835183806)
        self.assertEqual(round(res["firt"], 8), -5.89707436)
        self.assertEqual(round(res["firtt"], 8), -2.45187285)
        self.assertEqual(round(res["firdt"], 8), -1.13178440)

        # Table 7, Pag 12, Single phase region
        st = D2O(T=300, rhom=55.126)
        self.assertEqual(round(st.P, 10), 0.0529123711)
        self.assertEqual(round(st.cvM, 7), 83.3839128)
        self.assertEqual(round(st.w, 5), 1403.74625)
        self.assertEqual(round(st.sM, 8), 6.73910582)

        st = D2O(T=300, rhom=60)
        self.assertEqual(round(st.P, 6), 238.222326)
        self.assertEqual(round(st.cvM, 7), 73.8561038)
        self.assertEqual(round(st.w, 5), 1772.79674)
        self.assertEqual(round(st.sM, 8), 5.40117148)

        st = D2O(T=300, rhom=65)
        self.assertEqual(round(st.P, 6), 626.176781)
        self.assertEqual(round(st.cvM, 7), 69.9125978)
        self.assertEqual(round(st.w, 5), 2296.97942)
        self.assertEqual(round(st.sM, 8), 2.71566150)

        st = D2O(T=500, rhom=0.05)
        self.assertEqual(round(st.P, 9), 0.206052588)
        self.assertEqual(round(st.cvM, 7), 29.4298102)
        self.assertEqual(round(st.w, 6), 514.480413)
        self.assertEqual(round(st.sM, 6), 140.879085)

        st = D2O(T=500, rhom=0.5)
        self.assertEqual(round(st.P, 8), 1.88967446)
        self.assertEqual(round(st.cvM, 7), 36.6460545)
        self.assertEqual(round(st.w, 6), 489.633254)
        self.assertEqual(round(st.sM, 6), 120.227024)

        st = D2O(T=500, rhom=46.26)
        self.assertEqual(round(st.P, 8), 8.35329492)
        self.assertEqual(round(st.cvM, 7), 62.6885994)
        self.assertEqual(round(st.w, 5), 1178.88631)
        self.assertEqual(round(st.sM, 7), 49.5587000)

        st = D2O(T=500, rhom=50)
        self.assertEqual(round(st.P, 6), 107.462884)
        self.assertEqual(round(st.cvM, 7), 61.7372286)
        self.assertEqual(round(st.w, 5), 1483.74868)
        self.assertEqual(round(st.sM, 7), 46.9453826)

        st = D2O(T=500, rhom=60)
        self.assertEqual(round(st.P, 6), 721.798322)
        self.assertEqual(round(st.cvM, 7), 57.6860681)
        self.assertEqual(round(st.w, 5), 2413.93520)
        self.assertEqual(round(st.sM, 7), 39.3599094)

        st = D2O(T=643.8, rhom=20)
        self.assertEqual(round(st.P, 7), 21.6503820)
        self.assertEqual(round(st.cvM, 7), 99.2661842)
        self.assertEqual(round(st.w, 6), 256.043612)
        self.assertEqual(round(st.sM, 7), 81.7656125)

        st = D2O(T=800, rhom=0.01)
        self.assertEqual(round(st.P, 10), 0.0664864175)
        self.assertEqual(round(st.cvM, 7), 34.0033604)
        self.assertEqual(round(st.w, 6), 642.794634)
        self.assertEqual(round(st.sM, 6), 169.067586)

        st = D2O(T=800, rhom=0.25)
        self.assertEqual(round(st.P, 8), 1.64466177)
        self.assertEqual(round(st.cvM, 7), 34.4327932)
        self.assertEqual(round(st.w, 6), 639.281410)
        self.assertEqual(round(st.sM, 6), 142.125615)

        # Table 8, Pag 13, Saturation state
        st = D2O(T=280, x=0.5)
        self.assertEqual(round(st.P, 12), 0.000823054058)
        self.assertEqual(round(st.Liquid.rhoM, 7), 55.2072786)
        self.assertEqual(round(st.Gas.rhoM, 12), 0.000353747143)
        self.assertEqual(round(st.Liquid.hM, 6), 257.444444)
        self.assertEqual(round(st.Gas.hM, 4), 46610.6716)
        self.assertEqual(round(st.Liquid.sM, 9), 0.924406091)
        self.assertEqual(round(st.Gas.sM, 6), 166.471646)

        st = D2O(T=450, x=0.5)
        self.assertEqual(round(st.P, 9), 0.921212105)
        self.assertEqual(round(st.Liquid.rhoM, 7), 49.2937575)
        self.assertEqual(round(st.Gas.rhoM, 9), 0.264075691)
        self.assertEqual(round(st.Liquid.hM, 4), 14512.7149)
        self.assertEqual(round(st.Gas.hM, 4), 51501.9146)
        self.assertEqual(round(st.Liquid.sM, 7), 40.6584121)
        self.assertEqual(round(st.Gas.sM, 6), 122.856634)

        st = D2O(T=625, x=0.5)
        self.assertEqual(round(st.P, 7), 17.2118129)
        self.assertEqual(round(st.Liquid.rhoM, 7), 30.6770554)
        self.assertEqual(round(st.Gas.rhoM, 8), 6.94443339)
        self.assertEqual(round(st.Liquid.hM, 4), 32453.3556)
        self.assertEqual(round(st.Gas.hM, 4), 47246.0343)
        self.assertEqual(round(st.Liquid.sM, 7), 73.1042291)
        self.assertEqual(round(st.Gas.sM, 7), 96.7725149)

        # Sublimation-pressure equation
        # Inline point in section 6, pag 10
        P = _D2O_Sublimation_Pressure(245)
        self.assertEqual(round(P, 13), 3.27390934e-5)

        # Melting-pressure equation
        # Inline point in section 6, pag 10
        P = _D2O_Melting_Pressure(270)
        self.assertEqual(round(P, 7), 83.7888413)

        P = _D2O_Melting_Pressure(255, "III")
        self.assertEqual(round(P, 6), 236.470168)

        P = _D2O_Melting_Pressure(275, "V")
        self.assertEqual(round(P, 6), 619.526971)

        P = _D2O_Melting_Pressure(300, "VI")
        self.assertEqual(round(P, 6), 959.203594)

    def test_D2O_Viscosity(self) -> None:
        """Table A5 pag 10"""
        mur = 55.2651e-6
        Tr = 643.847
        rhor = 358
        self.assertEqual(round(_D2O_Viscosity(3.09*rhor, 0.431*Tr)/mur, 10), 36.9123166244)
        self.assertEqual(round(_D2O_Viscosity(3.23*rhor, 0.431*Tr)/mur, 10), 34.1531546602)
        self.assertEqual(round(_D2O_Viscosity(0.0002*rhor, 0.5*Tr)/mur, 10), 0.1972984225)
        self.assertEqual(round(_D2O_Viscosity(3.07*rhor, 0.5*Tr)/mur, 10), 12.0604912273)
        self.assertEqual(round(_D2O_Viscosity(3.18*rhor, 0.5*Tr)/mur, 10), 12.4679405772)
        self.assertEqual(round(_D2O_Viscosity(0.0027*rhor, 0.6*Tr)/mur, 10), 0.2365829037)
        self.assertEqual(round(_D2O_Viscosity(2.95*rhor, 0.6*Tr)/mur, 10), 5.2437249935)
        self.assertEqual(round(_D2O_Viscosity(3.07*rhor, 0.6*Tr)/mur, 10), 5.7578399754)
        self.assertEqual(round(_D2O_Viscosity(0.0295*rhor, 0.75*Tr)/mur, 10), 0.2951479769)
        self.assertEqual(round(_D2O_Viscosity(2.65*rhor, 0.75*Tr)/mur, 10), 2.6275043948)
        self.assertEqual(round(_D2O_Viscosity(2.83*rhor, 0.75*Tr)/mur, 10), 3.0417583586)
        self.assertEqual(round(_D2O_Viscosity(0.08*rhor, 0.9*Tr)/mur, 10), 0.3685472578)
        self.assertEqual(round(_D2O_Viscosity(0.163*rhor, 0.9*Tr)/mur, 10), 0.3619649145)
        self.assertEqual(round(_D2O_Viscosity(2.16*rhor, 0.9*Tr)/mur, 10), 1.6561616211)
        self.assertEqual(round(_D2O_Viscosity(2.52*rhor, 0.9*Tr)/mur, 10), 2.1041364724)
        self.assertEqual(round(_D2O_Viscosity(0.3*rhor, Tr)/mur, 10), 0.4424816849)
        self.assertEqual(round(_D2O_Viscosity(0.7*rhor, Tr)/mur, 10), 0.5528693914)
        self.assertEqual(round(_D2O_Viscosity(1.55*rhor, Tr)/mur, 10), 1.1038442411)
        self.assertEqual(round(_D2O_Viscosity(2.26*rhor, Tr)/mur, 10), 1.7569585722)
        self.assertEqual(round(_D2O_Viscosity(0.49*rhor, 1.1*Tr)/mur, 10), 0.5633038063)
        self.assertEqual(round(_D2O_Viscosity(0.98*rhor, 1.1*Tr)/mur, 10), 0.7816387903)
        self.assertEqual(round(_D2O_Viscosity(1.47*rhor, 1.1*Tr)/mur, 10), 1.1169456968)
        self.assertEqual(round(_D2O_Viscosity(1.96*rhor, 1.1*Tr)/mur, 10), 1.5001420619)
        self.assertEqual(round(_D2O_Viscosity(0.4*rhor, 1.2*Tr)/mur, 10), 0.6094539064)
        self.assertEqual(round(_D2O_Viscosity(0.8*rhor, 1.2*Tr)/mur, 10), 0.7651099154)
        self.assertEqual(round(_D2O_Viscosity(1.2*rhor, 1.2*Tr)/mur, 10), 0.9937870139)
        self.assertEqual(round(_D2O_Viscosity(1.61*rhor, 1.2*Tr)/mur, 10), 1.2711900131)

    def test_D2O_ThCond(self) -> None:
        """Table B4 pag 17"""
        lr = 0.742128e-3
        Tr = 643.847
        rhor = 358
        self.assertEqual(round(_D2O_ThCond(3.09*rhor, 0.431*Tr)/lr, 9), 762.915707396)
        self.assertEqual(round(_D2O_ThCond(3.23*rhor, 0.431*Tr)/lr, 9), 833.912049618)
        self.assertEqual(round(_D2O_ThCond(0.0002*rhor, 0.5*Tr)/lr, 9), 27.006536978)
        self.assertEqual(round(_D2O_ThCond(3.07*rhor, 0.5*Tr)/lr, 9), 835.786416818)
        self.assertEqual(round(_D2O_ThCond(3.18*rhor, 0.5*Tr)/lr, 9), 891.181752526)
        self.assertEqual(round(_D2O_ThCond(0.0027*rhor, 0.6*Tr)/lr, 9), 35.339949553)
        self.assertEqual(round(_D2O_ThCond(2.95*rhor, 0.6*Tr)/lr, 9), 861.240794445)
        self.assertEqual(round(_D2O_ThCond(3.07*rhor, 0.6*Tr)/lr, 9), 919.859094854)
        self.assertEqual(round(_D2O_ThCond(0.0295*rhor, 0.75*Tr)/lr, 9), 55.216750017)
        self.assertEqual(round(_D2O_ThCond(2.65*rhor, 0.75*Tr)/lr, 9), 790.442563472)
        self.assertEqual(round(_D2O_ThCond(2.83*rhor, 0.75*Tr)/lr, 9), 869.672292625)
        self.assertEqual(round(_D2O_ThCond(0.08*rhor, 0.9*Tr)/lr, 9), 74.522283066)
        self.assertEqual(round(_D2O_ThCond(0.163*rhor, 0.9*Tr)/lr, 9), 106.301972320)
        self.assertEqual(round(_D2O_ThCond(2.16*rhor, 0.9*Tr)/lr, 9), 627.777590127)
        self.assertEqual(round(_D2O_ThCond(2.52*rhor, 0.9*Tr)/lr, 9), 761.055043002)
        self.assertEqual(round(_D2O_ThCond(0.3*rhor, Tr)/lr, 9), 143.422002971)
        self.assertEqual(round(_D2O_ThCond(0.7*rhor, Tr)/lr, 9), 469.015122112)
        self.assertEqual(round(_D2O_ThCond(1.55*rhor, Tr)/lr, 9), 502.846952426)
        self.assertEqual(round(_D2O_ThCond(2.26*rhor, Tr)/lr, 9), 668.743524402)
        self.assertEqual(round(_D2O_ThCond(0.49*rhor, 1.1*Tr)/lr, 9), 184.813462109)
        self.assertEqual(round(_D2O_ThCond(0.98*rhor, 1.1*Tr)/lr, 9), 326.652382218)
        self.assertEqual(round(_D2O_ThCond(1.47*rhor, 1.1*Tr)/lr, 9), 438.370305052)
        self.assertEqual(round(_D2O_ThCond(1.96*rhor, 1.1*Tr)/lr, 9), 572.014411428)
        self.assertEqual(round(_D2O_ThCond(0.4*rhor, 1.2*Tr)/lr, 9), 160.059403824)
        self.assertEqual(round(_D2O_ThCond(0.8*rhor, 1.2*Tr)/lr, 9), 259.605241187)
        self.assertEqual(round(_D2O_ThCond(1.2*rhor, 1.2*Tr)/lr, 9), 362.179570932)
        self.assertEqual(round(_D2O_ThCond(1.61*rhor, 1.2*Tr)/lr, 9), 471.747729424)
        self.assertEqual(round(_D2O_ThCond(0.3*rhor, 1.27*Tr)/lr, 9), 145.249914694)
        self.assertEqual(round(_D2O_ThCond(0.6*rhor, 1.27*Tr)/lr, 9), 211.996299238)
        self.assertEqual(round(_D2O_ThCond(0.95*rhor, 1.27*Tr)/lr, 9), 299.251471210)
        self.assertEqual(round(_D2O_ThCond(1.37*rhor, 1.27*Tr)/lr, 9), 409.359675394)

    def test_D2O_Tension(self) -> None:
        """Selected values from table 1"""
        self.assertRaises(NotImplementedError, _D2O_Tension, 250)
        self.assertEqual(round(_D2O_Tension(273.15+3.8)*1000, 2), 74.93)
        self.assertEqual(round(_D2O_Tension(283.15)*1000, 2), 74.06)
        self.assertEqual(round(_D2O_Tension(293.15)*1000, 2), 72.61)
        self.assertEqual(round(_D2O_Tension(303.15)*1000, 2), 71.09)
        self.assertEqual(round(_D2O_Tension(313.15)*1000, 2), 69.52)
        self.assertEqual(round(_D2O_Tension(323.15)*1000, 2), 67.89)
        self.assertEqual(round(_D2O_Tension(333.15)*1000, 2), 66.21)
        self.assertEqual(round(_D2O_Tension(343.15)*1000, 2), 64.47)
        self.assertEqual(round(_D2O_Tension(353.15)*1000, 2), 62.67)
        self.assertEqual(round(_D2O_Tension(363.15)*1000, 2), 60.82)
        self.assertEqual(round(_D2O_Tension(373.15)*1000, 2), 58.93)
        self.assertEqual(round(_D2O_Tension(393.15)*1000, 2), 54.99)
        self.assertEqual(round(_D2O_Tension(413.15)*1000, 2), 50.87)
        self.assertEqual(round(_D2O_Tension(433.15)*1000, 2), 46.59)
        self.assertEqual(round(_D2O_Tension(453.15)*1000, 2), 42.16)
        self.assertEqual(round(_D2O_Tension(473.15)*1000, 2), 37.61)
        self.assertEqual(round(_D2O_Tension(523.15)*1000, 2), 25.84)
        self.assertEqual(round(_D2O_Tension(573.15)*1000, 2), 13.99)
        self.assertEqual(round(_D2O_Tension(623.15)*1000, 2), 3.17)
        self.assertEqual(round(_D2O_Tension(643.15)*1000, 2), 0.05)

    def test_Ice(self) -> None:
        """Table 6, pag 12"""
        ice = _Ice(273.16, 0.000611657)
        self.assertEqual(round(ice["g"], 12), 0.000611784135)
        self.assertEqual(round(ice["gp"], 11), 1.09085812737)
        self.assertEqual(round(ice["gt"], 11), 1.22069433940)
        self.assertEqual(round(ice["gpp"], 15), -0.000128495941571)
        self.assertEqual(round(ice["gtp"], 15), 0.000174387964700)
        self.assertEqual(round(ice["gtt"], 14), -0.00767602985875)
        self.assertEqual(round(ice["h"], 9), -333.444253966)
        self.assertEqual(round(ice["a"], 12), -0.000055446875)
        self.assertEqual(round(ice["u"], 9), -333.444921197)
        self.assertEqual(round(ice["s"], 11), -1.22069433940)
        self.assertEqual(round(ice["cp"], 11), 2.09678431622)
        self.assertEqual(round(ice["rho"], 9), 916.709492200)
        self.assertEqual(round(ice["alfav"], 15), 0.000159863102566)
        self.assertEqual(round(ice["beta"], 11), 1.35714764659)
        self.assertEqual(round(ice["xkappa"], 15), 0.000117793449348)
        self.assertEqual(round(ice["ks"], 15), 0.000114161597779)

        ice = _Ice(273.152519, 0.101325)
        self.assertEqual(round(ice["g"], 11), 0.10134274069)
        self.assertEqual(round(ice["gp"], 11), 1.09084388214)
        self.assertEqual(round(ice["gt"], 11), 1.22076932550)
        self.assertEqual(round(ice["gpp"], 15), -0.000128485364928)
        self.assertEqual(round(ice["gtp"], 15), 0.000174362219972)
        self.assertEqual(round(ice["gtt"], 14), -0.00767598233365)
        self.assertEqual(round(ice["h"], 9), -333.354873637)
        self.assertEqual(round(ice["a"], 11), -0.00918701567)
        self.assertEqual(round(ice["u"], 9), -333.465403393)
        self.assertEqual(round(ice["s"], 11), -1.22076932550)
        self.assertEqual(round(ice["cp"], 11), 2.09671391024)
        self.assertEqual(round(ice["rho"], 9), 916.721463419)
        self.assertEqual(round(ice["alfav"], 15), 0.000159841589458)
        self.assertEqual(round(ice["beta"], 11), 1.35705899321)
        self.assertEqual(round(ice["xkappa"], 15), 0.000117785291765)
        self.assertEqual(round(ice["ks"], 15), 0.000114154442556)

        ice = _Ice(100, 100.)
        self.assertEqual(round(ice["g"], 9), -222.296513088)
        self.assertEqual(round(ice["gp"], 11), 1.06193389260)
        self.assertEqual(round(ice["gt"], 11), 2.61195122589)
        self.assertEqual(round(ice["gpp"], 16), -0.0000941807981761)
        self.assertEqual(round(ice["gtp"], 16), 0.0000274505162488)
        self.assertEqual(round(ice["gtt"], 14), -0.00866333195517)
        self.assertEqual(round(ice["h"], 9), -483.491635676)
        self.assertEqual(round(ice["a"], 9), -328.489902347)
        self.assertEqual(round(ice["u"], 9), -589.685024936)
        self.assertEqual(round(ice["s"], 11), -2.61195122589)
        self.assertEqual(round(ice["cp"], 12), 0.866333195517)
        self.assertEqual(round(ice["rho"], 9), 941.678203297)
        self.assertEqual(round(ice["alfav"], 16), 0.0000258495528207)
        self.assertEqual(round(ice["beta"], 12), 0.291466166994)
        self.assertEqual(round(ice["xkappa"], 16), 0.0000886880048115)
        self.assertEqual(round(ice["ks"], 16), 0.0000886060982687)

        # Test check input
        self.assertRaises(NotImplementedError, _Ice, *(270, 300))
        if major == 3:
            self.assertWarns(Warning, _Ice, *(300, 1))
            self.assertWarns(Warning, _Ice, *(273, 3))
            self.assertWarns(Warning, _Ice, *(272, 1e-4))

    def test_SeaWater(self) -> None:
        """Table 8, pag 17-19"""
        # Part a, pag 17
        fluid = SeaWater(T=273.15, P=0.101325, S=0.03516504)
        state = fluid._water(273.15, 0.101325)
        self.assertEqual(round(state["g"], 8), 0.10134274)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 11), 0.00014764338)
        self.assertEqual(round(state["gp"], 11), 0.00100015694)
        self.assertEqual(round(state["gsp"], 9), 0.0)
        self.assertEqual(round(state["gtt"], 10), -0.0154473542)
        self.assertEqual(round(state["gtp"], 16), -0.677700318e-7)
        self.assertEqual(round(state["gpp"], 15), -0.508928895e-6)

        state = fluid._saline(273.15, 0.101325, 0.03516504)
        self.assertEqual(round(state["g"], 9), -0.101342742)
        self.assertEqual(round(state["gs"], 7), 63.9974067)
        self.assertEqual(round(state["gt"], 12), -0.000147643376)
        self.assertEqual(round(state["gp"], 13), -0.0000274957224)
        self.assertEqual(round(state["gsp"], 12), -0.000759615412)
        self.assertEqual(round(state["gtt"], 12), 0.000852861151)
        self.assertEqual(round(state["gtp"], 15), 0.119286787e-6)
        self.assertEqual(round(state["gpp"], 16), 0.581535172e-7)

        self.assertEqual(round(fluid.g, 5), 0.0)
        self.assertEqual(round(fluid.gs, 7), 63.9974067)
        self.assertEqual(round(fluid.gt, 5), 0.0)
        self.assertEqual(round(fluid.gp, 12), 0.000972661217)
        self.assertEqual(round(fluid.gsp, 12), -0.000759615412)
        self.assertEqual(round(fluid.gtt, 10), -0.0145944931)
        self.assertEqual(round(fluid.gtp, 16), 0.515167556e-7)
        self.assertEqual(round(fluid.gpp, 15), -0.450775377e-6)
        self.assertEqual(round(fluid.h, 6), 0.0)
        # self.assertEqual(round(fluid.a, 10), -0.0985548978)
        # self.assertEqual(round(fluid.u, 10), -0.0985548978)
        self.assertEqual(round(fluid.s, 6), 0.0)
        self.assertEqual(round(fluid.rho, 5), 1028.10720)
        self.assertEqual(round(fluid.cp, 8), 3.98648579)
        self.assertEqual(round(fluid.w, 5), 1449.00246)
        self.assertEqual(round(fluid.muw, 8), -2.25047137)

        # Part b, pag 18
        fluid = SeaWater(T=353, P=0.101325, S=0.1)
        state = fluid._water(353, 0.101325)
        self.assertEqual(round(state["g"], 9), -44.6114969)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 8), -1.07375993)
        self.assertEqual(round(state["gp"], 11), 0.00102892956)
        self.assertEqual(round(state["gsp"], 9), 0.0)
        self.assertEqual(round(state["gtt"], 10), -0.0118885000)
        self.assertEqual(round(state["gtp"], 15), 0.659051552e-6)
        self.assertEqual(round(state["gpp"], 14), -0.47467282e-6)

        state = fluid._saline(353, 0.101325, 0.1)
        self.assertEqual(round(state["g"], 7), 15.0871740)
        self.assertEqual(round(state["gs"], 6), 251.957276)
        self.assertEqual(round(state["gt"], 9), 0.156230907)
        self.assertEqual(round(state["gp"], 13), -0.0000579227286)
        self.assertEqual(round(state["gsp"], 12), -0.000305957802)
        self.assertEqual(round(state["gtt"], 11), 0.00127922649)
        self.assertEqual(round(state["gtp"], 15), 0.803061596e-6)
        self.assertEqual(round(state["gpp"], 15), 0.213086154e-6)

        self.assertEqual(round(fluid.g, 7), -29.5243229)
        self.assertEqual(round(fluid.gs, 6), 251.957276)
        self.assertEqual(round(fluid.gt, 9), -0.917529024)
        self.assertEqual(round(fluid.gp, 12), 0.000971006828)
        self.assertEqual(round(fluid.gsp, 12), -0.000305957802)
        self.assertEqual(round(fluid.gtt, 10), -0.0106092735)
        self.assertEqual(round(fluid.gtp, 14), 0.146211315e-5)
        self.assertEqual(round(fluid.gpp, 15), -0.261586665e-6)
        self.assertEqual(round(fluid.h, 6), 294.363423)
        self.assertEqual(round(fluid.a, 7), -29.6227102)
        self.assertEqual(round(fluid.u, 6), 294.265035)
        self.assertEqual(round(fluid.s, 9), 0.917529024)
        self.assertEqual(round(fluid.rho, 5), 1029.85888)
        self.assertEqual(round(fluid.cp, 8), 3.74507355)
        self.assertEqual(round(fluid.w, 5), 3961.27835)
        self.assertEqual(round(fluid.muw, 7), -54.7200505)

        # Part c, pag 19
        fluid = SeaWater(T=273.15, P=100, S=0.03516504)
        state = fluid._water(273.15, 100)
        self.assertEqual(round(state["g"], 7), 97.7303862)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 11), 0.00851466502)
        self.assertEqual(round(state["gp"], 12), 0.000956683329)
        self.assertEqual(round(state["gsp"], 9), 0.0)
        self.assertEqual(round(state["gtt"], 10), -0.0142969873)
        self.assertEqual(round(state["gtp"], 15), 0.199079571e-6)
        self.assertEqual(round(state["gpp"], 15), -0.371530889e-6)

        state = fluid._saline(273.15, 100, 0.03516504)
        self.assertEqual(round(state["g"], 8), -2.60093051)
        self.assertEqual(round(state["gs"], 8), -5.45861581)
        self.assertEqual(round(state["gt"], 11), 0.00754045685)
        self.assertEqual(round(state["gp"], 13), -0.0000229123842)
        self.assertEqual(round(state["gsp"], 12), -0.000640757619)
        self.assertEqual(round(state["gtt"], 12), 0.000488076974)
        self.assertEqual(round(state["gtp"], 16), 0.466284412e-7)
        self.assertEqual(round(state["gpp"], 16), 0.357345736e-7)

        self.assertEqual(round(fluid.g, 7), 95.1294557)
        self.assertEqual(round(fluid.gs, 8), -5.45861581)
        self.assertEqual(round(fluid.gt, 10), 0.0160551219)
        self.assertEqual(round(fluid.gp, 12), 0.000933770945)
        self.assertEqual(round(fluid.gsp, 12), -0.000640757619)
        self.assertEqual(round(fluid.gtt, 10), -0.0138089104)
        self.assertEqual(round(fluid.gtp, 15), 0.245708012e-6)
        self.assertEqual(round(fluid.gpp, 15), -0.335796316e-6)
        self.assertEqual(round(fluid.h, 7), 90.7439992)
        self.assertEqual(round(fluid.a, 8), 1.75236121)
        self.assertEqual(round(fluid.u, 8), -2.63309532)
        self.assertEqual(round(fluid.s, 10), -0.0160551219)
        self.assertEqual(round(fluid.rho, 5), 1070.92645)
        self.assertEqual(round(fluid.cp, 8), 3.77190387)
        self.assertEqual(round(fluid.w, 5), 1621.98998)
        self.assertEqual(round(fluid.muw, 7), 95.3214082)

        # Custom derivative implementation
        fluid = SeaWater(T=353, P=0.101325, S=0)
        wat = IAPWS95(T=353, P=0.101325)
        self.assertEqual(round(fluid.derivative("T", "P", "h")*1000-wat.joule, 7), 0)

    def test_SeaWater_supp(self) -> None:
        """Table 6, pag 9"""
        fluid = SeaWater(T=273.15, P=0.101325, S=0, fast=True)
        state = fluid._waterSupp(273.15, 0.101325)
        self.assertEqual(round(state["g"], 9), 0.101342743)
        self.assertEqual(round(state["gt"], 12), 0.000147644587)
        self.assertEqual(round(state["gp"], 11), 0.00100015695)
        self.assertEqual(round(state["gtt"], 10), -0.0154472324)
        self.assertEqual(round(state["gtp"], 16), -0.677459513e-7)
        self.assertEqual(round(state["gpp"], 15), -0.508915308e-6)
        self.assertEqual(round(fluid.h, 10), 0.0610136242)
        self.assertEqual(round(fluid.a, 14), 1.83980891e-6)
        self.assertEqual(round(fluid.u, 10), -0.0403272791)
        self.assertEqual(round(fluid.s, 12), -0.000147644587)
        self.assertEqual(round(fluid.rho, 6), 999.843071)
        self.assertEqual(round(fluid.cp, 8), 4.21941153)
        self.assertEqual(round(fluid.w, 5), 1402.40099)

        fluid = SeaWater(T=273.15, P=100, S=0, fast=True)
        state = fluid._waterSupp(273.15, 100)
        self.assertEqual(round(state["g"], 7), 97.7303868)
        self.assertEqual(round(state["gt"], 11), 0.00851506346)
        self.assertEqual(round(state["gp"], 12), 0.000956683354)
        self.assertEqual(round(state["gtt"], 10), -0.0142970174)
        self.assertEqual(round(state["gtp"], 16), 1.99088060e-7)
        self.assertEqual(round(state["gpp"], 15), -0.371527164e-6)
        self.assertEqual(round(fluid.h, 7), 95.4044973)
        self.assertEqual(round(fluid.a, 8), 2.06205140)
        self.assertEqual(round(fluid.u, 9), -0.263838183)
        self.assertEqual(round(fluid.s, 11), -0.00851506346)
        self.assertEqual(round(fluid.rho, 5), 1045.27793)
        self.assertEqual(round(fluid.cp, 8), 3.90523030)
        self.assertEqual(round(fluid.w, 5), 1575.43089)

        fluid = SeaWater(T=313.15, P=0.101325, S=0, fast=True)
        state = fluid._waterSupp(313.15, 0.101325)
        self.assertEqual(round(state["g"], 7), -11.6198898)
        self.assertEqual(round(state["gt"], 9), -0.572365181)
        self.assertEqual(round(state["gp"], 11), 0.00100784471)
        self.assertEqual(round(state["gtt"], 10), -0.0133463968)
        self.assertEqual(round(state["gtp"], 15), 0.388499694e-6)
        self.assertEqual(round(state["gpp"], 15), -0.445841077e-6)
        self.assertEqual(round(fluid.h, 6), 167.616267)
        self.assertEqual(round(fluid.a, 7), -11.7220097)
        self.assertEqual(round(fluid.u, 6), 167.514147)
        self.assertEqual(round(fluid.s, 9), 0.572365181)
        self.assertEqual(round(fluid.rho, 6), 992.216354)
        self.assertEqual(round(fluid.cp, 8), 4.17942416)
        self.assertEqual(round(fluid.w, 5), 1528.91242)

    def test_SeaWaterIF97(self) -> None:
        """Table A1, pag 19-21"""
        fluid = SeaWater(T=273.15, P=0.101325, S=0.03516504, IF97=True)
        state = fluid._waterIF97(273.15, 0.101325)
        self.assertEqual(round(state["g"], 9), 0.101359446)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 12), 0.147711823e-3)
        self.assertEqual(round(state["gp"], 11), 0.100015572e-2)
        self.assertEqual(round(state["gsp"], 9), 0.0)
        self.assertEqual(round(state["gtt"], 10), -0.154473013e-1)
        self.assertEqual(round(state["gtp"], 16), -0.676992620e-7)
        self.assertEqual(round(state["gpp"], 15), -0.508885499e-6)

        state = fluid._saline(273.15, 0.101325, 0.03516504)
        self.assertEqual(round(state["g"], 9), -0.101342742)
        self.assertEqual(round(state["gs"], 7), 0.639974067e2)
        self.assertEqual(round(state["gt"], 12), -0.147643376e-3)
        self.assertEqual(round(state["gp"], 13), -0.274957224e-4)
        self.assertEqual(round(state["gsp"], 12), -0.759615412e-3)
        self.assertEqual(round(state["gtt"], 12), 0.852861151e-3)
        self.assertEqual(round(state["gtp"], 15), 0.119286787e-6)
        self.assertEqual(round(state["gpp"], 16), 0.581535172e-7)

        self.assertEqual(round(fluid.g, 9), 0.16705e-4)
        self.assertEqual(round(fluid.gs, 7), 0.639974067e2)
        self.assertEqual(round(fluid.gt, 12), 0.68447e-7)
        self.assertEqual(round(fluid.gp, 12), 0.972659995e-3)
        self.assertEqual(round(fluid.gsp, 12), -0.759615412e-3)
        self.assertEqual(round(fluid.gtt, 10), -0.145944401e-1)
        self.assertEqual(round(fluid.gtp, 16), 0.515875254e-7)
        self.assertEqual(round(fluid.gpp, 15), -0.450731982e-6)
        self.assertEqual(round(fluid.v, 12), 0.972659995e-3)
        self.assertEqual(round(fluid.u, 10), -0.985567655e-1)
        self.assertEqual(round(fluid.h, 10), -0.19916e-5)
        self.assertEqual(round(fluid.s, 12), -0.68447e-7)
        self.assertEqual(round(fluid.cp, 8), 0.398647132e1)
        self.assertEqual(round(fluid.w, 5), 0.144907123e4)
        self.assertEqual(round(fluid.muw, 8), -0.225045466e1)

        fluid = SeaWater(T=353, P=0.101325, S=0.1, IF97=True)
        state = fluid._waterIF97(353, 0.101325)
        self.assertEqual(round(state["g"], 7), -0.446091363e2)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 8), -0.107357342e1)
        self.assertEqual(round(state["gp"], 11), 0.102891627e-2)
        self.assertEqual(round(state["gsp"], 9), 0.0)
        self.assertEqual(round(state["gtt"], 10), -0.118849543e-1)
        self.assertEqual(round(state["gtp"], 15), 0.659344070e-6)
        self.assertEqual(round(state["gpp"], 15), -0.473220602e-6)

        state = fluid._saline(353, 0.101325, 0.1)
        self.assertEqual(round(state["g"], 7), 0.150871740e2)
        self.assertEqual(round(state["gs"], 6), 0.251957276e3)
        self.assertEqual(round(state["gt"], 9), 0.156230907)
        self.assertEqual(round(state["gp"], 13), -0.579227286e-4)
        self.assertEqual(round(state["gsp"], 12), -0.305957802e-3)
        self.assertEqual(round(state["gtt"], 11), 0.127922649e-2)
        self.assertEqual(round(state["gtp"], 15), 0.803061596e-6)
        self.assertEqual(round(state["gpp"], 15), 0.213086154e-6)

        self.assertEqual(round(fluid.g, 7), -0.295219623e2)
        self.assertEqual(round(fluid.gs, 6), 0.251957276e3)
        self.assertEqual(round(fluid.gt, 9), -0.917342513)
        self.assertEqual(round(fluid.gp, 12), 0.970993546e-3)
        self.assertEqual(round(fluid.gsp, 12), -0.305957802e-3)
        self.assertEqual(round(fluid.gtt, 10), -0.106057278e-1)
        self.assertEqual(round(fluid.gtp, 14), 0.146240567e-5)
        self.assertEqual(round(fluid.gpp, 15), -0.260134448e-6)
        self.assertEqual(round(fluid.v, 12), 0.970993546e-3)
        self.assertEqual(round(fluid.u, 6), 0.294201559e3)
        self.assertEqual(round(fluid.h, 6), 0.294299945e3)
        self.assertEqual(round(fluid.s, 9), 0.917342513)
        self.assertEqual(round(fluid.cp, 8), 0.374382192e1)
        self.assertEqual(round(fluid.w, 5), 0.401505044e4)
        self.assertEqual(round(fluid.muw, 7), -0.547176899e2)

        # Part c, pag 19
        fluid = SeaWater(T=273.15, P=100, S=0.03516504, IF97=True)
        state = fluid._waterIF97(273.15, 100)
        self.assertEqual(round(state["g"], 7), 0.977302204e2)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 11), 0.858228709e-2)
        self.assertEqual(round(state["gp"], 12), 0.956686939e-3)
        self.assertEqual(round(state["gsp"], 9), 0.0)
        self.assertEqual(round(state["gtt"], 10), -0.142987096e-1)
        self.assertEqual(round(state["gtp"], 15), 0.202974451e-6)
        self.assertEqual(round(state["gpp"], 15), -0.371594622e-6)

        state = fluid._saline(273.15, 100, 0.03516504)
        self.assertEqual(round(state["g"], 8), -0.260093051e1)
        self.assertEqual(round(state["gs"], 8), -0.545861581e1)
        self.assertEqual(round(state["gt"], 11), 0.754045685e-2)
        self.assertEqual(round(state["gp"], 13), -0.229123842e-4)
        self.assertEqual(round(state["gsp"], 12), -0.640757619e-3)
        self.assertEqual(round(state["gtt"], 12), 0.488076974e-3)
        self.assertEqual(round(state["gtp"], 16), 0.466284412e-7)
        self.assertEqual(round(state["gpp"], 16), 0.357345736e-7)

        self.assertEqual(round(fluid.g, 7), 0.951292899e2)
        self.assertEqual(round(fluid.gs, 8), -0.545861581e1)
        self.assertEqual(round(fluid.gt, 10), 0.161227439e-1)
        self.assertEqual(round(fluid.gp, 12), 0.933774555e-3)
        self.assertEqual(round(fluid.gsp, 12), -0.640757619e-3)
        self.assertEqual(round(fluid.gtt, 10), -0.138106326e-1)
        self.assertEqual(round(fluid.gtp, 15), 0.249602892e-6)
        self.assertEqual(round(fluid.gpp, 15), -0.335860049e-6)
        self.assertEqual(round(fluid.v, 12), 0.933774555e-3)
        self.assertEqual(round(fluid.u, 8), -0.265209313e1)
        self.assertEqual(round(fluid.h, 7), 0.907253624e2)
        self.assertEqual(round(fluid.s, 10), -0.161227439e-1)
        self.assertEqual(round(fluid.cp, 8), 0.377237430e1)
        self.assertEqual(round(fluid.w, 5), 0.162218081e4)
        self.assertEqual(round(fluid.muw, 7), 0.953212423e2)

        # Table A2
        self.assertEqual(round(_Tb(0.001, 0), 2), 280.12)
        self.assertEqual(round(_Tb(0.001, 0.02), 2), 280.27)
        self.assertEqual(round(_Tb(0.001, 0.04), 2), 280.43)
        self.assertEqual(round(_Tb(0.001, 0.06), 2), 280.61)
        self.assertEqual(round(_Tb(0.001, 0.08), 2), 280.80)
        self.assertEqual(round(_Tb(0.001, 0.10), 2), 281.01)
        self.assertEqual(round(_Tb(0.001, 0.12), 2), 281.25)

        self.assertEqual(round(_Tb(0.005, 0), 2), 306.02)
        self.assertEqual(round(_Tb(0.005, 0.02), 2), 306.21)
        self.assertEqual(round(_Tb(0.005, 0.04), 2), 306.41)
        self.assertEqual(round(_Tb(0.005, 0.06), 2), 306.63)
        self.assertEqual(round(_Tb(0.005, 0.08), 2), 306.87)
        self.assertEqual(round(_Tb(0.005, 0.10), 2), 307.14)
        self.assertEqual(round(_Tb(0.005, 0.12), 2), 307.44)

        self.assertEqual(round(_Tb(0.01, 0), 2), 318.96)
        self.assertEqual(round(_Tb(0.01, 0.02), 2), 319.16)
        self.assertEqual(round(_Tb(0.01, 0.04), 2), 319.38)
        self.assertEqual(round(_Tb(0.01, 0.06), 2), 319.62)
        self.assertEqual(round(_Tb(0.01, 0.08), 2), 319.89)
        self.assertEqual(round(_Tb(0.01, 0.10), 2), 320.19)
        self.assertEqual(round(_Tb(0.01, 0.12), 2), 320.52)

        self.assertEqual(round(_Tb(0.02, 0), 2), 333.21)
        self.assertEqual(round(_Tb(0.02, 0.02), 2), 333.44)
        self.assertEqual(round(_Tb(0.02, 0.04), 2), 333.68)
        self.assertEqual(round(_Tb(0.02, 0.06), 2), 333.95)
        self.assertEqual(round(_Tb(0.02, 0.08), 2), 334.24)
        self.assertEqual(round(_Tb(0.02, 0.10), 2), 334.57)
        self.assertEqual(round(_Tb(0.02, 0.12), 2), 334.94)

        self.assertEqual(round(_Tb(0.04, 0), 2), 349.01)
        self.assertEqual(round(_Tb(0.04, 0.02), 2), 349.26)
        self.assertEqual(round(_Tb(0.04, 0.04), 2), 349.53)
        self.assertEqual(round(_Tb(0.04, 0.06), 2), 349.83)
        self.assertEqual(round(_Tb(0.04, 0.08), 2), 350.15)
        self.assertEqual(round(_Tb(0.04, 0.10), 2), 350.52)
        self.assertEqual(round(_Tb(0.04, 0.12), 2), 350.94)

        # Table A2
        self.assertEqual(round(_Tf(0.001, 0), 4), 273.1600)
        self.assertEqual(round(_Tf(0.001, 0.02), 4), 272.0823)
        self.assertEqual(round(_Tf(0.001, 0.04), 4), 270.9611)
        self.assertEqual(round(_Tf(0.001, 0.06), 4), 269.7618)
        self.assertEqual(round(_Tf(0.001, 0.08), 4), 268.4609)
        self.assertEqual(round(_Tf(0.001, 0.1), 4), 267.0397)
        self.assertEqual(round(_Tf(0.001, 0.12), 4), 265.4900)

        self.assertEqual(round(_Tf(0.005, 0), 4), 273.1597)
        self.assertEqual(round(_Tf(0.005, 0.02), 4), 272.0820)
        self.assertEqual(round(_Tf(0.005, 0.04), 4), 270.9608)
        self.assertEqual(round(_Tf(0.005, 0.06), 4), 269.7615)
        self.assertEqual(round(_Tf(0.005, 0.08), 4), 268.4606)
        self.assertEqual(round(_Tf(0.005, 0.1), 4), 267.0394)
        self.assertEqual(round(_Tf(0.005, 0.12), 4), 265.4897)

        self.assertEqual(round(_Tf(0.01, 0), 4), 273.1593)
        self.assertEqual(round(_Tf(0.01, 0.02), 4), 272.0816)
        self.assertEqual(round(_Tf(0.01, 0.04), 4), 270.9604)
        self.assertEqual(round(_Tf(0.01, 0.06), 4), 269.7612)
        self.assertEqual(round(_Tf(0.01, 0.08), 4), 268.4602)
        self.assertEqual(round(_Tf(0.01, 0.12), 4), 265.4893)

        self.assertEqual(round(_Tf(0.1, 0), 3), 273.153)
        self.assertEqual(round(_Tf(0.1, 0.02), 3), 272.075)
        self.assertEqual(round(_Tf(0.1, 0.04), 3), 270.954)
        self.assertEqual(round(_Tf(0.1, 0.06), 3), 269.754)
        self.assertEqual(round(_Tf(0.1, 0.08), 3), 268.453)
        self.assertEqual(round(_Tf(0.1, 0.1), 3), 267.032)
        self.assertEqual(round(_Tf(0.1, 0.12), 3), 265.482)

        self.assertEqual(round(_Tf(1, 0), 2), 273.09)
        self.assertEqual(round(_Tf(1, 0.02), 2), 272.01)
        self.assertEqual(round(_Tf(1, 0.04), 2), 270.89)

        self.assertEqual(round(_Tf(10, 0), 1), 272.4)
        self.assertEqual(round(_Tf(10, 0.02), 2), 271.32)
        self.assertEqual(round(_Tf(10, 0.04), 2), 270.20)

        self.assertEqual(round(_Tf(100, 0), 2), 264.21)
        self.assertEqual(round(_Tf(100, 0.02), 2), 263.09)
        self.assertEqual(round(_Tf(100, 0.04), 2), 261.92)

        # Triple point, Table A4
        Tt = _Triple(0)
        self.assertEqual(round(Tt["Tt"], 2), 273.16)
        self.assertEqual(round(Tt["Pt"], 8), 6.1168e-4)

        Tt = _Triple(0.02)
        self.assertEqual(round(Tt["Tt"], 2), 272.08)
        self.assertEqual(round(Tt["Pt"], 8), 5.5953e-4)

        Tt = _Triple(0.04)
        self.assertEqual(round(Tt["Tt"], 2), 270.96)
        self.assertEqual(round(Tt["Pt"], 8), 5.0961e-4)

        Tt = _Triple(0.06)
        self.assertEqual(round(Tt["Tt"], 2), 269.76)
        self.assertEqual(round(Tt["Pt"], 8), 4.6073e-4)

        Tt = _Triple(0.08)
        self.assertEqual(round(Tt["Tt"], 2), 268.46)
        self.assertEqual(round(Tt["Pt"], 8), 4.1257e-4)

        Tt = _Triple(0.1)
        self.assertEqual(round(Tt["Tt"], 2), 267.04)
        self.assertEqual(round(Tt["Pt"], 8), 3.6524e-4)

        Tt = _Triple(0.12)
        self.assertEqual(round(Tt["Tt"], 2), 265.49)
        self.assertEqual(round(Tt["Pt"], 8), 3.1932e-4)

        # Osmotic pressure, i have no test to do
        self.assertEqual(round(_OsmoticPressure(300, 0.1, 0), 5), 0.0)

    def test_SeaWater_thcond(self) -> None:
        """Table 2, pag 5"""
        fluid = SeaWater(T=293.15, P=0.1, S=0.035)
        self.assertEqual(round(_ThCond_SeaWater(T=293.15, P=0.1, S=0.035), 9), -0.004186040)
        self.assertEqual(round(fluid.k, 9), 0.593825535)

        fluid = SeaWater(T=293.15, P=120, S=0.035)
        self.assertEqual(round(_ThCond_SeaWater(T=293.15, P=120, S=0.035), 9), -0.004317350)
        self.assertEqual(round(fluid.k, 9), 0.651692949)

        fluid = SeaWater(T=333.15, P=0.1, S=0.035)
        self.assertEqual(round(_ThCond_SeaWater(T=333.15, P=0.1, S=0.035), 9), -0.004124057)
        self.assertEqual(round(fluid.k, 9), 0.646875533)

        fluid = SeaWater(T=333.15, P=120, S=0.035)
        self.assertEqual(round(_ThCond_SeaWater(T=333.15, P=120, S=0.035), 9), -0.004264405)
        self.assertEqual(round(fluid.k, 9), 0.702484548)

        fluid = SeaWater(T=293.15, P=0.1, S=0.1)
        self.assertEqual(round(_ThCond_SeaWater(T=293.15, P=0.1, S=0.1), 9), -0.013819821)
        self.assertEqual(round(fluid.k, 9), 0.584191754)

        fluid = SeaWater(T=373.15, P=1, S=0.1)
        self.assertEqual(round(_ThCond_SeaWater(T=373.15, P=1, S=0.1), 9), -0.013094107)
        self.assertEqual(round(fluid.k, 9), 0.664627314)

        fluid = SeaWater(T=293.15, P=0.1, S=0.12)
        self.assertEqual(round(_ThCond_SeaWater(T=293.15, P=0.1, S=0.12), 9), -0.017005302)
        self.assertEqual(round(fluid.k, 9), 0.581006273)

        fluid = SeaWater(T=293.15, P=120, S=0.12)
        self.assertEqual(round(_ThCond_SeaWater(T=293.15, P=120, S=0.12), 9), -0.020194816)
        self.assertEqual(round(fluid.k, 9), 0.635815483)

        fluid = SeaWater(T=333.15, P=120, S=0.12)
        self.assertEqual(round(_ThCond_SeaWater(T=333.15, P=120, S=0.12), 9), -0.019722469)
        self.assertEqual(round(fluid.k, 9), 0.687026483)

        fluid = SeaWater(T=270, P=1, S=0.12)
        self.assertRaises(NotImplementedError, _ThCond_SeaWater, *(270, 1, 0))

    def test_SeaWater_tension(self) -> None:
        """Table 2, pag 4"""
        self.assertEqual(round(_Tension_SeaWater(253.15, 0.035)*1e3, 9), 79.225179610)
        self.assertEqual(round(_Tension_SeaWater(298.15, 0.035)*1e3, 9), 73.068674787)
        self.assertEqual(round(_Tension_SeaWater(353.15, 0.035)*1e3, 9), 63.910806802)
        self.assertEqual(round(_Tension_SeaWater(298.15, 0.060)*1e3, 9), 73.851867328)
        self.assertEqual(round(_Tension_SeaWater(353.15, 0.060)*1e3, 9), 64.795058112)
        self.assertEqual(round(_Tension_SeaWater(293.15, 0.120)*1e3, 9), 76.432940211)
        self.assertEqual(round(_Tension_SeaWater(353.15, 0.120)*1e3, 9), 66.917261258)

    def test_na2so4(self) -> None:
        """Selected point from Table 1, pag 5"""
        self.assertEqual(round(_solNa2SO4(523.15, 0, 0), 2), 3.54)
        self.assertEqual(round(_solNa2SO4(523.15, 0.25, 0.083), 2), 3.31)
        self.assertEqual(round(_solNa2SO4(523.15, 0.75, 2.25), 2), 2.85)
        self.assertEqual(round(_solNa2SO4(548.15, 0, 0), 2), 2.51)
        self.assertEqual(round(_solNa2SO4(548.15, 0.75, 2.25), 2), 2.62)
        self.assertEqual(round(_solNa2SO4(548.15, 0.25, 0.083), 2), 2.58)
        self.assertEqual(round(_solNa2SO4(573.15, 0, 0), 2), 1.61)
        self.assertEqual(round(_solNa2SO4(573.15, 0.75, 2.25), 2), 2.42)
        self.assertEqual(round(_solNa2SO4(598.15, 0, 0), 2), 0.81)
        self.assertEqual(round(_solNa2SO4(598.15, 0.25, 0.083), 2), 1.2)
        self.assertEqual(round(_solNa2SO4(598.15, 0.75, 2.25), 2), 2.26)
        self.assertEqual(round(_solNa2SO4(623.15, 0, 0), 2), 0.13)
        self.assertEqual(round(_solNa2SO4(623.15, 0.25, 0.083), 2), 0.55)
        self.assertEqual(round(_solNa2SO4(623.15, 0.75, 2.25), 2), 2.13)
        self.assertRaises(NotImplementedError, _solNa2SO4, *(500, 0, 0))

    def test_critNaCl(self) -> None:
        """Table II, page 6"""
        crit = _critNaCl(0)
        self.assertEqual(round(crit["Tc"], 6), 647.096000)
        self.assertEqual(round(crit["Pc"], 7), 22.0640000)
        self.assertEqual(round(crit["rhoc"], 6), 322.000000)
        crit = _critNaCl(0.0005)
        self.assertEqual(round(crit["Tc"], 6), 651.858942)
        self.assertEqual(round(crit["Pc"], 7), 23.0502156)
        self.assertEqual(round(crit["rhoc"], 6), 341.467388)
        crit = _critNaCl(0.001)
        self.assertEqual(round(crit["Tc"], 6), 653.957959)
        self.assertEqual(round(crit["Pc"], 7), 23.5003231)
        self.assertEqual(round(crit["rhoc"], 6), 355.403431)
        crit = _critNaCl(0.0015)
        self.assertEqual(round(crit["Tc"], 6), 656.214478)
        self.assertEqual(round(crit["Pc"], 7), 23.9942848)
        self.assertEqual(round(crit["rhoc"], 6), 366.631874)
        crit = _critNaCl(0.002)
        self.assertEqual(round(crit["Tc"], 6), 658.266685)
        self.assertEqual(round(crit["Pc"], 7), 24.4522876)
        self.assertEqual(round(crit["rhoc"], 6), 376.071640)
        crit = _critNaCl(0.003)
        self.assertEqual(round(crit["Tc"], 6), 661.792675)
        self.assertEqual(round(crit["Pc"], 7), 25.2578916)
        self.assertEqual(round(crit["rhoc"], 6), 391.370335)
        crit = _critNaCl(0.004)
        self.assertEqual(round(crit["Tc"], 6), 664.850727)
        self.assertEqual(round(crit["Pc"], 7), 25.9748152)
        self.assertEqual(round(crit["rhoc"], 6), 403.503882)
        crit = _critNaCl(0.005)
        self.assertEqual(round(crit["Tc"], 6), 667.640852)
        self.assertEqual(round(crit["Pc"], 7), 26.6429248)
        self.assertEqual(round(crit["rhoc"], 6), 413.572545)
        crit = _critNaCl(0.006)
        self.assertEqual(round(crit["Tc"], 6), 670.274404)
        self.assertEqual(round(crit["Pc"], 7), 27.2851928)
        self.assertEqual(round(crit["rhoc"], 6), 422.220364)
        crit = _critNaCl(0.007)
        self.assertEqual(round(crit["Tc"], 6), 672.818265)
        self.assertEqual(round(crit["Pc"], 7), 27.9158160)
        self.assertEqual(round(crit["rhoc"], 6), 429.857769)
        crit = _critNaCl(0.008)
        self.assertEqual(round(crit["Tc"], 6), 675.314254)
        self.assertEqual(round(crit["Pc"], 7), 28.5438775)
        self.assertEqual(round(crit["rhoc"], 6), 436.760441)
        crit = _critNaCl(0.009)
        self.assertEqual(round(crit["Tc"], 6), 677.789002)
        self.assertEqual(round(crit["Pc"], 7), 29.1752574)
        self.assertEqual(round(crit["rhoc"], 6), 443.120149)
        crit = _critNaCl(0.01)
        self.assertEqual(round(crit["Tc"], 6), 680.259476)
        self.assertEqual(round(crit["Pc"], 7), 29.8137278)
        self.assertEqual(round(crit["rhoc"], 6), 449.073533)
        crit = _critNaCl(0.02)
        self.assertEqual(round(crit["Tc"], 6), 705.710570)
        self.assertEqual(round(crit["Pc"], 7), 36.7725468)
        self.assertEqual(round(crit["rhoc"], 6), 499.162456)
        crit = _critNaCl(0.03)
        self.assertEqual(round(crit["Tc"], 6), 731.830060)
        self.assertEqual(round(crit["Pc"], 7), 44.3508666)
        self.assertEqual(round(crit["rhoc"], 6), 542.631406)
        crit = _critNaCl(0.04)
        self.assertEqual(round(crit["Tc"], 6), 757.383936)
        self.assertEqual(round(crit["Pc"], 7), 51.8585660)
        self.assertEqual(round(crit["rhoc"], 6), 579.897786)
        crit = _critNaCl(0.05)
        self.assertEqual(round(crit["Tc"], 6), 782.719671)
        self.assertEqual(round(crit["Pc"], 7), 59.1490976)
        self.assertEqual(round(crit["rhoc"], 6), 609.816085)
        crit = _critNaCl(0.06)
        self.assertEqual(round(crit["Tc"], 6), 809.415999)
        self.assertEqual(round(crit["Pc"], 7), 66.4909864)
        self.assertEqual(round(crit["rhoc"], 6), 632.701609)
        crit = _critNaCl(0.07)
        self.assertEqual(round(crit["Tc"], 6), 839.687528)
        self.assertEqual(round(crit["Pc"], 7), 74.2862762)
        self.assertEqual(round(crit["rhoc"], 6), 649.832063)
        crit = _critNaCl(0.08)
        self.assertEqual(round(crit["Tc"], 6), 875.954487)
        self.assertEqual(round(crit["Pc"], 7), 82.9284097)
        self.assertEqual(round(crit["rhoc"], 6), 662.792131)
        crit = _critNaCl(0.09)
        self.assertEqual(round(crit["Tc"], 6), 920.557281)
        self.assertEqual(round(crit["Pc"], 7), 93.0389707)
        self.assertEqual(round(crit["rhoc"], 6), 673.036769)
        crit = _critNaCl(0.1)
        self.assertEqual(round(crit["Tc"], 6), 975.571016)
        self.assertEqual(round(crit["Pc"], 6), 106.692174)
        self.assertEqual(round(crit["rhoc"], 6), 681.662669)
        crit = _critNaCl(0.11)
        self.assertEqual(round(crit["Tc"], 5), 1042.68691)
        self.assertEqual(round(crit["Pc"], 6), 130.966478)
        self.assertEqual(round(crit["rhoc"], 6), 689.331958)
        crit = _critNaCl(0.12)
        self.assertEqual(round(crit["Tc"], 5), 1123.13874)
        self.assertEqual(round(crit["Pc"], 6), 186.176548)
        self.assertEqual(round(crit["rhoc"], 6), 696.300163)

        self.assertRaises(NotImplementedError, _critNaCl, 0.2)

    def test_Henry(self) -> None:
        """Table 6, Henry constants."""
        # Table 6 for Henry constants
        self.assertRaises(NotImplementedError, _Henry, *(300, "He", "He"))
        self.assertRaises(NotImplementedError, _Henry, *(300, "SF6", "D2O"))
        self.assertEqual(round(log(_Henry(300, "He")/1000), 4), 2.6576)
        self.assertEqual(round(log(_Henry(400, "He")/1000), 4), 2.1660)
        self.assertEqual(round(log(_Henry(500, "He")/1000), 4), 1.1973)
        self.assertEqual(round(log(_Henry(600, "He")/1000), 4), -0.1993)
        self.assertEqual(round(log(_Henry(300, "Ne")/1000), 4), 2.5134)
        self.assertEqual(round(log(_Henry(400, "Ne")/1000), 4), 2.3512)
        self.assertEqual(round(log(_Henry(500, "Ne")/1000), 4), 1.5952)
        self.assertEqual(round(log(_Henry(600, "Ne")/1000), 4), 0.4659)
        self.assertEqual(round(log(_Henry(300, "Ar")/1000), 4), 1.4061)
        self.assertEqual(round(log(_Henry(400, "Ar")/1000), 4), 1.8079)
        self.assertEqual(round(log(_Henry(500, "Ar")/1000), 4), 1.1536)
        self.assertEqual(round(log(_Henry(600, "Ar")/1000), 4), 0.0423)
        self.assertEqual(round(log(_Henry(300, "Kr")/1000), 4), 0.8210)
        self.assertEqual(round(log(_Henry(400, "Kr")/1000), 4), 1.4902)
        self.assertEqual(round(log(_Henry(500, "Kr")/1000), 4), 0.9798)
        self.assertEqual(round(log(_Henry(600, "Kr")/1000), 4), 0.0006)
        self.assertEqual(round(log(_Henry(300, "Xe")/1000), 4), 0.2792)
        self.assertEqual(round(log(_Henry(400, "Xe")/1000), 4), 1.1430)
        self.assertEqual(round(log(_Henry(500, "Xe")/1000), 4), 0.5033)
        self.assertEqual(round(log(_Henry(600, "Xe")/1000), 4), -0.7081)
        self.assertEqual(round(log(_Henry(300, "H2")/1000), 4), 1.9702)
        self.assertEqual(round(log(_Henry(400, "H2")/1000), 4), 1.8464)
        self.assertEqual(round(log(_Henry(500, "H2")/1000), 4), 1.0513)
        self.assertEqual(round(log(_Henry(600, "H2")/1000), 4), -0.1848)
        self.assertEqual(round(log(_Henry(300, "N2")/1000), 4), 2.1716)
        self.assertEqual(round(log(_Henry(400, "N2")/1000), 4), 2.3509)
        self.assertEqual(round(log(_Henry(500, "N2")/1000), 4), 1.4842)
        self.assertEqual(round(log(_Henry(600, "N2")/1000), 4), 0.1647)
        self.assertEqual(round(log(_Henry(300, "O2")/1000), 4), 1.5024)
        self.assertEqual(round(log(_Henry(400, "O2")/1000), 4), 1.8832)
        self.assertEqual(round(log(_Henry(500, "O2")/1000), 4), 1.1630)
        self.assertEqual(round(log(_Henry(600, "O2")/1000), 4), -0.0276)
        self.assertEqual(round(log(_Henry(300, "CO")/1000), 4), 1.7652)
        self.assertEqual(round(log(_Henry(400, "CO")/1000), 4), 1.9939)
        self.assertEqual(round(log(_Henry(500, "CO")/1000), 4), 1.1250)
        self.assertEqual(round(log(_Henry(600, "CO")/1000), 4), -0.2382)
        self.assertEqual(round(log(_Henry(300, "CO2")/1000), 4), -1.7508)
        self.assertEqual(round(log(_Henry(400, "CO2")/1000), 4), -0.5450)
        self.assertEqual(round(log(_Henry(500, "CO2")/1000), 4), -0.6524)
        self.assertEqual(round(log(_Henry(600, "CO2")/1000), 4), -1.3489)
        self.assertEqual(round(log(_Henry(300, "H2S")/1000), 4), -2.8784)
        self.assertEqual(round(log(_Henry(400, "H2S")/1000), 4), -1.7083)
        self.assertEqual(round(log(_Henry(500, "H2S")/1000), 4), -1.6074)
        self.assertEqual(round(log(_Henry(600, "H2S")/1000), 4), -2.1319)
        self.assertEqual(round(log(_Henry(300, "CH4")/1000), 4), 1.4034)
        self.assertEqual(round(log(_Henry(400, "CH4")/1000), 4), 1.7946)
        self.assertEqual(round(log(_Henry(500, "CH4")/1000), 4), 1.0342)
        self.assertEqual(round(log(_Henry(600, "CH4")/1000), 4), -0.2209)
        self.assertEqual(round(log(_Henry(300, "C2H6")/1000), 4), 1.1418)
        self.assertEqual(round(log(_Henry(400, "C2H6")/1000), 4), 1.8495)
        self.assertEqual(round(log(_Henry(500, "C2H6")/1000), 4), 0.8274)
        self.assertEqual(round(log(_Henry(600, "C2H6")/1000), 4), -0.8141)
        self.assertEqual(round(log(_Henry(300, "SF6")/1000), 4), 3.1445)
        self.assertEqual(round(log(_Henry(400, "SF6")/1000), 4), 3.6919)
        self.assertEqual(round(log(_Henry(500, "SF6")/1000), 4), 2.6749)
        self.assertEqual(round(log(_Henry(600, "SF6")/1000), 4), 1.2402)
        self.assertEqual(round(log(_Henry(300, "He", "D2O")/1000), 4), 2.5756)
        self.assertEqual(round(log(_Henry(400, "He", "D2O")/1000), 4), 2.1215)
        self.assertEqual(round(log(_Henry(500, "He", "D2O")/1000), 4), 1.2748)
        self.assertEqual(round(log(_Henry(600, "He", "D2O")/1000), 4), -0.0034)
        self.assertEqual(round(log(_Henry(300, "Ne", "D2O")/1000), 4), 2.4421)
        self.assertEqual(round(log(_Henry(400, "Ne", "D2O")/1000), 4), 2.2525)
        self.assertEqual(round(log(_Henry(500, "Ne", "D2O")/1000), 4), 1.5554)
        self.assertEqual(round(log(_Henry(600, "Ne", "D2O")/1000), 4), 0.4664)
        self.assertEqual(round(log(_Henry(300, "Ar", "D2O")/1000), 4), 1.3316)
        self.assertEqual(round(log(_Henry(400, "Ar", "D2O")/1000), 4), 1.7490)
        self.assertEqual(round(log(_Henry(500, "Ar", "D2O")/1000), 4), 1.1312)
        self.assertEqual(round(log(_Henry(600, "Ar", "D2O")/1000), 4), 0.0360)
        self.assertEqual(round(log(_Henry(300, "Kr", "D2O")/1000), 4), 0.8015)
        self.assertEqual(round(log(_Henry(400, "Kr", "D2O")/1000), 4), 1.4702)
        self.assertEqual(round(log(_Henry(500, "Kr", "D2O")/1000), 4), 0.9505)
        self.assertEqual(round(log(_Henry(600, "Kr", "D2O")/1000), 4), -0.0661)
        self.assertEqual(round(log(_Henry(300, "Xe", "D2O")/1000), 4), 0.2750)
        self.assertEqual(round(log(_Henry(400, "Xe", "D2O")/1000), 4), 1.1251)
        self.assertEqual(round(log(_Henry(500, "Xe", "D2O")/1000), 4), 0.4322)
        self.assertEqual(round(log(_Henry(600, "Xe", "D2O")/1000), 4), -0.8730)
        self.assertEqual(round(log(_Henry(300, "D2", "D2O")/1000), 4), 1.6594)
        self.assertEqual(round(log(_Henry(400, "D2", "D2O")/1000), 4), 1.6762)
        self.assertEqual(round(log(_Henry(500, "D2", "D2O")/1000), 4), 0.9042)
        self.assertEqual(round(log(_Henry(600, "D2", "D2O")/1000), 4), -0.3665)
        self.assertEqual(round(log(_Henry(300, "CH4", "D2O")/1000), 4), 1.3624)
        self.assertEqual(round(log(_Henry(400, "CH4", "D2O")/1000), 4), 1.7968)
        self.assertEqual(round(log(_Henry(500, "CH4", "D2O")/1000), 4), 1.0491)
        self.assertEqual(round(log(_Henry(600, "CH4", "D2O")/1000), 4), -0.2186)

        # Table 7 for Kd
        self.assertRaises(NotImplementedError, _Kvalue, *(300, "He", "He"))
        self.assertRaises(NotImplementedError, _Kvalue, *(300, "SF6", "D2O"))
        self.assertEqual(round(log(_Kvalue(300, "He")), 4), 15.2250)
        self.assertEqual(round(log(_Kvalue(400, "He")), 4), 10.4364)
        self.assertEqual(round(log(_Kvalue(500, "He")), 4), 6.9971)
        self.assertEqual(round(log(_Kvalue(600, "He")), 4), 3.8019)
        self.assertEqual(round(log(_Kvalue(300, "Ne")), 4), 15.0743)
        self.assertEqual(round(log(_Kvalue(400, "Ne")), 4), 10.6379)
        self.assertEqual(round(log(_Kvalue(500, "Ne")), 4), 7.4116)
        self.assertEqual(round(log(_Kvalue(600, "Ne")), 4), 4.2308)
        self.assertEqual(round(log(_Kvalue(300, "Ar")), 4), 13.9823)
        self.assertEqual(round(log(_Kvalue(400, "Ar")), 4), 10.0558)
        self.assertEqual(round(log(_Kvalue(500, "Ar")), 4), 6.9869)
        self.assertEqual(round(log(_Kvalue(600, "Ar")), 4), 3.9861)
        self.assertEqual(round(log(_Kvalue(300, "Kr")), 4), 13.3968)
        self.assertEqual(round(log(_Kvalue(400, "Kr")), 4), 9.7362)
        self.assertEqual(round(log(_Kvalue(500, "Kr")), 4), 6.8371)
        self.assertEqual(round(log(_Kvalue(600, "Kr")), 4), 3.9654)
        self.assertEqual(round(log(_Kvalue(300, "Xe")), 4), 12.8462)
        self.assertEqual(round(log(_Kvalue(400, "Xe")), 4), 9.4268)
        self.assertEqual(round(log(_Kvalue(500, "Xe")), 4), 6.3639)
        self.assertEqual(round(log(_Kvalue(600, "Xe")), 4), 3.3793)
        self.assertEqual(round(log(_Kvalue(300, "H2")), 4), 14.5286)
        self.assertEqual(round(log(_Kvalue(400, "H2")), 4), 10.1484)
        self.assertEqual(round(log(_Kvalue(500, "H2")), 4), 6.8948)
        self.assertEqual(round(log(_Kvalue(600, "H2")), 4), 3.7438)
        self.assertEqual(round(log(_Kvalue(300, "N2")), 4), 14.7334)
        self.assertEqual(round(log(_Kvalue(400, "N2")), 4), 10.6221)
        self.assertEqual(round(log(_Kvalue(500, "N2")), 4), 7.2923)
        self.assertEqual(round(log(_Kvalue(600, "N2")), 4), 4.0333)
        self.assertEqual(round(log(_Kvalue(300, "O2")), 4), 14.0716)
        self.assertEqual(round(log(_Kvalue(400, "O2")), 4), 10.1676)
        self.assertEqual(round(log(_Kvalue(500, "O2")), 4), 6.9979)
        self.assertEqual(round(log(_Kvalue(600, "O2")), 4), 3.8707)
        self.assertEqual(round(log(_Kvalue(300, "CO")), 4), 14.3276)
        self.assertEqual(round(log(_Kvalue(400, "CO")), 4), 10.2573)
        self.assertEqual(round(log(_Kvalue(500, "CO")), 4), 7.1218)
        self.assertEqual(round(log(_Kvalue(600, "CO")), 4), 4.0880)
        self.assertEqual(round(log(_Kvalue(300, "CO2")), 4), 10.8043)
        self.assertEqual(round(log(_Kvalue(400, "CO2")), 4), 7.7705)
        self.assertEqual(round(log(_Kvalue(500, "CO2")), 4), 5.2123)
        self.assertEqual(round(log(_Kvalue(600, "CO2")), 4), 2.7293)
        self.assertEqual(round(log(_Kvalue(300, "H2S")), 4), 9.6846)
        self.assertEqual(round(log(_Kvalue(400, "H2S")), 4), 6.5840)
        self.assertEqual(round(log(_Kvalue(500, "H2S")), 4), 4.2781)
        self.assertEqual(round(log(_Kvalue(600, "H2S")), 4), 2.2200)
        self.assertEqual(round(log(_Kvalue(300, "CH4")), 4), 13.9659)
        self.assertEqual(round(log(_Kvalue(400, "CH4")), 4), 10.0819)
        self.assertEqual(round(log(_Kvalue(500, "CH4")), 4), 6.8559)
        self.assertEqual(round(log(_Kvalue(600, "CH4")), 4), 3.7238)
        self.assertEqual(round(log(_Kvalue(300, "C2H6")), 4), 13.7063)
        self.assertEqual(round(log(_Kvalue(400, "C2H6")), 4), 10.1510)
        self.assertEqual(round(log(_Kvalue(500, "C2H6")), 4), 6.8453)
        self.assertEqual(round(log(_Kvalue(600, "C2H6")), 4), 3.6493)
        self.assertEqual(round(log(_Kvalue(300, "SF6")), 4), 15.7067)
        self.assertEqual(round(log(_Kvalue(400, "SF6")), 4), 11.9887)
        self.assertEqual(round(log(_Kvalue(500, "SF6")), 4), 8.5550)
        self.assertEqual(round(log(_Kvalue(600, "SF6")), 4), 4.9599)
        self.assertEqual(round(log(_Kvalue(300, "He", "D2O")), 4), 15.2802)
        self.assertEqual(round(log(_Kvalue(400, "He", "D2O")), 4), 10.4217)
        self.assertEqual(round(log(_Kvalue(500, "He", "D2O")), 4), 7.0674)
        self.assertEqual(round(log(_Kvalue(600, "He", "D2O")), 4), 3.9539)
        self.assertEqual(round(log(_Kvalue(300, "Ne", "D2O")), 4), 15.1473)
        self.assertEqual(round(log(_Kvalue(400, "Ne", "D2O")), 4), 10.5331)
        self.assertEqual(round(log(_Kvalue(500, "Ne", "D2O")), 4), 7.3435)
        self.assertEqual(round(log(_Kvalue(600, "Ne", "D2O")), 4), 4.2800)
        self.assertEqual(round(log(_Kvalue(300, "Ar", "D2O")), 4), 14.0517)
        self.assertEqual(round(log(_Kvalue(400, "Ar", "D2O")), 4), 10.0632)
        self.assertEqual(round(log(_Kvalue(500, "Ar", "D2O")), 4), 6.9498)
        self.assertEqual(round(log(_Kvalue(600, "Ar", "D2O")), 4), 3.9094)
        self.assertEqual(round(log(_Kvalue(300, "Kr", "D2O")), 4), 13.5042)
        self.assertEqual(round(log(_Kvalue(400, "Kr", "D2O")), 4), 9.7854)
        self.assertEqual(round(log(_Kvalue(500, "Kr", "D2O")), 4), 6.8035)
        self.assertEqual(round(log(_Kvalue(600, "Kr", "D2O")), 4), 3.8160)
        self.assertEqual(round(log(_Kvalue(300, "Xe", "D2O")), 4), 12.9782)
        self.assertEqual(round(log(_Kvalue(400, "Xe", "D2O")), 4), 9.4648)
        self.assertEqual(round(log(_Kvalue(500, "Xe", "D2O")), 4), 6.3074)
        self.assertEqual(round(log(_Kvalue(600, "Xe", "D2O")), 4), 3.1402)
        self.assertEqual(round(log(_Kvalue(300, "D2", "D2O")), 4), 14.3520)
        self.assertEqual(round(log(_Kvalue(400, "D2", "D2O")), 4), 10.0178)
        self.assertEqual(round(log(_Kvalue(500, "D2", "D2O")), 4), 6.6975)
        self.assertEqual(round(log(_Kvalue(600, "D2", "D2O")), 4), 3.5590)
        self.assertEqual(round(log(_Kvalue(300, "CH4", "D2O")), 4), 14.0646)
        self.assertEqual(round(log(_Kvalue(400, "CH4", "D2O")), 4), 10.1013)
        self.assertEqual(round(log(_Kvalue(500, "CH4", "D2O")), 4), 6.9021)
        self.assertEqual(round(log(_Kvalue(600, "CH4", "D2O")), 4), 3.8126)

    def xest_Conductivity(self) -> None:
        """Selected values from table II"""
        self.assertEqual(round(_Conductivity(600, 673.15), 9), 1.57e-6)
        self.assertEqual(round(_Conductivity(800, 1073.15), 9), 103e-6)
        self.assertEqual(round(_Conductivity(900, 473.15), 9), 4.19e-6)
        self.assertEqual(round(_Conductivity(1100, 273.16), 9), 0.0333e-6)
        self.assertEqual(round(_Conductivity(1100, 473.15), 9), 22.8e-6)

    def test_virial(self) -> None:
        """Tables 7 & 8, page 10"""
        # Table 7, page 10
        st = _virial(200)
        self.assertEqual(round(st["Baa"], 13), -0.392722567e-4)
        self.assertEqual(round(st["Baw"], 13), -0.784874278e-4)
        self.assertEqual(round(st["Bww"], 10), -0.186282737e-1)
        self.assertEqual(round(st["Caaa"], 17), 0.227113063e-8)
        self.assertEqual(round(st["Caaw"], 17), 0.105493575e-8)
        self.assertEqual(round(st["Caww"], 14), -0.349872634e-5)
        self.assertEqual(round(st["Cwww"], 12), -0.263959706e-3)

        st = _virial(300)
        self.assertEqual(round(st["Baa"], 14), -0.776210977e-5)
        self.assertEqual(round(st["Baw"], 13), -0.295672747e-4)
        self.assertEqual(round(st["Bww"], 11), -0.120129928e-2)
        self.assertEqual(round(st["Caaa"], 17), 0.181166638e-8)
        self.assertEqual(round(st["Caaw"], 18), 0.801977741e-9)
        self.assertEqual(round(st["Caww"], 15), -0.115552784e-6)
        self.assertEqual(round(st["Cwww"], 14), -0.420419196e-5)

        st = _virial(400)
        self.assertEqual(round(st["Baa"], 14), 0.603953176e-5)
        self.assertEqual(round(st["Baw"], 13), -0.100804610e-4)
        self.assertEqual(round(st["Bww"], 12), -0.348784166e-3)
        self.assertEqual(round(st["Caaa"], 17), 0.162604635e-8)
        self.assertEqual(round(st["Caaw"], 18), 0.672018172e-9)
        self.assertEqual(round(st["Caww"], 16), -0.200806021e-7)
        self.assertEqual(round(st["Cwww"], 15), -0.217733298e-6)

        # Table 8, page 10
        self.assertEqual(round(_fugacity(300, 0.01, 0.1), 12), 0.000998917199)
        self.assertEqual(round(_fugacity(300, 0.01, 0.9), 11), 0.00895677892)
        self.assertEqual(round(_fugacity(300, 0.1, 0.1), 11), 0.00989090701)
        self.assertEqual(round(_fugacity(300, 0.1, 0.9), 9), 0.085431837)
        self.assertEqual(round(_fugacity(300, 1, 0.1), 9), 0.088406169)
        self.assertEqual(round(_fugacity(300, 1, 0.9), 8), 0.36007512)

        if major == 3:
            self.assertWarns(Warning, _virial, 50)
        self.assertRaises(NotImplementedError, _fugacity, *(190, 1, 0.1))

    def test_Air(self) -> None:
        """Tables A1 & A2, page 363 & 366."""
        # Table A1, Pag 363
        self.assertEqual(round(Air._bubbleP(59.75), 6), 0.005265)
        self.assertEqual(round(Air._bubbleP(59.75), 6), 0.005265)
        self.assertEqual(round(Air._dewP(59.75), 5), 0.00243)
        self.assertEqual(round(Air._bubbleP(70), 5), 0.03191)
        self.assertEqual(round(Air._dewP(70), 5), 0.01943)
        self.assertEqual(round(Air._bubbleP(80), 5), 0.11462)
        self.assertEqual(round(Air._dewP(80), 5), 0.08232)
        self.assertEqual(round(Air._bubbleP(100), 5), 0.66313)
        self.assertEqual(round(Air._dewP(100), 5), 0.56742)
        self.assertEqual(round(Air._bubbleP(120), 5), 2.15573)
        self.assertEqual(round(Air._dewP(120), 5), 2.00674)
        self.assertEqual(round(Air._bubbleP(130), 5), 3.42947)
        self.assertEqual(round(Air._dewP(130), 5), 3.30835)

        # Table A2, Pag 366
        st = Air(T=100, P=0.101325)
        self.assertEqual(round(st.rhoM, 5), 0.12449)
        self.assertEqual(round(st.cvM, 2), 21.09)
        self.assertEqual(round(st.cpM, 2), 30.13)
        self.assertEqual(round(st.w, 1), 198.2)

        st = Air(T=500, P=0.2)
        self.assertEqual(round(st.rhoM, 6), 0.048077)
        self.assertEqual(round(st.cvM, 2), 21.51)
        self.assertEqual(round(st.cpM, 2), 29.84)
        self.assertEqual(round(st.w, 1), 446.6)

        st = Air(T=130, P=1)
        self.assertEqual(round(st.rhoM, 4), 1.0295)
        self.assertEqual(round(st.cvM, 2), 22.05)
        self.assertEqual(round(st.cpM, 2), 34.69)
        self.assertEqual(round(st.w, 1), 216.8)

        st = Air(T=2000, P=10)
        self.assertEqual(round(st.rhoM, 5), 0.59094)
        self.assertEqual(round(st.cvM, 2), 27.93)
        self.assertEqual(round(st.cpM, 2), 36.25)
        self.assertEqual(round(st.w, 1), 878.5)

        st = Air(T=2000, P=500)
        self.assertEqual(round(st.rhoM, 2), 16.48)
        self.assertEqual(round(st.cvM, 2), 29.07)
        self.assertEqual(round(st.cpM, 2), 37.27)
        self.assertEqual(round(st.w, 1), 1497.4)

        # Zero point enthalpy and entropy
        st0 = Air(T=273.15, P=0.101325)
        self.assertEqual(round(st0.h, 9), 0)
        self.assertEqual(round(st0.s, 9), 0)

        # Custom cycle
        P = 50   # MPa
        T = 470  # K
        f_pt = Air(P=P, T=T)
        f_prho = Air(P=f_pt.P, rho=f_pt.rho)
        self.assertEqual(round(f_prho.P-P, 6), 0)
        self.assertEqual(round(f_prho.T-T, 6), 0)

    def test_AirTransport(self) -> None:
        """Table V, pag 28"""
        st = Air()
        self.assertEqual(round(st._visco(0, 100), 11), 7.09559e-6)
        self.assertEqual(round(st._visco(0, 300), 10), 18.523e-6)
        self.assertEqual(round(st._visco(28*28.9586, 100), 9), 107.923e-6)
        self.assertEqual(round(st._visco(10*28.9586, 200), 10), 21.1392e-6)
        self.assertEqual(round(st._visco(5*28.9586, 300), 10), 21.3241e-6)
        self.assertEqual(round(st._visco(10.4*28.9586, 132.64), 10), 17.7623e-6)

        self.assertEqual(round(st._thermo(0, 100), 8), 9.35902e-3)
        self.assertEqual(round(st._thermo(0, 300), 7), 26.3529e-3)
        self.assertEqual(round(Air(rho=28*28.9586, T=100).k, 6), 119.222e-3)
        self.assertEqual(round(Air(rho=10*28.9586, T=200).k, 7), 35.3186e-3)
        self.assertEqual(round(Air(rho=5*28.9586, T=300).k, 7), 32.6062e-3)
        # self.assertEqual(round(Air(rho=10.4*28.9586, T=132.64).k, 7), 75.6231e-3)

    def test_HumidAir(self) -> None:
        """Tables 13-15 from page 19"""
        # Table 13
        A = 0.892247719
        T = 200
        rho = 1.63479657e-5
        psy = HumidAir()
        fa = psy._fav(T, rho, A)
        self.assertEqual(round(fa["fir"], 6), -0.682093392e3)
        self.assertEqual(round(fa["fira"], 6), -0.572680404e3)
        self.assertEqual(round(fa["firt"], 8), -0.405317966e1)
        self.assertEqual(round(fa["fird"], 2), 0.374173101e7)
        self.assertEqual(round(fa["firaa"], 6), 0.920967684e3)
        self.assertEqual(round(fa["firat"], 8), 0.915653743e1)
        self.assertEqual(round(fa["firad"], 2), -0.213442099e7)
        self.assertEqual(round(fa["firtt"], 11), -0.394011921e-2)
        self.assertEqual(round(fa["firdt"], 4), 0.187087034e5)
        self.assertEqual(round(fa["firdd"]*1e-6, 3), -0.228880603e6)
        colig = psy._coligative(rho, A, fa)
        self.assertEqual(round(colig["muw"], 6), -0.109950917e3)
        prop = psy._prop(T, rho, fa)
        self.assertEqual(round(prop["P"], 15), 0.999999998e-6)
        self.assertEqual(round(prop["h"], 6), 0.189712231e3)
        self.assertEqual(round(prop["g"], 6), -0.620923701e3)
        self.assertEqual(round(prop["s"], 8), 0.405317966e1)
        self.assertEqual(round(prop["cp"], 8), 0.109387397e1)
        self.assertEqual(round(prop["w"], 6), 0.291394959e3)

        A = 0.977605798
        T = 300
        rho = 1.14614216
        psy = HumidAir()
        fa = psy._fav(T, rho, A)
        self.assertEqual(round(fa["fir"], 7), -0.927718178e2)
        self.assertEqual(round(fa["fira"], 9), -0.263453864)
        self.assertEqual(round(fa["firt"], 9), -0.296711481)
        self.assertEqual(round(fa["fird"], 7), 0.761242496e2)
        self.assertEqual(round(fa["firaa"], 5), 0.624886233e4)
        self.assertEqual(round(fa["firat"], 8), 0.822733446e1)
        self.assertEqual(round(fa["firad"], 7), -0.450004399e2)
        self.assertEqual(round(fa["firtt"], 11), -0.244742952e-2)
        self.assertEqual(round(fa["firdt"], 9), 0.254456302)
        self.assertEqual(round(fa["firdd"], 7), -0.664465525e2)
        colig = psy._coligative(rho, A, fa)
        self.assertEqual(round(colig["muw"], 8), -0.526505193e1)
        prop = psy._prop(T, rho, fa)
        self.assertEqual(round(prop["P"], 9), 0.1)
        self.assertEqual(round(prop["h"], 7), 0.834908383e2)
        self.assertEqual(round(prop["g"], 8), -0.552260595e1)
        self.assertEqual(round(prop["s"], 9), 0.296711481)
        self.assertEqual(round(prop["cp"], 8), 0.102681324e1)
        self.assertEqual(round(prop["w"], 6), 0.349234196e3)

        A = 0.825565291
        T = 400
        rho = 0.793354063e1
        psy = HumidAir()
        fa = psy._fav(T, rho, A)
        self.assertEqual(round(fa["fir"], 7), 0.240345570e2)
        self.assertEqual(round(fa["fira"], 6), 0.311096733e3)
        self.assertEqual(round(fa["firt"], 8), -0.106891931e1)
        self.assertEqual(round(fa["fird"], 7), 0.158878781e2)
        self.assertEqual(round(fa["firaa"], 5), 0.113786423e4)
        self.assertEqual(round(fa["firat"], 8), 0.702631471e1)
        self.assertEqual(round(fa["firad"], 8), -0.727972651e1)
        self.assertEqual(round(fa["firtt"], 11), -0.222449294e-2)
        self.assertEqual(round(fa["firdt"], 10), 0.414350772e-1)
        self.assertEqual(round(fa["firdd"], 8), -0.201886184e1)
        colig = psy._coligative(rho, A, fa)
        self.assertEqual(round(colig["muw"], 6), -0.106748981e3)
        prop = psy._prop(T, rho, fa)
        self.assertEqual(round(prop["P"], 8), 1)
        self.assertEqual(round(prop["h"], 6), 0.577649408e3)
        self.assertEqual(round(prop["g"], 6), 0.150081684e3)
        self.assertEqual(round(prop["s"], 8), 0.106891931e1)
        self.assertEqual(round(prop["cp"], 8), 0.123552454e1)
        self.assertEqual(round(prop["w"], 6), 0.416656820e3)

        # Table 14
        A = 0.892247719
        T = 200
        rho = 1.63479657e-5
        rhoa = A*rho
        rhov = (1-A)*rho
        air = Air()
        fa = air._derivDimensional(0, T)
        self.assertEqual(round(fa["fir"], 6), 0)

        fa = air._derivDimensional(rhoa, T)
        self.assertEqual(round(rhoa, 13), 1.45864351e-5)
        self.assertEqual(round(fa["fir"], 6), -0.740041144e3)
        self.assertEqual(round(fa["firt"], 8), -0.304774177e1)
        self.assertEqual(round(fa["fird"], 2), 0.393583654e7)
        self.assertEqual(round(fa["firtt"], 11), -0.357677878e-2)
        self.assertEqual(round(fa["firdt"], 4), 0.196791837e5)
        self.assertEqual(round(fa["firdd"]*1e-6, 3), -0.269828549e6)
        water = IAPWS95()
        fv = water._derivDimensional(rhov, T)
        self.assertEqual(round(rhov, 14), 1.76153059e-6)
        self.assertEqual(round(fv["fir"], 6), -0.202254351e3)
        self.assertEqual(round(fv["firt"], 7), -0.123787544e2)
        self.assertEqual(round(fv["fird"], 1), 0.523995674e8)
        self.assertEqual(round(fv["firtt"], 11), -0.694877601e-2)
        self.assertEqual(round(fv["firdt"], 3), 0.262001885e6)
        self.assertEqual(round(fv["firdd"]*1e-6, 1), -0.297466671e8)

        A = 0.977605798
        T = 300
        rho = 1.14614216
        rhoa = A*rho
        rhov = (1-A)*rho
        air = Air()
        fa = air._derivDimensional(rhoa, T)
        self.assertEqual(round(rhoa, 8), 1.12047522)
        self.assertEqual(round(fa["fir"], 7), -0.916103453e2)
        self.assertEqual(round(fa["firt"], 9), -0.108476220)
        self.assertEqual(round(fa["fird"], 7), 0.768326795e2)
        self.assertEqual(round(fa["firtt"], 11), -0.239319940e-2)
        self.assertEqual(round(fa["firdt"], 9), 0.256683306)
        self.assertEqual(round(fa["firdd"], 7), -0.685917373e2)
        water = IAPWS95()
        fv = water._derivDimensional(rhov, T)
        self.assertEqual(round(rhov, 10), 0.256669391e-1)
        self.assertEqual(round(fv["fir"], 6), -0.143157426e3)
        self.assertEqual(round(fv["firt"], 8), -0.851598213e1)
        self.assertEqual(round(fv["fird"], 5), 0.538480619e4)
        self.assertEqual(round(fv["firtt"], 11), -0.480817011e-2)
        self.assertEqual(round(fv["firdt"], 7), 0.181489502e2)
        self.assertEqual(round(fv["firdd"], 3), -0.210184992e6)

        A = 0.825565291
        T = 400
        rho = 0.793354063e1
        rhoa = A*rho
        rhov = (1-A)*rho
        air = Air()
        fa = air._derivDimensional(rhoa, T)
        self.assertEqual(round(rhoa, 8), 0.654965578e1)
        self.assertEqual(round(fa["fir"], 7), 0.895561286e2)
        self.assertEqual(round(fa["firt"], 9), 0.193271394)
        self.assertEqual(round(fa["fird"], 7), 0.175560114e2)
        self.assertEqual(round(fa["firtt"], 11), -0.181809877e-2)
        self.assertEqual(round(fa["firdt"], 10), 0.442769673e-1)
        self.assertEqual(round(fa["firdd"], 8), -0.267635928e1)
        water = IAPWS95()
        fv = water._derivDimensional(rhov, T)
        self.assertEqual(round(rhov, 8), 0.138388485e1)
        self.assertEqual(round(fv["fir"], 6), -0.285137534e3)
        self.assertEqual(round(fv["firt"], 8), -0.705288048e1)
        self.assertEqual(round(fv["fird"], 6), 0.129645039e3)
        self.assertEqual(round(fv["firtt"], 11), -0.411710659e-2)
        self.assertEqual(round(fv["firdt"], 9), 0.361784086)
        self.assertEqual(round(fv["firdd"], 7), -0.965539462e2)

        # Table 15
        A = 0.892247719
        T = 200
        rho = 1.63479657e-5
        hum = HumidAir()
        fmix = hum._fmix(T, rho, A)
        self.assertEqual(round(fmix["fir"], 15), -0.786231899e-6)
        self.assertEqual(round(fmix["fira"], 14), 0.641550398e-5)
        self.assertEqual(round(fmix["firt"], 17), 0.456438658e-8)
        self.assertEqual(round(fmix["fird"], 10), -0.480937188e-1)
        self.assertEqual(round(fmix["firaa"], 13), 0.163552956e-4)
        self.assertEqual(round(fmix["firat"], 16), -0.372455576e-7)
        self.assertEqual(round(fmix["firad"], 9), 0.392437132)
        self.assertEqual(round(fmix["firtt"], 19), -0.378875706e-10)
        self.assertEqual(round(fmix["firdt"], 12), 0.279209778e-3)
        self.assertEqual(round(fmix["firdd"], 10), -0.192042557e-1)
        vir = _virial(T)
        self.assertEqual(round(vir["Baw"], 13), -0.784874278e-4)
        self.assertEqual(round(vir["Bawt"], 15), 0.848076624e-6)
        self.assertEqual(round(vir["Bawtt"], 16), -0.122622146e-7)
        self.assertEqual(round(vir["Caaw"], 17), 0.105493575e-8)
        self.assertEqual(round(vir["Caawt"], 20), -0.152535e-11)
        self.assertEqual(round(vir["Caawtt"], 21), -0.113436375e-12)
        self.assertEqual(round(vir["Caww"], 14), -0.349872634e-5)
        self.assertEqual(round(vir["Cawwt"], 15), 0.188025052e-6)
        self.assertEqual(round(vir["Cawwtt"], 16), -0.124996856e-7)

        A = 0.977605798
        T = 300
        rho = 0.114614216e1
        hum = HumidAir()
        fmix = hum._fmix(T, rho, A)
        self.assertEqual(round(fmix["fir"], 11), -0.711677596e-2)
        self.assertEqual(round(fmix["fira"], 9), 0.311844020)
        self.assertEqual(round(fmix["firt"], 13), 0.441247962e-4)
        self.assertEqual(round(fmix["fird"], 11), -0.623030392e-2)
        self.assertEqual(round(fmix["firaa"], 9), 0.534234669)
        self.assertEqual(round(fmix["firat"], 11), -0.195073372e-2)
        self.assertEqual(round(fmix["firad"], 9), 0.274155508)
        self.assertEqual(round(fmix["firtt"], 15), -0.148783177e-6)
        self.assertEqual(round(fmix["firdt"], 13), 0.390012443e-4)
        self.assertEqual(round(fmix["firdd"], 13), -0.365975429e-4)
        vir = _virial(T)
        self.assertEqual(round(vir["Baw"], 13), -0.295672747e-4)
        self.assertEqual(round(vir["Bawt"], 15), 0.280097360e-6)
        self.assertEqual(round(vir["Bawtt"], 17), -0.242599241e-8)
        self.assertEqual(round(vir["Caaw"], 18), 0.801977741e-9)
        self.assertEqual(round(vir["Caawt"], 20), -0.196103457e-11)
        self.assertEqual(round(vir["Caawtt"], 22), 0.170055638e-13)
        self.assertEqual(round(vir["Caww"], 15), -0.115552784e-6)
        self.assertEqual(round(vir["Cawwt"], 17), 0.261363278e-8)
        self.assertEqual(round(vir["Cawwtt"], 19), -0.751334582e-10)

        A = 0.825565291
        T = 400
        rho = 0.793354063e1
        hum = HumidAir()
        fmix = hum._fmix(T, rho, A)
        self.assertEqual(round(fmix["fir"], 9), -0.161991543)
        self.assertEqual(round(fmix["fira"], 9), 0.831044354)
        self.assertEqual(round(fmix["firt"], 11), 0.178968942e-2)
        self.assertEqual(round(fmix["fird"], 10), -0.223330257e-1)
        self.assertEqual(round(fmix["firaa"], 8), 0.135814949e1)
        self.assertEqual(round(fmix["firat"], 11), -0.916854756e-2)
        self.assertEqual(round(fmix["firad"], 9), 0.125834930)
        self.assertEqual(round(fmix["firtt"], 14), -0.536741578e-5)
        self.assertEqual(round(fmix["firdt"], 12), 0.249580143e-3)
        self.assertEqual(round(fmix["firdd"], 12), -0.482623664e-3)
        vir = _virial(T)
        self.assertEqual(round(vir["Baw"], 13), -0.100804610e-4)
        self.assertEqual(round(vir["Bawt"], 15), 0.135021228e-6)
        self.assertEqual(round(vir["Bawtt"], 18), -0.839901729e-9)
        self.assertEqual(round(vir["Caaw"], 18), 0.672018172e-9)
        self.assertEqual(round(vir["Caawt"], 21), -0.812416406e-12)
        self.assertEqual(round(vir["Caawtt"]*1e12, 11), 0.683147461e-2)
        self.assertEqual(round(vir["Caww"], 16), -0.200806021e-7)
        self.assertEqual(round(vir["Cawwt"], 18), 0.274535403e-9)
        self.assertEqual(round(vir["Cawwtt"], 20), -0.491763910e-11)

        # Humid air class custom testing
        hum = HumidAir(T=300, P=0.1, A=0.977605798)
        hum2 = HumidAir(T=hum.T, P=hum.P, W=hum.W)
        hum3 = HumidAir(T=hum2.T, P=hum2.P, xa=hum2.xa)
        hum4 = HumidAir(T=hum3.T, P=hum3.P, xw=hum3.xw)
        hum5 = HumidAir(P=hum4.P, rho=hum4.rho, xw=hum4.xw)
        hum6 = HumidAir(T=hum5.T, rho=hum5.rho, xw=hum5.xw)
        hum7 = HumidAir(T=hum6.T, v=hum6.v, xw=hum6.xw)
        self.assertEqual(round(hum.cp-hum7.cp, 8), 0)

        hum = HumidAir(T=200, rho=1.63479657e-5, A=0.892247719)
        self.assertEqual(round(hum.cp, 8), 1.09387397)
        self.assertEqual(round(hum.cp-200*hum.derivative("s", "T", "P"), 8), 0)

    def test_Ammonia(self) -> None:
        """Selected point front table of pag 42"""
        st = NH3(T=-77.65+273.15, x=0.5)
        self.assertEqual(round(st.P, 5), 0.00609)
        self.assertEqual(round(st.Liquid.rho, 2), 732.90)
        self.assertEqual(round(st.Gas.rho, 4), 0.0641)
        # self.assertEqual(round(st.Liquid.h, 2), -143.14)
        self.assertEqual(round(st.Hvap, 1), 1484.4)
        self.assertEqual(round(st.Gas.h, 1), 1341.2)
        self.assertEqual(round(st.Liquid.s, 4), -0.4715)
        # self.assertEqual(round(st.Svap, 4), 7.5928)
        # self.assertEqual(round(st.Gas.s, 4), 7.1213)

        st = NH3(T=273.15, x=0.5)
        self.assertEqual(round(st.P, 5), 0.42938)
        self.assertEqual(round(st.Liquid.rho, 2), 638.57)
        self.assertEqual(round(st.Gas.rho, 4), 3.4567)
        self.assertEqual(round(st.Liquid.h, 2), 200.00)
        self.assertEqual(round(st.Hvap, 1), 1262.2)
        self.assertEqual(round(st.Gas.h, 1), 1462.2)
        self.assertEqual(round(st.Liquid.s, 4), 1.0000)
        self.assertEqual(round(st.Svap, 4), 4.6210)
        self.assertEqual(round(st.Gas.s, 4), 5.6210)

        st = NH3(T=125+273.15, x=0.5)
        self.assertEqual(round(st.P, 5), 9.97022)
        self.assertEqual(round(st.Liquid.rho, 2), 357.80)
        self.assertEqual(round(st.Gas.rho, 2), 120.73)
        self.assertEqual(round(st.Liquid.h, 2), 919.68)
        self.assertEqual(round(st.Hvap, 2), 389.44)
        self.assertEqual(round(st.Gas.h, 1), 1309.1)
        self.assertEqual(round(st.Liquid.s, 4), 3.0702)
        self.assertEqual(round(st.Svap, 4), 0.9781)
        self.assertEqual(round(st.Gas.s, 4), 4.0483)

        st = NH3(P=1, x=0.5)
        self.assertEqual(round(st.T-273.15, 2), 24.89)
        self.assertEqual(round(st.Liquid.rho, 2), 602.92)
        # self.assertEqual(round(st.Gas.rho, 4), 7.7821)
        self.assertEqual(round(st.Liquid.h, 2), 317.16)
        self.assertEqual(round(st.Hvap, 1), 1166.2)
        self.assertEqual(round(st.Gas.h, 1), 1483.4)
        self.assertEqual(round(st.Liquid.s, 4), 1.4072)
        # self.assertEqual(round(st.Svap, 4), 3.9129)
        self.assertEqual(round(st.Gas.s, 4), 5.3200)

        if major == 3:
            self.assertWarns(Warning, st._thermo, *(235, st.Tc, st))

    def test_AmmoniaVisco(self) -> None:
        """Appendix II & III, page 1664 & 1667."""
        # Appendix II, pag 1664
        st = NH3(T=680, P=0.1)
        self.assertEqual(round(st.mu*1e6, 2), 24.66)
        st = NH3(T=290, P=1)
        self.assertEqual(round(st.mu*1e6, 2), 142.93)
        st = NH3(T=680, P=50)
        self.assertEqual(round(st.mu*1e6, 2), 31.90)

        # Appendix III, pag 1667
        st = NH3(T=196, x=0.5)
        self.assertEqual(round(st.P, 4), 0.0063)
        self.assertEqual(round(st.Gas.rhoM, 4), 0.0039)
        self.assertEqual(round(st.Gas.mu*1e6, 2), 6.85)
        self.assertEqual(round(st.Liquid.rhoM, 4), 43.0041)
        self.assertEqual(round(st.Liquid.mu*1e6, 2), 553.31)

        st = NH3(T=300, x=0.5)
        self.assertEqual(round(st.P, 4), 1.0617)
        self.assertEqual(round(st.Gas.rhoM, 4), 0.4845)
        self.assertEqual(round(st.Gas.mu*1e6, 2), 9.89)
        self.assertEqual(round(st.Liquid.rhoM, 4), 35.2298)
        self.assertEqual(round(st.Liquid.mu*1e6, 2), 129.33)

        st = NH3(T=402, x=0.5)
        self.assertEqual(round(st.P, 4), 10.6777)
        self.assertEqual(round(st.Gas.rhoM, 4), 8.5479)
        self.assertEqual(round(st.Gas.mu*1e6, 2), 19.69)
        self.assertEqual(round(st.Liquid.rhoM, 4), 19.1642)
        self.assertEqual(round(st.Liquid.mu*1e6, 2), 39.20)

    def test_nh3h2o(self) -> None:
        """Test outstanding problems in H2ONH3."""
        # Range of validity
        Tt1 = Ttr(0)
        Tt2 = Ttr(0.5)
        Tt3 = Ttr(0.7)
        Tt4 = Ttr(0.9)
        if major == 3:
            self.assertGreater(Tt1, Tt2)
            self.assertGreater(Tt3, Tt4)

        self.assertRaises(NotImplementedError, Ttr, 1.1)

        cl = H2ONH3()
        # Pure fluid reference state
        water = IAPWS95(T=IAPWS95.Tt, x=0)
        st = cl._prop(water.rho, IAPWS95.Tt, 0)
        # self.assertEqual(round(st["u"], 5), 0)
        # self.assertEqual(round(st["s"], 5), 0)

        # Suppress flake8 error until tests are working.
        self.assertIsInstance(st["u"], float)

        nh3 = NH3(T=NH3.Tt, x=0)
        st = cl._prop(nh3.rho, NH3.Tt, 1)
        # self.assertEqual(round(st["u"], 5), 0)
        # self.assertEqual(round(st["s"], 5), 0)

        # Table 6
        x = 0.1
        rhoM = 35.0
        M = (1-x)*IAPWS95.M+x*NH3.M
        rho = rhoM*M
        st = cl._prop(rho, 600, x)
        # FIXME: The values are good, bad difer by 1%, a error I can find
        # In Pressure happen and only use fird
        # self.assertEqual(round(st["a"]*M, 7), -13734.1763)
        # self.assertEqual(round(st["P"], 7), 32.1221333)
        # self.assertEqual(round(st["cv"]*M, 7), 53.3159544)
        # self.assertEqual(round(st["w"], 6), 883.925596)

        x = 0.1
        rhoM = 4.0
        M = (1-x)*IAPWS95.M+x*NH3.M
        rho = rhoM*M
        st = cl._prop(rho, 600, x)
        # self.assertEqual(round(st["a"]*M, 7), -16991.6697)
        # self.assertEqual(round(st["P"], 7), 12.7721090)
        # self.assertEqual(round(st["cv"]*M, 7), 52.7644553)
        # self.assertEqual(round(st["w"], 6), 471.762394)

        x = 0.5
        rhoM = 32.0
        M = (1-x)*IAPWS95.M+x*NH3.M
        rho = rhoM*M
        st = cl._prop(rho, 500, x)
        # self.assertEqual(round(st["a"]*M, 7), -12109.5369)
        # self.assertEqual(round(st["P"], 7), 21.3208159)
        # self.assertEqual(round(st["cv"]*M, 7), 58.0077346)
        # self.assertEqual(round(st["w"], 6), 830.295833)

        x = 0.5
        rhoM = 1.0
        M = (1-x)*IAPWS95.M+x*NH3.M
        rho = rhoM*M
        st = cl._prop(rho, 500, x)
        # self.assertEqual(round(st["a"]*M, 7), -18281.3020)
        # self.assertEqual(round(st["P"], 7), 3.6423080)
        # self.assertEqual(round(st["cv"]*M, 7), 36.8228098)
        # self.assertEqual(round(st["w"], 6), 510.258362)

        x = 0.9
        rhoM = 30.0
        M = (1-x)*IAPWS95.M+x*NH3.M
        rho = rhoM*M
        st = cl._prop(rho, 400, x)
        # self.assertEqual(round(st["a"]*M, 7), -6986.4869)
        # self.assertEqual(round(st["P"], 7), 22.2830797)
        # self.assertEqual(round(st["cv"]*M, 7), 51.8072415)
        # self.assertEqual(round(st["w"], 6), 895.748711)

        x = 0.9
        rhoM = 0.5
        M = (1-x)*IAPWS95.M+x*NH3.M
        rho = rhoM*M
        st = cl._prop(rho, 400, x)
        # self.assertEqual(round(st["a"]*M, 7), -13790.6278)
        # self.assertEqual(round(st["P"], 7), 1.5499708)
        # self.assertEqual(round(st["cv"]*M, 7), 32.9703870)
        # self.assertEqual(round(st["w"], 6), 478.608147)


if __name__ == "__main__":
    if major == 2 and minor == 6:
        unittest.main()
    else:
        unittest.main(verbosity=2)
