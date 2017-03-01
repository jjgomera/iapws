#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from iapws.iapws08 import SeaWater
from iapws._iapws import (_Ice, _Sublimation_Pressure, _Melting_Pressure,
                          _Viscosity, _ThCond, _Tension, _Kw, _Liquid,
                          _D2O_Viscosity, _D2O_ThCond, _D2O_Tension)


# Test
class Test(unittest.TestCase):
    """
    Global unittest for module
    Run for python2 and python3 before to release distribution
    """
    def test_Helmholtz(self):
        """Table 6 from IAPWS95, pag 14"""
        T = 500
        rho = 838.025
        fluid = IAPWS95()
        delta = rho/fluid.rhoc
        tau = fluid.Tc/T

        fio, fiot, fiott, fiod, fiodd, fiodt = fluid._phi0(tau, delta)
        self.assertEqual(round(fio, 8), 2.04797733)
        self.assertEqual(round(fiod, 9), 0.384236747)
        self.assertEqual(round(fiodd, 9), -0.147637878)
        self.assertEqual(round(fiot, 8), 9.04611106)
        self.assertEqual(round(fiott, 8), -1.93249185)
        self.assertEqual(round(fiodt, 8), 0.0)

        fir, firt, firtt, fird, firdd, firdt, firdtt, B, C = fluid._phir(tau, delta)
        self.assertEqual(round(fir, 8), -3.42693206)
        self.assertEqual(round(fird, 9), -0.364366650)
        self.assertEqual(round(firdd, 9), 0.856063701)
        self.assertEqual(round(firt, 8), -5.81403435)
        self.assertEqual(round(firtt, 8), -2.23440737)
        self.assertEqual(round(firdt, 8), -1.12176915)

    def test_phase(self):
        """Table 7 from IAPWS95, pag 14"""
        fluid = IAPWS95()

        state = fluid._Helmholtz(996.556, 300)
        # See footnote for imprecise P value in last significant figures
        self.assertEqual(round(state["P"], 7), 99.241835)
        self.assertEqual(round(state["cv"], 8), 4.13018112)
        self.assertEqual(round(state["w"], 5), 1501.51914)
        self.assertEqual(round(state["s"], 9), 0.393062643)

        state = fluid._Helmholtz(1005.308, 300)
        self.assertEqual(round(state["P"], 4), 20002.2515)
        self.assertEqual(round(state["cv"], 8), 4.06798347)
        self.assertEqual(round(state["w"], 5), 1534.92501)
        self.assertEqual(round(state["s"], 9), 0.387405401)

        state = fluid._Helmholtz(1188.202, 300)
        self.assertEqual(round(state["P"], 3), 700004.704)
        self.assertEqual(round(state["cv"], 8), 3.46135580)
        self.assertEqual(round(state["w"], 5), 2443.57992)
        self.assertEqual(round(state["s"], 9), 0.132609616)

        state = fluid._Helmholtz(0.435, 500)
        self.assertEqual(round(state["P"], 7), 99.9679423)
        self.assertEqual(round(state["cv"], 8), 1.50817541)
        self.assertEqual(round(state["w"], 6), 548.314253)
        self.assertEqual(round(state["s"], 8), 7.94488271)

        state = fluid._Helmholtz(4.532, 500)
        self.assertEqual(round(state["P"], 6), 999.938125)
        self.assertEqual(round(state["cv"], 8), 1.66991025)
        self.assertEqual(round(state["w"], 6), 535.739001)
        self.assertEqual(round(state["s"], 8), 6.82502725)

        state = fluid._Helmholtz(838.025, 500)
        self.assertEqual(round(state["P"], 4), 10000.3858)
        self.assertEqual(round(state["cv"], 8), 3.22106219)
        self.assertEqual(round(state["w"], 5), 1271.28441)
        self.assertEqual(round(state["s"], 8), 2.56690919)

        state = fluid._Helmholtz(1084.564, 500)
        self.assertEqual(round(state["P"], 3), 700000.405)
        self.assertEqual(round(state["cv"], 8), 3.07437693)
        self.assertEqual(round(state["w"], 5), 2412.00877)
        self.assertEqual(round(state["s"], 8), 2.03237509)

        state = fluid._Helmholtz(358., 647)
        self.assertEqual(round(state["P"], 4), 22038.4756)
        self.assertEqual(round(state["cv"], 8), 6.18315728)
        self.assertEqual(round(state["w"], 6), 252.145078)
        self.assertEqual(round(state["s"], 8), 4.32092307)

        state = fluid._Helmholtz(0.241, 900)
        self.assertEqual(round(state["P"], 6), 100.062559)
        self.assertEqual(round(state["cv"], 8), 1.75890657)
        self.assertEqual(round(state["w"], 6), 724.027147)
        self.assertEqual(round(state["s"], 8), 9.16653194)

        state = fluid._Helmholtz(52.615, 900)
        self.assertEqual(round(state["P"], 3), 20000.069)
        self.assertEqual(round(state["cv"], 8), 1.93510526)
        self.assertEqual(round(state["w"], 6), 698.445674)
        self.assertEqual(round(state["s"], 8), 6.59070225)

        state = fluid._Helmholtz(870.769, 900)
        self.assertEqual(round(state["P"], 3), 700000.006)
        self.assertEqual(round(state["cv"], 8), 2.66422350)
        self.assertEqual(round(state["w"], 5), 2019.33608)
        self.assertEqual(round(state["s"], 8), 4.17223802)

    def test_saturation(self):
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

    def test_Melting(self):
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

    def test_Viscosity_1(self):
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

    def test_Viscosity_2(self):
        """Table 5, pag 9"""
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

    def test_ThCond_1(self):
        """Table 4, pag 10"""
        self.assertEqual(round(_ThCond(0, 298.15)*1000, 7), 18.4341883)
        self.assertEqual(round(_ThCond(998, 298.15)*1000, 6), 607.712868)
        self.assertEqual(round(_ThCond(1200, 298.15)*1000, 6), 799.038144)
        self.assertEqual(round(_ThCond(0, 873.15)*1000, 7), 79.1034659)

    def test_ThCond_2(self):
        """Table 5, pag 10"""
        fluid = IAPWS95(rho=1, T=647.35)
        self.assertEqual(round(fluid.k*1000, 7), 51.9298924)
        fluid = IAPWS95(rho=122, T=647.35)
        self.assertEqual(round(fluid.k*1000, 6), 130.922885)
        fluid = IAPWS95(rho=222, T=647.35)
        self.assertEqual(round(fluid.k*1000, 6), 367.787459)
        fluid = IAPWS95(rho=272, T=647.35)
        self.assertEqual(round(fluid.k*1000, 6), 757.959776)
        fluid = IAPWS95(rho=322, T=647.35)
        self.assertEqual(round(fluid.k*1000, 5), 1443.75556)
        fluid = IAPWS95(rho=372, T=647.35)
        self.assertEqual(round(fluid.k*1000, 6), 650.319402)
        fluid = IAPWS95(rho=422, T=647.35)
        self.assertEqual(round(fluid.k*1000, 6), 448.883487)
        fluid = IAPWS95(rho=750, T=647.35)
        self.assertEqual(round(fluid.k*1000, 6), 600.961346)

    def test_Tension(self):
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

    def test_Dielect(self):
        """Table 4, pag 8"""
        fluid = IAPWS95(P=0.101325, T=240)
        self.assertEqual(round(fluid.epsilon, 5), 104.34982)
        fluid = IAPWS95(P=0.101325, T=300)
        self.assertEqual(round(fluid.epsilon, 5), 77.74735)
        fluid = IAPWS95(P=10, T=300)
        self.assertEqual(round(fluid.epsilon, 5), 78.11269)
        fluid = IAPWS95(P=1000, T=300)
        self.assertEqual(round(fluid.epsilon, 5), 103.69632)
        fluid = IAPWS95(P=10, T=650)
        self.assertEqual(round(fluid.epsilon, 5), 1.26715)
        fluid = IAPWS95(P=100, T=650)
        self.assertEqual(round(fluid.epsilon, 5), 17.71733)
        fluid = IAPWS95(P=500, T=650)
        self.assertEqual(round(fluid.epsilon, 5), 26.62132)
        fluid = IAPWS95(P=10, T=870)
        self.assertEqual(round(fluid.epsilon, 5), 1.12721)
        fluid = IAPWS95(P=100, T=870)
        self.assertEqual(round(fluid.epsilon, 5), 4.98281)
        fluid = IAPWS95(P=500, T=870)
        self.assertEqual(round(fluid.epsilon, 5), 15.09746)

    def test_Refractive(self):
        """Selected values from table 3, pag 6"""
        fluid = IAPWS95(P=0.1, T=273.15, l=0.2265)
        self.assertEqual(round(fluid.n, 6), 1.394527)
        fluid = IAPWS95(P=10., T=273.15, l=0.2265)
        self.assertEqual(round(fluid.n, 6), 1.396526)
        fluid = IAPWS95(P=1., T=373.15, l=0.2265)
        self.assertEqual(round(fluid.n, 6), 1.375622)
        fluid = IAPWS95(P=100., T=373.15, l=0.2265)
        self.assertEqual(round(fluid.n, 6), 1.391983)
        fluid = IAPWS95(P=0.1, T=473.15, l=0.589)
        self.assertEqual(round(fluid.n, 7), 1.0001456)
        fluid = IAPWS95(P=1., T=773.15, l=0.589)
        self.assertEqual(round(fluid.n, 7), 1.0008773)
        fluid = IAPWS95(P=10., T=273.15, l=1.01398)
        self.assertEqual(round(fluid.n, 6), 1.327710)
        fluid = IAPWS95(P=100., T=473.15, l=1.01398)
        self.assertEqual(round(fluid.n, 6), 1.298369)

    def test_kw(self):
        """Table 3, pag 5"""
        self.assertRaises(NotImplementedError, _Kw, *(1000, 270))
        self.assertEqual(round(_Kw(1000, 300), 6), 13.906565)
        self.assertEqual(round(_Kw(70, 600), 6), 21.048874)
        self.assertEqual(round(_Kw(700, 600), 6), 11.203153)
        self.assertEqual(round(_Kw(200, 800), 6), 15.089765)
        self.assertEqual(round(_Kw(1200, 800), 6), 6.438330)

    def test_liquid(self):
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

        self.assertWarns(Warning, _Liquid, *(375, 0.2))
        self.assertRaises(NotImplementedError, _Liquid, *(375, 0.4))

    def test_IAPWS97_1(self):
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

    def test_IAPWS97_2(self):
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

    def test_IAPWS97_3(self):
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

    def test_IAPWS97_3_Sup03(self):
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

    def test_IAPWS97_3_Sup04(self):
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

    def test_IAPWS97_3_Sup05(self):
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

    def test_IAPWS97_4(self):
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

    def test_IAPWS97_5(self):
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

    def test_IAPWS97_custom(self):
        """Cycle input parameter from selected point for IAPWS97"""
        # Region 1
        P = 50   # MPa
        T = 470  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20   # MPa
        T = 370  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Region 2
        P = 25   # MPa
        T = 700  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 10   # MPa
        T = 700  # K
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

        P = 0.01   # MPa
        T = 1000  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 2   # MPa
        T = 1000  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # Region 3
        P = 50   # MPa
        T = 700  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20   # MPa
        s = 4  # kJ/kgK
        f_ps = IAPWS97(P=P, s=s)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_pt = IAPWS97(P=f_ph.P, T=f_ph.T)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.s-s, 6), 0)

        P = 19   # MPa
        f_px = IAPWS97(P=P, x=0)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 3), 0)
        self.assertEqual(round(f_tx.x, 6), 0)
        P = 19   # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 3), 0)
        self.assertEqual(round(f_tx.x, 6), 1)

        P = 21   # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 1)

        P = 21.5   # MPa
        f_px = IAPWS97(P=P, x=0)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 0)
        P = 21.5   # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 1)

        P = 22.02   # MPa
        f_px = IAPWS97(P=P, x=0)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 0)
        P = 22.02   # MPa
        f_px = IAPWS97(P=P, x=1)
        f_tx = IAPWS97(T=f_px.T, x=f_px.x)
        self.assertEqual(round(f_tx.P-P, 2), 0)
        self.assertEqual(round(f_tx.x, 3), 1)

        P = 24.   # MPa
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

        P = 17  # MPa
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
        P = 25   # MPa
        T = 1100  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 10   # MPa
        T = 1100  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        P = 20   # MPa
        T = 1100  # K
        f_pt = IAPWS97(P=P, T=T)
        f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
        f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
        f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 6), 0)
        self.assertEqual(round(f_hs.T-T, 6), 0)

        # P-T subregion 3
        P = [17, 21, 21, 21, 21, 22, 23.2, 23.2, 23.2, 23.2, 23.2, 23.2, 23.2,
             23, 23, 22.2, 22.065, 22.065, 22.065, 22.065, 21.8, 22]
        T = [625, 625, 640, 643, 645, 630, 640, 650, 651, 652, 653, 656, 660,
             640, 652, 647, 646, 647.05, 647.1, 647.2, 647, 647]
        for p, t in zip(P, T):
            f_pt = IAPWS97(P=p, T=t)
            f_ph = IAPWS97(h=f_pt.h, P=f_pt.P)
            f_ps = IAPWS97(P=f_ph.P, s=f_ph.s)
            f_hs = IAPWS97(h=f_ps.h, s=f_ps.s)
            self.assertEqual(round(f_hs.P-p, 6), 0)
            self.assertEqual(round(f_hs.T-t, 6), 0)

        # Other h-s region
        h = [2700, 2700, 1500, 2500, 2000, 2000, 3000, 2400, 2500, 2850, 2600]
        s = [5.15, 5.87, 3.5, 5, 5.5, 7, 6, 5.1, 5.05, 5.25, 5.25]
        for H, S in zip(h, s):
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

    def test_IAPWS95_custom(self):
        """Cycle input parameter from selected point for IAPWS95"""
        P = 50   # MPa
        T = 470  # K
        f_pt = IAPWS95_PT(P, T)
        f_ph = IAPWS95_Ph(f_pt.P, f_pt.h)
        f_ps = IAPWS95_Ps(f_ph.P, f_ph.s)
        f_hs = IAPWS95(h=f_ps.h, s=f_ps.s)
        self.assertEqual(round(f_hs.P-P, 5), 0)
        self.assertEqual(round(f_hs.T-T, 5), 0)

        P = 2   # MPa
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

        P = 50   # MPa
        T = 770  # K
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

        P = 0.1   # MPa
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

        P = 0.1   # MPa
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

        P = 2   # MPa
        f_px = IAPWS95_Px(P, 0)
        f_tx = IAPWS95_Tx(f_px.T, f_px.x)
        self.assertEqual(round(f_tx.P-P, 5), 0)
        f_px = IAPWS95_Px(P, 1)
        f_tx = IAPWS95_Tx(f_px.T, f_px.x)
        self.assertEqual(round(f_tx.P-P, 5), 0)

        P = 0.1   # MPa
        T = 300  # K
        f_pt = D2O(P=P, T=T)
        self.assertEqual(round(f_pt.P-P, 5), 0)
        self.assertEqual(round(f_pt.T-T, 5), 0)

        self.assertRaises(NotImplementedError, IAPWS95, **{"T": 700, "x": 0})
        self.assertRaises(NotImplementedError, IAPWS95, **{"P": 25, "x": 1})

    def xest_D2O(self):
        """Table 5 pag 11"""
        fluid = D2O()
        Tr = 643.847
        rhor = 358
        ar = 21.671*1000/358
        sr = 21.671*1000/358./643.847
        pr = 21.671*1000

        state = fluid._Helmholtz(0.0002*rhor, 0.5*Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -2.644979)
        self.assertEqual(round(state["P"]/pr, 7), 0.0004402)
        self.assertEqual(round(state["cv"]/sr, 4), 14.2768)

        state = fluid._Helmholtz(3.18*rhor, 0.5*Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -0.217388)
        self.assertEqual(round(state["P"]/pr, 7), 4.3549719)
        self.assertEqual(round(state["cv"]/sr, 4), 41.4463)

        state = fluid._Helmholtz(0.0295*rhor, 0.75*Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -7.272543)
        self.assertEqual(round(state["P"]/pr, 7), 0.0870308)
        self.assertEqual(round(state["cv"]/sr, 4), 20.1586)

        state = fluid._Helmholtz(2.83*rhor, 0.75*Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -4.292707)
        self.assertEqual(round(state["P"]/pr, 7), 4.4752958)
        self.assertEqual(round(state["cv"]/sr, 4), 33.4367)

        state = fluid._Helmholtz(0.3*rhor, Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -15.163326)
        self.assertEqual(round(state["P"]/pr, 7), 0.8014044)
        self.assertEqual(round(state["cv"]/sr, 4), 30.8587)

        state = fluid._Helmholtz(1.55*rhor, Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -12.643811)
        self.assertEqual(round(state["P"]/pr, 7), 1.0976283)
        self.assertEqual(round(state["cv"]/sr, 4), 33.0103)

        state = fluid._Helmholtz(0.4*rhor, 1.2*Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -25.471535)
        self.assertEqual(round(state["P"]/pr, 7), 1.4990994)
        self.assertEqual(round(state["cv"]/sr, 4), 23.6594)

        state = fluid._Helmholtz(1.61*rhor, 1.2*Tr)
        self.assertEqual(round((state["h"]-state["P"]*1000*state["v"]-state["T"]*state["s"])/ar, 6), -21.278164)
        self.assertEqual(round(state["P"]/pr, 7), 4.5643798)
        self.assertEqual(round(state["cv"]/sr, 4), 25.4800)

    def test_D2O_Viscosity(self):
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

    def test_D2O_ThCond(self):
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

    def test_D2O_Tension(self):
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

    def test_Ice(self):
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
        self.assertEqual(round(ice["kt"], 15), 0.000117793449348)
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
        self.assertEqual(round(ice["kt"], 15), 0.000117785291765)
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
        self.assertEqual(round(ice["kt"], 16), 0.0000886880048115)
        self.assertEqual(round(ice["ks"], 16), 0.0000886060982687)

        # Test check input
        self.assertRaises(NotImplementedError, _Ice, *(270, 300))
        self.assertRaises(NotImplementedError, _Ice, *(300, 1))
        self.assertRaises(NotImplementedError, _Ice, *(273, 3))
        self.assertRaises(NotImplementedError, _Ice, *(272, 1e-4))

    def test_SeaWater(self):
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
        self.assertEqual(round(state["gpp"], 15), -0.474672819e-6)

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

    def test_SeaWater_supp(self):
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


if __name__ == "__main__":
    major = sys.version_info[0]
    minor = sys.version_info[1]
    if major == 2 and minor == 6:
        unittest.main()
    else:
        unittest.main(verbosity=2)
