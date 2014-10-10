#!/usr/bin/python
# -*- coding: utf-8 -*-

__version__ = "1.1"

import unittest

from iapws97 import IAPWS97
from iapws97 import (_Region1, _Region2, _Region3, _Region5,
                     _Backward1_T_Ph, _Backward1_T_Ps, _Backward1_P_hs,
                     _Backward2_T_Ph, _Backward2_T_Ps, _Backward2_P_hs,
                     _h_3ab, _Backward3_T_Ph, _Backward3_v_Ph, _Backward3_T_Ps,
                     _Backward3_v_Ps, _PSat_h, _PSat_s, _Backward3_P_hs, _h1_s,
                     _h3a_s, _h2ab_s, _h2c3b_s, _PSat_T, _TSat_P, _h13_s,
                     _t_hs, _Backward4_T_hs, _tab_P, _top_P, _twx_P, _tef_P,
                     _txx_P, _Backward3_v_PT)
from iapws95 import IAPWS95, D2O
from iapws08 import SeaWater
from _iapws import (_Ice, _Sublimation_Pressure, _Melting_Pressure,
                    _Viscosity, _ThCond)


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
        self.assertEqual(round(_Sublimation_Pressure(230), 11), 8.94735e-6)
        self.assertEqual(round(_Melting_Pressure(260, 200), 3), 138.268)
        self.assertEqual(round(_Melting_Pressure(254, 350), 3), 268.685)
        self.assertEqual(round(_Melting_Pressure(265, 500), 3), 479.640)
        self.assertEqual(round(_Melting_Pressure(320, 1200), 2), 1356.76)
        self.assertEqual(round(_Melting_Pressure(550, 3000), 2), 6308.71)

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
        fluid = IAPWS95()
        self.assertEqual(round(fluid._Tension(273.16)*1000, 2), 75.65)
        self.assertEqual(round(fluid._Tension(283.15)*1000, 2), 74.22)
        self.assertEqual(round(fluid._Tension(293.15)*1000, 2), 72.74)
        self.assertEqual(round(fluid._Tension(303.15)*1000, 2), 71.19)
        self.assertEqual(round(fluid._Tension(313.15)*1000, 2), 69.60)
        self.assertEqual(round(fluid._Tension(323.15)*1000, 2), 67.94)
        self.assertEqual(round(fluid._Tension(333.15)*1000, 2), 66.24)
        self.assertEqual(round(fluid._Tension(343.15)*1000, 2), 64.48)
        self.assertEqual(round(fluid._Tension(353.15)*1000, 2), 62.67)
        self.assertEqual(round(fluid._Tension(363.15)*1000, 2), 60.82)
        self.assertEqual(round(fluid._Tension(373.15)*1000, 2), 58.91)
        self.assertEqual(round(fluid._Tension(393.15)*1000, 2), 54.97)
        self.assertEqual(round(fluid._Tension(413.15)*1000, 2), 50.86)
        self.assertEqual(round(fluid._Tension(433.15)*1000, 2), 46.59)
        self.assertEqual(round(fluid._Tension(453.15)*1000, 2), 42.19)
        self.assertEqual(round(fluid._Tension(473.15)*1000, 2), 37.67)
        self.assertEqual(round(fluid._Tension(523.15)*1000, 2), 26.04)
        self.assertEqual(round(fluid._Tension(573.15)*1000, 2), 14.36)
        self.assertEqual(round(fluid._Tension(623.15)*1000, 2), 3.67)
        self.assertEqual(round(fluid._Tension(643.15)*1000, 2), 0.39)

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
        self.assertEqual(round(_PSat_h(1700), 8), 17.24175718)
        self.assertEqual(round(_PSat_h(2000), 8), 21.93442957)
        self.assertEqual(round(_PSat_h(2400), 8), 20.18090839)

        # _PSat_s Table 20 pag 19
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
        self.assertEqual(round(_h1_s(1), 7), 308.5509647)
        self.assertEqual(round(_h1_s(2), 7), 700.6304472)
        self.assertEqual(round(_h1_s(3), 6), 1198.359754)
        self.assertEqual(round(_h3a_s(3.8), 6), 1685.025565)
        self.assertEqual(round(_h3a_s(4), 6), 1816.891476)
        self.assertEqual(round(_h3a_s(4.2), 6), 1949.352563)

        # _h2ab_s _h2c3b_s Table 18 pag 21
        self.assertEqual(round(_h2ab_s(7), 6), 2723.729985)
        self.assertEqual(round(_h2ab_s(8), 6), 2599.047210)
        self.assertEqual(round(_h2ab_s(9), 6), 2511.861477)
        self.assertEqual(round(_h2c3b_s(5.5), 6), 2687.693850)
        self.assertEqual(round(_h2c3b_s(5.0), 6), 2451.623609)
        self.assertEqual(round(_h2c3b_s(4.5), 6), 2144.360448)

        # _h13_s Table 18 pag 21
        self.assertEqual(round(_h13_s(3.7), 6), 1632.525047)
        self.assertEqual(round(_h13_s(3.6), 6), 1593.027214)
        self.assertEqual(round(_h13_s(3.5), 6), 1566.104611)

        # _t_hs Table 26 pag 26
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
        self.assertEqual(round(_PSat_T(300), 11), 0.00353658941)
        self.assertEqual(round(_PSat_T(500), 8), 2.63889776)
        self.assertEqual(round(_PSat_T(600), 7), 12.3443146)

        # _TSat_P Table 36 pag 36
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

    def test_D2O(self):
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
        self.assertEqual(round(D2O._visco(3.09*rhor, 0.431*Tr)/mur, 10), 36.9123166244)
        self.assertEqual(round(D2O._visco(3.23*rhor, 0.431*Tr)/mur, 10), 34.1531546602)
        self.assertEqual(round(D2O._visco(0.0002*rhor, 0.5*Tr)/mur, 10), 0.1972984225)
        self.assertEqual(round(D2O._visco(3.07*rhor, 0.5*Tr)/mur, 10), 12.0604912273)
        self.assertEqual(round(D2O._visco(3.18*rhor, 0.5*Tr)/mur, 10), 12.4679405772)
        self.assertEqual(round(D2O._visco(0.0027*rhor, 0.6*Tr)/mur, 10), 0.2365829037)
        self.assertEqual(round(D2O._visco(2.95*rhor, 0.6*Tr)/mur, 10), 5.2437249935)
        self.assertEqual(round(D2O._visco(3.07*rhor, 0.6*Tr)/mur, 10), 5.7578399754)
        self.assertEqual(round(D2O._visco(0.0295*rhor, 0.75*Tr)/mur, 10), 0.2951479769)
        self.assertEqual(round(D2O._visco(2.65*rhor, 0.75*Tr)/mur, 10), 2.6275043948)
        self.assertEqual(round(D2O._visco(2.83*rhor, 0.75*Tr)/mur, 10), 3.0417583586)
        self.assertEqual(round(D2O._visco(0.08*rhor, 0.9*Tr)/mur, 10), 0.3685472578)
        self.assertEqual(round(D2O._visco(0.163*rhor, 0.9*Tr)/mur, 10), 0.3619649145)
        self.assertEqual(round(D2O._visco(2.16*rhor, 0.9*Tr)/mur, 10), 1.6561616211)
        self.assertEqual(round(D2O._visco(2.52*rhor, 0.9*Tr)/mur, 10), 2.1041364724)
        self.assertEqual(round(D2O._visco(0.3*rhor, Tr)/mur, 10), 0.4424816849)
        self.assertEqual(round(D2O._visco(0.7*rhor, Tr)/mur, 10), 0.5528693914)
        self.assertEqual(round(D2O._visco(1.55*rhor, Tr)/mur, 10), 1.1038442411)
        self.assertEqual(round(D2O._visco(2.26*rhor, Tr)/mur, 10), 1.7569585722)
        self.assertEqual(round(D2O._visco(0.49*rhor, 1.1*Tr)/mur, 10), 0.5633038063)
        self.assertEqual(round(D2O._visco(0.98*rhor, 1.1*Tr)/mur, 10), 0.7816387903)
        self.assertEqual(round(D2O._visco(1.47*rhor, 1.1*Tr)/mur, 10), 1.1169456968)
        self.assertEqual(round(D2O._visco(1.96*rhor, 1.1*Tr)/mur, 10), 1.5001420619)
        self.assertEqual(round(D2O._visco(0.4*rhor, 1.2*Tr)/mur, 10), 0.6094539064)
        self.assertEqual(round(D2O._visco(0.8*rhor, 1.2*Tr)/mur, 10), 0.7651099154)
        self.assertEqual(round(D2O._visco(1.2*rhor, 1.2*Tr)/mur, 10), 0.9937870139)
        self.assertEqual(round(D2O._visco(1.61*rhor, 1.2*Tr)/mur, 10), 1.2711900131)

    def test_D2O_ThCond(self):
        """Table B4 pag 17"""
        lr = 0.742128e-3
        Tr = 643.847
        rhor = 358
        self.assertEqual(round(D2O._thermo(3.09*rhor, 0.431*Tr)/lr, 9), 762.915707396)
        self.assertEqual(round(D2O._thermo(3.23*rhor, 0.431*Tr)/lr, 9), 833.912049618)
        self.assertEqual(round(D2O._thermo(0.0002*rhor, 0.5*Tr)/lr, 9), 27.006536978)
        self.assertEqual(round(D2O._thermo(3.07*rhor, 0.5*Tr)/lr, 9), 835.786416818)
        self.assertEqual(round(D2O._thermo(3.18*rhor, 0.5*Tr)/lr, 9), 891.181752526)
        self.assertEqual(round(D2O._thermo(0.0027*rhor, 0.6*Tr)/lr, 9), 35.339949553)
        self.assertEqual(round(D2O._thermo(2.95*rhor, 0.6*Tr)/lr, 9), 861.240794445)
        self.assertEqual(round(D2O._thermo(3.07*rhor, 0.6*Tr)/lr, 9), 919.859094854)
        self.assertEqual(round(D2O._thermo(0.0295*rhor, 0.75*Tr)/lr, 9), 55.216750017)
        self.assertEqual(round(D2O._thermo(2.65*rhor, 0.75*Tr)/lr, 9), 790.442563472)
        self.assertEqual(round(D2O._thermo(2.83*rhor, 0.75*Tr)/lr, 9), 869.672292625)
        self.assertEqual(round(D2O._thermo(0.08*rhor, 0.9*Tr)/lr, 9), 74.522283066)
        self.assertEqual(round(D2O._thermo(0.163*rhor, 0.9*Tr)/lr, 9), 106.301972320)
        self.assertEqual(round(D2O._thermo(2.16*rhor, 0.9*Tr)/lr, 9), 627.777590127)
        self.assertEqual(round(D2O._thermo(2.52*rhor, 0.9*Tr)/lr, 9), 761.055043002)
        self.assertEqual(round(D2O._thermo(0.3*rhor, Tr)/lr, 9), 143.422002971)
        self.assertEqual(round(D2O._thermo(0.7*rhor, Tr)/lr, 9), 469.015122112)
        self.assertEqual(round(D2O._thermo(1.55*rhor, Tr)/lr, 9), 502.846952426)
        self.assertEqual(round(D2O._thermo(2.26*rhor, Tr)/lr, 9), 668.743524402)
        self.assertEqual(round(D2O._thermo(0.49*rhor, 1.1*Tr)/lr, 9), 184.813462109)
        self.assertEqual(round(D2O._thermo(0.98*rhor, 1.1*Tr)/lr, 9), 326.652382218)
        self.assertEqual(round(D2O._thermo(1.47*rhor, 1.1*Tr)/lr, 9), 438.370305052)
        self.assertEqual(round(D2O._thermo(1.96*rhor, 1.1*Tr)/lr, 9), 572.014411428)
        self.assertEqual(round(D2O._thermo(0.4*rhor, 1.2*Tr)/lr, 9), 160.059403824)
        self.assertEqual(round(D2O._thermo(0.8*rhor, 1.2*Tr)/lr, 9), 259.605241187)
        self.assertEqual(round(D2O._thermo(1.2*rhor, 1.2*Tr)/lr, 9), 362.179570932)
        self.assertEqual(round(D2O._thermo(1.61*rhor, 1.2*Tr)/lr, 9), 471.747729424)
        self.assertEqual(round(D2O._thermo(0.3*rhor, 1.27*Tr)/lr, 9), 145.249914694)
        self.assertEqual(round(D2O._thermo(0.6*rhor, 1.27*Tr)/lr, 9), 211.996299238)
        self.assertEqual(round(D2O._thermo(0.95*rhor, 1.27*Tr)/lr, 9), 299.251471210)
        self.assertEqual(round(D2O._thermo(1.37*rhor, 1.27*Tr)/lr, 9), 409.359675394)

    def test_D2O_Tension(self):
        """Selected values from table 1"""
        fluid = D2O()
        self.assertEqual(round(fluid._Tension(273.15+3.8)*1000, 2), 74.93)
        self.assertEqual(round(fluid._Tension(283.15)*1000, 2), 74.06)
        self.assertEqual(round(fluid._Tension(293.15)*1000, 2), 72.61)
        self.assertEqual(round(fluid._Tension(303.15)*1000, 2), 71.09)
        self.assertEqual(round(fluid._Tension(313.15)*1000, 2), 69.52)
        self.assertEqual(round(fluid._Tension(323.15)*1000, 2), 67.89)
        self.assertEqual(round(fluid._Tension(333.15)*1000, 2), 66.21)
        self.assertEqual(round(fluid._Tension(343.15)*1000, 2), 64.47)
        self.assertEqual(round(fluid._Tension(353.15)*1000, 2), 62.67)
        self.assertEqual(round(fluid._Tension(363.15)*1000, 2), 60.82)
        self.assertEqual(round(fluid._Tension(373.15)*1000, 2), 58.93)
        self.assertEqual(round(fluid._Tension(393.15)*1000, 2), 54.99)
        self.assertEqual(round(fluid._Tension(413.15)*1000, 2), 50.87)
        self.assertEqual(round(fluid._Tension(433.15)*1000, 2), 46.59)
        self.assertEqual(round(fluid._Tension(453.15)*1000, 2), 42.16)
        self.assertEqual(round(fluid._Tension(473.15)*1000, 2), 37.61)
        self.assertEqual(round(fluid._Tension(523.15)*1000, 2), 25.84)
        self.assertEqual(round(fluid._Tension(573.15)*1000, 2), 13.99)
        self.assertEqual(round(fluid._Tension(623.15)*1000, 2), 3.17)
        self.assertEqual(round(fluid._Tension(643.15)*1000, 2), 0.05)

    def test_Ice(self):
        """Table, pag 12"""
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

    def test_SeaWater(self):
        """Table 8, pag 17-19"""

        # Part a, pag 17
        fluid = SeaWater(T=273.15, P=0.101325, S=0.03516504)
        state = fluid._water(273.15, 0.101325)
        self.assertEqual(round(state["g"], 9), 0.101342742)
        self.assertEqual(round(state["gs"], 9), 0.0)
        self.assertEqual(round(state["gt"], 12), 0.000147643376)
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
        self.assertEqual(round(fluid.a, 10), -0.0985548978)
        self.assertEqual(round(fluid.u, 10), -0.0985548978)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
