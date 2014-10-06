#!/usr/bin/python
# -*- coding: utf-8 -*-

__version__ = "1.0.5"

import unittest

from iapws97 import IAPWS97
from iapws95 import IAPWS95, D2O
from _iapws import (_Ice, _Sublimation_Pressure, _Melting_Pressure, M, 
                    _Viscosity, _ThCond, _Tension, _Dielectric)

# Test
class Test(unittest.TestCase):

    def test_Helmholtz(self):
        """Table 6 from IAPWS95, pag 14"""
        T = 500
        rho = 838.025
        fluid = IAPWS95()
        delta = rho/fluid.rhoc
        tau = fluid.Tc/T
        
        fio, fiot, fiott, fiod, fiodd, fiodt = fluid._phi0(tau, delta)
        self.assertEquals(round(fio, 8), 2.04797733)
        self.assertEquals(round(fiod, 9), 0.384236747)
        self.assertEquals(round(fiodd, 9), -0.147637878)
        self.assertEquals(round(fiot, 8), 9.04611106)
        self.assertEquals(round(fiott, 8), -1.93249185)
        self.assertEquals(round(fiodt, 8), 0.0)
        
        fir, firt, firtt, fird, firdd, firdt, firdtt, B, C = fluid._phir(tau, delta)
        self.assertEquals(round(fir, 8), -3.42693206)
        self.assertEquals(round(fird, 9), -0.364366650)
        self.assertEquals(round(firdd, 9), 0.856063701)
        self.assertEquals(round(firt, 8), -5.81403435)
        self.assertEquals(round(firtt, 8), -2.23440737)
        self.assertEquals(round(firdt, 8), -1.12176915)

    def test_phase(self):
        """Table 7 from IAPWS95, pag 14"""
        fluid = IAPWS95()

        state = fluid._Helmholtz(996.556, 300)
        # See footnote for imprecise P value in last significant figures
        self.assertEquals(round(state["P"], 7), 99.241835)
        self.assertEquals(round(state["cv"], 8), 4.13018112)
        self.assertEquals(round(state["w"], 5), 1501.51914)
        self.assertEquals(round(state["s"], 9), 0.393062643)
        
        state = fluid._Helmholtz(1005.308, 300)
        self.assertEquals(round(state["P"], 4), 20002.2515)
        self.assertEquals(round(state["cv"], 8), 4.06798347)
        self.assertEquals(round(state["w"], 5), 1534.92501)
        self.assertEquals(round(state["s"], 9), 0.387405401)

        state = fluid._Helmholtz(1188.202, 300)
        self.assertEquals(round(state["P"], 3), 700004.704)
        self.assertEquals(round(state["cv"], 8), 3.46135580)
        self.assertEquals(round(state["w"], 5), 2443.57992)
        self.assertEquals(round(state["s"], 9), 0.132609616)

        state = fluid._Helmholtz(0.435, 500)
        self.assertEquals(round(state["P"], 7), 99.9679423)
        self.assertEquals(round(state["cv"], 8), 1.50817541)
        self.assertEquals(round(state["w"], 6), 548.314253)
        self.assertEquals(round(state["s"], 8), 7.94488271)

        state = fluid._Helmholtz(4.532, 500)
        self.assertEquals(round(state["P"], 6), 999.938125)
        self.assertEquals(round(state["cv"], 8), 1.66991025)
        self.assertEquals(round(state["w"], 6), 535.739001)
        self.assertEquals(round(state["s"], 8), 6.82502725)

        state = fluid._Helmholtz(838.025, 500)
        self.assertEquals(round(state["P"], 4), 10000.3858)
        self.assertEquals(round(state["cv"], 8), 3.22106219)
        self.assertEquals(round(state["w"], 5), 1271.28441)
        self.assertEquals(round(state["s"], 8), 2.56690919)

        state = fluid._Helmholtz(1084.564, 500)
        self.assertEquals(round(state["P"], 3), 700000.405)
        self.assertEquals(round(state["cv"], 8), 3.07437693)
        self.assertEquals(round(state["w"], 5), 2412.00877)
        self.assertEquals(round(state["s"], 8), 2.03237509)

        state = fluid._Helmholtz(358., 647)
        self.assertEquals(round(state["P"], 4), 22038.4756)
        self.assertEquals(round(state["cv"], 8), 6.18315728)
        self.assertEquals(round(state["w"], 6), 252.145078)
        self.assertEquals(round(state["s"], 8), 4.32092307)

        state = fluid._Helmholtz(0.241, 900)
        self.assertEquals(round(state["P"], 6), 100.062559)
        self.assertEquals(round(state["cv"], 8), 1.75890657)
        self.assertEquals(round(state["w"], 6), 724.027147)
        self.assertEquals(round(state["s"], 8), 9.16653194)

        state = fluid._Helmholtz(52.615, 900)
        self.assertEquals(round(state["P"], 3), 20000.069)
        self.assertEquals(round(state["cv"], 8), 1.93510526)
        self.assertEquals(round(state["w"], 6), 698.445674)
        self.assertEquals(round(state["s"], 8), 6.59070225)

        state = fluid._Helmholtz(870.769, 900)
        self.assertEquals(round(state["P"], 3), 700000.006)
        self.assertEquals(round(state["cv"], 8), 2.66422350)
        self.assertEquals(round(state["w"], 5), 2019.33608)
        self.assertEquals(round(state["s"], 8), 4.17223802)

#    def test_saturation(self):
#        """Table 8 from IAPWS95, pag 14"""
#        fluid = IAPWS95()
#        
#        rhol, rhov, Ps = fluid._saturation(275)
#        liquid = fluid._Helmholtz(rhol, 275)
#        vapor = fluid._Helmholtz(rhov, 275)
#        self.assertEquals(round(Ps, 9), 0.698451167)
#        self.assertEquals(round(rhol, 6), 999.887406)
#        self.assertEquals(round(rhov, 11), 0.00550664919)
#        self.assertEquals(round(liquid["h"], 8), 7.75972202)
#        self.assertEquals(round(vapor["h"], 5), 2504.28995)
#        self.assertEquals(round(liquid["s"], 10), 0.0283094670)
#        self.assertEquals(round(vapor["s"], 8), 9.10660121)
#
#        rhol, rhov, Ps = fluid._saturation(450)
#        liquid = fluid._Helmholtz(rhol, 450)
#        vapor = fluid._Helmholtz(rhov, 450)
#        self.assertEquals(round(Ps, 6), 932.203564)
#        self.assertEquals(round(rhol, 6), 890.341250)
#        self.assertEquals(round(rhov, 8), 4.81200360)
#        self.assertEquals(round(liquid["h"], 6), 749.161585)
#        self.assertEquals(round(vapor["h"], 5), 2774.41078)
#        self.assertEquals(round(liquid["s"], 8), 2.10865845)
#        self.assertEquals(round(vapor["s"], 8), 6.60921221)
#
#        rhol, rhov, Ps = fluid._saturation(625)
#        liquid = fluid._Helmholtz(rhol, 625)
#        vapor = fluid._Helmholtz(rhov, 625)
#        self.assertEquals(round(Ps, 4), 16908.2693)
#        self.assertEquals(round(rhol, 6), 567.090385)
#        self.assertEquals(round(rhov, 6), 118.290280)
#        self.assertEquals(round(liquid["h"], 5), 1686.26976)
#        self.assertEquals(round(vapor["h"], 5), 2550.71625)
#        self.assertEquals(round(liquid["s"], 8), 3.80194683)
#        self.assertEquals(round(vapor["s"], 8), 5.18506121)



    def test_Ice(self):
        """Table, pag 12"""
        ice = _Ice(273.16, 0.000611657)
        self.assertEquals(round(ice["g"], 12), 0.000611784135)
        self.assertEquals(round(ice["gp"], 11), 1.09085812737)
        self.assertEquals(round(ice["gt"], 11), 1.22069433940)
        self.assertEquals(round(ice["gpp"], 15), -0.000128495941571)
        self.assertEquals(round(ice["gtp"], 15), 0.000174387964700)
        self.assertEquals(round(ice["gtt"], 14), -0.00767602985875)
        self.assertEquals(round(ice["h"], 9), -333.444253966)
        self.assertEquals(round(ice["a"], 12), -0.000055446875)
        self.assertEquals(round(ice["u"], 9), -333.444921197)
        self.assertEquals(round(ice["s"], 11), -1.22069433940)
        self.assertEquals(round(ice["cp"], 11), 2.09678431622)
        self.assertEquals(round(ice["rho"], 9), 916.709492200)
        self.assertEquals(round(ice["alfav"], 15), 0.000159863102566)
        self.assertEquals(round(ice["beta"], 11), 1.35714764659)
        self.assertEquals(round(ice["kt"], 15), 0.000117793449348)
        self.assertEquals(round(ice["ks"], 15), 0.000114161597779)

        ice = _Ice(273.152519, 0.101325)
        self.assertEquals(round(ice["g"], 11), 0.10134274069)
        self.assertEquals(round(ice["gp"], 11), 1.09084388214)
        self.assertEquals(round(ice["gt"], 11), 1.22076932550)
        self.assertEquals(round(ice["gpp"], 15), -0.000128485364928)
        self.assertEquals(round(ice["gtp"], 15), 0.000174362219972)
        self.assertEquals(round(ice["gtt"], 14), -0.00767598233365)
        self.assertEquals(round(ice["h"], 9), -333.354873637)
        self.assertEquals(round(ice["a"], 11), -0.00918701567)
        self.assertEquals(round(ice["u"], 9), -333.465403393)
        self.assertEquals(round(ice["s"], 11), -1.22076932550)
        self.assertEquals(round(ice["cp"], 11), 2.09671391024)
        self.assertEquals(round(ice["rho"], 9), 916.721463419)
        self.assertEquals(round(ice["alfav"], 15), 0.000159841589458)
        self.assertEquals(round(ice["beta"], 11), 1.35705899321)
        self.assertEquals(round(ice["kt"], 15), 0.000117785291765)
        self.assertEquals(round(ice["ks"], 15), 0.000114154442556)

        ice = _Ice(100, 100.)
        self.assertEquals(round(ice["g"], 9), -222.296513088)
        self.assertEquals(round(ice["gp"], 11), 1.06193389260)
        self.assertEquals(round(ice["gt"], 11), 2.61195122589)
        self.assertEquals(round(ice["gpp"], 16), -0.0000941807981761)
        self.assertEquals(round(ice["gtp"], 16), 0.0000274505162488)
        self.assertEquals(round(ice["gtt"], 14), -0.00866333195517)
        self.assertEquals(round(ice["h"], 9), -483.491635676)
        self.assertEquals(round(ice["a"], 9), -328.489902347)
        self.assertEquals(round(ice["u"], 9), -589.685024936)
        self.assertEquals(round(ice["s"], 11), -2.61195122589)
        self.assertEquals(round(ice["cp"], 12), 0.866333195517)
        self.assertEquals(round(ice["rho"], 9), 941.678203297)
        self.assertEquals(round(ice["alfav"], 16), 0.0000258495528207)
        self.assertEquals(round(ice["beta"], 12), 0.291466166994)
        self.assertEquals(round(ice["kt"], 16), 0.0000886880048115)
        self.assertEquals(round(ice["ks"], 16), 0.0000886060982687)

    def test_Melting(self):
        """Table 3, pag 7"""
        self.assertEquals(round(_Sublimation_Pressure(230), 11), 8.94735e-6)
        self.assertEquals(round(_Melting_Pressure(260, 200), 3), 138.268)
        self.assertEquals(round(_Melting_Pressure(254, 350), 3), 268.685)
        self.assertEquals(round(_Melting_Pressure(265, 500), 3), 479.640)
        self.assertEquals(round(_Melting_Pressure(320, 1200), 2), 1356.76)
        self.assertEquals(round(_Melting_Pressure(550, 3000), 2), 6308.71)

    def test_Viscosity_1(self):
        """Table 4, pag 8"""
        self.assertEquals(round(_Viscosity(998, 298.15)*1e6, 6), 889.735100)
        self.assertEquals(round(_Viscosity(1200, 298.15)*1e6, 6), 1437.649467)
        self.assertEquals(round(_Viscosity(1000, 373.15)*1e6, 6), 307.883622)
        self.assertEquals(round(_Viscosity(1, 433.15)*1e6, 6), 14.538324)
        self.assertEquals(round(_Viscosity(1000, 433.15)*1e6, 6), 217.685358)
        self.assertEquals(round(_Viscosity(1, 873.15)*1e6, 6), 32.619287)
        self.assertEquals(round(_Viscosity(100, 873.15)*1e6, 6), 35.802262)
        self.assertEquals(round(_Viscosity(600, 873.15)*1e6, 6), 77.430195)
        self.assertEquals(round(_Viscosity(1, 1173.15)*1e6, 6), 44.217245)
        self.assertEquals(round(_Viscosity(100, 1173.15)*1e6, 6), 47.640433)
        self.assertEquals(round(_Viscosity(400, 1173.15)*1e6, 6), 64.154608)

    def test_Viscosity_2(self):
        """Table 5, pag 9"""
        fluid = IAPWS95(rho=122, T=647.35)
        self.assertEquals(round(fluid.mu*1e6, 6), 25.520677)
        fluid = IAPWS95(rho=222, T=647.35)
        self.assertEquals(round(fluid.mu*1e6, 6), 31.337589)
        fluid = IAPWS95(rho=272, T=647.35)
        self.assertEquals(round(fluid.mu*1e6, 6), 36.228143)
        fluid = IAPWS95(rho=322, T=647.35)
        self.assertEquals(round(fluid.mu*1e6, 6), 42.961579)
        fluid = IAPWS95(rho=372, T=647.35)
        self.assertEquals(round(fluid.mu*1e6, 6), 45.688204)
        fluid = IAPWS95(rho=422, T=647.35)
        self.assertEquals(round(fluid.mu*1e6, 6), 49.436256)

    def test_ThCond_1(self):
        """Table 4, pag 10"""
        self.assertEquals(round(_ThCond(0, 298.15)*1000, 7), 18.4341883)
        self.assertEquals(round(_ThCond(998, 298.15)*1000, 6), 607.712868)
        self.assertEquals(round(_ThCond(1200, 298.15)*1000, 6), 799.038144)
        self.assertEquals(round(_ThCond(0, 873.15)*1000, 7), 79.1034659)

    def test_ThCond_2(self):
        """Table 5, pag 10"""
        fluid = IAPWS95(rho=1, T=647.35)
        self.assertEquals(round(fluid.k*1000, 7), 51.9298924)
        fluid = IAPWS95(rho=122, T=647.35)
        self.assertEquals(round(fluid.k*1000, 6), 130.922885)
        fluid = IAPWS95(rho=222, T=647.35)
        self.assertEquals(round(fluid.k*1000, 6), 367.787459)
        fluid = IAPWS95(rho=272, T=647.35)
        self.assertEquals(round(fluid.k*1000, 6), 757.959776)
        fluid = IAPWS95(rho=322, T=647.35)
        self.assertEquals(round(fluid.k*1000, 5), 1443.75556)
        fluid = IAPWS95(rho=372, T=647.35)
        self.assertEquals(round(fluid.k*1000, 6), 650.319402)
        fluid = IAPWS95(rho=422, T=647.35)
        self.assertEquals(round(fluid.k*1000, 6), 448.883487)
        fluid = IAPWS95(rho=750, T=647.35)
        self.assertEquals(round(fluid.k*1000, 6), 600.961346)

    def test_Tension(self):
        """Selected values from table 1"""
        self.assertEquals(round(_Tension(273.16)*1000, 2), 75.65)
        self.assertEquals(round(_Tension(283.15)*1000, 2), 74.22)
        self.assertEquals(round(_Tension(293.15)*1000, 2), 72.74)
        self.assertEquals(round(_Tension(303.15)*1000, 2), 71.19)
        self.assertEquals(round(_Tension(313.15)*1000, 2), 69.60)
        self.assertEquals(round(_Tension(323.15)*1000, 2), 67.94)
        self.assertEquals(round(_Tension(333.15)*1000, 2), 66.24)
        self.assertEquals(round(_Tension(343.15)*1000, 2), 64.48)
        self.assertEquals(round(_Tension(353.15)*1000, 2), 62.67)
        self.assertEquals(round(_Tension(363.15)*1000, 2), 60.82)
        self.assertEquals(round(_Tension(373.15)*1000, 2), 58.91)
        self.assertEquals(round(_Tension(393.15)*1000, 2), 54.97)
        self.assertEquals(round(_Tension(413.15)*1000, 2), 50.86)
        self.assertEquals(round(_Tension(433.15)*1000, 2), 46.59)
        self.assertEquals(round(_Tension(453.15)*1000, 2), 42.19)
        self.assertEquals(round(_Tension(473.15)*1000, 2), 37.67)
        self.assertEquals(round(_Tension(523.15)*1000, 2), 26.04)
        self.assertEquals(round(_Tension(573.15)*1000, 2), 14.36)
        self.assertEquals(round(_Tension(623.15)*1000, 2), 3.67)
        self.assertEquals(round(_Tension(643.15)*1000, 2), 0.39)

#    def test_Dielect(self):
#        """Table 4, pag 8"""
#        fluid = IAPWS95(P=0.101325, T=240)
#        self.assertEquals(round(fluid.epsilon, 5), 104.34982)
#        fluid = IAPWS95(P=0.101325, T=300)
#        self.assertEquals(round(fluid.epsilon, 5), 77.74735)
#        fluid = IAPWS95(P=10, T=300)
#        self.assertEquals(round(fluid.epsilon, 5), 78.11269)
#        fluid = IAPWS95(P=1000, T=300)
#        self.assertEquals(round(fluid.epsilon, 5), 103.69632)
#        fluid = IAPWS95(P=10, T=650)
#        self.assertEquals(round(fluid.epsilon, 5), 1.26715)
#        fluid = IAPWS95(P=100, T=650)
#        self.assertEquals(round(fluid.epsilon, 5), 17.71733)
#        fluid = IAPWS95(P=500, T=650)
#        self.assertEquals(round(fluid.epsilon, 5), 26.62132)
#        fluid = IAPWS95(P=10, T=870)
#        self.assertEquals(round(fluid.epsilon, 5), 1.12721)
#        fluid = IAPWS95(P=100, T=870)
#        self.assertEquals(round(fluid.epsilon, 5), 4.98281)
#        fluid = IAPWS95(P=500, T=870)
#        self.assertEquals(round(fluid.epsilon, 5), 15.09746)
#        
#    def test_Refractive(self):
#        """Selected values from table 3, pag 6"""
#        fluid = IAPWS95(P=0.1, T=273.15, l=0.2265)
#        self.assertEquals(round(fluid.n, 6), 1.394527)
#        fluid = IAPWS95(P=10., T=273.15, l=0.2265)
#        self.assertEquals(round(fluid.n, 6), 1.396526)
#        fluid = IAPWS95(P=1., T=373.15, l=0.2265)
#        self.assertEquals(round(fluid.n, 6), 1.375622)
#        fluid = IAPWS95(P=100., T=373.15, l=0.2265)
#        self.assertEquals(round(fluid.n, 6), 1.391983)
#        fluid = IAPWS95(P=0.1, T=473.15, l=0.589)
#        self.assertEquals(round(fluid.n, 7), 1.0001456)
#        fluid = IAPWS95(P=1., T=773.15, l=0.589)
#        self.assertEquals(round(fluid.n, 7), 1.0008773)
#        fluid = IAPWS95(P=10., T=273.15, l=1.01398)
#        self.assertEquals(round(fluid.n, 6), 1.327710)
#        fluid = IAPWS95(P=100., T=473.15, l=1.01398)
#        self.assertEquals(round(fluid.n, 6), 1.298369)
        
    def test_IAPWS97_1(self):
        """"""
        fluid = IAPWS97(T=300, P=3)
        self.assertEquals(round(fluid.v, 11), 0.00100215168)
        self.assertEquals(round(fluid.h, 6), 115.331273)
        self.assertEquals(round(fluid.u, 6), 112.324818)
        self.assertEquals(round(fluid.s, 9), 0.392294792)
        self.assertEquals(round(fluid.cp, 8), 4.17301218)
        self.assertEquals(round(fluid.w, 5), 1507.73921)

        fluid = IAPWS97(T=300, P=80)
        self.assertEquals(round(fluid.v, 12), 0.000971180894)
        self.assertEquals(round(fluid.h, 6), 184.142828)
        self.assertEquals(round(fluid.u, 6), 106.448356)
        self.assertEquals(round(fluid.s, 9), 0.368563852)
        self.assertEquals(round(fluid.cp, 8), 4.01008987)
        self.assertEquals(round(fluid.w, 5), 1634.69054)

        fluid = IAPWS97(T=500, P=3)
        self.assertEquals(round(fluid.v, 10), 0.0012024180)
        self.assertEquals(round(fluid.h, 6), 975.542239)
        self.assertEquals(round(fluid.u, 6), 971.934985)
        self.assertEquals(round(fluid.s, 9), 2.58041912)
        self.assertEquals(round(fluid.cp, 8), 4.65580682)
        self.assertEquals(round(fluid.w, 5), 1240.71337)


if __name__ == "__main__":
    unittest.main(verbosity=2)
