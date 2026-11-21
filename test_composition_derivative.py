import unittest
from math import log

from iapws import IAPWS95
from iapws.ammonia import H2ONH3, NH3


class MixtureCompositionDerivative(unittest.TestCase):
    def test_departure_derivative_at_fixed_reduced_state(self):
        for tau, delta, x in [(1, 1, .5), (1.2, 2, .2), (.8, .2, .75)]:
            with self.subTest(tau=tau, delta=delta, x=x):
                h = 1e-5
                left = H2ONH3._Dphir(tau, delta, x-h)["fir"]
                right = H2ONH3._Dphir(tau, delta, x+h)["fir"]
                self.assertAlmostEqual(
                    H2ONH3._Dphir(tau, delta, x)["firx"],
                    (right-left)/(2*h), places=8)

    def test_fugacity_ratio_at_fixed_temperature_and_molar_density(self):
        model = H2ONH3()
        states = [(600, 35, .1), (600, 4, .1), (500, 32, .5),
                  (500, 1, .5), (400, 30, .9), (400, .5, .9)]
        for temperature, molar_density, x in states:
            with self.subTest(T=temperature, c=molar_density, x=x):
                # _phir takes mass density; preserve molar density when
                # changing composition. Table 4 of IAPWS G4-01 gives the
                # log ratio of the two fugacity coefficients as F.
                def mass_density(composition):
                    return molar_density*((1-composition)*IAPWS95.M
                                          + composition*NH3.M)

                state = model._prop(mass_density(x), temperature, x)
                actual = log(state["fugNH3"])-log(state["fugH2O"])
                h = 1e-5
                left = model._phir(
                    mass_density(x-h), temperature, x-h)["fir"]
                right = model._phir(
                    mass_density(x+h), temperature, x+h)["fir"]
                self.assertAlmostEqual(actual, (right-left)/(2*h),
                                       places=7)


if __name__ == "__main__":
    unittest.main()
