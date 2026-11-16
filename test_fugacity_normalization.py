import unittest
from math import exp, log
from iapws.ammonia import H2ONH3, NH3
from iapws import IAPWS95

class FugacityNormalization(unittest.TestCase):
    def test_table4_normalization(self):
        # IAPWS G4-01, Table 4: log(Z*phi_i), not log(phi_i/Z).
        for T, c, x in [(600,35,.1),(600,4,.1),(500,32,.5),
                         (500,1,.5),(400,30,.9),(400,.5,.9)]:
            with self.subTest(T=T,c=c,x=x):
                rho=c*((1-x)*IAPWS95.M+x*NH3.M)
                model=H2ONH3()
                p=model._phir(rho,T,x)
                state=model._prop(rho,T,x)
                z=1+p['delta']*p['fird']
                residual=p['fir']+z-1
                self.assertAlmostEqual(log(z*state['fugH2O']), residual-x*p['F'], places=12)
                self.assertAlmostEqual(log(z*state['fugNH3']), residual+(1-x)*p['F'], places=12)
                weighted=(1-x)*log(state['fugH2O'])+x*log(state['fugNH3'])
                self.assertAlmostEqual(weighted,residual-log(z),places=12)

if __name__=='__main__': unittest.main()
