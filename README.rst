iapws
=====

Python implementation of international-standard IAPWS-IF97 steam tables 


dependences
--------------------

* python 2x;3x, compatible with both versions
* Numpy-scipy: library with mathematic and scientific tools


install
--------------------

In debian you can find in oficial repositories in testing and sid. In stable you can install using pip::

	pip install iapws

In other SO you can download from its webpage in `pypi <http://pypi.python.org/pypi/iapws>`_ and unzipped in python folder dist-packages.


Use
--------------------

Class to model a state for liquid water or steam with the Industrial Formulation IAPWS-IF97

Incoming properties:

* T, Temperature, K
* P, Pressure, MPa
* h, Specific enthalpy, kJ/kg
* s, Specific entropy, kJ/kg·K
* x, Quality, [-]
    
Definitions options:

* T, P (Not valid for two-phases region)
* P, h
* P, s
* h, s
* T, x (Only for two-phases region)
* P, x (Only for two-phases region)
    
Properties:

* P, Pressure, MPa
* T, Temperature, K
* g, Specific Gibbs free energy, kJ/kg
* a, Specific Helmholtz free energy, kJ/kg
* v, Specific volume, m³/kg
* rho, Density, kg/m³
* x, quality, [-]
* h, Specific enthalpy, kJ/kg
* u, Specific internal energy, kJ/kg
* s, Specific entropy, kJ/kg·K
* cp, Specific isobaric heat capacity, kJ/kg·K
* cv, Specific isochoric heat capacity, kJ/kg·K
* Z, Compression factor. [-]
* gamma, Isoentropic exponent, [-]
* alfav, Isobaric cubic expansion coefficient, 1/K
* kt, Isothermal compressibility, 1/MPa
* alfap, Relative pressure coefficient, 1/K
* betap, Isothermal stress coefficient, kg/m³
* joule, Joule-Thomson coefficient, K/MPa
* deltat, Isothermal throttling coefficient, kJ/kg·MPa
* region, Region

* v0, Ideal specific volume, m³/kg
* u0, Ideal specific internal energy, kJ/kg
* h0, Ideal specific enthalpy, kJ/kg
* s0, Ideal specific entropy, kJ/kg·K
* a0, Ideal specific Helmholtz free energy, kJ/kg
* g0, Ideal specific Gibbs free energy, kJ/kg
* cp0, Ideal specific isobaric heat capacity, kJ/kg·K
* cv0, Ideal specific isochoric heat capacity, kJ/kg·K
* w0, Ideal speed of sound, m/s
* gamma0, Ideal isoentropic exponent [-]
    
* w, Speed of sound, m/s
* mu, Dynamic viscosity, Pa·s
* nu, Kinematic viscosity, m²/s
* k, Thermal conductivity, W/m·K
* alfa, Thermal diffusivity, m²/s
* sigma, Surface tension, N/m
* epsilon, Dielectric constant, [-]
* n, Refractive index, [-]
* Prandt, Prandtl number, [-]
* Tr, Reduced temperature, [-]
* Pr, Reduced pressure, [-]


Usage::

	from iapws import IAPWS97
	sat_steam=IAPWS97(P=1,x=1)                #saturated steam with known P
	sat_liquid=IAPWS97(T=370, x=0)            #saturated liquid with known T
	steam=IAPWS97(P=2.5, T=500)               #steam with known P and T
	print(sat_steam.h, sat_liquid.h, steam.h) #calculated enthalpies
    
    
Ice Ih properties
-------------------------

There is too implemented a function to calculate properties of ice Ih from 2009 revision, in this case only let temperature and pressure as input for calculate properties, the function return a dict with properties available:

* P, Pressure, MPa
* T, Temperature, K
* v, Specific volume, m³/kg
* rho, Density, kg/m³
* g, Specific Gibbs free energy, kJ/kg
* a, Specific Helmholtz free energy, kJ/kg
* h, Specific enthalpy, kJ/kg
* u, Specific internal energy, kJ/kg
* s, Specific entropy, kJ/kg·K
* cp, Specific isobaric heat capacity, kJ/kg·K
* alfa, Cubic expansion coefficient, 1/K
* beta, Pressure coefficient, MPa/K
* kt, Isothermal compressibility, MPa
* ks, Isentropic compressibility, MPa

    
Usage::
    
    from iapws import _Ice
    ice=_Ice(273.15, 0.101325)            #Ice at normal melting point
    print(ice["rho"])                     #Calculated density
