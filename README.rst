iapws
=====

Python implementation of standard from IAPWS (http://www.iapws.org/release.html). The available standard are::

    IAPWS-IF97
    IAPWS-95
    IAPWS-06 for Ice
    IAPWS-08 for seawater
    IAPWS-05 for Heavy water
    
dependences
--------------------

* python 2x, 3x, compatible with both versions
* Numpy-scipy: library with mathematic and scientific tools


install
--------------------

In debian you can find in official repositories in testing and sid. In stable you can install using pip::

	pip install iapws

In ubuntu it's in official repositories from ubuntu saucy (13.10)

In other SO you can download from its webpage in `pypi <http://pypi.python.org/pypi/iapws>`_ and unzipped in python folder dist-packages. This is the recommended options to have the latest version.


TODO
--------------------

Improve convergence in two phase region for IAPWS95 and D2O class


IAPWS-IF97
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
    
    
    
IAPWS-95
--------------------------------

Class to model a state for liquid water or steam with the general and scientific formulation IAPWS-95

Incoming properties:

* T, Temperature, K
* P, Pressure, MPa
* rho, Density, kg/m3
* v, Specific volume, m3/kg
* h, Specific enthalpy, kJ/kg
* s, Specific entropy, kJ/kg·K
* x, Quality, [-]
* l, Optional parameter to light wavelength for Refractive index, mm

rho and v are equivalent, only one can be defined
Definitions options:

* T, P (Not valid for two-phases region)
* T, rho
* T, h
* T, s
* T, u
* P, rho
* P, h
* P, s
* P, u
* rho, h
* rho, s
* rho, u
* h, s
* h, u
* s, u
* T, x (Only for two-phases region)
* P, x (Only for two-phases region) Very slow

Properties:

* P,  Pressure, MPa
* Pr, Reduced pressure, [-]
* T, Temperature, K
* Tr, Reduced temperature, [-]
* x, Quality, [-]
* v, Specific volume, m³/kg
* rho, Density, kg/m³
* h, Specific enthalpy, kJ/kg
* s, Specific entropy, kJ/kg·K
* u, Specific internal energy, kJ/kg
* g, Specific Gibbs free energy, kJ/kg
* a, Specific Helmholtz free energy, kJ/kg
* cp, Specific isobaric heat capacity, kJ/kg·K
* cv, Specific isochoric heat capacity, kJ/kg·K
* cp_cv, Heat capacity ratio, [-]
* w, Speed of sound, m/s
* Z, Compression factor, [-]
* fi, Fugacity coefficient, [-]
* f, Fugacity, MPa
* gamma, Isoentropic exponent, [-]

* alfav, Thermal expansion coefficient (Volume expansivity), 1/K
* kappa, Isothermal compressibility, 1/MPa
* alfap, Relative pressure coefficient, 1/K
* betap, Isothermal stress coefficient, kg/m³
* betas, Isoentropic temperature-pressure coefficient, [-]
* joule, Joule-Thomson coefficient, K/MPa
* Gruneisen, Gruneisen parameter, [-]
* virialB, Second virial coefficient, m³/kg
* virialC, Third virial coefficient, m⁶/kg²
* dpdT_rho, Derivatives, dp/dT at constant rho, MPa/K
* dpdrho_T, Derivatives, dp/drho at constant T, MPa·m³/kg
* drhodT_P, Derivatives, drho/dT at constant P, kg/m³·K
* drhodP_T, Derivatives, drho/dP at constant T, kg/m³·MPa
* dhdT_rho, Derivatives, dh/dT at constant rho, kJ/kg·K
* dhdP_T, Isothermal throttling coefficient, kJ/kg·MPa
* dhdT_P, Derivatives, dh/dT at constant P, kJ/kg·K
* dhdrho_T, Derivatives, dh/drho at constant T, kJ·m³/kg²
* dhdrho_P, Derivatives, dh/drho at constant P, kJ·m³/kg²
* dhdP_rho, Derivatives, dh/dP at constant rho, kJ/kg·MPa
* kt, Isothermal Expansion Coefficient, [-]
* ks, Adiabatic Compressibility, 1/MPa
* Ks, Adiabatic bulk modulus, MPa
* Kt, Isothermal bulk modulus, MPa

* Hvap, Vaporization heat, kJ/kg
* Z_rho, (Z-1) over the density, m³/kg
* IntP,  Internal pressure, MPa
* invT, Negative reciprocal temperature, 1/K
* hInput, Specific heat input, kJ/kg

* mu, Dynamic viscosity, Pa·s
* nu, Kinematic viscosity, m²/s
* k, Thermal conductivity, W/m·K
* sigma, Surface tension, N/m
* alfa, Thermal diffusivity, m²/s
* Pramdt, Prandtl number, [-]
* epsilon, Dielectric constant, [-]
* n, Refractive index, [-]

* v0, Ideal gas Specific volume, m³/kg
* rho0, Ideal gas Density, kg/m³
* h0, Ideal gas Specific enthalpy, kJ/kg
* u0, Ideal gas Specific internal energy, kJ/kg
* s0, Ideal gas Specific entropy, kJ/kg·K
* a0, Ideal gas Specific Helmholtz free energy, kJ/kg
* g0, Ideal gas Specific Gibbs free energy, kJ/kg
* cp0, Ideal gas Specific isobaric heat capacity, kJ/kg·K
* cv0, Ideal gas Specific isochoric heat capacity, kJ/kg·K
* cp0_cv, Ideal gas Heat capacity ratio, [-]
* gamma0, Ideal gas Isoentropic exponent, [-]


Usage::

	from iapws import IAPWS95
	sat_steam=IAPWS95(P=1,x=1)                #saturated steam with known P
	sat_liquid=IAPWS95(T=370, x=0)            #saturated liquid with known T
	steam=IAPWS95(P=2.5, T=500)               #steam with known P and T
	print(sat_steam.h, sat_liquid.h, steam.h) #calculated enthalpies


    
IAPWS-06 for Ice Ih
--------------------------------------------

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

    
IAPWS-05 for Heavy water
--------------------------------------------

Same properties as for  IAPWS-95
Reference state set at liquid at normal boiling point (1 atm)

Usage::

	from iapws import D2O
	sat_liquid=D2O(T=370, x=0)            #saturated liquid with known T
	print(sat_liquid.h) #calculated enthalpy

    
IAPWS-08 for seawater
--------------------------------------------

Incoming properties:

* T: Temperature, K
* P: Pressure, MPa
* S: Salinity, kg/kg

S is the Reference-Composition Salinity as defined in Millero, F.J., R. Feistel, D.G. Wright and T.J. McDougall, "The composition of Standard Seawater and the definition of the Reference-Composition Salinity Scale", Deep-Sea Res. I 55, 50 (2008).

Calculated properties:

* T: Temperature, K
* P: Pressure, MPa
* rho: Density, kg/m³
* v: Specific volume, m³/kg
* h: Specific enthalpy, kJ/kg
* s: Specific entropy, kJ/kg·K
* u: Specific internal energy, kJ/kg
* g: Specific Gibbs free energy, kJ/kg
* a: Specific Helmholtz free energy, kJ/kg
* cp: Specific isobaric heat capacity, kJ/kg·K

* gt: Derivative Gibbs energy with temperature, kJ/kg·K
* gp: Derivative Gibbs energy with pressure, m³/kg
* gtt: Derivative Gibbs energy with temperature square, kJ/kg·K²
* gtp: Derivative Gibbs energy with pressure and temperature, m³/kg·K
* gpp: Derivative Gibbs energy with temperature square, m³/kg·MPa
* gs: Derivative Gibbs energy with salinity, kJ/kg
* gsp: Derivative Gibbs energy with salinity and pressure, m³/kg

* alfa: Thermal expansion coefficient, 1/K
* betas: Isentropic temperature-pressure coefficient, K/MPa
* kt: Isothermal compressibility, 1/MPa
* ks: Isentropic compressibility, 1/MPa
* w: Sound Speed, m/s

* mu: Relative chemical potential, kJ/kg
* muw: Chemical potential of H2O, kJ/kg
* mus: Chemical potential of sea salt, kJ/kg
* osm: Osmotic coefficient, [-]
* haline: Haline contraction coefficient, kg/kg

        
Usage::
    
    from iapws import SeaWater
    state = SeaWater(T=300, P=0.101325, S=0.001)    #Seawater with 0.1% Salinity
    print(state.cp)     # Get cp
