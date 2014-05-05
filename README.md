iapws
=====

Librería de python que implementa el estándar IAPWS-IF97, siendo capaz de calcular las propiedades físicas del agua líquida o vapor en un amplio rango de presiónes y temperaturas


dependencias
--------------------

* python 2,3, es compatible con las dos ramas de desarrollo de python.
* Numpy-scipy: librería con herramientas para el cálculo mátemático


instalación
--------------------

En debian se encuentra en los repositorios oficiales tanto en la rama testing como en sid. En estable se puede instalar usando pip::

	pip install iapws

En cualquier otro sistema operativo se puede instalar descargandolo de su página en `pypi <https://pypi.python.org/pypi?name=iapws&version=1.0.2>`_
y descomprimiendolo en la carpate dist-packages de python.


características
--------------------

Class to model a state for liquid water or steam with the Industrial Formulation IAPWS-IF97

Incoming properties:

* T, Temperature, K
* P, Pressure, MPa
* h, Specific enthalpy, kJ/kg
* s, Specific entropy, kJ/kg·K
* x, Quality
    
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
* h, Specific enthalpy, kJ/kg
* u, Specific internal energy, kJ/kg
* s, Specific entropy, kJ/kg·K
* cp, Specific isobaric heat capacity, kJ/kg·K
* cv, Specific isochoric heat capacity, kJ/kg·K
* Z, Compression factor
* gamma, Isoentropic exponent
* alfav, Isobaric cubic expansion coefficient, 1/K
* kt, Isothermal compressibility, 1/MPa
* alfap, Relative pressure coefficient, 1/K
* betap, Isothermal stress coefficient, kg/m³
* joule, Joule-Thomson coefficient, K/MPa
* deltat, Isothermal throttling coefficient, kJ/kg·MPa
* region, Region
    
* w, Speed of sound, m/s
* mu, Dynamic viscosity, Pa·s
* nu, Kinematic viscosity, m²/s
* k, Thermal conductivity, W/m·K
* alfa, Thermal diffusivity, m²/s
* sigma, Surface tension, N/m
* epsilon, Dielectric constant
* n, Refractive index
* Pr, Prandtl number



Usage::


   from iapws import IAPWS97

   sat_steam=IAPWS97(P=1,x=1)                #saturated steam with known P
   sat_liquid=IAPWS97(T=370, x=0)            #saturated liquid with known T
   steam=IAPWS97(P=2.5, T=500)               #steam with known P and T
   print(sat_steam.h, sat_liquid.h, steam.h) #calculated enthalpy
