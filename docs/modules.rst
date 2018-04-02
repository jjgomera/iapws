
.. include:: header.rst


Introduction
============

Python implementation of standard from IAPWS (http://www.iapws.org/release.html).

- Home: https://github.com/jjgomera/iapws
- Author: Juan José Gómez Romera <jjgomera@gmail.com>
- License: GPL-3 
- Documentation: http://iapws.readthedocs.io/


Dependences
===========

* python 2x, 3x, compatible with both versions
* Numpy-scipy: library with mathematic and scientific tools


Installation
============

In debian you can find in official repositories in jessie, testing and sid. In ubuntu it's in official repositories from ubuntu saucy (13.10). In other system you can install using pip::

	pip install iapws
 
or directly cloning the github repository::

    git clone https://github.com/jjgomera/iapws.git

and adding the folder to a python path.


Features
========

This module implements almost the full set of standards:

Releases:

* R1-76(2014): Revised Release on the Surface Tension of Ordynary Water Substance, :func:`iapws._iapws._Tension`
* R2-83(1992): Release on the Values of Temperature, Pressure and Density of Ordynary and Heavy Water Substances at their Respectives Critical Points, :func:`iapws._iapws`
* R4-84(2007): Revised Release on Viscosity and Thermal Conductivity of Heavy Water Substance, :func:`iapws._iapws._D2O_Viscosity`, :func:`iapws._iapws._D2O_ThCond`
* R5-85(1994): Release on Surface Tension of Heavy Water Substance, :func:`iapws._iapws._D2O_Tension`
* R6-95(2016): Revised Release on the IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use, :func:`iapws.iapws95.IAPWS95`
* R7-97(2012): Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam, :func:`iapws.iapws97`
* R8-97: Release on the Static Dielectric Constant of Ordinary Water Substance for Temperatures from 238 K to 873 K and Pressures up to 1000 MPa, :func:`iapws._iapws._Dielectric`
* R9-97: Release on the Refractive Index of Ordinary Water Substance as a Function of Wavelength, Temperature and Pressure, :func:`iapws._iapws._Refractive`
* R10-06(2009): Revised Release on the Equation of State 2006 for H2O Ice Ih, :func:`iapws._iapws._Ice`
* R11-07: Release on the Ionization Constant of H2O, :func:`iapws._iapws._Kw`
* R12-08: Release on the IAPWS Formulation 2008 for the Viscosity of Ordinary Water Substance, :func:`iapws._iapws._Viscosity`
* R13-08: Release on the IAPWS Formulation 2008 for the Thermodynamic Properties of Seawater, :func:`iapws.iapws08`
* R14-08(2011): Revised Release on the Pressure along the Melting and Sublimation Curves of Ordinary Water Substance, :func:`iapws._iapws._Melting_Pressure`, :func:`iapws._iapws._Sublimation_Pressure`
* R15-11: Release on the IAPWS Formulation 2011 for the Thermal Conductivity of Ordinary Water Substance, :func:`iapws._iapws._ThCond`
* R16-17: Release on the IAPWS Formulation 2017 for the Thermodynamic Properties of Heavy Water, :func:`iapws.iapws95.D2O`

Supplementary Releases:

* SR1-86(1992): Revised Supplementary Release on Saturation Properties of Ordinary Water Substance, :func:`iapws.iapws95.MEoS._Liquid_Density`, :func:`iapws.iapws95.MEoS._Vapor_Density`, :func:`iapws.iapws95.MEoS._Vapor_Pressure`
* SR2-01(2014): Revised Supplementary Release on Backward Equations for Pressure as a Function of Enthalpy and Entropy p(h,s) for Regions 1 and 2 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam, :func:`iapws.iapws97._Backward1_P_hs`, :func:`iapws.iapws97._Backward2_P_hs` 
* SR3-03(2014): Revised Supplementary Release on Backward Equations for the Functions T(p,h), v(p,h), and T(p,s), v(p,s) for Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam, :func:`iapws.iapws97._Backward3_T_Ph`, :func:`iapws.iapws97._Backward3_T_Ps`, :func:`iapws.iapws97._Backward3_v_Ph`, :func:`iapws.iapws97._Backward3_v_Ps`
* SR4-04(2014): Revised Supplementary Release on Backward Equations p(h,s) for Region 3, Equations as a Function of h and s for the Region Boundaries, and an Equation Tsat(h,s) for Region 4 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam, :func:`iapws.iapws97._Backward3_P_hs`
* SR5-05(2016): Revised Supplementary Release on Backward Equations for Specific Volume as a Function of Pressure and Temperature v(p,T) for Region 3 of the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam, :func:`iapws.iapws97._Backward3_v_PT`
* SR6-08(2011): Revised Supplementary Release on Properties of Liquid Water at 0.1 MPa, :func:`iapws._iapws._Liquid`
* SR7-09: Supplementary Release on a Computationally Efficient Thermodynamic Formulation for Liquid Water for Oceanographic Use, :func:`iapws.iapws08.SeaWater._waterSupp`


Guidelines:

* G1-90: Electrolytic Conductivity (Specific Conductance) of Liquid and Dense Supercritical Water from 0°C to 800°C and Pressures up to 1000 MPa, :func:`iapws._iapws._Conductivity`
* G2-90(1994): Solubility of Sodium Sulfate in Aqueous Mixtures of Sodium Chloride and Sulfuric Acid from Water to Concentrated Solutions, from 250 °C to 350 °C, :func:`iapws.iapws08._solNa2SO4`
* G3-00(2012): Revised Guideline on the Critical Locus of Aqueous Solutions of Sodium Chloride, :func:`iapws.iapws08._critNaCl`
* G4-01: Guideline on the IAPWS Formulation 2001 for the Thermodynamic Properties of Ammonia-Water Mixtures, :func:`iapws.ammonia`
* G5-01(2016): Guideline on the Use of Fundamental Physical Constants and Basic Constants of Water, :func:`iapws._iapws`
* G6-03: Guideline on the Tabular Taylor Series Expansion (TTSE) Method for Calculation of Thermodynamic Properties of Water and Steam Applied to IAPWS-95 as an Example (Not implemented)
* G7-04: Guideline on the Henry's Constant and Vapor-Liquid Distribution Constant for Gases in H2O and D2O at High Temperatures, :func:`iapws._iapws._Henry`, :func:`iapws._iapws._Kvalue`
* G8-10: Guideline on an Equation of State for Humid Air in Contact with Seawater and Ice, Consistent with the IAPWS Formulation 2008 for the Thermodynamic Properties of Seawater, :func:`iapws.humidAir.HumidAir`
* G9-12: Guideline on a Low-Temperature Extension of the IAPWS-95 Formulation for Water Vapor, :func:`iapws.iapws95.IAPWS95._phiex`
* G10-15: Guideline on the Thermal Conductivity of Seawater, :func:`iapws.iapws08._ThCond_SeaWater`
* G11-15: Guideline on a Virial Equation for the Fugacity of H2O in Humid Air, :func:`iapws.humidAir._virial`
* G12-15: Guideline on Thermodynamic Properties of Supercooled Water, :func:`iapws._iapws._Supercooled`
* G13-15: Guideline on the Fast Calculation of Steam and Water Properties with the Spline-Based Table Look-Up Method (SBTL) (Not implemented)


Advisory Notes:

* AN1-03: Uncertainties in Enthalpy for the IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use (IAPWS-95) and the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam (IAPWS-IF97)
* AN2-04(2013): Role of Various IAPWS Documents Concerning the Thermodynamic Properties of Ordinary Water Substance
* AN3-07(2014): Thermodynamic Derivatives from IAPWS Formulations, :func:`iapws._utils.deriv_G`, :func:`iapws._utils.deriv_H`
* AN4-09: Roles of IAPWS and CIPM Standards for the Density of Water
* AN5-13(2016): Industrial Calculation of the Thermodynamic Properties of Seawater, :func:`iapws.iapws08.Seawater._waterIF97`, :func:`iapws.iapws08._Tb`, :func:`iapws.iapws08._Tf`, :func:`iapws.iapws08._Triple`, :func:`iapws.iapws08._OsmoticPressure`
* AN6-16: Relationship between Various IAPWS Documents and the International Thermodynamic Equation of Seawater - 2010 (TEOS-10)


Documentation
=============
 
You can navigate the full documentation of package:

.. toctree::
   :maxdepth: 10

   iapws


.. include:: ../README.rst
  :start-after: inclusion-marker-do-not-remove

