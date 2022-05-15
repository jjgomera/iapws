---
title: 'iapws: A Python package for implement the IAPWS standards'
tags:
  - Python
  - chemistry
  - physics
authors:
  - name: Juan José Gómez-Romera^[jjgomera@gmail.com]
    orcid: 0000-0002-0406-1180
    affiliation: 1
affiliations:
 - name: Independent Researcher, Spain
   index: 1

date: 15 May 2022
bibliography: paper.bib
---

# Summary

Water is possibly the most important chemical compound. Due to both its uses
and its abundance, being able to predict its physical properties is essential
to be able to simulate any process in which water participates.

The International Association for the Properties of Water and Steam,
[IAPWS](http://iapws.org/), provides internationally accepted formulations for
the properties of light and heavy steam for scientific and industrial
applications.

iapws is a python module that implements those formulations defined by the
IAPWS for the calculation of the properties of water and steam.

# Description

This module implements almost the full set of standards:

* R1-76(2014): Revised Release on the Surface Tension of Ordynary Water Substance, [@Surf].
* R2-83(1992): Release on the Values of Temperature, Pressure and Density of Ordynary and Heavy Water Substances at their Respectives Critical Points.
* R5-85(1994): Release on Surface Tension of Heavy Water Substance.
* R6-95(2018): Revised Release on the IAPWS Formulation 1995 for the Thermodynamic Properties of Ordinary Water Substance for General and Scientific Use, [@IAPWS95].
* R7-97(2012): Revised Release on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties of Water and Steam, [@IF97].
* R8-97: Release on the Static Dielectric Constant of Ordinary Water Substance for Temperatures from 238 K to 873 K and Pressures up to 1000 MPa, [@Dielec].
* R9-97: Release on the Refractive Index of Ordinary Water Substance as a Function of Wavelength, Temperature and Pressure, [@Rindex].
* R10-06(2009): Revised Release on the Equation of State 2006 for H2O Ice Ih, [@Ice].
* R11-07(2019): Release on the Ionization Constant of H2O, [@Ionization].
* R12-08: Release on the IAPWS Formulation 2008 for the Viscosity of Ordinary Water Substance, [@Visc].
* R13-08: Release on the IAPWS Formulation 2008 for the Thermodynamic Properties of Seawater, [@Seawater].
* R14-08(2011): Revised Release on the Pressure along the Melting and Sublimation Curves of Ordinary Water Substance, [@MeltSub].
* R15-11: Release on the IAPWS Formulation 2011 for the Thermal Conductivity of Ordinary Water Substance, [@ThCond].
* R16-17(2018): Release on the IAPWS Formulation 2017 for the Thermodynamic Properties of Heavy Water, [@Heavy].
* R17-20: Release on the IAPWS Formulation 2020 for the Viscosity of Heavy Water, [@D2OVisc].
* R18-21: Release on the IAPWS Formulation 2021 for the Thermal Conductivity of Heavy Water, [@D2OThCond].


# References
