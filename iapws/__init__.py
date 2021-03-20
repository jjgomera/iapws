#!/usr/bin/python
# -*- coding: utf-8 -*-
"""International Association for the Properties of Water and Steam (IAPWS)."""

import os

from ._iapws import (_Ice, _Sublimation_Pressure, _Melting_Pressure,  # noqa
                     _Viscosity, _ThCond, _Tension, _Dielectric, _Refractive)
from .iapws97 import IAPWS97  # noqa
from .iapws95 import IAPWS95, D2O  # noqa
from .iapws08 import SeaWater  # noqa
from .humidAir import HumidAir  # noqa
from .ammonia import H2ONH3  # noqa

basepath = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(basepath, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

__doi__ = {
    "R1-76":
        {"autor": "IAPWS",
         "title": "Revised Release on the Surface Tension of Ordynary Water "
                  "Substance",
         "ref": "2014",
         "doi": ""},
    "R2-83":
        {"autor": "IAPWS",
         "title": "Release on the Values of Temperature, Pressure and Density "
                  "of Ordynary and Heavy Water Substances at their Respectives"
                  " Critical Points",
         "ref": "1992",
         "doi": ""},
    "R4-84":
        {"autor": "IAPWS",
         "title": "Revised Release on Viscosity and Thermal Conductivity of "
                  "Heavy Water Substance",
         "ref": "2007",
         "doi": ""},
    "R5-85":
        {"autor": "IAPWS",
         "title": "Release on Surface Tension of Heavy Water Substance",
         "ref": "1994",
         "doi": ""},
    "R6-95":
        {"autor": "IAPWS",
         "title": "Revised Release on the IAPWS Formulation 1995 for the "
                  "Thermodynamic Properties of Ordinary Water Substance for "
                  "General and Scientific Use",
         "ref": "2006",
         "doi": ""},
    "R7-97":
        {"autor": "IAPWS",
         "title": "Revised Release on the IAPWS Industrial Formulation 1997 "
                  "for the Thermodynamic Properties of Water and Steam",
         "ref": "2012",
         "doi": ""},
    "R8-97":
        {"autor": "IAPWS",
         "title": "Release on the Static Dielectric Constant of Ordinary Water"
                  "Substance for Temperatures from 238 K to 873 K and "
                  "Pressures up to 1000 MPa",
         "ref": "1997",
         "doi": ""},
    "R9-97":
        {"autor": "IAPWS",
         "title": "Release on the Refractive Index of Ordinary Water Substance"
                  " as a Function of Wavelength, Temperature and Pressure",
         "ref": "1997",
         "doi": ""},
    "R10-06":
        {"autor": "IAPWS",
         "title": "Revised Release on the Equation of State 2006 for H2O Ice "
                  "Ih",
         "ref": "2009",
         "doi": ""},
    "R11-19":
        {"autor": "IAPWS",
         "title": "Release on the Ionization Constant of H2O",
         "ref": "2019",
         "doi": ""},
    "R12-08":
        {"autor": "IAPWS",
         "title": "Release on the IAPWS Formulation 2008 for the Viscosity of "
                  "Ordinary Water Substance",
         "ref": "2008",
         "doi": ""},
    "R13-08":
        {"autor": "IAPWS",
         "title": "Release on the IAPWS Formulation 2008 for the Thermodynamic"
                  " Properties of Seawater",
         "ref": "2008",
         "doi": ""},
    "R14-08":
        {"autor": "IAPWS",
         "title": "Revised Release on the Pressure along the Melting and "
                  "Sublimation Curves of Ordinary Water Substance",
         "ref": "2011",
         "doi": ""},
    "R15-11":
        {"autor": "IAPWS",
         "title": "Release on the IAPWS Formulation 2011 for the Thermal "
                  "Conductivity of Ordinary Water Substance",
         "ref": "2011",
         "doi": ""},
    "R16-17":
        {"autor": "IAPWS",
         "title": "Release on the IAPWS Formulation 2017 for the Thermodynamic"
                  " Properties of Heavy Water",
         "ref": "2017",
         "doi": ""},
    "SR1-86":
        {"autor": "IAPWS",
         "title": "Revised Supplementary Release on Saturation Properties of "
                  "Ordinary Water Substance",
         "ref": "1992",
         "doi": ""},
    "SR2-01":
        {"autor": "IAPWS",
         "title": "Revised Supplementary Release on Backward Equations for "
                  "Pressure as a Function of Enthalpy and Entropy p(h,s) for "
                  "Regions 1 and 2 of the IAPWS Industrial Formulation 1997 "
                  "for the Thermodynamic Properties of Water and Steam",
         "ref": "2014",
         "doi": ""},
    "SR3-03":
        {"autor": "IAPWS",
         "title": "Revised Supplementary Release on Backward Equations for "
                  "the Functions T(p,h), v(p,h), and T(p,s), v(p,s) for "
                  "Region 3 of the IAPWS Industrial Formulation 1997 for the "
                  "Thermodynamic Properties of Water and Steam",
         "ref": "2014",
         "doi": ""},
    "SR4-04":
        {"autor": "IAPWS",
         "title": "Revised Supplementary Release on Backward Equations p(h,s) "
                  "for Region 3, Equations as a Function of h and s for the "
                  "Region Boundaries, and an Equation Tsat(h,s) for Region 4 "
                  "of the IAPWS Industrial Formulation 1997 for the "
                  "Thermodynamic Properties of Water and Steam",
         "ref": "2014",
         "doi": ""},
    "SR5-05":
        {"autor": "IAPWS",
         "title": "Revised Supplementary Release on Backward Equations for "
                  "Specific Volume as a Function of Pressure and Temperature "
                  "v(p,T) for Region 3 of the IAPWS Industrial Formulation "
                  "1997 for the Thermodynamic Properties of Water and Steam",
         "ref": "2016",
         "doi": ""},
    "SR6-08":
        {"autor": "IAPWS",
         "title": "Revised Supplementary Release on Properties of Liquid "
                  "Water at 0.1 MPa",
         "ref": "2011",
         "doi": ""},

    "SR7-09":
        {"autor": "IAPWS",
         "title": "Supplementary Release on a Computationally Efficient "
                  "Thermodynamic Formulation for Liquid Water for "
                  "Oceanographic Use",
         "ref": "2009",
         "doi": ""},
    "G1-90":
        {"autor": "IAPWS",
         "title": "Electrolytic Conductivity (Specific Conductance) of Liquid "
                  "and Dense Supercritical Water from 0°C to 800°C and "
                  "Pressures up to 1000 MPa",
         "ref": "1990",
         "doi": ""},
    "G2-90":
        {"autor": "IAPWS",
         "title": "Solubility of Sodium Sulfate in Aqueous Mixtures of Sodium "
                  "Chloride and Sulfuric Acid from Water to Concentrated "
                  "Solutions, from 250 °C to 350 °C",
         "ref": "1994",
         "doi": ""},
    "G3-00":
        {"autor": "IAPWS",
         "title": "Revised Guideline on the Critical Locus of Aqueous "
                  "Solutions of Sodium Chloride",
         "ref": "2012",
         "doi": ""},
    "G4-01":
        {"autor": "IAPWS",
         "title": "Guideline on the IAPWS Formulation 2001 for the "
                  "Thermodynamic Properties of Ammonia-Water Mixtures",
         "ref": "2001",
         "doi": ""},
    "G5-01":
        {"autor": "IAPWS",
         "title": "Guideline on the Use of Fundamental Physical Constants and "
                  "Basic Constants of Water",
         "ref": "2016",
         "doi": ""},
    "G7-04":
        {"autor": "IAPWS",
         "title": "Guideline on the Henry's Constant and Vapor-Liquid "
                  "Distribution Constant for Gases in H2O and D2O at High "
                  "Temperatures",
         "ref": "2004",
         "doi": ""},

    "G8-10":
        {"autor": "IAPWS",
         "title": "Guideline on an Equation of State for Humid Air in Contact "
                  "with Seawater and Ice, Consistent with the IAPWS "
                  "Formulation 2008 for the Thermodynamic Properties of "
                  "Seawater",
         "ref": "2010",
         "doi": ""},
    "G9-12":
        {"autor": "IAPWS",
         "title": "Guideline on a Low-Temperature Extension of the IAPWS-95 "
                  "Formulation for Water Vapor",
         "ref": "2012",
         "doi": ""},
    "G10-15":
        {"autor": "IAPWS",
         "title": "Guideline on the Thermal Conductivity of Seawater",
         "ref": "2015",
         "doi": ""},
    "G11-15":
        {"autor": "IAPWS",
         "title": "Guideline on a Virial Equation for the Fugacity of H2O in "
                  "Humid Air",
         "ref": "2015",
         "doi": ""},
    "G12-15":
        {"autor": "IAPWS",
         "title": "Guideline on Thermodynamic Properties of Supercooled Water",
         "ref": "2015",
         "doi": ""},
    "AN3-07":
        {"autor": "IAPWS",
         "title": "Thermodynamic Derivatives from IAPWS Formulations",
         "ref": "2014",
         "doi": ""},
    "AN5-13":
        {"autor": "IAPWS",
         "title": "Industrial Calculation of the Thermodynamic Properties of "
                  "Seawater",
         "ref": "2016",
         "doi": ""},
}
