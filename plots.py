#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Generate various plots using IAPWS."""

from math import pi, atan, log
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

import iapws
from iapws._iapws import Pt, Pc, Tc
from iapws.iapws97 import _PSat_T, _P23_T


###############################################################################
# Configuration section
###############################################################################

# Define standard to use in plot, IAPWS95 very slow!
fluid = iapws.IAPWS97
# fluid = iapws.IAPWS95

# Define kind of plot
xAxis = "s"
yAxis = "P"

# Point count for line, high value get more definition but slow calculate time
points = 50

# Saturation line format
isosat_kw = {"ls": "-", "color": "black", "lw": 1}

# Isoquality lines to plot
isoq = np.arange(0.1, 1, 0.1)
isoq_kw = {"ls": "--", "color": "black", "lw": 0.5}
labelq_kw = {"size": "xx-small", "ha": "right", "va": "center"}

# Isotherm lines to plot, values in ºC
isoT = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 1200, 1600, 2000]
isoT_kw = {"ls": "-", "color": "red", "lw": 0.5}
labelT_kw = {"size": "xx-small", "ha": "right", "va": "bottom"}

# Isobar lines to plot
isoP = [Pt, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100]
isoP_kw = {"ls": "-", "color": "blue", "lw": 0.5}
labelP_kw = {"size": "xx-small", "ha": "center", "va": "center"}

# Isoenthalpic lines to plot
isoh = np.arange(200, 4400, 200)
isoh_kw = {"ls": "-", "color": "green", "lw": 0.5}
labelh_kw = {"size": "xx-small", "ha": "center", "va": "center"}

# Isoentropic lines to plot
isos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
isos_kw = {"ls": "-", "color": "brown", "lw": 0.5}
labels_kw = {"size": "xx-small", "ha": "center", "va": "center"}

# # Isochor lines to plot
isov = [0.1, 1, 10, 100]
isov_kw = {"ls": "-", "color": "green", "lw": 0.5}

# Show region limits
regionBoundary = True

# Show region5
region5 = False

###############################################################################
# Calculate
###############################################################################

# Set plot label
title = {
    "T": "T, K",
    "P": "P, MPa",
    "v": "v, m³/kg",
    "h": "h, kJ/kg",
    "s": "s, kJ/kgK"}


# Check axis correct definition
validAxis = ", ".join(title.keys())
if xAxis not in title:
    raise ValueError("X axis variable don´t supported, valid only ", validAxis)
if yAxis not in title:
    raise ValueError("Y axis variable don´t supported, valid only ", validAxis)
if xAxis == yAxis:
    raise ValueError("X and Y axis can't show same variable")


# Set plot legend
plt.title("%s-%s Diagram" % (yAxis, xAxis))
xtitle = title[xAxis]
plt.xlabel(xtitle)
ytitle = title[yAxis]
plt.ylabel(ytitle)


# Set logaritmic scale if apropiate
if xAxis in ["P", "v"]:
    plt.xscale("log")
if yAxis in ["P", "v"]:
    plt.yscale("log")
plt.grid(True)


# Calculate point of isolines
Ps = list(np.concatenate([
    np.logspace(np.log10(Pt), np.log10(0.1*Pc), points),
    np.linspace(0.1*Pc, 0.9*Pc, points),
    np.linspace(0.9*Pc, 0.99*Pc, points),
    np.linspace(0.99*Pc, Pc, points)]))
Pl = list(np.concatenate([
    np.logspace(np.log10(Pt), np.log10(0.1*Pc), points),
    np.linspace(0.1*Pc, 0.5*Pc, points),
    np.linspace(0.5*Pc, 0.9*Pc, points),
    np.linspace(0.9*Pc, 0.99*Pc, points),
    np.linspace(0.99*Pc, Pc, points),
    np.linspace(Pc, 1.01*Pc, points),
    np.linspace(1.01*Pc, 1.1*Pc, points),
    np.linspace(1.1*Pc, 50, points),
    np.linspace(50, 100, points)]))
Tl = list(np.concatenate([
    np.linspace(0, 25, points),
    np.linspace(25, 0.5*Tc, points),
    np.linspace(0.5*Tc, 0.9*Tc, points),
    np.linspace(0.9*Tc, Tc, points),
    np.linspace(Tc, 1.1*Tc, points),
    np.linspace(1.1*Tc, 1.1*Tc, points),
    np.linspace(1.1*Tc, 800, points),
    np.linspace(800, 2000, points)]))


# Calculate saturation line
print("Calculating saturation lines...")
liq = [fluid(P=p, x=0) for p in Ps]
xliq = [l.__getattribute__(xAxis) for l in liq]
yliq = [l.__getattribute__(yAxis) for l in liq]
plt.plot(xliq, yliq, **isosat_kw)
vap = [fluid(P=p, x=1) for p in Ps]
xvap = [v.__getattribute__(xAxis) for v in vap]
yvap = [v.__getattribute__(yAxis) for v in vap]
plt.plot(xvap, yvap, **isosat_kw)


# Calculate isoquality lines
print("Calculating isoquality lines...")
Q: Dict[str, Dict[str, List[Any]]] = {}
for q in isoq:
    Q["%s" % q] = {}
    txt = "x=%s" % q
    print("    %s" % txt)
    pts = [fluid(P=p, x=q) for p in Ps]
    x = [p.__getattribute__(xAxis) for p in pts]
    y = [p.__getattribute__(yAxis) for p in pts]
    Q["%s" % q]["x"] = x
    Q["%s" % q]["y"] = y
    plt.plot(x, y, **isoq_kw)


# Calculate isotherm lines
if xAxis != "T" and yAxis != "T":
    print("Calculating isotherm lines...")
    T_: Dict[str, Dict[str, List[Any]]] = {}
    for T in isoT:
        T_["%s" % T] = {}
        print("    T=%sºC" % T)
        # Calculate the saturation point if available
        if T+273.15 < Tc:
            liqsat = fluid(T=T+273.15, x=0)
            vapsat = fluid(T=T+273.15, x=1)
            sat = True
        else:
            sat = False
        pts = []
        for pressure in Pl:
            try:
                point = fluid(P=pressure, T=T+273.15)
                if fluid == iapws.IAPWS97 and not region5 and \
                        point.region == 5:
                    continue
                # Add saturation point if neccesary
                if sat and T+273.15 < Tc and point.s < vapsat.s:
                    pts.append(vapsat)
                    pts.append(liqsat)
                    sat = False
                pts.append(point)
            except NotImplementedError:
                pass
        x = []
        y = []
        for p in pts:
            if p.status:
                x.append(p.__getattribute__(xAxis))
                y.append(p.__getattribute__(yAxis))
        plt.plot(x, y, **isoT_kw)
        T_["%s" % T]["x"] = x
        T_["%s" % T]["y"] = y


# Calculate isobar lines
if xAxis != "P" and yAxis != "P":
    print("Calculating isobar lines...")
    P_: Dict[str, Dict[str, List[Any]]] = {}
    for P in isoP:
        print("    P=%sMPa" % P)
        P_["%s" % P] = {}
        # Calculate the saturation point if available
        if P < Pc:
            liqsat = fluid(P=P, x=0)
            vapsat = fluid(P=P, x=1)
            sat = True
        else:
            sat = False
        pts = []
        for t in Tl:
            try:
                point = fluid(P=P, T=t+273.15)
                if fluid == iapws.IAPWS97 and not region5 and \
                        point.region == 5:
                    continue
                # Add saturation point if neccesary
                if sat and P < Pc and point.status and point.s > vapsat.s:
                    pts.append(liqsat)
                    pts.append(vapsat)
                    sat = False
                pts.append(point)
            except NotImplementedError:
                pass

        x = []
        y = []
        for p in pts:
            if p.status:
                x.append(p.__getattribute__(xAxis))
                y.append(p.__getattribute__(yAxis))
        plt.plot(x, y, **isoP_kw)
        P_["%s" % P]["x"] = x
        P_["%s" % P]["y"] = y

# Calculate isoenthalpic lines
if xAxis != "h" and yAxis != "h":
    print("Calculating isoenthalpic lines...")
    H_: Dict[str, Dict[str, List[Any]]] = {}
    for h in isoh:
        print("    h=%skJ/kg" % h)
        H_["%s" % h] = {}
        pts = []
        for pressure in Pl:
            try:
                point = fluid(P=pressure, h=h)
                if fluid == iapws.IAPWS97 and not region5 and \
                        point.region == 5:
                    continue
                pts.append(point)
            except NotImplementedError:
                pass
        x = []
        y = []
        for p in pts:
            if p.status:
                x.append(p.__getattribute__(xAxis))
                y.append(p.__getattribute__(yAxis))
        plt.plot(x, y, **isoh_kw)
        H_["%s" % h]["x"] = x
        H_["%s" % h]["y"] = y

# Calculate isoentropic lines
if xAxis != "s" and yAxis != "s":
    print("Calculating isoentropic lines...")
    S_: Dict[str, Dict[str, List[Any]]] = {}
    for s in isos:
        print("    s=%skJ/kgK" % s)
        S_["%s" % s] = {}
        pts = []
        for pressure in Pl:
            try:
                point = fluid(P=pressure, s=s)
                if fluid == iapws.IAPWS97 and not region5 and \
                        point.region == 5:
                    continue
                pts.append(point)
            except NotImplementedError:
                pass
        x = []
        y = []
        for p in pts:
            if p.status:
                x.append(p.__getattribute__(xAxis))
                y.append(p.__getattribute__(yAxis))
        plt.plot(x, y, **isos_kw)
        S_["%s" % s]["x"] = x
        S_["%s" % s]["y"] = y

# Calculate isochor lines
if xAxis != "v" and yAxis != "v":
    print("Calculating isochor lines...")
    for v in isov:
        print("    v=%s" % v)
        pts95 = [iapws.IAPWS95(T=t, v=v) for t in Tl]
        x = []
        y = []
        for p95 in pts95:
            if p95.status:
                x.append(p95.__getattribute__(xAxis))
                y.append(p95.__getattribute__(yAxis))
        plt.plot(x, y, **isov_kw)


# Plot region limits
if regionBoundary:
    # Boundary 1-3
    Po = _PSat_T(623.15)
    numpy_pressure_points = np.linspace(Po, 100, points)
    pts = [fluid(P=float(p), T=623.15) for p in numpy_pressure_points]
    x = [p.__getattribute__(xAxis) for p in pts]
    y = [p.__getattribute__(yAxis) for p in pts]
    plt.plot(x, y, **isosat_kw)

    # Boundary 2-3
    numpy_temp_points = list(map(float, np.linspace(623.15, 863.15)))
    Psf = [_P23_T(t) for t in numpy_temp_points]
    Psf[-1] = 100.0  # Avoid round problem with value out of range > 100 MPa
    pts = [fluid(P=p, T=t) for p, t in zip(Psf, numpy_temp_points)]
    x = [p.__getattribute__(xAxis) for p in pts]
    y = [p.__getattribute__(yAxis) for p in pts]
    plt.plot(x, y, **isosat_kw)


# Show annotate in plot
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
for q in isoq:
    x = Q["%s" % q]["x"]
    y = Q["%s" % q]["y"]

    txt = "x=%s" % q
    i = 0
    j = i+1

    if xAxis in ["P", "v"]:
        fx = (log(x[i])-log(x[j]))/(log(xmax)-log(xmin))
    else:
        fx = (x[i]-x[j])/(xmax-xmin)
    if yAxis in ["P", "v"]:
        fy = (log(y[i])-log(y[j]))/(log(ymax)-log(ymin))
    else:
        fy = (y[i]-y[j])/(ymax-ymin)
    rot = atan(fy/fx)*360/2/pi
    plt.annotate(txt, (x[i], y[i]), rotation=rot, **labelq_kw)

if xAxis != "T" and yAxis != "T":
    for T in isoT:
        x = T_["%s" % T]["x"]
        y = T_["%s" % T]["y"]

        if not x:
            continue

        txt = "%sºC" % T
        i = 0
        j = i+2

        if xAxis in ["P", "v"]:
            fx = (log(x[i])-log(x[j]))/(log(xmax)-log(xmin))
        else:
            fx = (x[i]-x[j])/(xmax-xmin)
        if yAxis in ["P", "v"]:
            fy = (log(y[i])-log(y[j]))/(log(ymax)-log(ymin))
        else:
            fy = (y[i]-y[j])/(ymax-ymin)
        rot = atan(fy/fx)*360/2/pi
        plt.annotate(txt, (x[i], y[i]), rotation=rot, **labelT_kw)

if xAxis != "P" and yAxis != "P":
    for P in isoP:
        x = P_["%s" % P]["x"]
        y = P_["%s" % P]["y"]

        if not x:
            continue

        txt = "%sMPa" % P
        i = len(x)-15
        j = i-2

        if xAxis in ["P", "v"]:
            fx = (log(x[i])-log(x[j]))/(log(xmax)-log(xmin))
        else:
            fx = (x[i]-x[j])/(xmax-xmin)
        if yAxis in ["P", "v"]:
            fy = (log(y[i])-log(y[j]))/(log(ymax)-log(ymin))
        else:
            fy = (y[i]-y[j])/(ymax-ymin)
        rot = atan(fy/fx)*360/2/pi
        plt.annotate(txt, (x[i], y[i]), rotation=rot, **labelP_kw)

if xAxis != "h" and yAxis != "h":
    for h in isoh:
        x = H_["%s" % h]["x"]
        y = H_["%s" % h]["y"]

        if not x:
            continue
        if h % 1000:
            continue

        txt = "%s J/g" % h
        i = points
        j = i+2

        if xAxis in ["P", "v"]:
            fx = (log(x[i])-log(x[j]))/(log(xmax)-log(xmin))
        else:
            fx = (x[i]-x[j])/(xmax-xmin)
        if yAxis in ["P", "v"]:
            fy = (log(y[i])-log(y[j]))/(log(ymax)-log(ymin))
        else:
            fy = (y[i]-y[j])/(ymax-ymin)
        rot = atan(fy/fx)*360/2/pi
        plt.annotate(txt, (x[i], y[i]), rotation=rot, **labelh_kw)

if xAxis != "s" and yAxis != "s":
    for s in isos:
        x = S_["%s" % s]["x"]
        y = S_["%s" % s]["y"]

        txt = "%s J/gK" % s
        i = len(x)//2
        if s > 10:
            j = i+1
        else:
            j = i+5

        if xAxis in ["P", "v"]:
            fx = (log(x[i])-log(x[j]))/(log(xmax)-log(xmin))
        else:
            fx = (x[i]-x[j])/(xmax-xmin)
        if yAxis in ["P", "v"]:
            fy = (log(y[i])-log(y[j]))/(log(ymax)-log(ymin))
        else:
            fy = (y[i]-y[j])/(ymax-ymin)
        rot = atan(fy/fx)*360/2/pi
        plt.annotate(txt, (x[i], y[i]), rotation=rot, **labels_kw)

plt.show()
