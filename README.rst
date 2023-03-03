.. image:: https://dl.circleci.com/status-badge/img/gh/jjgomera/iapws/tree/master.svg?style=svg
    :target: https://dl.circleci.com/status-badge/redirect/gh/jjgomera/iapws/tree/master
    :alt: Build Status

.. image:: https://ci.appveyor.com/api/projects/status/a128sh8e50cjsiya?svg=true
    :target: https://ci.appveyor.com/project/jjgomera/iapws
    :alt: Windows Build Status

.. image:: https://coveralls.io/repos/github/jjgomera/iapws/badge.svg?branch=master
    :target: https://coveralls.io/github/jjgomera/iapws?branch=master
    :alt: coveralls.io analysis

.. image:: https://codecov.io/gh/jjgomera/iapws/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/jjgomera/iapws
    :alt: codecov.io analysis

.. image:: https://app.codacy.com/project/badge/Grade/bb92d537dfa1461d919a0782f3c398b9
    :target: https://www.codacy.com/gh/jjgomera/iapws/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jjgomera/iapws&amp;utm_campaign=Badge_Grade
    :alt: Code Quality

.. image:: http://readthedocs.org/projects/iapws/badge/?version=latest
    :target: http://iapws.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4744318.svg
   :target: https://doi.org/10.5281/zenodo.4744318
   :alt: DOI


iapws
=====

Python implementation of standard from IAPWS (http://www.iapws.org/release.html). The module implements the full set of standards, including::

    IAPWS-IF97
    IAPWS-95
    IAPWS-06 for Ice
    IAPWS-08 for seawater
    IAPWS-17 for Heavy water
    ...
    

dependences
--------------------

* Support for both python branch::

  * python 2.7
  * python 3.4 or later

* Numpy-scipy: library with mathematic and scientific tools


install
--------------------

In debian you can find in official repositories in jessie, testing and sid. In ubuntu it's in official repositories from ubuntu saucy (13.10). In other system you can install using pip::

	pip install iapws
 
or directly from the github repository::

    pip install git+https://github.com/jjgomera/iapws.git

This is the recommended option to have the latest version.


documentation
--------------------
 
To see the full documentation of package, see `readthedocs <http://iapws.readthedocs.io/>`__

.. inclusion-marker-do-not-remove

For a rapid usage demostration, see this examples 


IAPWS-IF97 (`see full documentation <https://iapws.readthedocs.io/en/latest/iapws.iapws97.html#iapws.iapws97.IAPWS97>`__)

.. code:: python

	from iapws import IAPWS97
	sat_steam=IAPWS97(P=1,x=1)                #saturated steam with known P
	sat_liquid=IAPWS97(T=370, x=0)            #saturated liquid with known T
	steam=IAPWS97(P=2.5, T=500)               #steam with known P and T
	print(sat_steam.h, sat_liquid.h, steam.h) #calculated enthalpies
    

IAPWS-95 (`see full documentation <https://iapws.readthedocs.io/en/latest/iapws.iapws95.html#iapws.iapws95.IAPWS95>`__)

.. code:: python

	from iapws import IAPWS95
	sat_steam=IAPWS95(P=1,x=1)                #saturated steam with known P
	sat_liquid=IAPWS95(T=370, x=0)            #saturated liquid with known T
	steam=IAPWS95(P=2.5, T=500)               #steam with known P and T
	print(sat_steam.h, sat_liquid.h, steam.h) #calculated enthalpies
    

IAPWS-17 for Heavy water (`see full documentation <https://iapws.readthedocs.io/en/latest/iapws.iapws95.html#iapws.iapws95.D2O>`__)

.. code:: python

	from iapws import D2O
	sat_liquid=D2O(T=370, x=0)            #saturated liquid with known T
	print(sat_liquid.h)                   #calculated enthalpy


IAPWS-06 for Ice Ih (`see full documentation <https://iapws.readthedocs.io/en/latest/iapws._iapws.html#iapws._iapws._Ice>`__)

.. code:: python

    from iapws import _Ice
    ice=_Ice(273.15, 0.101325)            #Ice at normal melting point
    print(ice["rho"])                     #Calculated density


IAPWS-08 for seawater (`see full documentation <https://iapws.readthedocs.io/en/latest/iapws.iapws08.html#iapws.iapws08.SeaWater>`__)

.. code:: python

    from iapws import SeaWater
    state = SeaWater(T=300, P=0.101325, S=0.001)    #Seawater with 0.1% Salinity
    print(state.cp)                                 # Get cp


TODO
====

* TODO: Improve convergence in two phase region for IAPWS95 and D2O class
* TODO: Implement SBTL method for fast calculation
* TODO: Implement TTSE method for fast calculation

Ammonia-water mixture:

* TODO: Add equilibrium routine

I've tried to test all code and use all values for computer verification the standards give, but anyway the code can have hidden problem.
For any suggestions, comments, bugs ... you can usage the `github issue section <https://github.com/jjgomera/iapws/issues>`__, or contact directly with me at `email <jjgomera@gmail.com>`__.
