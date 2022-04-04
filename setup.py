#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Install IAPWS module using setuptools."""

from setuptools import setup

import io  # for backwards compatibility with Python 2
import os


with open(os.path.join("iapws", "VERSION")) as version_file:
    __version__ = version_file.read().strip()

with io.open('README.rst', encoding="utf8") as file:
    long_description = file.read()

setup(
    name='iapws',
    version=__version__,
    packages=['iapws'],
    include_package_data=True,
    author='jjgomera',
    author_email='jjgomera@gmail.com',
    url='https://github.com/jjgomera/iapws',
    download_url='https://github.com/jjgomera/iapws/tarball/v' + __version__,
    description='Python implementation of standards from The International'
                'Association for the Properties of Water and Steam',
    long_description=long_description,
    license="gpl v3",
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
    install_requires=["scipy>=1.2"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
