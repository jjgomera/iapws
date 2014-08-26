from setuptools import setup

from iapws.iapws import __version__

with open('README.rst') as file:
    long_description = file.read()

setup(
    name='iapws',
    version=__version__,
    packages=['iapws'],
    author='jjgomera',
    author_email='jjgomera@gmail.com',
    url='https://github.com/jjgomera/iapws',
    description='Python implementation of international-standard IAPWS-IF97',
    long_description=long_description,
    license="gpl v3",
    install_requires=["scipy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules"
        ]
    )
