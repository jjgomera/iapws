from distutils.core import setup

from iapws import __version__

setup(
    name='iapws',
    version=__version__,
    py_modules=['iapws'],
    author='jjgomera',
    author_email='jjgomera@gmail.com',
    url='https://github.com/jjgomera/iapws',
    description='Python implementation of international-standard IAPWS-IF97 steam tables',
    license="gpl v3")
