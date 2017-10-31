import sys
from distutils.core import setup
from setuptools import find_packages

# TODO add loggers and warnings
# TODO lazy module improting (https://github.com/bwesterb/py-demandimport)

if sys.version_info < (3, 5, 2):
    print("SciLK requires Python >= 3.5.2")
    sys.exit(1)

# from Cython.Build import cythonize
#
# os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="scilk",
    version="0.1a1",
    packages=find_packages("./"),
    scripts=[],
    requires=["numpy",
              "h5py",
              "fn",
              "pyrsistent",
              "keras",
              "scikit-learn",
              "pandas",
              "hypothesis",
              "frozendict",
              "tensorflow"]
)
