import sys
from distutils.core import setup
from setuptools import find_packages

# TODO add loggers and warnings
# TODO lazy module importing (https://github.com/bwesterb/py-demandimport)

if sys.version_info < (3, 5, 2):
    print("SciLK requires Python >= 3.5.2")
    sys.exit(1)

# from Cython.Build import cythonize
#
# os.environ['CFLAGS'] = '-O3 -Wall'

setup(
    name="scilk",
    version="0.1a1",
    packages=find_packages(),
    scripts=[],
    install_requires=["numpy==1.14.0",
                      "h5py==2.7.1",
                      "fn",
                      "pyrsistent",
                      "scikit-learn==0.19.1",
                      "pandas==0.22.0",
                      "hypothesis",
                      "frozendict",
                      "joblib==0.11",
                      "tensorflow==1.4.1",
                      "keras==2.1.3",
                      "binpacking==1.3"]
)
