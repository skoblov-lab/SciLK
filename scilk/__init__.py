import sys

if sys.version_info < (3, 5, 2):
    print("ChemPred required Python >= 3.5.2")
    sys.exit(1)
