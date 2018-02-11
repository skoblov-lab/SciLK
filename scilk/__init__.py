"""



"""


import pathlib
import sys
import os


if sys.version_info < (3, 5, 2):
    print("ChemPred required Python >= 3.5.2")
    sys.exit(1)


SCILK_ROOT = os.path.abspath(os.environ.get('SCILK_ROOT') or
                             os.path.expanduser('~/.scilk'))
os.makedirs(SCILK_ROOT, exist_ok=True)


if __name__ == '__main__':
    raise RuntimeError
