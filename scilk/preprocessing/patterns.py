import re


numeric = re.compile("[0-9]*\.?[0-9]+")
wordlike = re.compile("\w+")
misc = re.compile("[^\s\w]")


if __name__ == "__main__":
    raise RuntimeError
