import typing
from infgain.preprocessing import remove_separators


def load_corpus(filename: str) -> str:
    """
    Remove separators from `filename` corpus.
    :param filename:
    :return:
    >>> raw_data = load_corpus('data/text_small.txt')
    >>> len(raw_data) == 2940422
    True
    """
    with open(filename, 'r') as f:
        raw_data = f.read()
        # Get rid of underscores and tildas
        raw_data = remove_separators(raw_data)
        return raw_data


def load_example_corpus() -> str:
    """
    :return:
    >>> raw_data = load_example_corpus()
    >>> len(raw_data) == 2940422
    True
    """
    return load_corpus('data/text_small.txt')


if __name__ == "__main__":
    raise RuntimeError
