import csv
from itertools import chain
from typing import Sequence, Iterable, TypeVar, List, Tuple, Callable, Mapping, Union

import numpy as np
from fn import F
import pandas as pd

from scilk.util import preprocessing
from scilk.util.binning import unbin, unmerge_bins

T = TypeVar('T')

TextEncoder = Callable[[Union[str, Iterable[str]]], np.ndarray]


def asciicharset(strings: Iterable[str]) -> List[str]:
    """
    Return a sorted list of unique ascii characters
    :param strings: an iterable of strings to extract characters from
    :return:
    """
    characters = chain.from_iterable(strings)
    return sorted(set(filter(lambda x: ord(x) < 128, characters)))


# TODO specify all exception in the docs

def build_charencoder(corpus: Iterable[str], wordlen: int=None) \
        -> Tuple[int,  Mapping[str, int], TextEncoder]:
    """
    Create a char-level encoder: a Callable, mapping strings into integer arrays.
    Encoders dispatch on input type: if you pass a single string, you will get
    a 1D array, if you pass an Iterable of strings, you will get a 2D array
    where row i encodes the i-th string in the Iterable.
    :param corpus: an Iterable of strings to extract characters from. The
    encoder will map any non-ASCII character into the OOV code.
    :param wordlen: when `wordlen` is None and an encoder receives an Iterable of
    strings, the second dimension in the output array will be as long as the
    longest string, otherwise it will be `wordlen` long. In the latter case
    words exceeding `wordlen` will be trimmed. In both cases empty-spaces are
    filled with zeros.
    in the Iterable. If wordlen is not
    :return: the OOV code, a character mapping representing non-OOV character
    encodings, an encoder
    """
    if wordlen and wordlen < 1:
        raise ValueError('`wordlen` must be positive')
    try:
        charmap = {char: i + 1 for i, char in enumerate(asciicharset(corpus))}
    except TypeError:
        raise ValueError('`corpus` can be either a string or an Iterable of '
                         'strings')
    if not charmap:
        raise ValueError('the `corpus` is empty')
    oov = len(charmap) + 1

    def encode_string(string: str) -> np.ndarray:
        if not string:
            raise ValueError("can't encode empty strings")
        return np.fromiter((charmap.get(char, oov) for char in string), np.int32,
                           len(string))

    def charencoder(target: Union[str, Iterable[str]]):
        if isinstance(target, str):
            return encode_string(target)
        encoded_strings = list(map(encode_string, target))
        if not encoded_strings:
            raise ValueError('there are no `target`')
        return preprocessing.stack(
            encoded_strings, [wordlen or -1], np.int32, 0, True)[0]

    return oov, charmap, charencoder


def build_wordencoder(embeddings: pd.DataFrame, transform: Callable[[str], str]) \
        -> TextEncoder:
    """
    Create a word-level encoder: a Callable, mapping strings into integer arrays.
    Encoders dispatch on input type: if you pass a single string, you will get
    a 1D array, if you pass an Iterable of strings, you will get a 2D array,
    where row i encodes the i-th string in the Iterable.
    :param embeddings: a dataframe of word vectors indexed by words. The last
    vector (row) is used to encode OOV words.
    :return:
    """
    wordmap = {word: i for i, word in enumerate(embeddings.index)}
    if not wordmap:
        raise ValueError('empty `embeddings`')
    if not all(isinstance(word, str) for word in wordmap):
        raise ValueError('`embeddings` can be indexed by strings alone')
    oov = wordmap[embeddings.index[-1]]
    vectors = embeddings.as_matrix()

    def index(word: str) -> int:
        if not word:
            raise ValueError("can't encode empty words")
        return wordmap.get(transform(word), oov)

    def wordencoder(target: Union[str, Iterable[str]]) -> np.ndarray:
        if isinstance(target, str):
            return vectors[index(target)]
        indices = list(map(index, target))
        if not indices:
            raise ValueError('there are no `target`s')
        return np.vstack(vectors[indices])

    return wordencoder


def read_glove(path: str) -> pd.DataFrame:
    """
    Read Glove embeddings in text format. The file can be compressed.
    :param path:
    :return:
    """
    return pd.read_table(
        path, sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE,
        na_values=None, keep_default_na=False
    ).astype(np.float32)


def decode_merged_predictions(merged: np.ndarray, bins: Sequence[Sequence[int]],
                              lengths: Sequence[int]) -> List[Sequence[int]]:
    """
    :param merged: merged predictions
    :param bins: bins
    :param lengths: text lengths
    """
    unmerged = unmerge_bins(merged, bins, lengths)
    unbined = (F(map, preprocessing.reverse) >> list)(unbin(unmerged, bins))
    return [np.nonzero(anno > 0.5)[0] for anno in unbined]


if __name__ == '__main__':
    raise RuntimeError
