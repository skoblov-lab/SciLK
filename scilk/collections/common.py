import operator as op
import csv
from itertools import chain
from typing import Sequence, Iterable, TypeVar, List, Tuple, Callable, Mapping, Union

import numpy as np
from multipledispatch import dispatch
from fn import F
import pandas as pd

from scilk.util import preprocessing, intervals

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
# TODO patch dispatcher namespace overlapping


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

    @dispatch(str)
    def charencoder(string: str) -> np.ndarray:
        return encode_string(string)

    @dispatch(Iterable)
    def charencoder(strings: Iterable[str]):
        encoded_strings = list(map(encode_string, strings))
        if not encoded_strings:
            raise ValueError('there are no `strings`')
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
    vectors = embeddings.as_matrix().astype(np.float32)

    def index(word: str) -> int:
        if not word:
            raise ValueError("can't encode empty words")
        return wordmap.get(transform(word), oov)

    @dispatch(str)
    def wordencoder(word: str) -> np.ndarray:
        return vectors[index(word)]

    @dispatch(Iterable)
    def wordencoder(words: Iterable[str]) -> np.ndarray:
        indices = list(map(index, words))
        if not indices:
            raise ValueError('there are no `words`')
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


def merge_bins(sources: Union[np.ndarray, Sequence[np.ndarray]],
               bins: Sequence[Sequence[int]], dtype=None) -> np.ndarray:
    """
    Merge sources within bins and stack them on top of each other.
    :param sources: a Sequence of source arrays.
    :param bins: a Sequence of bins: Sequences of indices referencing
    arrays in `sources`.
    :param dtype: numpy data type; if None `sources[0].dtype` will be used
    instead
    :return: a merged arrays
    """
    if not len(sources):
        raise ValueError('no `sources`')
    extracted = (
            F(preprocessing.binextract) >> (map, np.concatenate) >> list
    )(sources, bins)
    return preprocessing.stack(extracted, None,
                               dtype=(dtype or sources[0].dtype))[0]


def unbin(binned: Iterable[Iterable[T]], bins: Iterable[Iterable[int]]) \
        -> List[T]:
    """
    Revert binning: transform a nested Iterable of objects (i.e. objects packed
    into bins) into a list of objects ordered the same way as the original
    Sequence
    :param binned: a nested Iterable of binned objects
    :param bins: a nested Iterable of bins: Iterables of indices referencing
    objects in the original Sequence
    :return:
    """
    return (F(map, chain.from_iterable) >>
            (lambda x: zip(*x)) >>
            F(sorted, key=op.itemgetter(0)) >>
            (map, op.itemgetter(1)) >> list)([bins, binned])


def unmerge_bins(merged: np.ndarray, bins: Sequence[Sequence[int]],
                 lengths: Sequence[int]) -> List[List[np.ndarray]]:
    """
    Breaks `merged` into binned objects corresponding to the original objects
    in a binned Sequence
    :param merged: a merged representation of binned data
    :param bins: a Sequence of bins: Sequences of indices referencing
    :param lengths: lengths of the original source objects
    :return:
    """
    lengths_ = np.array(lengths)
    indices = [lengths_[bin_] for bin_ in bins]
    return [list(np.split(line, np.cumsum(l_indices)))[:-1]
            for line, l_indices in zip(merged, indices)]


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
