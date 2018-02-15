import operator as op
from itertools import chain
from typing import Sequence, Iterable, TypeVar, List, Tuple, Callable, Mapping

import numpy as np
from fn import F

from scilk.util import preprocessing, intervals

T = TypeVar('T')


def build_string_encoder(chars: str) \
        -> Tuple[int,  Mapping[str, int], Callable[[str], np.ndarray]]:
    """
    Create a string encoder: a Callable from strings to integer arrays
    :param chars: a sequence of characters to consider; the function will return
    the OOV code for any non-ASCII character.
    :return: the OOV code, a character mapping representing non-OOV character
    encodings, an encoder
    """
    charset = sorted(set(filter(lambda x: ord(x) < 128, chars)))
    charmap = {char: i + 1 for i, char in enumerate(charset)}
    oov = len(charmap) + 1

    def stringencoder(text: str) -> np.ndarray:
        return np.fromiter((charmap.get(char, oov) for char in text), np.int32,
                           len(text))

    return oov, charmap, stringencoder


def encode_tokens(stringencoder: Callable[[str], np.ndarray], maxlen: int,
                  tokens: Sequence[intervals.Interval[str]]) -> np.ndarray:
    """
    Encode tokens
    :param stringencoder: a function mapping strings into integer arrays
    :param maxlen: token length limit
    :param tokens: either a Sequence of Intervals loaded with token strings
    """
    tokens_strings = (tk.data for tk in tokens)
    return preprocessing.stack(
        list(map(stringencoder, tokens_strings)), [maxlen], np.int32, 0, True
    )[0]


def merge_bins(sources: Sequence[np.ndarray], bins: Sequence[Sequence[int]]) \
        -> np.ndarray:
    """
    Merge sources within bins and stack them on top of each other.
    :param sources: a Sequence of source arrays.
    :param bins: a Sequence of bins: Sequences of indices referencing
    arrays in `sources`.
    :return: a merged arrays
    """
    extracted = (
            F(preprocessing.binextract) >> (map, np.concatenate) >> list
    )(sources, bins)
    return preprocessing.stack(extracted, None, dtype=np.int32)[0]


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
    unbined = (F(map, reverse) >> list)(unbin(unmerged, bins))
    return [np.nonzero(anno > 0.5)[0] for anno in unbined]


if __name__ == '__main__':
    raise RuntimeError
