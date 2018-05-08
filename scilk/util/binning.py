import operator as op
from itertools import chain
from numbers import Number
from typing import Union, Sequence, Iterable, List, TypeVar, Callable

import numpy as np
from binpacking import to_constant_bin_number
from fn import F

from scilk.util import preprocessing


T = TypeVar('T')


def binpack(nbins: int, weight: Callable[[T], Number], items: Sequence[T]) \
        -> List[List[int]]:
    """
    Pack items into n bins while minimising the variance of weight accumulated
    in each bin. The function uses a greedy algorithm, which doesn't not
    guarantee a perfect result.
    :param nbins: the number of bins to create
    :param weight: a weight function
    :param items: items to pack; since the function returns bins packed with
    positions inferred from iteration order, iteration over `items` must be
    stable for the output to be useful.
    :return: a nested list of integers representing positions in `items`
    """
    if len(items) < nbins:
        raise ValueError('There should be at lest `nbins` items')
    weighted = [(i, weight(item)) for i, item in enumerate(items)]
    return (F(map, F(map, op.itemgetter(0)) >> list) >> list)(
        to_constant_bin_number(weighted, nbins, weight_pos=1)
    )


def binextract(source: Union[Sequence[T], np.ndarray], bins: Sequence[Sequence[int]]) \
        -> Union[List[List[T]], List[np.ndarray]]:
    """
    'Materialise' bins, i.e. transform a nested list of indices into bins of
    source items. See `binpack` for additional info.
    :param source: source items
    :param bins: a nested sequence if integers - indices referring to object
    from `source`
    :return:
    """
    if not isinstance(source, (Sequence, np.ndarray)):
        raise ValueError('`source` must be either a Sequence or a numpy array')
    try:
        return (
            [source[bin_] for bin_ in bins] if isinstance(source, np.ndarray) else
            [[source[i] for i in bin_] for bin_ in bins]
        )
    except IndexError:
        raise ValueError('`bins` contain indices outside of the `source` range')


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
            F(binextract) >> (map, np.concatenate) >> list
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


if __name__ == '__main__':
    raise RuntimeError
