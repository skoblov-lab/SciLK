"""



"""


import operator as op
from itertools import chain, repeat, count
from math import ceil
from typing import List, Tuple, Optional, TypeVar, Sequence

import numpy as np
from fn import F


T = TypeVar('T')


homogenous = F(map) >> set >> len >> F(op.contains, [0, 1])
flatmap = F(map) >> chain.from_iterable
strictmap = F(map) >> list


def flatzip(flat, nested):
    flatrep = map(F(map, repeat), flat)
    iterables = (*flatrep, *nested)
    return (F(zip) >> F(map, lambda x: zip(*x)) >> chain.from_iterable)(*iterables)


def maxshape(arrays: Sequence[np.ndarray]) -> Tuple[int]:
    """
    :param arrays: a nonempty sequence of arrays; the sequence must be
    homogeneous with respect to dimensionality.
    :raises ValueError: if `arrays` sequence is empty; if arrays have different
    dimensionality.
    """
    if not arrays:
        raise ValueError('`arrays` should not be empty')
    if not homogenous(np.ndim, arrays):
        raise ValueError('`arrays` must have homogeneous dimensionality')
    return tuple(np.array([array.shape for array in arrays]).max(axis=0))


def stack(arrays: Sequence[np.ndarray], shape: Optional[Sequence[int]], dtype,
          filler=0, trim=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack N-dimensional arrays with variable sizes across dimensions.
    :param arrays: a nonempty sequence of arrays; the sequence must be
    homogeneous with respect to dimensionality.
    :param shape: target shape to broadcast each array to. The shape must
    specify one integer per dimension – the output will thus have shape
    `[len(arrays), *shape]`. If None the function will infer the maximal size
    per dimension from `arrays`. To infer size for individual dimension(s)
    use -1.
    :param dtype: output data type
    :param filler: a value to fill in the empty space.
    :param trim: trim arrays to fit the `shape`.
    :raises ValueError: if `len(shape)` doesn't match the dimensionality of
    arrays in `arrays`; if an array can't be broadcasted to `shape` without
    trimming, while trimming is disabled; + all cases specified in function
    `maxshape`
    :return: stacked arrays, a boolean mask (empty positions are False).
    >>> from random import choice
    >>> maxlen = 100
    >>> ntests = 10000
    >>> lengths = range(10, maxlen+1, 2)
    >>> arrays = [
    ...    np.random.randint(0, 127, size=choice(lengths)).reshape((2, -1))
    ...    for _ in range(ntests)
    ... ]
    >>> stacked, masks = stack(arrays, [-1, maxlen], np.int)
    >>> all((arr.flatten() == s[m].flatten()).all()
    ...     for arr, s, m in zip(arrays, stacked, masks))
    True
    >>> stacked, masks = stack(arrays, [2, -1], np.int)
    >>> all((arr.flatten() == s[m].flatten()).all()
    ...     for arr, s, m in zip(arrays, stacked, masks))
    True
    """
    def slices(limits: Tuple[int], array: np.ndarray) -> List[slice]:
        stops = [min(limit, size) for limit, size in zip(limits, array.shape)]
        return [slice(0, stop) for stop in stops]

    if not isinstance(arrays, Sequence):
        raise ValueError('`arrays` must be a Sequence object')
    ndim = arrays[0].ndim
    if shape is not None and len(shape) != ndim:
        raise ValueError("`shape`'s dimensionality doesn't match that of "
                         "`arrays`")
    if shape is not None and any(s < 1 and s != -1 for s in shape):
        raise ValueError('the only allowed non-positive value in `shape` is -1')
    # infer size across all dimensions
    inferred = np.array(maxshape(arrays))
    # mix inferred and requested sizes where requested
    limits = (inferred if shape is None else
              np.where(np.array(shape) == -1, inferred, shape))
    # make sure everything fits fine
    if not (shape is None or trim or (inferred <= limits).all()):
        raise ValueError("can't broadcast all arrays to `shape` without "
                         "trimming")
    stacked = np.full([len(arrays), *limits], filler, dtype=dtype)
    mask = np.zeros([len(arrays), *limits], dtype=bool)
    for i, arr, slices_ in zip(count(), arrays, map(F(slices, limits), arrays)):
        op.setitem(stacked, [i, *slices_], op.getitem(arr, slices_))
        op.setitem(mask, [i, *slices_], True)
    stacked[~mask] = filler
    return stacked, mask


def maskfalse(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace False-masked items with zeros.
    >>> array = np.arange(10)
    >>> mask = np.random.binomial(1, 0.5, len(array)).astype(bool)
    >>> masked = maskfalse(array, mask)
    >>> (masked[mask] == array[mask]).all()
    True
    >>> (masked[~mask] == 0).all()
    True
    """
    if not np.issubdtype(mask.dtype, np.bool):
        raise ValueError("Masks are supposed to be boolean")
    copy = array.copy()
    copy[~mask] = 0
    return copy


def chunksteps(size: int, array: np.ndarray, filler=0) -> np.ndarray:
    """
    Chunk time steps, that is break an array into fixed-size slices along the
    second dimension (array.shape[1]).
    :param size: chunk size
    :param array: an array to chunk. The array must have at lest two dimensions
    :param filler: a value to fill in the empty space in the last chunk if
    `array.shape[1] % size != 0`
    :return:
    """
    nchunks = int(ceil(array.shape[1] / size))
    chunks = [array[:, start:start+size] for start in range(0, size*nchunks, size)]
    assert chunks[-1].shape[1] <= size
    if chunks[-1].shape[1] < size:
        chunk = np.full((array.shape[0], size, *array.shape[2:]), filler,
                         dtype=array.dtype)
        chunk[:, :chunks[-1].shape[1]] = chunks[-1]
        chunks[-1] = chunk
    return np.array(chunks)


reverse = op.itemgetter(slice(None, None, -1))  # reverse a Sequence or an array


if __name__ == '__main__':
    raise RuntimeError
