import operator as op
from functools import reduce
from itertools import chain, repeat, groupby
from typing import List, Tuple, Optional, Mapping, Union

import numpy as np
from fn import F
from sklearn.utils import class_weight

homogenous = F(map) >> set >> len >> F(op.eq, 1)
flatmap = F(map) >> chain.from_iterable
oldmap = F(map) >> list


def flatzip(flat, nested):
    flatrep = map(F(map, repeat), flat)
    iterables = (*flatrep, *nested)
    return (F(zip) >> F(map, lambda x: zip(*x)) >> chain.from_iterable)(*iterables)


def join(arrays: List[np.ndarray], length: int, padval=0, trim=False) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Join 1D or 2D arrays. The function uses zero-padding to bring all arrays to the
    same length. The dtypes will be coerced to `dtype`
    :param arrays: arrays to join
    :param length: final sample length
    :param padval: padding value
    :param dtype: output data type (must be a numpy integral type)
    :return: (joined and padded arrays, boolean array masks); masks are
    positive, i.e. padded regions are False
    >>> import random
    >>> length = 100
    >>> ntests = 10000
    >>> arrays = [np.random.randint(0, 127, size=random.randint(1, length))
    ...           for _ in range(ntests)]
    >>> joined, masks = join(arrays, length)
    >>> all((arr == j[m]).all() for arr, j, m in zip(arrays, joined, masks))
    True
    """
    if length < max(map(len, arrays)) and not trim:
        raise ValueError("Some arrays are longer than `length`")
    ndim = set(arr.ndim for arr in arrays)
    if ndim not in ({1}, {2}):
        raise ValueError("`arrays` must be a nonempty list of 2D or 3D arrays ")
    masks = np.zeros((len(arrays), length), dtype=bool)
    shape = ((len(arrays), length) if ndim == {1} else
             (len(arrays), length, arrays[0].shape[1]))
    dtype = arrays[0].dtype
    joined = (
        np.repeat([padval], reduce(op.mul, shape)).reshape(shape).astype(dtype))
    for i, arr in enumerate(arrays):
        joined[i, :len(arr)] = arr[:length]
        masks[i, :len(arr)] = True
    return joined, masks


def one_hot(ncls: int, array: np.ndarray) -> np.ndarray:
    """
    One-hot encode an integer array; the output inherits the array's dtype.
    >>> nclasses = 10
    >>> permutations = np.vstack([np.random.permutation(nclasses)
    ...                           for _ in range(nclasses)])
    >>> (one_hot(permutations).argmax(permutations.ndim) == permutations).all()
    True
    """
    if not np.issubdtype(array.dtype, np.int):
        raise ValueError("`array.dtype` must be integral")
    if not len(array):
        return array
    vectors = np.eye(ncls, dtype=array.dtype)
    return vectors[array]


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


def balance_class_weights(y: np.ndarray, mask: Optional[np.ndarray]=None) \
        -> Optional[Mapping[int, float]]:
    # TODO update docs
    # TODO tests
    """
    :param y: a numpy array encoding sample classes; samples are encoded along
    the 0-axis
    :param mask: a boolean array of shape compatible with `y`, wherein True
    shows that the corresponding value(s) in `y` should be used to calculate
    weights; if `None` the function will consider all values in `y`
    :return: class weights
    """
    if not len(y):
        raise ValueError("`y` is empty")
    if y.ndim == 2:
        y_flat = (y.flatten() if mask is None else
                  np.concatenate([sample[mask] for sample, mask in zip(y, mask)]))
    elif y.ndim == 3:
        y_flat = (y.nonzero()[-1] if mask is None else
                  y[mask].nonzero()[-1])
    else:
        raise ValueError("`y` should be either a 2D or a 3D array")
    classes = np.unique(y_flat)
    weights = class_weight.compute_class_weight("balanced", classes, y_flat)
    weights_scaled = weights / weights.min()
    return {cls: weight for cls, weight in zip(classes, weights_scaled)}


def sample_weights(y: np.ndarray, class_weights: Mapping[int, float]) \
        -> np.ndarray:
    # TODO update docs
    # TODO tests
    """
    :param y: a 2D array encoding sample classes; each sample is a row of
    integers representing class code
    :param class_weights: a class to weight mapping
    :return: a 2D array of the same shape as `y`, wherein each position stores
    a weight for the corresponding position in `y`
    """
    weights_mask = np.zeros(shape=y.shape, dtype=np.float32)
    for cls, weight in class_weights.items():
        weights_mask[y == cls] = weight
    return weights_mask


def group(ids, sources, *args):
    """
    Group args by id and source
    :param ids:
    :param sources:
    :param args:
    :return:
    """
    records = zip(ids, sources, *args)
    id_groups = groupby(records, op.itemgetter(0))
    return [[list(grp) for _, grp in src_grps] for src_grps in
            (groupby(list(grp), op.itemgetter(1)) for _, grp in id_groups)]


if __name__ == "__main__":
    raise RuntimeError
