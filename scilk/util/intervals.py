import sys
from typing import TypeVar, Container, Generic, Optional, Sequence, Iterable, \
    List, Iterator, overload

import numpy as np

_slots_supported = (sys.version_info >= (3, 6, 2) or
                    (3, 5, 3) <= sys.version_info < (3, 6))
T = TypeVar("T")


class Interval(Container, Generic[T]):

    if _slots_supported:
        __slots__ = ("start", "stop", "data")

    def __init__(self, start: int, stop: int, data: Optional[T]=None):
        self.start = start
        self.stop = stop
        self.data = data

    def __contains__(self, item: T) -> bool:
        return False if self.data is None or item is None else self.data == item

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __eq__(self, other: "Interval"):
        return (self.start, self.stop, self.data) == (other.start, other.stop, other.data)

    def __hash__(self):
        return hash((self.start, self.stop, self.data))

    def __len__(self):
        return self.stop - self.start

    def __bool__(self):
        return bool(len(self))

    def __and__(self, other: "Interval"):
        # TODO docs
        first, second = sorted([self, other], key=lambda iv: iv.start)
        return type(self)(first.start, second.stop, [first.data, second.data])

    def __repr__(self):
        return "{}(start={}, stop={}, data={})".format(type(self).__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.data)

    def reload(self, value: T):
        return type(self)(self.start, self.stop, value)


def extract(sequence: Sequence[T], ivs: Iterable[Interval], offset=0) \
        -> List[Sequence[T]]:
    return [sequence[iv.start-offset:iv.stop-offset] for iv in ivs]


def span(ivs: Sequence[Interval]) -> Optional[Interval]:
    """
    Intervals must be presorted
    :param ivs:
    :return:
    """
    return Interval(ivs[0].start, ivs[-1].stop) if len(ivs) else None


def unload(intervals: Iterable[Interval[T]]) -> Iterator[T]:
    return (iv.data for iv in intervals)


@overload
def unextract(ivs: Sequence[Interval], extracted: Sequence[Sequence[T]], fill: T) \
        -> Sequence[T]:
    pass


@overload
def unextract(ivs: Sequence[Interval], extracted: Sequence[np.ndarray], fill) \
        -> Sequence[T]:
    pass


def unextract(ivs, extracted, fill):
    if not len(ivs) or not len(extracted):
        return None
    if all(isinstance(ext, np.ndarray) for ext in extracted):
        return _unextract_arr(ivs, extracted, fill)
    if isinstance(extracted, Sequence):
        return _unextract_sequence(ivs, extracted, fill)
    raise ValueError("Extracted must be either a sequence of numpy arrays or "
                     "a sequence of Sequence objects")


def _unextract_sequence(ivs: Sequence[Interval],
                        extracted: Sequence[Sequence[T]],
                        fill: T) -> Sequence[T]:
    sorted_ivs = sorted(ivs, key=lambda x: x.start)
    res = [fill] * len(span(sorted_ivs))
    offset = sorted_ivs[0].start
    for iv, ext in zip(ivs, extracted):
        if len(iv) != len(ext):
            raise ValueError("Intervals and extracted data are not aligned "
                             "with respect to length")
        for i, val in zip(iv, ext):
            res[i-offset] = val
    return res


def _unextract_arr(ivs: Sequence[Interval], extracted: Sequence[np.ndarray], fill) \
        -> Optional[np.ndarray]:
    ndims = set(map(np.ndim, extracted))
    dtypes = set(ext.dtype for ext in extracted)
    if not len(ndims) == len(dtypes) == 1:
        raise ValueError("Arrays must be homogeneous")
    if isinstance(fill, np.ndarray) and fill.shape != extracted[0].shape[1:]:
        raise ValueError("fill is incompatible with extracted arrays")
    sorted_ivs = sorted(ivs, key=lambda x: x.start)
    res = np.array([fill]*len(span(sorted_ivs)), dtype=dtypes.pop())
    offset = sorted_ivs[0].start
    for iv, ext in zip(ivs, extracted):
        if len(iv) != len(ext):
            raise ValueError("Intervals and extracted data are not aligned "
                             "with respect to length")
        res[iv.start-offset:iv.stop-offset] = ext
    return res


if __name__ == "__main__":
    raise ValueError
