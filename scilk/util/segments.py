from typing import Sequence, List, Iterable, Tuple
from itertools import dropwhile
from functools import reduce

from fn import F
from fn.iters import splitby, droplast

from .intervals import Interval


def breaks(intervals: Iterable[Interval]) -> List[int]:
    """
    Find breakpoints between intervals
    :param intervals:
    :return:
    """
    return [iv.stop - 1 for iv in intervals]


def stitches(intervals: Sequence[Interval], targets: Sequence[Interval]):
    """
    Find breakpoints that have separate to be stitched in order to recover
    target intervals from finer subintervals.
    :param intervals:
    :param targets:
    :return:
    """
    intervals_ = sorted(intervals, key=lambda iv: iv.start)
    stitched_ = sorted(targets, key=lambda iv: iv.start)
    inbreaks = F(droplast, 1) >> breaks

    def grouper(acc: Tuple[List[int], Iterable[Interval]], iv: Interval):
        # find breakpoints to stitch
        breakpoints, tail = acc
        group, tail = splitby(
            iv.intersects, dropwhile(lambda x: x.stop <= iv.start, tail)
        )
        return breakpoints.extend(inbreaks(group)) or breakpoints, list(tail)

    return reduce(grouper, stitched_, ([], intervals_))[0]


if __name__ == '__main__':
    raise RuntimeError
