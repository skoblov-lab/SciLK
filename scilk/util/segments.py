from typing import Sequence, List, Iterable, Tuple
from itertools import dropwhile
from functools import reduce
import operator as op

from fn import F
from fn.iters import splitby, droplast
import numpy as np

from .intervals import Interval


def breakpoints(intervals: Iterable[Interval]) -> List[int]:
    """
    Find breakpoints between intervals
    :param intervals:
    :return:
    """
    return [iv.stop - 1 for iv in intervals]


def stitchpoints(intervals: Sequence[Interval], targets: Sequence[Interval]):
    """
    Find breakpoints that have to be stitched in order to recover target
    intervals from finer subintervals. For a set of intervals [iv_1, ..., iv_n]
    that must be stitched to obtain a target t1, the function returns
    [(iv_1).stop-1, ..., (iv_n-1).stop-1]. The function groups all intervals
    intersecting a target together and merges them. Take note, that an ideal
    reconstruction might not be achievable. In this case it is only guaranteed
    that all merged intervals will contain the entire span of the corresponding
    target, but not the other way around.
    :param intervals:
    :param targets:
    :return:
    """
    intervals_ = sorted(intervals, key=lambda iv: iv.start)
    stitched_ = sorted(targets, key=lambda iv: iv.start)
    inbreaks = F(droplast, 1) >> breakpoints

    def grouper(acc: Tuple[List[int], Iterable[Interval]], iv: Interval):
        # find breakpoints to stitch
        breaks, ivs = acc
        grouped, remainder = splitby(
            iv.intersects, dropwhile(lambda x: x.stop <= iv.start, ivs)
        )
        return breaks.extend(inbreaks(grouped)) or breaks, list(remainder)

    return reduce(grouper, stitched_, ([], intervals_))[0]


def stitch(intervals: Sequence[Interval], points: Sequence[int]) \
        -> List[Interval]:
    """
    Stitch intervals. If any point in `points` falls into an interval at
    position i this interval will be stitched to interval at position i+1.
    :param intervals:
    :param points:
    :return:
    """
    # extract annotations
    ivs = sorted(intervals, key=lambda iv: iv.start)
    length = max(iv.stop for iv in ivs)
    annotations = np.zeros(length, dtype=np.int32)
    annotations[points] = 1
    iv_anno = [annotations[iv.start:iv.stop].any() for iv in ivs]

    # group intervals to stitch
    def group(acc: Tuple[List[List[Interval]], bool],
              step: Tuple[Interval, bool]) \
            -> Tuple[List[List[Interval]], bool]:
        groups, takethis = acc
        iv, takenext = step
        if takethis:
            groups[-1].append(iv)
        else:
            groups.append([iv])
        return groups, takenext

    grouped = reduce(group, zip(ivs, iv_anno), ([], False))[0]
    # stitch intervals
    return [reduce(op.and_, group) for group in grouped]


if __name__ == '__main__':
    raise RuntimeError
