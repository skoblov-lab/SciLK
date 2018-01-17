from typing import Sequence, Iterator, Iterable, Callable

import numpy as np

from scilk.encoding.encoding import EncodingError
from scilk.util.intervals import Interval

Sample = Sequence[Interval]
Sampler = Callable[[Sequence[Interval]], Iterable[Sample]]
Annotator = Callable[[Sample], np.ndarray]


class AmbiguousAnnotation(EncodingError):
    pass


def sample_windows(window: int, step: int, text_intervals: Sequence[Interval]) \
        -> Iterator[Sequence[Interval]]:
    # TODO update docs
    # TODO test
    """
    Sample windows using a sliding window approach. Sampling windows start at
    the beginning of each interval in `intervals`
    :param text_intervals: a sequence (preferable a numpy array) of interval objects
    :param window: sampling window width in tokens
    """
    if len(text_intervals) <= window:
        return iter([text_intervals])
    steps = list(range(0, len(text_intervals) - window + 1, step))
    if steps[-1] + window < len(text_intervals):
        steps.append(steps[-1] + step)
    return (text_intervals[i:i + window] for i in steps)


def sample_sentences(borders, text_intervals):
    # TODO docs
    # TODO tests
    if not len(text_intervals) or not len(borders):
        raise ValueError("empty intervals and/or borders")
    ends = iter(sorted(border.stop for border in borders))
    end = next(ends)
    samples = [[]]
    for iv in sorted(text_intervals, key=lambda x: x.start):
        if iv.stop <= end:
            samples[-1].append(iv)
        else:
            end = next(ends, None)
            if end is None:
                raise RuntimeError
            samples.append([iv])
    if len(samples) != len(borders):
        raise RuntimeError
    return samples


if __name__ == "__main__":
    raise RuntimeError
