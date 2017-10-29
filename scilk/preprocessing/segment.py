from functools import reduce
from typing import Callable, Text, Tuple, Pattern, List, Iterable

from scilk.structures.intervals import Interval


Segmenter = Callable[[Text], Iterable[Text]]


def ptokenise(patterns: List[Pattern], text: Text, mask=" ") \
        -> List[Interval[Text]]:
    """
    Return intervals matched by `patterns`. The patterns are applied
    in iteration order. Before applying pattern `i+1`, the function replaces
    each region `r` matched by pattern `i` with `mask * len(r)`. This means
    the output might be sensitive to pattern order.
    :param patterns: a list of patterns to search for
    :param text: a unicode string
    :param mask: the masking value
    :return: a list of intervals storing the corresponding string
    """
    def repl(match) -> Text:
        return mask * (match.end() - match.start())

    def match_mask(acc: Tuple[List[Tuple[int, int]], Text],
                   patt: Pattern) -> Tuple[List[Tuple[int, int]], Text]:
        spans, s = acc
        spans.extend(m.span() for m in patt.finditer(s))
        return spans, patt.sub(repl, s)

    return [Interval(start, stop, text[start:stop]) for start, stop in
            sorted(reduce(match_mask, patterns, ([], text))[0])]


if __name__ == "__main__":
    raise RuntimeError
