
import re
from functools import reduce
from typing import Text, Tuple, Pattern, List, Iterable, Callable, Union

from scilk.structures.intervals import Interval

# patterns
numeric = re.compile("[0-9]*\.?[0-9]+")
wordlike = re.compile("\w+")
misc = re.compile("[^\s\w]")


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


def ptransform(transforms: Iterable[Tuple[Pattern, Union[Text, Callable]]],
               text: Text) -> Text:
    """
    Pattern transform. The patterns are applied in iteration order with no
    intermediate masking.
    :param transforms: pairs of patterns and replacements (refer to `re.sub`'s
    documentation for more information on possible replacements);
    :param text: text to transform
    :return: transformed text
    """
    return reduce(lambda s, t: t[0].sub(t[1], s), transforms, text)


if __name__ == "__main__":
    raise RuntimeError
