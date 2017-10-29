import re
from functools import reduce
from itertools import chain
from typing import Callable, Iterable, Text, Sequence

import numpy as np
import spacy
from fn import F
from pyrsistent import PVector, pvector

from scilk.structures import intervals
from scilk.util.func import flatmap

spacy_tokeniser = (F(spacy.load("en").tokenizer) >>
                   (map, lambda tk: tk.text) >>
                   (flatmap, re.compile(r"[&/|]|[^&/|]+").findall))


def tointervals(tokeniser: Callable[[str], Iterable[str]], text: Text) \
        -> Sequence[intervals.Interval[Text]]:
    # TODO docs
    def mark_boundaries(boundaries: PVector, token: str):
        if not boundaries:
            return boundaries.append(intervals.Interval(0, len(token), token))
        prev = boundaries[-1]
        start = prev.stop
        stop = start + len(token)
        return boundaries.append(intervals.Interval(start, stop, token))

    if not text:
        return np.array([])
    all_tk = re.compile("\S+|\s+")
    ws = re.compile("\s")
    tokens = all_tk.findall(text)
    fine_grained = chain.from_iterable(
        tokeniser(tk) if not ws.match(tk) else [tk] for tk in tokens)
    intervals_ = reduce(mark_boundaries, fine_grained, pvector())
    ws_less = (iv for iv in intervals_ if not ws.match(iv.data))
    return np.array([iv.reload(text[iv.start:iv.stop]) for iv in ws_less])


if __name__ == "__main__":
    raise RuntimeError
