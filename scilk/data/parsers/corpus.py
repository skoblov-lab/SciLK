from numbers import Integral
from typing import Sequence, NamedTuple, Text, Iterable, Tuple, List, \
    Mapping, Optional

from scilk.structures import intervals

OTHER = "OTHER"
TITLE = "T"
BODY = "A"
ClassMapping = Mapping[Text, Integral]
LabeledInterval = intervals.Interval[Text]
Annotation = Sequence[LabeledInterval]
SentenceBorders = Sequence[intervals.Interval]

AbstractText = NamedTuple("Abstract",
                          [("id", int), ("title", Text), ("body", Text)])
AbstractAnnotation = NamedTuple("AbstractAnnotation", [("id", int),
                                                       ("title", Annotation),
                                                       ("body", Annotation)])
AbstractSentenceBorders = NamedTuple("AbstractSentenceBorders",
                                     [("id", int), ("title", SentenceBorders),
                                      ("body", SentenceBorders)])
Abstract = Tuple[AbstractText, Optional[AbstractAnnotation],
                 Optional[AbstractSentenceBorders]]
# Record: (abstract id, part type, text, annotation, sentence borders)
Record = Tuple[int, Text, Text, Optional[Annotation], Optional[SentenceBorders]]


class AnnotationError(ValueError):
    pass


def flatten_abstract(abstract: Abstract) -> List[Record]:
    """
    :return: list[(abstract id, source, text, annotation)]
    """
    abstract_id, title, body = abstract[0]
    anno_id, title_anno, body_anno = abstract[1]
    borders_id, title_borders, body_borders = abstract[2]
    if abstract_id != anno_id:
        raise AnnotationError("Abstract ids do not match")
    return [(abstract_id, TITLE, title, title_anno, title_borders),
            (abstract_id, BODY, body, body_anno, body_borders)]


def parse_mapping(classmaps: Iterable[str]) -> ClassMapping:
    """
    :param classmaps:
    :return:
    >>> classmaps = ["a:1", "b:1", "c:2"]
    >>> parse_mapping(classmaps) == dict(a=1, b=1, c=2)
    True
    """
    try:
        return {cls: int(val)
                for cls, val in [classmap.split(":") for classmap in classmaps]}
    except ValueError as err:
        raise AnnotationError("Badly formatted mapping: {}".format(err))


if __name__ == "__main__":
    raise RuntimeError
