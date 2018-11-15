"""

Parsers, preprocessors and type annotations for the chemdner dataset.

"""

import operator as op
from itertools import groupby
from typing import List, Tuple, Text, Iterable, Iterator

import pandas as pd
from fn import F

from scilk.corpora.corpus import TITLE, BODY, Abstract, AbstractAnnotation, \
    AbstractText, AbstractSentenceBorders, UTF8
from scilk.util.intervals import Interval


def parse_abstracts(path: Text, encoding=UTF8) -> List[AbstractText]:
    """
    Read chemdner abstracts
    :return: list[(abstract id, title, body)]
    >>> path = "testdata/abstracts.txt"
    >>> abstracts = parse_abstracts(path)
    >>> ids = {21826085, 22080034, 22080035, 22080037}
    >>> all(id_ in ids for id_, *_ in abstracts)
    True
    """
    with open(path, encoding=encoding) as buffer:
        parsed_buffer = (line.strip().split('\t') for line in buffer)
        return [AbstractText(int(id_), title.rstrip(), body.rstrip())
                for id_, title, body in parsed_buffer]


def parse_annotations(path: Text, encoding=UTF8) -> List[AbstractAnnotation]:
    # TODO log empty annotations
    # TODO more tests
    """
    Read chemdner annotations
    :param path: path to a CHEMDNER-formatted annotation files
    >>> path = "testdata/annotations.txt"
    >>> anno = parse_annotations(path)
    >>> ids = {21826085, 22080034, 22080035, 22080037}
    >>> all(id_ in ids for id_, *_ in anno)
    True
    >>> nonempty_anno = [id_ for id_, title, _ in anno if title]
    >>> nonempty_anno
    [22080037]
    >>> [len(title) for _, title, _ in anno]
    [0, 0, 0, 2]
    >>> [len(body) for _, _, body in anno]
    [1, 6, 9, 5]
    """
    def wrap_interval(record: Tuple[str, str, str, str, str, str]) \
            -> Interval:
        _, _, start, stop, text, label = record
        return Interval(int(start), int(stop), label)

    def parse_line(line):
        id_, src, start, stop, text, label = line.split('\t')
        return int(id_), src, int(start), int(stop), text, label

    with open(path, encoding=encoding) as buffer:
        parsed_lines = map(parse_line, map(str.strip, buffer))
        lines_sorted = sorted(
            parsed_lines, key=lambda x: (-x[0], x[1], -x[2]), reverse=True)
        # separate abstracts
        abstract_groups = groupby(lines_sorted, op.itemgetter(0))
        # separate parts (title and body)
        part_groups = ((id_, groupby(group, op.itemgetter(1)))
                       for id_, group in abstract_groups)
        # filter zero-length intervals and `None`s
        wrapper = F(map, wrap_interval) >> (filter, bool) >> list
        mapped_parts = ((id_, {part: wrapper(recs) for part, recs in parts})
                        for id_, parts in part_groups)
        return [AbstractAnnotation(int(id_),
                                   list(parts.get(TITLE, [])),
                                   list(parts.get(BODY, [])))
                for id_, parts in mapped_parts]


def parse_borders(path: Text, encoding=UTF8) -> List[AbstractSentenceBorders]:
    def pack_borders(id_: int, borders_: pd.DataFrame):
        src_mapped = {
            src: [Interval(*map(int, b_str.split(':'))) for b_str in bs[2]]
            for src, bs in borders_.groupby(1)
        }
        title_borders = src_mapped.get(TITLE, [])
        body_borders = src_mapped.get(BODY, [])
        return AbstractSentenceBorders(id_, title_borders, body_borders)

    borders = pd.read_csv(path, sep='\t', header=None, encoding=encoding)
    return ([] if not len(borders) else
            [pack_borders(id_, bs) for id_, bs in borders.groupby(0)])


def align_abstracts(abstracts: Iterable[AbstractText],
                    annotations: Iterable[AbstractAnnotation]=None,
                    borders: Iterable[AbstractSentenceBorders]=None) \
        -> Iterator[Abstract]:
    # TODO tests
    """
    Align abstracts and annotations (i.e. match abstract ids)
    :param abstracts: parsed abstracts (e.g. produces by `read_abstracts`)
    :param annotations: parsed annotations (e.g. produces by `read_annotations`)
    :return: Iterator[(parsed abstract, parsed annotation)]
    """
    def empty_anno(id_: int) -> AbstractAnnotation:
        return AbstractAnnotation(id_, [], [])

    def empty_borders(id_: int) -> AbstractSentenceBorders:
        return AbstractSentenceBorders(id_, [], [])

    anno_mapping = {anno.id: anno for anno in annotations or []}
    borders_mapping = {b.id: b for b in borders or []}

    return ((abstract,
             anno_mapping.get(abstract.id, empty_anno(abstract.id)),
             borders_mapping.get(abstract.id, empty_borders(abstract.id)))
            for abstract in abstracts)


def parse(abstracts: str, annotations: str=None, borders: str=None, encoding=UTF8) \
        -> List[Abstract]:
    return list(
        align_abstracts(
            parse_abstracts(abstracts, encoding),
            parse_annotations(annotations, encoding) if annotations else None,
            parse_borders(borders, encoding) if borders else None
        )
    )


if __name__ == '__main__':
    raise RuntimeError
