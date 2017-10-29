from itertools import starmap
from typing import Sequence, NamedTuple, Tuple, Iterable, Text, Optional, List, \
    Iterator
from xml.etree import ElementTree as ETree

import operator as op
import re
from functools import reduce
from pyrsistent import v, pvector

from scilk.data.parsers.corpus import AbstractAnnotation, AbstractText, AbstractSentenceBorders, \
    AnnotationError, LabeledInterval, Abstract, SentenceBorders
from scilk.structures.intervals import Interval

ANNO_PATT = re.compile("G#(\w+)")
SENTENCE_TAG = "sentence"
ANNO_TAG = "sem"
ARTICLE_TAG = "article"

LevelAnnotation = NamedTuple("Annotation", [("level", int),
                                            ("anno", Sequence[Optional[Text]]),
                                            ("terminal", bool)])


def _flatten_sentence(sentence: ETree.Element) \
        -> List[Tuple[Text, Sequence[LevelAnnotation]]]:
    # TODO docs
    """
    Turn `sentence` Element from xml format to normal text.
    :param sentence:
    :return: list of strings with corresponding annotations
    """

    def isterminal(element: ETree.Element):
        return next(iter(element), None) is None

    def getanno(element: ETree.Element):
        return element.get(ANNO_TAG, None)

    stack = [(sentence, iter(sentence), v())]
    texts = [sentence.text]
    annotations = [stack[0][2]]
    while stack:
        node, children, anno = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            texts.append(node.tail)
            annotations.append(anno[:-1])
            continue
        child_anno = anno.append(
            LevelAnnotation(len(anno), getanno(child), isterminal(child)))
        texts.append(child.text)
        annotations.append(child_anno)
        stack.append((child, iter(child), child_anno))

    return list(zip(texts, annotations))


def _segment_borders(texts: Iterable[Text]) -> List[Tuple[int, int]]:
    # TODO docs
    """
    Returns list of start/stop positions of segments' starts/ends in `texts`.
    :param texts: list of strings
    :return: list of (start position, stop position)
    >>> _segment_borders(['amino acid', 'is any']) == [(0, 10), (10, 16)]
    True
    """

    def aggregate_boundaries(boundaries: pvector, text):
        return (
            boundaries + [(boundaries[-1][1], boundaries[-1][1] + len(text))]
            if boundaries else v((0, len(text)))
        )

    return list(reduce(aggregate_boundaries, texts, v()))


def _sentences_borders(sentences: Iterable[ETree.Element]) -> SentenceBorders:
    """
    Applies _segment_borders to sentences
    and corrects intervals to handle \n at the ends
    of sentences in AbstractText returned by _parse_sentences
    :param sentences: Iterable of ETree.Element objects each containing sentence's text.
    Correct sentence segmentation is assumed
    :return: List of Intervals with sentence borders and no Interval.data.
    ~List[Interval[start, stop, None]]
    """
    sentences_j = ["".join(x.itertext()) for x in sentences]
    borders = ((start + i, stop + i) for i, (start, stop) in enumerate(_segment_borders(sentences_j)))
    return [Interval(start, stop) for (start, stop), _ in zip(borders, sentences_j)]


def _parse_sentences(root: ETree.Element) -> Tuple[Text, List[LabeledInterval], SentenceBorders]:
    # TODO docs
    """
    Get text from `root` Element with given mapping dictionary.
    :param root:
    :return: joined text along with its annotations
    """

    def wrap_iv(start: int, stop: int, levels: Sequence[LevelAnnotation]) \
            -> LabeledInterval:
        """
        Wrap `start`, `stop` and `levels` into an Interval.
        :param start: start position
        :param stop: stop position
        :param levels: list of annotations
        :return: Interval(`start`, `stop`, mappings)
        """
        # get the first nonempty annotation bottom to top
        anno = next(filter(bool, (l.anno for l in reversed(levels))), "")
        codes = set(ANNO_PATT.findall(anno))
        if not len(codes) == 1:
            raise AnnotationError(
                "The annotation is either ambiguous or empty: {}".format(codes))
        return Interval(start, stop, codes.pop())

    sentences = root.findall(SENTENCE_TAG)
    flattened = reduce(op.iadd, map(_flatten_sentence, sentences), [])
    texts, annotations = zip(*((txt, anno) for txt, anno in flattened
                               if txt is not None))
    boundaries = _segment_borders(texts)
    intervals = [wrap_iv(start, stop, levels)
                 for (start, stop), levels in zip(boundaries, annotations)
                 if levels and levels[-1].terminal]
    return "".join(texts), list(filter(bool, intervals)), _sentences_borders(sentences)


def parse(path: Text) \
        -> List[Tuple[AbstractText, AbstractAnnotation, AbstractSentenceBorders]]:
    """
    Extract text from xml file `path`.
    :param path: xml file's path
    :return:
    """

    def getid(article: ETree.Element) -> int:
        raw = article.find("articleinfo").find("bibliomisc").text
        return int(raw.replace("MEDLINE:", ""))

    def accumulate_articles(root: ETree.Element) \
            -> Iterator[Tuple[int, ETree.Element, ETree.Element]]:
        """
        Collects articles inside `root`.
        :param root:
        :return:
        """
        articles_ = root.findall(ARTICLE_TAG)
        ids = map(getid, articles_)
        title_roots = [article.find("title") for article in articles_]
        body_roots = [article.find("abstract") for article in articles_]
        return zip(ids, title_roots, body_roots)

    def parse_article(id_: int, title_root: ETree.Element,
                      body_root: ETree.Element) \
            -> Tuple[AbstractText, AbstractAnnotation, AbstractSentenceBorders]:
        """
        Extract title and body texts from `title_root` and `body_root`.
        :param id_: article's id
        :param title_root:
        :param body_root:
        :return:
        """
        title_text, title_anno, title_sent = _parse_sentences(title_root)
        body_text, body_anno, body_sent = _parse_sentences(body_root)
        abstract = AbstractText(id_, title_text, body_text)
        annotation = AbstractAnnotation(id_, title_anno, body_anno)
        sent_borders = AbstractSentenceBorders(id_, title_sent, body_sent)
        return abstract, annotation, sent_borders

    corpus = ETree.parse(path)
    articles = accumulate_articles(corpus)
    return list(starmap(parse_article, articles))


if __name__ == "__main__":
    raise RuntimeError
