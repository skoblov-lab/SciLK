import operator as op
import re
from functools import reduce
from itertools import starmap
from numbers import Integral
from typing import Sequence, NamedTuple, Tuple, Iterable, Text, Optional, List, \
    Iterator
from xml.etree.ElementTree import Element, parse

from fn import F
from pyrsistent import v, pvector

from scilk.data.parsers.corpus import AbstractAnnotation, AbstractText, \
    ClassMapping, \
    AnnotationError, LabeledInterval
from scilk.structures.intervals import Interval

ANNO_PATT = re.compile("G#(\w+)")
SENTENCE_TAG = "sentence"
ANNO_TAG = "sem"
ARTICLE = "article"

LevelAnnotation = NamedTuple("Annotation", [("level", int),
                                            ("anno", Sequence[Optional[Text]]),
                                            ("terminal", bool)])


def flatten_sentence(sentence: Element) \
        -> List[Tuple[Text, Sequence[LevelAnnotation]]]:
    # TODO docs
    """
    Turn `sentence` Element from xml format to normal text.
    :param sentence:
    :return: list of strings with corresponding annotations
    """
    def isterminal(element: Element):
        return next(iter(element), None) is None

    def getanno(element: Element):
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


def text_boundaries(texts: Iterable[Text]) -> List[Tuple[int, int]]:
    # TODO docs
    """
    Returns list of start/stop positions of words' starts/ends in `texts`.
    :param texts: list of strings
    :return: list of (start position, stop position)
    >>> text_boundaries(['amino acid', 'is any']) == [(0, 10), (10, 16)]
    True
    """
    def aggregate_boundaries(boundaries: pvector, text):
        return (
            boundaries + [(boundaries[-1][1], boundaries[-1][1] + len(text))]
            if boundaries else v((0, len(text)))
        )

    return list(reduce(aggregate_boundaries, texts, v()))


def parse_sentences(root: Element, mapping: ClassMapping,
                    default: Integral = None) \
        -> Tuple[Text, List[LabeledInterval]]:
    # TODO docs
    """
    Get text form `root` Element with given mapping dictionary.
    :param root:
    :param mapping:
    :param default:
    :return:
    """
    def wrap_interval(start: int, stop: int, levels: Sequence[LevelAnnotation]) -> LabeledInterval:
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
        encoded = mapping.get(codes.pop(), default)
        return Interval(start, stop, encoded) if encoded is not None else None

    sentences = root.findall(SENTENCE_TAG)
    flattened = reduce(op.iadd, map(flatten_sentence, sentences), [])
    texts, annotations = zip(*((txt, anno) for txt, anno in flattened
                               if txt is not None))
    boundaries = text_boundaries(texts)
    intervals = [wrap_interval(start, stop, levels)
                 for (start, stop), levels in zip(boundaries, annotations)
                 if levels and levels[-1].terminal]
    return "".join(texts), list(filter(bool, intervals))


def parse_corpus(path: Text, mapping: ClassMapping, default: Integral = None) \
        -> List[Tuple[AbstractText, AbstractAnnotation]]:
    """
    Extract text from xml file `path`.
    :param path: xml file's path
    :param mapping: mapping entity to number
    :param default:
    :return:
    """
    parser = F(parse_sentences, mapping=mapping, default=default)

    def getid(article: Element) -> int:
        raw = article.find("articleinfo").find("bibliomisc").text
        return int(raw.replace("MEDLINE:", ""))

    def accumulate_articles(root: Element) -> Iterator[Tuple[int, Element, Element]]:
        """
        Collects articles inside `root`.
        :param root:
        :return:
        """
        articles_ = root.findall(ARTICLE)
        ids = map(getid, articles_)
        title_roots = [article.find("title") for article in articles_]
        body_roots = [article.find("abstract") for article in articles_]
        return zip(ids, title_roots, body_roots)

    def parse_article(id_: int, title_root: Element, body_root: Element) \
            -> Tuple[AbstractText, AbstractAnnotation]:
        """
        Extract title and body texts from `title_root` and `body_root`.
        :param id_: article's id
        :param title_root:
        :param body_root:
        :return:
        """
        title_text, title_anno = parser(title_root)
        body_text, body_anno = parser(body_root)
        abstract = AbstractText(id_, title_text, body_text)
        annotation = AbstractAnnotation(id_, title_anno, body_anno)
        return abstract, annotation

    corpus = parse(path)
    articles = accumulate_articles(corpus)
    return list(starmap(parse_article, articles))


if __name__ == "__main__":
    raise RuntimeError
