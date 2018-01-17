"""



"""

import re
from itertools import chain
from typing import Mapping, Tuple, Text, Iterable, List, overload

import numpy as np
from fn.func import identity
from frozendict import frozendict
from multipledispatch import dispatch

from scilk.util.intervals import Interval
from scilk.util.func import oldmap, homogenous

MAXLABEL = 255


class EncodingError(ValueError):
    pass


class WordEncoder:
    """
    Zero is reserved for padding
    """

    def __init__(self, path: Text, oov: Text, transform=None):
        self._oov = oov
        self._transform = transform if transform else identity
        word_index, vectors = self._read_embeddings(path)
        self._vocab = word_index
        self._vectors = vectors

    def __str__(self):
        return "<Vocabulary> with {} entries".format(len(self._vocab))

    def __len__(self):
        return len(self._vectors)  # including the pad and oov words

    @property
    def vocab(self):
        return self._vocab

    @property
    def vectors(self):
        return self._vectors

    @property
    def transform(self):
        return self._transform

    @property
    def oov(self):
        return self._oov

    def encode(self, words: Iterable[Text], vectors=False) -> np.ndarray:
        oov = self._vocab[self._oov]
        ids = [self._vocab.get(w, oov) for w in map(self._transform, words)]
        return self._vectors[ids] if vectors else np.array(ids, dtype=np.int32)

    @staticmethod
    def _read_embeddings(path) -> Tuple[Mapping[str, int], np.ndarray]:
        with open(path) as lines:
            parsed = map(str.split, lines)
            words, vectors = zip(*((w, oldmap(float, v)) for w, *v in parsed))
        if not words:
            raise EncodingError("File {} is empty".format(path))
        if not homogenous(len, vectors):
            raise EncodingError("Word vectors must be homogeneous")
        ndim = len(vectors[0])
        padvec = [0.0] * ndim
        word_index = frozendict({word: i+1 for i, word in enumerate(words)})
        vectors_ = np.array(list(chain([padvec], vectors)))
        vectors_.setflags(write=False)
        return word_index, vectors_


class CharEncoder:
    """
    Zero is reserved for the padding value
    """

    def __init__(self, text: Text):
        if not text or not isinstance(text, Text):
            raise ValueError("`text` must be a nonempty string")
        self._chars = self._read_characters(text)
        self._oov = len(self._chars) + 1

    def __len__(self):
        return len(self._chars) + 2  # including the pad and oov characters

    @overload
    def __call__(self, text: str) -> np.ndarray:
        pass

    @overload
    def __call__(self, text: Iterable[str]) -> List[np.ndarray]:
        pass

    def __call__(self, text):
        return (self.encode(text) if isinstance(text, str) else
                [self.encode(item) for item in text])

    @property
    def oov(self):
        return self._oov

    @property
    def characters(self):
        return self._chars

    @staticmethod
    def _read_characters(text):
        nows = re.sub("\s", "", text)
        char_index = frozendict(
            {char: i+1 for i, char in enumerate(set(nows))})
        return char_index

    def encode(self, text: Text) -> np.ndarray:
        oov = self._oov
        return np.array([self._chars.get(c, oov) for c in text], dtype=np.int32)


def encode_annotation(mapping, annotations: Iterable[Interval], size: int,
                      start_only: bool=False, default: int=0) -> np.ndarray:
    # TODO update docs
    """
    :param mapping:
    :param annotations:
    :param size:
    :param start_only: only encode the first character in an entity
    :param default:
    :return:
    """
    encoded_anno = np.zeros(size, dtype=np.int32)
    for anno in annotations:
        if anno.stop > size:
            raise EncodingError("annotation `size` is insufficient")
        if start_only:
            encoded_anno[anno.start] = mapping.get(anno.data, default)
        else:
            encoded_anno[anno.start:anno.stop] = mapping.get(anno.data, default)
    return encoded_anno


if __name__ == "__main__":
    pass
