import unittest
from typing import Sequence, Iterable, cast, Mapping
import tempfile
import os

import numpy as np
import joblib
from hypothesis import given, note
from hypothesis import settings, strategies as st

from scilk.corpora import genia
from scilk.util import intervals
from scilk.collections import _collections
import scilk

MAX_TESTS = 1000


# strategies

texts = st.text(st.characters(min_codepoint=32, max_codepoint=255), 0, 500, 1000)


def loader_caller(collection: _collections.Collection, data=None):

    def caller(value: str):
        return collection.translate(value)

    return caller


def loader_translate(collection: _collections.Collection, data: dict):
    mapping = joblib.load(data['mapping'])

    def translate(value: str):
        return mapping.get(value)

    return translate


# test cases

class TestText(unittest.TestCase):

    @staticmethod
    def unparse(txt, intervals_: Sequence[intervals.Interval]):
        if not len(intervals_):
            return ""
        codes = np.repeat([ord(" ")], intervals_[-1].stop)
        for iv in intervals_:
            token = intervals.extract(txt, [iv])[0]
            codes[iv.start:iv.stop] = list(map(ord, token))
        return "".join(map(chr, codes))

    # @given(texts)
    # @settings(max_examples=MAX_TESTS)
    # def test_parse_text(self, txt):
    #     parsed = text.tointervals(text.fine_tokeniser, txt)
    #     mod_text = re.sub("\s", " ", txt)
    #     self.assertEqual(self.unparse(txt, parsed), mod_text.rstrip())


class TestGenia(unittest.TestCase):

    @given(st.lists(st.text()))
    @settings(max_examples=MAX_TESTS)
    def test_text_boundaries(self, texts: list):
        """
        Test of text_boundaries() function.
        :return:
        """
        boundaries = genia._segment_borders(texts)
        note(boundaries)

        self.assertTrue(all([boundaries[i][1] == boundaries[i + 1][0] for i in
                             range(len(boundaries) - 1)]))
        self.assertTrue(all([boundaries[i][0] <= boundaries[i][1] for i in
                            range(len(boundaries) - 1)]))
        if boundaries:
            self.assertTrue(boundaries[0][0] == 0)


class TestCollection(unittest.TestCase):
    def test_collection(self):
        with tempfile.TemporaryDirectory() as dirpath:
            scilk.SCILK_ROOT = dirpath
            mapping = dict(test='OK')
            mapping_path = os.path.join(dirpath, 'mapping.joblib')
            joblib.dump(mapping, mapping_path)
            collection = _collections.Collection()
            collection.add('translate', loader_translate, dict(mapping=mapping_path))
            collection.add('caller', loader_caller)
            self.assertAlmostEqual(collection.caller('test'), 'OK')
            collection.save(name='test')
            collection = _collections.Collection.load('test')
            self.assertAlmostEqual(collection.caller('test'), 'OK')
            self.assertEquals({'translate', 'caller'}, set(collection.entries))


if __name__ == '__main__':
    unittest.main()
