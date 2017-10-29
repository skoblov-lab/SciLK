import unittest
from typing import Sequence, Iterable, cast

import numpy as np
from hypothesis import given, note
from hypothesis import settings, strategies as st

from scilk.data.parsers import genia
from scilk.preprocessing import preprocessing
from scilk.structures import intervals

MAX_TESTS = 1000


# strategies

texts = st.text(st.characters(min_codepoint=32, max_codepoint=255), 0, 500, 1000)


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


class TestSampling(unittest.TestCase):

    @given(st.integers(1, 10), st.integers(100, 500), st.integers(10, 50))
    @settings(max_examples=MAX_TESTS)
    def test_annotate_sample(self, ncls, length, nintervals):
        anno = np.random.choice(ncls, length)
        split_points = sorted(np.random.choice(length, nintervals, False))
        ivs = [intervals.Interval(arr[0], arr[-1] + 1) for arr in
               np.split(np.arange(length), split_points)[1:-1]]

        sample_anno = preprocessing.annotate_sample(ncls, anno, ivs)
        sample_anno_cls = [set(iv_anno.nonzero()[-1])
                           for iv_anno in cast(Iterable[np.ndarray], sample_anno)]
        self.assertSequenceEqual([set(anno[iv.start:iv.stop]) for iv in ivs],
                                 sample_anno_cls)
        self.assertEqual(len(ivs), len(sample_anno))

    # @given(st.integers(1, 10), st.integers(1, 100), st.integers(0, 100))
    # @settings(max_examples=MAX_TESTS)
    # def test_flatten_multilabel_annotation(self, ncls, nsteps, maxmixed):
    #     anno = np.random.choice(ncls, nsteps)
    #     nested = util.one_hot(ncls, anno)
    #     mixed_steps = np.random.choice(nsteps, min(nsteps, maxmixed), False)
    #     nested[mixed_steps, 0] = 1
    #     self.assertTrue(
    #         (anno == sampling.flatten_multilabel_annotation(nested)).all()
    #     )
    #     nested[mixed_steps, np.random.choice(ncls)] = 1
    #     if ncls > 2 and (nested[:, 1:].sum(1) > 1).any():
    #         with self.assertRaises(sampling.AmbiguousAnnotation):
    #             sampling.flatten_multilabel_annotation(nested)


if __name__ == "__main__":
    unittest.main()
