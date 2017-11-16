################
# Maxim Holmatov
################

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

__all__ = ["tf_idf"]


def tf_idf(corpus, n, l=10000):
    """
    Take corpus and return a sorted list of scored ngrams

    param n: number of words in n-gram
    type n: int
    param l: number of terms to output
    type l: int
    return: sorted list of scored ngrams
    rtype: list
    """
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(n, n),
                                 input="content", stop_words="english")
    x = vectorizer.fit_transform(corpus)
    sums = np.asarray(x.sum(axis=0))[0]
    indeces = sums.argsort()[::-1]

    names = vectorizer.get_feature_names()

    return [(names[index], sums[index]) for index in indeces[:l]]
