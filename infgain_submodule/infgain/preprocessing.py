import typing
import nltk
import re


NUMBER_TOKEN = 'NUMBER'
NGRAM_SEP = '_'
TILDA_SEP = '~'
UNK_TOKEN = 'UNK'

EN_STOPWORDS = nltk.corpus.stopwords.words('english')


def remove_separators(string: str) -> str:
    """
    :param string:
    :return:
    >>> clean_str = remove_separators('crf01_ae~recombinant')
    >>> clean_str == 'crf01 ae recombinant'
    True
    """
    string = string.replace('_', ' ').replace('~', ' ')
    return string


def sent_tokenize(string: str) -> list:
    """
    Tokenize `string` into sentences.
    :param string:
    :return:
    """
    return nltk.tokenize.sent_tokenize(string)


def regexp_word_tokenize(string: str) -> list:
    #TODO: it doesn't split by underscore!
    """
    Tokenize `string` into words.
    :param string:
    :return:
    >>> words = regexp_word_tokenize('Mean age was 74.5+/-9.0 years (men: 63.2%) and mean CHADS2 score (+/-SD) was 1.8+/-1.2.')
    >>> words == ['Mean','age','was','74','5','9','0','years','men','63','2','and','mean','CHADS2','score','SD','was','1','8','1','2']
    True
    """
    rgx = re.compile("([\w]*\w)")
    return rgx.findall(string)


def merge_numbers(words: list) -> list:
    """
    Merge a sequence of numbers into one `NUMBER_TOKEN`.
    :param words:
    :return:
    >>> words = regexp_word_tokenize('Mean age was 74.5+/-9.0 years (men: 63.2%) and mean CHADS2 score (+/-SD) was 1.8+/-1.2.')
    >>> merge_numbers(words) == ['Mean','age','was','NUMBER','years','men','NUMBER','and','mean','CHADS2','score','SD','was','NUMBER']
    True
    """
    def startswith_digit(string: str) -> bool:
        return string[0].isdigit()

    ans = []
    i = 0
    while i < len(words):
        j = 0
        while i + j < len(words) and startswith_digit(words[i + j]):
            j += 1
        if i + j > 1 and startswith_digit(words[i + j - 1]):
            ans.append(NUMBER_TOKEN)
        if i + j < len(words):
            ans.append(words[i + j])
        i += j + 1
    return ans


def word_tokenize(string: str) -> list:
    """
    Tokenize `string` into words and replace numbers with `NUMBER_TOKEN`.
    :param string:
    :return:
    >>> tokenized = cool_tokenize('Mean age was 74.5+/-9.0 years (men: 63.2%) and mean CHADS2 score (+/-SD) was 1.8+/-1.2.')
    >>> tokenized == ['mean','age','was','NUMBER','years','men','NUMBER','and','mean','chads2','score','sd','was','NUMBER']
    True
    """
    string = string.lower()
    words = regexp_word_tokenize(string)
    return merge_numbers(words)


def make_ngrams(words: list, maxn=2) -> list:
    """
    :param words:
    :param maxn:
    :return:
    >>> ngrams = make_ngrams(['used','to','prevent','stroke'])
    >>> ngrams == [('used',),('to',),('prevent',),('stroke',),('used', 'to'),('to', 'prevent'),('prevent', 'stroke')]
    True
    """
    if len(words) > 1:
        res = []
        for i in range(1, maxn + 1):
            res.extend(nltk.ngrams(words, i))
        return res
    else:
        return [tuple(words)]


def contains_stopword(ngram: tuple, stopwords=EN_STOPWORDS) -> bool:
    """
    :param ngram:
    :param stopwords:
    :return:
    >>> contains_stopword(('atrial', 'fibrillation'))
    False
    >>> contains_stopword(('to', 'prevent', 'stroke'))
    True
    """
    list_of_bool = list(map(lambda x: x in stopwords, ngram))
    return any(list_of_bool)


def remove_bad_ngrams(ngrams: list) -> list:
    """
    :param ngrams:
    :return:
    >>> ngrams = [('atrial', 'fibrillation'), ('to', 'prevent', 'stroke')]
    >>> remove_bad_ngrams(ngrams) == [('atrial', 'fibrillation')]
    True
    """
    return list(filter(lambda x: contains_stopword(x) == False, ngrams))

if __name__ == "__main__":
    raise RuntimeError
