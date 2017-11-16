import typing
import nltk
import re
import itertools
import collections


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
    # TODO: it doesn't split by underscore!
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
    >>> tokenized = word_tokenize('Mean age was 74.5+/-9.0 years (men: 63.2%) and mean CHADS2 score (+/-SD) was 1.8+/-1.2.')
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


def get_unigrams(ngrams: list) -> list:
    """
    :param ngrams: tuples of ngrams
    :return: list of str
    >>> ngrams = [('background',),('rivaroxaban',),('currently',),('used',),('background', 'rivaroxaban'),('currently', 'used')]
    >>> get_unigrams(ngrams) == ['background', 'rivaroxaban', 'currently', 'used']
    True
    """
    unigram_tuples = list(filter(lambda x: len(x) == 1, ngrams))
    unigrams = list(map(lambda x: x[0], unigram_tuples))
    return unigrams


def make_sentence(ngrams: list) -> str:
    """
    :param ngrams:
    :return:
    >>> ngrams = [('background',),('rivaroxaban',),('currently',),('used',),('background', 'rivaroxaban')]
    >>> make_sentence(ngrams) == 'background rivaroxaban currently used'
    True
    """
    unigrams = get_unigrams(ngrams)
    return ' '.join(unigrams)


def ngrams_from_dict(dictionary: dict) -> list:
    return list(filter(lambda x: NGRAM_SEP in x, dictionary.keys()))


def tildas_from_dict(dictionary: dict) -> list:
    return list(filter(lambda x: TILDA_SEP in x, dictionary.keys()))


def make_dict(tokenized_sentences: list, maxn=3, threshold=3) -> tuple:
    """
    :param tokenized_sentences:
    :param maxn:
    :return: `dictionary` {token: id, ...} and `reversed_dictionary` {id: token, ...}
    """
    ngrams = list(map(lambda x: make_ngrams(x, maxn), tokenized_sentences))
    ngrams = list(map(lambda x: remove_bad_ngrams(x), ngrams))
    dictionary, reversed_dictionary = make_dict_from_ngrams(ngrams, threshold)
    return dictionary, reversed_dictionary


def make_dict_from_ngrams(ngrams: list, threshold=3) -> tuple:
    """
    :param ngrams: list of lists of ngram tuples
    :param threshold: min frequency of token
    :return: `dictionary` {token: id, ...} and `reversed_dictionary` {id: token, ...}
    """
    def ngrams_to_str(ngrams: list) -> list:
        """
        >>> ngrams = [('used',),('background', 'rivaroxaban')]
        >>> ngrams_to_str(ngrams) == ['used', 'background_rivaroxaban']
        True
        """
        return list(map(lambda x: NGRAM_SEP.join(x), ngrams))

    def merge_list(ngrams_list: list) -> list:
        """
        >>> merge_list([[1,2,3], [4]]) == [1, 2, 3, 4]
        True
        """
        return list(itertools.chain.from_iterable(ngrams_list))

    cntr = collections.Counter()
    cntr.update(ngrams_to_str(merge_list(ngrams)))
    cntr = list(filter(lambda x: x[1] > threshold, cntr.items()))

    dictionary = {UNK_TOKEN: 0}
    for item, _ in cntr:
        dictionary[item] = len(dictionary)

    def make_tilda_tokens(dictionary: dict) -> list:
        """
        Convert ngrams from `dictionary` into tilda tokens.
        :param dictionary:
        :return:
        """
        def to_tilda_tokens(ngrams: list) -> list:
            """
            >>> ngrams = [['atrial', 'fibrillation'], ['help', 'may']]
            >>> to_tilda_tokens(ngrams) == ['atrial~fibrillation', 'help~may']
            True
            """
            return list(map(lambda x: TILDA_SEP.join(x), ngrams))

        def sort_ngrams(ngrams: list) -> list:
            """
            >>> ngrams = [['atrial', 'fibrillation'], ['may', 'help']]
            >>> sort_ngrams(ngrams) == [['atrial', 'fibrillation'], ['help', 'may']]
            True
            """
            return list(map(lambda x: sorted(x), ngrams))

        def split_ngrams(ngrams: list) -> list:
            """
            >>> ngrams = ['currently_used', 'atrial_fibrillation', 'may_help']
            >>> split_ngrams(ngrams) == [['currently', 'used'], ['atrial', 'fibrillation'], ['may', 'help']]
            True
            """
            return list(map(lambda x: x.split(NGRAM_SEP), ngrams))

        return list(set(to_tilda_tokens(sort_ngrams(split_ngrams(ngrams_from_dict(dictionary))))))

    tilda_tokens = make_tilda_tokens(dictionary)

    for item in tilda_tokens:
        dictionary[item] = len(dictionary)

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reversed_dictionary


def replace_unk(string: str, dictionary: dict) -> str:
    """
    Replace out-of-vocabulary words with `UNK_TOKEN`.
    :param string:
    :param dictionary:
    :return:
    >>> dictionary = {'amino': 1}
    >>> replace_unk('to amino', dictionary) == 'UNK amino'
    True
    """
    list_unk = list(map(lambda x: x if x in dictionary else UNK_TOKEN, string.split()))
    return ' '.join(list_unk)


def replace_unk_list(words: list, dictionary: dict) -> str:
    """
    Replace out-of-vocabulary words with `UNK_TOKEN`.
    :param words:
    :param dictionary:
    :return:
    >>> dictionary = {'amino': 1}
    >>> replace_unk_list(['to', 'amino'], dictionary) == 'UNK amino'
    True
    """
    list_unk = list(map(lambda x: x if x in dictionary else UNK_TOKEN, words))
    return ' '.join(list_unk)


def indices_for_ngram(ngram: str, dictionary: dict) -> tuple:
    indices = tuple(list(map(lambda x: dictionary[x], ngram.split(NGRAM_SEP))))
    return tuple([tuple(indices), dictionary[ngram]])


def indices_for_tilda(ngram: str, dictionary: dict) -> tuple:
    indices = sorted(list(map(lambda x: dictionary[x], ngram.split(TILDA_SEP))))
    indices = tuple(indices)
    return tuple([tuple(sorted(indices)), dictionary[ngram]])


def make_tupled_dicts(dictionary: dict) -> tuple:
    """
    :param dictionary:
    :return: `dict_ngram_id_by_tuple`, `dict_tilda_id_by_tuple`
    """
    dict_ngram_id_by_tuple = {}
    dict_tilda_id_by_tuple = {}

    # ngrams
    for ngram in ngrams_from_dict(dictionary):
        tmp = indices_for_ngram(ngram, dictionary)
        words_tup, ngram_id = tmp
        dict_ngram_id_by_tuple[words_tup] = ngram_id

    # tildas
    for tilda in tildas_from_dict(dictionary):
        tmp = indices_for_tilda(tilda, dictionary)
        words_tup, tilda_id = tmp
        dict_tilda_id_by_tuple[words_tup] = tilda_id

    return dict_ngram_id_by_tuple, dict_tilda_id_by_tuple


def ngram_id(ngram_tuple: tuple, dict_ngram_id_by_tuple: dict) -> int:
    if ngram_tuple in dict_ngram_id_by_tuple:
        return dict_ngram_id_by_tuple[ngram_tuple]
    else:
        return 0


def tilda_id(tilda_tuple: tuple, dict_tilda_id_by_tuple: dict) -> int:
    tilda_tuple = tuple(sorted(tilda_tuple))
    if tilda_tuple in dict_tilda_id_by_tuple:
        return dict_tilda_id_by_tuple[tilda_tuple]
    else:
        return 0

if __name__ == "__main__":
    raise RuntimeError
