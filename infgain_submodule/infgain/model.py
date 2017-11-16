# import typing
# import infgain.preprocessing as preproc
#
#
# def kl_collocations(raw_text: str, n=2):
#     """
#     Execute following pipeline: `raw_text` -> sentences ->
#                                  ngrams -> dictionaries
#     :param raw_text:
#     :param n: max length of ngrams
#     :return:
#     """
#     sentences = preproc.sent_tokenize(raw_text)
#     sentences_tokenized = list(map(lambda x: preproc.word_tokenize(x), sentences))
#
#     ngrams = list(map(lambda x: preproc.make_ngrams(x, n), sentences_tokenized))
#     ngrams = list(map(lambda x: preproc.remove_bad_ngrams(x), ngrams))
#
#     dictionary, reversed_dictionary = preproc.make_dict(ngrams)
#
#     sentences_unked = list(map(lambda x: preproc.replace_unk_list(x, dictionary), sentences_tokenized))
#     sentences_unked = list(filter(lambda x: len(x) > 0, sentences_unked))
#
#     dict_ngram_id_by_tuple, dict_tilda_id_by_tuple = preproc.make_tupled_dicts(dictionary)

###
#
# DataReader <- Tokenizer
# Model <- TokenEmbedder
# Embedder, Tokenizer
#
###

class DataReader:
    def __init__(self):
        pass

if __name__ == "__main__":
    raise RuntimeError
