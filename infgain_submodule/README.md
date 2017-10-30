## Information gain submodule

#### Demo is here: [demo.ipynb](demo.ipynb)

#### Requirements

Install [word2gauss](https://github.com/seomoz/word2gauss):
```
cd word2gauss_py3
python3 setup.py install    
```

Check:
```
python3
>>> from word2gauss import GaussianEmbedding, iter_pairs
>>> from word2gauss.words import Vocabulary
```

#### Terms
- Ngram case: `ngram tokens` (e.g. 'amino_acid')
- Words encounter in the same sentence: `tilda tokens` (e.g. 'amino~acid')

#### Modifications of [word2gauss](https://github.com/seomoz/word2gauss)
- In file `word2gauss_py3/word2gauss/words.py` changed method `iter_pairs()`
- In file `word2gauss_py3/word2gauss/embeddings.pyx` changed method `text_to_pairs()`, added methods `get_contexts()` and `comb_len()`

#### TODO
- [ ] `text_to_pairs()` method: calculate `npairs` more precisely
- [ ] `text_to_pairs()` method: when the sentence is too long iterate over ngram/tilda tokens from dictionary
- [ ] `regexp_word_tokenize()` method: it doesn't split by underscore
