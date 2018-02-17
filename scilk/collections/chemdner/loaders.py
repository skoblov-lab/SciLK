from functools import reduce
from typing import List, Sequence, Callable
import re

from keras import layers, models
from fn import F
import joblib
import numpy as np

from scilk.util import preprocessing, intervals, patterns, segments
from scilk.util.networks import blocks, wrappers
from scilk.collections import common


def load_tokeniser(collection, data) \
        -> Callable[[List[str]], List[List[intervals.Interval[str]]]]:
    # load char encoder
    charmap = joblib.load(data['charmap'])
    oov, _, charencoder = common.build_charencoder(charmap)

    # build text encoder
    text_encoder = F(charencoder) >> preprocessing.reverse

    # build the model
    batchsize = 64
    chunksize = 256
    inputs = layers.Input(batch_shape=(batchsize, chunksize))
    embeddings = layers.Embedding(oov+1, 32)(inputs)
    l_cnn = blocks.cnn([256, 256], 3, [0.3, None])
    l_rnn1 = wrappers.HalfStatefulBidirectional(
        layers.GRU(16, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    l_rnn2 = wrappers.HalfStatefulBidirectional(
        layers.GRU(16, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    l_rnn3 = wrappers.HalfStatefulBidirectional(
        layers.GRU(16, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    labels = layers.Dense(1, activation='sigmoid')(
        reduce(lambda graph, layer: layer(graph),
               [l_cnn, l_rnn1, l_rnn2, l_rnn3], embeddings)
    )
    model = models.Model(inputs, labels)
    model.compile(optimizer='Adam', loss='binary_crossentropy')
    model.load_weights(data['tokeniser_weights'])

    # make a primary tokeniser
    primary_tokeniser = F(patterns.ptokenise, [re.compile('\w+|[^\s\w]')])

    def decode(merged: np.ndarray, bins: Sequence[Sequence[int]],
               lengths: Sequence[int]) -> List[Sequence[int]]:
        """
        Decode predictions
        :param merged: merged predictions
        :param bins: bins
        :lengths: text lengths
        """
        unmerged = common.unmerge_bins(merged, bins, lengths)
        unbined = (F(map, preprocessing.reverse) >> list)(common.unbin(unmerged, bins))
        return [np.nonzero(anno > 0.5)[0] for anno in unbined]

    def tokenise(texts: List[str]) -> List[List[intervals.Interval]]:
        if not texts:
            raise ValueError('there are no `texts`')
        if not all(texts):
            raise ValueError('empty strings are not allowed')
        # ensure that there are at lest `batchsize` texts_
        ndummy = min(0, batchsize - len(texts))
        texts_ = texts + [str(None)] * ndummy
        # encode data
        encoded_texts = np.array(list(map(text_encoder, texts_)))
        bins = preprocessing.binpack(batchsize, len, encoded_texts)
        merged = common.merge_bins(encoded_texts, bins)
        chunks = preprocessing.chunksteps(chunksize, merged)
        # primary tokenisation
        primary_tokens = (F(map, primary_tokeniser) >> list)(texts_)
        # reset stateful layers
        for layer in [l_rnn1, l_rnn2, l_rnn3]:
            layer.reset_states()
        # predict and decode stitch points
        predicted = model.predict(np.vstack(chunks), batch_size=batchsize)
        reshaped = np.concatenate(predicted.reshape((-1, 64, 256)), axis=1)
        textlens = list(map(len, texts_))
        stiches = decode(reshaped, bins, textlens)
        # stitch primary tokens
        stiched_ivs = [segments.stitch(tks, points)
                       for tks, points in zip(primary_tokens, stiches)]
        return [
            [iv.reload(text[iv.start:iv.stop]) for iv in stiched]
            for text, stiched in zip(texts_, stiched_ivs)
        ][:len(texts)]

    return tokenise


if __name__ == '__main__':
    raise RuntimeError