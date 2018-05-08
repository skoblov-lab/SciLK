from functools import reduce
from typing import List, Sequence, Callable
from itertools import chain, accumulate
import operator as op
import re

from keras import layers, models
from keras import backend as K
from fn import F
from fn.iters import droplast
import joblib
import numpy as np

import scilk.util.binning
from scilk.util import preprocessing, intervals, patterns, segments, binning
from scilk.util.networks import blocks, wrappers
from scilk.collections import common


def load_tokeniser(data) \
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
        unmerged = scilk.util.binning.unmerge_bins(merged, bins, lengths)
        unbined = (
            F(map, preprocessing.reverse) >> list
        )(binning.unbin(unmerged, bins))
        return [np.nonzero(anno > 0.5)[0] for anno in unbined]

    def tokenise(texts: List[str]) -> List[List[intervals.Interval]]:
        if not texts:
            raise ValueError('there are no `texts`')
        if not all(texts):
            raise ValueError('empty strings are not allowed')
        # ensure that there are at lest `batchsize` texts_
        ndummy = max(0, batchsize - len(texts))
        texts_ = [*texts, *[str(None)]*ndummy]
        # encode data
        encoded_texts = np.array(list(map(text_encoder, texts_)))
        bins = scilk.util.binning.binpack(batchsize, len, encoded_texts)
        merged = scilk.util.binning.merge_bins(encoded_texts, bins, np.int32)
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
            [iv.reload(text[iv.start:iv.stop]) for iv in stitched]
            for text, stitched in zip(texts_, stiched_ivs)
        ][:len(texts)]

    return tokenise


def load_detector(data):
    # load token encoders
    embeddings = common.read_glove(data['embeddings'])
    wordencoder = common.build_wordencoder(
        embeddings, lambda word: '<NUM>' if word.isnumeric() else word
    )
    wordlen = 32
    maxchar, charmap, charencoder = common.build_charencoder(
        joblib.load(data['charmap']), wordlen
    )
    # build the network
    chunksize = 128
    batchsize = 32
    wordemb_dim = embeddings.shape[1]
    charemb_dim = 32
    charemb_units = 16

    # character block
    inputs_char = layers.Input(batch_shape=(batchsize, chunksize, wordlen))
    char_embeddings = blocks.charemb(maxchar + 1, chunksize, charemb_dim,
                                     charemb_units, 0.3, 0.3, mask=False,
                                     layer=layers.GRU)(inputs_char)
    char_conv_narrow = blocks.cnn([256, 256], 3, [0.3, None],
                                  name_template='narrowcharconv{}')(
        char_embeddings)
    inputs_word = layers.Input(
        batch_shape=(batchsize, chunksize, wordemb_dim))
    word_conv_narrow = blocks.cnn([256, 256], 3, [0.3, None],
                                  name_template='narrowwordconv{}')(inputs_word)

    def reshape(shape, layer):
        return layers.Lambda(
            lambda incomming: K.reshape(incomming, shape=shape)
        )(layer)

    feat_shape = [batchsize, chunksize, -1]
    feat_layers = [char_conv_narrow, word_conv_narrow]
    features = reshape(feat_shape, layers.concatenate(feat_layers, axis=-1))
    rnn_common1 = wrappers.HalfStatefulBidirectional(
        layers.GRU(32, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    rnn_common2 = wrappers.HalfStatefulBidirectional(
        layers.GRU(32, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    common_rnn_layers = [rnn_common1, rnn_common2]
    rnn_common = reduce(lambda graph, layer: layer(graph), common_rnn_layers,
                        features)
    rnn_parts = wrappers.HalfStatefulBidirectional(
        layers.GRU(32, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    labels_parts = layers.Dense(1, activation='sigmoid')(rnn_parts(rnn_common))
    attention = reshape(feat_shape, layers.multiply([labels_parts, rnn_common]))
    rnn_starts = wrappers.HalfStatefulBidirectional(
        layers.GRU(32, stateful=True, dropout=0.3, recurrent_dropout=0.3,
                   return_sequences=True))
    labels_starts = layers.Dense(1, activation='sigmoid')(rnn_starts(attention))
    model = models.Model([inputs_char, inputs_word], [labels_parts, labels_starts])
    model.compile(optimizer='Adam', loss='binary_crossentropy')
    model.load_weights(data['weights'])
    stateful_layers = [*common_rnn_layers, rnn_parts, rnn_starts]

    def decode(samples, merged_starts: np.ndarray, merged_parts: np.ndarray,
               bins: Sequence[Sequence[int]],
               lengths: Sequence[int]) -> List[List[intervals.Interval]]:
        """
        Decode predictions
        :param samples: original unbinned data
        :param merged_starts: merged predictions (a boolean array)
        :param merged_parts: merged predictions (a boolean array)
        :param bins: bins
        :lengths: text lengths
        """
        offsets = F(map, len) >> accumulate >> (droplast, 1) >> (chain, [0]) >> list

        def extract_groups(starts, parts):
            runs = np.split(parts, np.nonzero(starts)[0])
            return (F(filter, len) >> list)(
                np.nonzero(run)[0] + offset
                for offset, run in zip(offsets(runs), runs)
            )

        anno_starts, anno_parts = map(
            F(common.unmerge_bins, bins=bins, lengths=lengths) >> F(common.unbin, bins=bins),
            [merged_starts, merged_parts]
        )

        sample_groups = list(map(extract_groups, anno_starts, anno_parts))
        return [
            [reduce(op.and_, [sample[i] for i in group]) for group in groups]
            for sample, groups in zip(samples, sample_groups)
        ]

    def detect(texts: List[List[str]]) -> List[List[intervals.Interval]]:
        if not texts:
            raise ValueError('there are no `texts`')
        if not all(texts):
            raise ValueError('empty texts are not allowed')
        # ensure that there are at lest `batchsize` texts_
        ndummy = max(0, batchsize - len(texts))
        texts_ = [*texts, *[[str(None)]]*ndummy]
        # bin texts
        bins = binning.binpack(batchsize, len, texts_)
        # encode data
        char_embs = np.array([charencoder(txt) for txt in texts_])
        word_embs = np.array([wordencoder(txt) for txt in texts_])
        x_char, x_word = map(
            F(binning.merge_bins, bins=bins) >> (preprocessing.chunksteps, chunksize),
            [char_embs, word_embs]
        )
        for layer in stateful_layers:
            layer.reset_states()
        inputs = [np.vstack(x_char), np.vstack(x_word)]
        # predict probabilities and assign labels
        pred_parts_prob, pred_starts_prob = model.predict(inputs)
        pred_starts, pred_parts = map(
            lambda y: (np.vstack(y) > 0.5).astype(int),
            [pred_starts_prob, pred_parts_prob]
        )
        # reshape labels
        original_shape = (-1, batchsize, chunksize)
        reshaped_starts = np.concatenate((pred_starts * pred_parts).reshape(original_shape), axis=1)
        reshaped_parts = np.concatenate(pred_parts.reshape(original_shape), axis=1)
        return decode(texts, reshaped_starts, reshaped_parts, bins, list(map(len, texts)))

    return detect

def load_sentence_segmenter(data):
    pass


if __name__ == '__main__':
    raise RuntimeError