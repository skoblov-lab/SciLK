"""

Utility functions for creating ChemPred deep learning models and working with
their predictions

"""
from functools import reduce
from typing import Sequence, Tuple, Optional, Union, Callable

import numpy as np
from keras import layers, backend as K


def cnn(nfilters: Sequence[int],
        filter_width: Union[int, Sequence[int]],
        dropout: Union[Optional[float], Sequence[Optional[float]]]=None,
        padding: Union[str, Sequence[str]]='same',
        name_template: str='conv{}') \
        -> Callable:
    # TODO extend documentation
    # TODO more tests
    # TODO make name_template Optional
    """

    :param nfilters:
    :param filter_width:
    :return:
    """
    def stack_conv(prev, param: Tuple[str, int, int, float, str]):
        name, nfilt, kern_size, drop_p, pad = param
        l = layers.Convolution1D(
            nfilt, kern_size, activation='relu', name=name, padding=pad
        )(prev)
        return layers.Dropout(drop_p)(l) if drop_p else l

    filter_width = (filter_width if isinstance(filter_width, Sequence) else
                    [filter_width] * len(nfilters))
    dropout = (dropout if isinstance(dropout, Sequence) else
               [dropout] * len(nfilters))
    padding = (padding if isinstance(padding, Sequence) and not isinstance(padding, str)
               else [padding] * len(nfilters))

    if not len(nfilters) == len(filter_width) == len(dropout) == len(padding):
        raise ValueError('Parameter sequences have different lengths')

    def conv(incomming):
        conv_names = (name_template.format(i+1) for i in range(0, len(nfilters)))
        parameters = zip(conv_names, nfilters, filter_width, dropout, padding)
        cnn = reduce(stack_conv, parameters, incomming)
        return cnn

    return conv


def rnn(nsteps: Sequence[int],
        inp_drop: Optional[Union[float, Sequence[float]]]=None,
        rec_drop: Optional[Union[float, Sequence[float]]]=None,
        bidirectional: Union[Optional[str], Sequence[Optional[str]]]=None,
        stateful=False, layer=layers.LSTM) -> Callable:
    # TODO extend documentation
    # TODO add name template argument
    # TODO tests
    """
    :param nsteps:
    :param inp_drop:
    :param rec_drop:
    :param bidirectional:
    :param stateful: use stateful RNN-cells
    :param layer: a recurrent layer to use
    :return:
    """

    def stack_layers(prev, param: Tuple[str, int, float, float, str]):
        """
        :param prev: incomming keras layer
        :param param: [layer name, steps, input dropout, recurrent dropout,
        bidirectional]
        """
        name, steps, indrop, recdrop, bidir = param
        layer_ = layer(steps, dropout=indrop, recurrent_dropout=recdrop,
                       return_sequences=True, stateful=stateful)
        return (layers.Bidirectional(layer_, bidir) if bidir else layer_)(prev)

    bidir_is_seq = (isinstance(bidirectional, Sequence)
                    and not isinstance(bidirectional, str))
    bi = (bidirectional if bidir_is_seq else [bidirectional] * len(nsteps))
    inp_drop = (inp_drop if isinstance(inp_drop, Sequence) else
                [inp_drop or 0] * len(nsteps))
    rec_drop = (rec_drop if isinstance(rec_drop, Sequence) else
                [rec_drop or 0] * len(nsteps))

    if not len(nsteps) == len(rec_drop) == len(inp_drop) == len(bi):
        raise ValueError('Parameter sequences have different length')

    def rec(incomming):
        rec_names = ('rec{}'.format(i) for i in range(1, len(nsteps) + 1))
        parameters = zip(rec_names, nsteps, inp_drop, rec_drop, bi)
        rnn = reduce(stack_layers, parameters, incomming)
        return rnn

    return rec


def wordemb(nwords: int, vectors: np.ndarray, mask: bool):
    # TODO docs
    def wordemb(incomming):
        emb = layers.embeddings.Embedding(input_dim=nwords,
                                          output_dim=vectors.shape[-1],
                                          mask_zero=mask,
                                          weights=[vectors])(incomming)
        return emb

    return wordemb


def charemb(input_dim: int, maxlen: int, embsize: int, nunits: int,
            indrop: float, recdrop: float, mask: bool, layer=layers.LSTM):
    # TODO docs
    def charemb(incomming):
        emb = layers.embeddings.Embedding(input_dim=input_dim,
                                          output_dim=embsize,
                                          mask_zero=mask)(incomming)
        shape = (K.shape(incomming)[0], maxlen, K.shape(incomming)[2], embsize)
        emb = layers.Lambda(
            lambda x: K.reshape(x, shape=(-1, shape[-2], embsize)))(emb)

        forward = layer(nunits,
                        return_state=True,
                        dropout=indrop,
                        recurrent_dropout=recdrop)(emb)[-2]
        reverse = layer(nunits,
                        return_state=True,
                        recurrent_dropout=recdrop,
                        dropout=indrop,
                        go_backwards=True)(emb)[-2]
        emb = layers.concatenate([forward, reverse], axis=-1)
        # shape = (batch size, max sentence length, char hidden size)
        embshape = [incomming.shape[0].value or -1, shape[1], 2 * nunits]
        return layers.Lambda(lambda x: K.reshape(x, shape=embshape))(emb)

    return charemb


if __name__ == '__main__':
    raise RuntimeError
