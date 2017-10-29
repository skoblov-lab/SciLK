import numpy as np
from keras import layers, backend as K


def wordemb(nwords: int, vectors: np.ndarray, mask: bool):
    # TODO docs
    def wordemb(incomming):
        emb = layers.embeddings.Embedding(input_dim=nwords,
                                          output_dim=vectors.shape[-1],
                                          mask_zero=mask,
                                          weights=[vectors])(incomming)
        return emb

    return wordemb


def charemb(nchar: int, maxlen: int, embsize: int, nunits: int,
            indrop: float, recdrop: float, mask: bool, layer=layers.LSTM):
    # TODO docs
    def charemb(incomming):
        emb = layers.embeddings.Embedding(input_dim=nchar,
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
        emb = layers.Lambda(
            lambda x: K.reshape(x, shape=[-1, shape[1], 2 * nunits]))(emb)
        return emb

    return charemb


if __name__ == "__main__":
    raise RuntimeError
