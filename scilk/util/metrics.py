"""


"""

from typing import Optional, Sequence, Text, Callable, Mapping, IO
from itertools import starmap
import sys

from keras import callbacks, backend as K
from fn.func import identity
import numpy as np


class Validator(callbacks.Callback):
    modes = ("max", "min")

    # TODO docs

    def __init__(self,
                 inputs: Sequence[np.ndarray],
                 output: np.ndarray,
                 batchsize: int,
                 metrics: Mapping[Text, Callable[[np.ndarray, np.ndarray], float]],
                 transform: Callable[[np.ndarray], np.ndarray]=identity,
                 monitor: Optional[Text]=None,
                 mode: Text="max",
                 prefix: Text=None,
                 stream: IO=sys.stderr):
        """
        :param inputs:
        :param output:
        :param batchsize:
        :param metrics: a mapping between names and functions; the functions
        must have the following signature: f(true, predicted) -> float
        :param transform:
        :param monitor:
        :param mode:
        :param prefix:
        """
        super().__init__()
        if mode not in self.modes:
            raise ValueError("`mode` must be either 'max' or 'min'")
        if monitor and monitor not in metrics:
            raise ValueError("`monitor` is not in metrics")
        if monitor and not prefix:
            raise ValueError("you must provide a path prefix when monitoring")
        self.inputs = inputs
        self.output = output
        self.epoch = None
        self.batchsize = batchsize
        self.metrics = metrics
        self.mode = mode
        self.transform = transform
        self.monitor = monitor
        self.best = float("-inf") if mode == "max" else float("inf")
        self.prefix = prefix
        self.stream = stream

    def _estimate_metrics(self):
        pred = self.transform(self.model.predict(self.inputs, self.batchsize))
        return {name: f(self.output, pred) for name, f in self.metrics.items()}

    @staticmethod
    def _format_score_log(scores: Mapping[Text, float]):
        template = "{} - {:.3f}"
        return " | ".join(starmap(template.format, scores.items()))

    def _improved(self, score: float):
        return score > self.best if self.mode == "max" else score < self.best

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        scores = self._estimate_metrics()
        log = self._format_score_log(scores)
        print("\n" + log, file=self.stream)
        if self.monitor and self._improved(scores[self.monitor]):
            path = "{}-{:02d}-{:.3f}.hdf5".format(self.prefix, self.epoch, scores[self.monitor])
            print("{} improved from {} to {}; saving weights to {}".format(
                self.monitor, self.best, scores[self.monitor], path),
                end="\n\n", file=self.stream)
            self.best = scores[self.monitor]
            self.model.save_weights(path)
        elif self.monitor:
            print("{} didn't improve".format(self.monitor), end="\n\n", file=self.stream)
        self.stream.flush()

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta):
    """
    Calculates the F score, the weighted harmonic mean of precision and recall.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """
    Calculates the f-measure, the harmonic mean of precision and recall.
    """
    return fbeta_score(y_true, y_pred, beta=1)


def recall_softmax(y_true, y_pred):
    labels_true = K.argmax(y_true, axis=-1)
    labels_pred = K.argmax(y_pred, axis=-1)
    positive_true = K.cast(K.equal(labels_true, 1), dtype=K.floatx())
    positive_pred = K.cast(K.equal(labels_pred, 1), dtype=K.floatx())
    true_positives = K.sum(positive_true * positive_pred) + K.epsilon()
    return true_positives / (K.sum(positive_true) + K.epsilon())


def precision_softmax(y_true, y_pred):
    labels_true = K.argmax(y_true, axis=-1)
    labels_pred = K.argmax(y_pred, axis=-1)
    positive_true = K.cast(K.equal(labels_true, 1), dtype=K.floatx())
    positive_pred = K.cast(K.equal(labels_pred, 1), dtype=K.floatx())
    true_positives = K.sum(positive_true * positive_pred) + K.epsilon()
    return true_positives / (K.sum(positive_pred) + K.epsilon())


def fmeasure_softmax(y_true, y_pred):
    p = precision_softmax(y_true, y_pred)
    r = recall_softmax(y_true, y_pred)
    return 2 * p * r / (p + r)


if __name__ == "__main__":
    raise RuntimeError
