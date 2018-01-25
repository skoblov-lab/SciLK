import sys
from itertools import starmap
from typing import Sequence, Mapping, Text, Callable, Optional, IO

import numpy as np
from fn.op import identity
from keras import callbacks


class Validator(callbacks.Callback):
    modes = ('max', 'min')

    # TODO docs

    def __init__(self,
                 inputs: Sequence[np.ndarray],
                 output: np.ndarray,
                 batchsize: int,
                 metrics: Mapping[Text, Callable[[np.ndarray, np.ndarray], float]],
                 transform: Callable[[np.ndarray], np.ndarray]=identity,
                 monitor: Optional[Text]=None,
                 mode: Text='max',
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
            raise ValueError('`mode` must be either "max" or "min"')
        if monitor and monitor not in metrics:
            raise ValueError('`monitor` is not in metrics')
        if monitor and not prefix:
            raise ValueError('you must provide a path prefix when monitoring')
        self.inputs = inputs
        self.output = output
        self.epoch = None
        self.batchsize = batchsize
        self.metrics = metrics
        self.mode = mode
        self.transform = transform
        self.monitor = monitor
        self.best = float('-inf') if mode == 'max' else float('inf')
        self.prefix = prefix
        self.stream = stream

    def _estimate_metrics(self):
        pred = self.transform(self.model.predict(self.inputs, self.batchsize))
        return {name: f(self.output, pred) for name, f in self.metrics.items()}

    @staticmethod
    def _format_score_log(scores: Mapping[Text, float]):
        template = '{} - {:.3f}'
        return " | ".join(starmap(template.format, scores.items()))

    def _improved(self, score: float):
        return score > self.best if self.mode == 'max' else score < self.best

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        scores = self._estimate_metrics()
        log = self._format_score_log(scores)
        print("\n" + log, file=self.stream)
        if self.monitor and self._improved(scores[self.monitor]):
            path = '{}-{:02d}-{:.3f}.hdf5'.format(self.prefix, self.epoch, scores[self.monitor])
            print('{} improved from {} to {}; saving weights to {}'.format(
                self.monitor, self.best, scores[self.monitor], path),
                end='\n\n', file=self.stream)
            self.best = scores[self.monitor]
            self.model.save_weights(path)
        elif self.monitor:
            print("{} didn't improve".format(self.monitor), end='\n\n', file=self.stream)
        self.stream.flush()


class Resetter(callbacks.Callback):
    def __init__(self, layers):
        # TODO docs
        super().__init__()
        self.layers = layers

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.layers:
            layer.reset_states()

    on_epoch_begin = on_epoch_end


if __name__ == '__main__':
    raise RuntimeError
