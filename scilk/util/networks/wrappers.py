from keras import layers
from keras.layers import wrappers
from keras import backend as K
import copy


class HalfStatefulBidirectional(wrappers.Wrapper):
    """
    Unlike the built-in keras.wrappers.Bidirectional, this wrapper only makes
    the forward reading layer stateful if an incoming layer is stateful. The
    backwards reading layer is always stateless, because it makes no sense to
    transfer state between batches evolving forward in time in a reversed
    layer.
    """
    def __init__(self, layer: layers.RNN, merge_mode='concat', weights=None, **kwargs):
        super().__init__(layer, **kwargs)
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        config = layer.get_config()
        forward_conf = {**config, 'go_backwards': False}
        backward_conf = {**config, 'go_backwards': True, 'stateful': False}
        self.forward_layer = layer.__class__.from_config(forward_conf)
        self.backward_layer = layer.__class__.from_config(backward_conf)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
        self.merge_mode = merge_mode
        if weights:
            self.forward_layer.initial_weights = weights[:len(weights) // 2]
            self.backward_layer.initial_weights = weights[len(weights) // 2:]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.return_state = layer.return_state
        self.supports_masking = True

    def get_weights(self):
        return self.forward_layer.get_weights() + self.backward_layer.get_weights()

    def set_weights(self, weights):
        self.forward_layer.set_weights(weights[:len(weights) // 2])
        self.backward_layer.set_weights(weights[len(weights) // 2:])

    def compute_output_shape(self, input_shape):
        output_shape = self.forward_layer.compute_output_shape(input_shape)
        if self.return_state:
            state_shape = output_shape[1:]
            output_shape = output_shape[0]

        if self.merge_mode == 'concat':
            output_shape = list(output_shape)
            output_shape[-1] *= 2
            output_shape = tuple(output_shape)
        elif self.merge_mode is None:
            output_shape = [output_shape, copy.copy(output_shape)]

        if self.return_state:
            if self.merge_mode is None:
                return output_shape + state_shape + copy.copy(state_shape)
            return [output_shape] + state_shape + copy.copy(state_shape)
        return output_shape

    def call(self, inputs, training=None, mask=None, initial_state=None):
        kwargs = {}
        if wrappers.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if wrappers.has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask

        if initial_state is not None and wrappers.has_arg(self.layer.call, 'initial_state'):
            if not isinstance(initial_state, list):
                raise ValueError(
                    'When passing `initial_state` to a Bidirectional RNN, the state '
                    'should be a list containing the states of the underlying RNNs. '
                    'Found: ' + str(initial_state))
            forward_state = initial_state[:len(initial_state) // 2]
            y = self.forward_layer.call(inputs, initial_state=forward_state, **kwargs)
            y_rev = self.backward_layer.call(inputs, **kwargs)
        else:
            y = self.forward_layer.call(inputs, **kwargs)
            y_rev = self.backward_layer.call(inputs, **kwargs)

        if self.return_state:
            states = y[1:] + y_rev[1:]
            y = y[0]
            y_rev = y_rev[0]

        if self.return_sequences:
            y_rev = K.reverse(y_rev, 1)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]

        # Properly set learning phase
        if (getattr(y, '_uses_learning_phase', False) or
           getattr(y_rev, '_uses_learning_phase', False)):
            if self.merge_mode is None:
                for out in output:
                    out._uses_learning_phase = True
            else:
                output._uses_learning_phase = True

        if self.return_state:
            if self.merge_mode is None:
                return output + states
            return [output] + states
        return output

    def reset_states(self):
        self.forward_layer.reset_states()

    def build(self, input_shape):
        with K.name_scope(self.forward_layer.name):
            self.forward_layer.build(input_shape)
        with K.name_scope(self.backward_layer.name):
            self.backward_layer.build(input_shape)
        self.built = True

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            if not self.merge_mode:
                return [mask, mask]
            else:
                return mask
        else:
            return None

    @property
    def trainable_weights(self):
        if hasattr(self.forward_layer, 'trainable_weights'):
            return (self.forward_layer.trainable_weights +
                    self.backward_layer.trainable_weights)
        return []

    @property
    def non_trainable_weights(self):
        if hasattr(self.forward_layer, 'non_trainable_weights'):
            return (self.forward_layer.non_trainable_weights +
                    self.backward_layer.non_trainable_weights)
        return []

    @property
    def updates(self):
        if hasattr(self.forward_layer, 'updates'):
            return self.forward_layer.updates + self.backward_layer.updates
        return []

    @property
    def losses(self):
        if hasattr(self.forward_layer, 'losses'):
            return self.forward_layer.losses + self.backward_layer.losses
        return []

    @property
    def constraints(self):
        constraints = {}
        if hasattr(self.forward_layer, 'constraints'):
            constraints.update(self.forward_layer.constraints)
            constraints.update(self.backward_layer.constraints)
        return constraints

    def get_config(self):
        return {**super().get_config(), 'merge_mode': self.merge_mode}


if __name__ == '__main__':
    raise RuntimeError
