import tensorflow as tf
import copy


class Dummy(tf.contrib.rnn.RNNCell):
    """Dummy RNN unit for use as filler.
    """

    def __init__(self, **kwargs):
        self._attrs = copy.copy(kwargs)
        self._attrs['state_size'] = 0

    def __getattr__(self, name):
        return self._attrs[name]

    @property
    def output_size(self):
        return self._attrs['output_size']

    @property
    def state_size(self):
        return self._attrs['state_size']

    def __call__(self, inputs, state):
        return None, None
