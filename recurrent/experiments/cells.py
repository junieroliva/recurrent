"""Helper classes to set up projected RNNCell of an output size based on a
passed config object.
"""
import tensorflow as tf
import recurrent.rnn_cell.sru as sru
import recurrent.rnn_cell.indexed as indexed
import recurrent.rnn_cell.dummy as dummy


class CellScoper(tf.contrib.rnn.RNNCell):
    """ Helper to ensure we get properly scoped cells. """

    def __init__(self, cell, scope):
        self._cell = cell
        self._scope = scope

    def __call__(self, inputs, state, scope=None):
        if scope is None:
            scope_name = self._scope
        else:
            scope_name = scope
        return self._cell(inputs, state, scope=scope_name)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size


class DummyCell:
    """Filler dummy cell."""

    def __init__(self, config):
        self.num_layers = config.num_layers
        self.units = config.units
        self.activation = config.activation

    def __call__(self, dim):
        return dummy.Dummy(output_size=dim, num_layers=self.num_layers,
                           units=self.units, activation=self.activation)


class LSTMCell:

    def __init__(self, config):
        self._do_dropout = config.dropout_keeprate is not None
        self._dropout_keeprate = config.dropout_keeprate
        self._units = config.units
        self._state_is_tuple = config.state_is_tuple

    def __call__(self, dim):
        if self._do_dropout:
            return tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(
                        self._units, state_is_tuple=self._state_is_tuple
                    ),
                    output_keep_prob=self._dropout_keeprate
                ), dim
            )
        return tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicLSTMCell(self._units), dim
        )


class GRUCell:

    def __init__(self, config):
        self._do_dropout = config.dropout_keeprate is not None
        self._dropout_keeprate = config.dropout_keeprate
        self._units = config.units
        self._num_layers = config.num_layers

    def __call__(self, dim):
        gru_cell = tf.contrib.rnn.GRUCell(self._units)
        if self._num_layers > 1:
            gru_cell = tf.contrib.rnn.MultiRNNCell(
                [gru_cell] * self._num_layers
            )
        if self._do_dropout:
            return tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.DropoutWrapper(
                    gru_cell, output_keep_prob=self._dropout_keeprate), dim
            )
        return tf.contrib.rnn.OutputProjectionWrapper(
            gru_cell, dim
        )


class GRUProjectionCell:

    def __init__(self, config):
        self._do_dropout = config.dropout_keeprate is not None
        self._dropout_keeprate = config.dropout_keeprate
        self._units = config.units

    def __call__(self, dim):
        unit_proj = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.GRUCell(self._units), self._units
        )
        unit = CellScoper(unit_proj, 'FirstProjection')
        if self._do_dropout:
            return tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.DropoutWrapper(
                    unit, output_keep_prob=self._dropout_keeprate
                ), dim
            )
        return tf.contrib.rnn.OutputProjectionWrapper(unit, dim)


class LSTMProjectionCell:

    def __init__(self, config):
        self._do_dropout = config.dropout_keeprate is not None
        self._dropout_keeprate = config.dropout_keeprate
        self._units = config.units
        self._state_is_tuple = config.state_is_tuple

    def __call__(self, dim):
        unit_proj = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicLSTMCell(
                self._units, state_is_tuple=self._state_is_tuple
            ),
            self._units
        )
        unit = CellScoper(unit_proj, 'FirstProjection')
        if self._do_dropout:
            return tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.DropoutWrapper(
                    unit, output_keep_prob=self._dropout_keeprate
                ), dim
            )
        return tf.contrib.rnn.OutputProjectionWrapper(unit, dim)


class SRUCell:

    def __init__(self, config):
        self._do_dropout = config.dropout_keeprate is not None
        self._dropout_keeprate = config.dropout_keeprate
        self._units = config.units
        self._alphas = config.alphas
        self._recur_dims = config.recur_dims
        self._num_stats = config.num_stats
        self._num_layers = config.num_layers

    def __call__(self, dim):
        sru_cell = sru.SimpleSRUCell(
            num_stats=self._num_stats,
            mavg_alphas=tf.constant(self._alphas),
            output_dims=self._units,
            recur_dims=self._recur_dims,
            linear_out=False,
            include_input=False,
        )
        if self._do_dropout:
            sru_cell = tf.contrib.rnn.DropoutWrapper(
                sru_cell, output_keep_prob=self._dropout_keeprate
            )
        if self._num_layers > 1:
            sru_cell = tf.contrib.rnn.MultiRNNCell(
                [sru_cell] * self._num_layers
            )
        return tf.contrib.rnn.OutputProjectionWrapper(sru_cell, dim)


class IMMECell:

    def __init__(self, config):
        self._do_dropout = config.dropout_keeprate is not None
        self._dropout_keeprate = config.dropout_keeprate
        self._units = config.units
        self._num_stats = config.num_stats
        self._num_layers = config.num_layers

    def __call__(self, dim):
        imme_cell = indexed.SimpleIndexedCell(
            num_stats=self._num_stats,
            output_dims=self._units,
            linear_out=False,
        )
        if self._do_dropout:
            imme_cell = tf.contrib.rnn.DropoutWrapper(
                imme_cell, output_keep_prob=self._dropout_keeprate
            )
        if self._num_layers > 1:
            imme_cell = tf.contrib.rnn.MultiRNNCell(
                [imme_cell] * self._num_layers
            )
        return tf.contrib.rnn.OutputProjectionWrapper(imme_cell, dim)
