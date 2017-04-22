import tensorflow as tf


def leaky_relu(x, alpha):
    return tf.maximum(x, alpha*x)


class Simple1dCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    """

    def __init__(self, state_size, alpha=0.01, state_activation=None):
        self._state_size = state_size
        self._output_dims = 1
        self._alpha = alpha
        if state_activation is not None:
            self._state_activation = state_activation
        else:
            def lr(x):
                return leaky_relu(x, self._alpha)

            self._state_activation = lr

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_dims

    def __call__(self, inputs, state, scope=None, return_only_state=False):
        with tf.variable_scope(scope or type(self).__name__):
            self._w_z_y = tf.get_variable('w_z_y', shape=(1,), dtype=tf.float32)
            # self._w_z_y = tf.Print(self._w_z_y, [self._w_z_y], 'w_z_y')

            self._w_z_h = tf.get_variable('w_z_h', shape=(self.state_size, 1),
                                          dtype=tf.float32)
            # self._w_z_h = tf.Print(self._w_z_h, [self._w_z_h], 'w_z_h')

            self._b_z = tf.get_variable('b_z', shape=(1,), dtype=tf.float32)
            # self._b_z = tf.Print(self._b_z, [self._b_z], 'b_z')

            self._w_h_y = tf.get_variable('w_h_y', shape=(1,), dtype=tf.float32)
            # self._w_h_y = tf.Print(self._w_h_y, [self._w_h_y], 'w_h_y')

            self._w_h_h = tf.get_variable(
                'w_h_h', shape=(self.state_size, self.state_size),
                dtype=tf.float32
            )
            # self._w_h_h = tf.Print(self._w_h_h, [self._w_h_h], 'w_h_h')

            self._b_h = tf.get_variable('b_h', shape=(self._state_size,),
                                        dtype=tf.float32)
            # self._b_h = tf.Print(self._b_h, [self._b_h], 'b_h')

            if not return_only_state:
                output = leaky_relu(
                    self._w_z_y*inputs + tf.matmul(state, self._w_z_h) +
                    self._b_z, self._alpha
                )
                output = tf.Print(output, [inputs, output], 'input/output')
            out_state = self._state_activation(
                self._w_h_y*inputs + tf.matmul(state, self._w_h_h) + self._b_h
            )
            out_state = tf.Print(out_state, [out_state], 'encode state')
            if return_only_state:
                out_state
        return (output, out_state)

    def inverse(self, output, scope=None, original=None):
        """ Computes the inverse mapping for this rnn.
        Args:
            output: ? x d tensors
        Returns:
            inverse: ? x d tensor of original values
        """
        batch_size = tf.shape(output)[0]
        d = int(output.get_shape()[1])
        state = self.zero_state(batch_size, dtype=tf.float32)
        y_list = []
        with tf.variable_scope(scope or type(self).__name__):
            for t in range(d):
                # this is because I want to be concise, unlike this very
                # verbose comment. if z_t is positive, then the min will return
                # z_t (assuming that alpha is in (0, 1].....
                if original is not None:
                    y_t_orig = original[:, t]
                z_t = tf.expand_dims(output[:, t], -1)
                # self._w_z_y = tf.Print(self._w_z_y, [self._w_z_y], 'inv w_z_y')
                # z_t = tf.Print(z_t, [z_t], 'z_t')
                z_t_scaled = tf.minimum(z_t, z_t/self._alpha)
                # z_t_positive = tf.cast(tf.greater(z_t, 0.0), tf.float32)
                # z_t_scaled = z_t*z_t_positive + \
                #     (z_t/self._alpha)*(1-z_t_positive)
                # z_t_scaled = tf.Print(z_t_scaled,
                #                       [z_t_scaled, z_t_scaled-z_t_scaled_org],
                #                       'z_t_s new/org')
                # self._w_h_h = tf.Print(self._w_h_h, [self._w_h_h], 'inv w_h_h')
                y_t = (z_t_scaled - tf.matmul(state, self._w_z_h) - self._b_z)
                y_t /= self._w_z_y
                y_t = tf.Print(y_t, [y_t, z_t], 'y_t/z_t')
                if original is not None:
                    y_t = tf.Print(y_t, [y_t, y_t_orig], 'y_t/y_t_orig')
                y_list.append(y_t)
                state = self._state_activation(
                    self._w_h_y*y_t + tf.matmul(state, self._w_h_h) +
                    self._b_h
                )
                state = tf.Print(state, [state], 'inverse state')
            y = tf.concat(y_list, 1)
        return y
