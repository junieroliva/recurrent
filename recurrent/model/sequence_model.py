import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.rnn import LSTMStateTuple # noqa


TRAIN = 'train'
VALID = 'valid'
TEST = 'test'


class SequenceModel:
    """Encapsulates the logic for training a sequence model.
    Args:
        fetchers: A dictionary of fetchers for training, validation, and testing
            datasets-
            {TRAIN: train_fetcher, VALID: valid_fetcher, TEST: test_fetcher}.
            Each fetcher implements functions
                set_variables(
                    session, input_data, initial_state, sequence_length,
                    targets, state_op, tensors
                )
                    where
                        session is the session to use to run iteration
                        input_data is the place holder for inputs
                        initial_state is the place_holder for states to start
                        with sequence_length is the place_holder for input
                        lengths targets is the place_holder for the target
                        output state_op is the operation that yeilds final
                            states after rnn
                        tensors is a list of default values for placeholders
                    and returns
                        None
                run_iter(eval_tuple)
                    where
                        eval_tuple is a tuple of operations to run
                    and returns
                        eval_tuple_op1, eval_tuple_op2, ...
            Each fetcher has properties:
                dim: the dimension of time-points
                window: the maximum len of sequences to feed through in a batch
                input_dtype: the type of inputs
                target_dtype: the type of output targets
        cell: RNNCell to use.
        loss: The loss to optimize.
        state_is_tuple: Indicates whether state will be LSTMStateTuple.
        loss_returns_valid: Indicates wheter the loss function also returns a
            value to validate with. E.g. an accuracy.
        input_process: Function to run on input sequences before the RNN.
            E.g. word embeddings.
        target_process: Functioln to run on target sequences before the loss.
            E.g. extracting a class.
        penalty: Multiplier to ridge penalty.
        max_grad_norm: Norm to clip gradients to.
        dropout_keeprate: placeholder for dropout value
        dropout_keeprate_val: real 0< <=1 of kept dropout rate for training
    """
    def __init__(self, fetchers, cell, loss,
                 state_is_tuple=False,
                 loss_returns_valid=False,
                 input_process=None,
                 output_process=None,
                 target_process=None,
                 penalty=0.0,
                 max_grad_norm=1e8,
                 init_lr=0.1,
                 lr_decay=0.9,
                 train_iters=100000,
                 hold_iters=1000,
                 test_iters=10000,
                 print_iters=100,
                 hold_interval=1000,
                 decay_interval=10000,
                 sess=None,
                 tensors=None,
                 optimizer_class=tf.train.GradientDescentOptimizer,
                 dropout_keeprate=None,
                 dropout_keeprate_val=1.0,
                 do_static=False,
                 do_check=False,
                 custom_rnn=None,
                 input_data=None,
                 targets=None,
                 rnn_input=None,
                 rnn_targets=None,
                 momentum=None,
                 momentum_iter=10000000000,
                 pretrain_scope=None,
                 pretrain_iters=5000,
                 ):
        # Training parameters.
        self._train_iters = train_iters
        self._valid_iters = hold_iters
        self._print_iters = print_iters
        self._hold_interval = hold_interval
        self._lr_decay = lr_decay
        self._decay_interval = decay_interval
        self._test_iters = test_iters
        # Make placeholders.
        self._dim = dim = fetchers[TRAIN].dim
        self._window = window = fetchers[TRAIN].window
        self._input_dtype = input_dtype = fetchers[TRAIN].input_dtype
        self._target_dtype = target_dtype = fetchers[TRAIN].target_dtype
        state_size = cell.state_size
        if state_is_tuple:
            initial_state = (
                tf.placeholder(tf.float32,
                               [None, state_size[0]], 'initial_state_0'),
                tf.placeholder(tf.float32,
                               [None, state_size[1]], 'initial_state_1'),
            )
        else:
            initial_state = tf.placeholder(tf.float32, [None, state_size],
                                           'initial_state')
        if input_data is None:
            if dim is not None:
                input_data = tf.placeholder(input_dtype, [None, window, dim],
                                            'input_data')
            else:
                input_data = tf.placeholder(input_dtype, [None, window],
                                            'input_data')
        if rnn_targets is None and targets is None:
            if dim is not None:
                targets = tf.placeholder(target_dtype, [None, window, dim],
                                         'targets')
            else:
                targets = tf.placeholder(target_dtype, [None, window],
                                         'targets')
        sequence_length = tf.placeholder(tf.int64, [None, ], 'sequnce_length')
        # Make losses based on RNN output.
        self._cell = cell
        self._sequence_length = sequence_length
        self._input_data = input_data
        self._initial_state = initial_state
        self._input_process = input_process
        if rnn_input is None:
            if input_process is not None:
                rnn_input = input_process(input_data)
            else:
                rnn_input = input_data
        self._rnn_input = rnn_input
        if custom_rnn is not None:
            outputs, state = \
                custom_rnn(cell, rnn_input, sequence_length=sequence_length,
                           initial_state=initial_state)
        elif do_check or do_static:
            # Tensorflow has an annoying bug with check_numerics and
            # dynammic rnns.
            split_rnn_input = tf.split(
                rnn_input, int(rnn_input.get_shape()[1]), 1
            )
            squeezed_rnn_input = [
                tf.squeeze(ri, 1) for ri in split_rnn_input
            ]
            outputs_list, state = \
                tf.contrib.rnn.static_rnn(cell, squeezed_rnn_input,
                                          sequence_length=sequence_length,
                                          initial_state=initial_state)
            outputs = tf.concat(
                [tf.expand_dims(oi, 1) for oi in outputs_list], 1
            )
        else:
            outputs, state = \
                tf.nn.dynamic_rnn(cell, rnn_input,
                                  sequence_length=sequence_length,
                                  initial_state=initial_state)
        self._outputs = outputs
        self._state_op = state
        self._output_process = output_process
        if output_process is not None:
            out_proc = output_process(outputs, input_data)
        else:
            out_proc = outputs
        self._out_proc = out_proc
        self._targets = targets
        self._target_process = target_process
        if rnn_targets is None:
            if target_process is not None:
                rnn_targets = target_process(targets)
            else:
                rnn_targets = targets
        self._rnn_targets = rnn_targets
        if loss_returns_valid:
            self._loss_op, self._valid_op = loss(out_proc, rnn_targets,
                                                 sequence_length)
        else:
            self._loss_op = loss(out_proc, rnn_targets, sequence_length)
            self._valid_op = self._loss_op
        if penalty > 0.0:
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self._loss_op += penalty*sum(reg_losses)
        # Training operations.
        self._lr = tf.Variable(init_lr, trainable=False)
        optimizer = optimizer_class(self._lr)
        self.tvars = tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss_op, tvars),
                                          max_grad_norm)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
        if do_check:
            check_op = tf.add_check_numerics_ops()
            self._train_op = tf.group(self._train_op, check_op,
                                      tf.check_numerics(self._loss_op, 'poop'))
        if pretrain_scope is not None and pretrain_iters is not None:
            self._do_pretrain = True
            self._pretrain_iters = pretrain_iters
            self.ptvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            pretrain_scope)
            ptgrads, _ = tf.clip_by_global_norm(
                tf.gradients(self._loss_op, self.ptvars), max_grad_norm
            )
            self._pretrain_op = optimizer.apply_gradients(
                zip(ptgrads, self.ptvars)
            )
        else:
            self._do_pretrain = False
        if momentum is not None:
            mom_optimizer = tf.train.MomentumOptimizer(self._lr, momentum)
            self._momentum_op = mom_optimizer.apply_gradients(zip(grads, tvars))
            self._momentum_iter = momentum_iter
        else:
            self._momentum_op = None
            self._momentum_iter = None
        self._lr_update = tf.assign(self._lr, self._lr * self._lr_decay)
        # Make session if needed.
        if sess is None:
            sess = tf.Session()
        self._sess = sess
        # Set up fetchers.
        self._dropout_keeprate = dropout_keeprate
        self._fetchers = fetchers
        for fetcher in self._fetchers:
            if dropout_keeprate is not None:
                if fetcher == TRAIN:
                    dkr = dropout_keeprate_val
                else:
                    dkr = 1.0
            else:
                dkr = None
            self.setup_fetcher(self._fetchers[fetcher], dkr)
        # Empty dictionary store easily retrieve tensors.
        self.model_tensors = {}

    def setup_fetcher(self, fetcher, dropout_keeprate_val=None):
        if self._dropout_keeprate is not None:
            tensors = {self._dropout_keeprate: dropout_keeprate_val}
        else:
            tensors = {}
        fetcher.set_variables(
            self._sess, self._input_data, self._initial_state,
            self._sequence_length, self._targets, self._state_op, self._cell,
            tensors=tensors
        )

    def update_lr(self):
        self._sess.run(self._lr_update)

    def main(self, summary_log_path=None, save_path=None):
        """Runs the model on the given data.
        Args:
            summary_log_path: path to save tensorboard summaries.
            save_path: path to save best validation set model.
            print_iters: number of iterations to print to screen at.
        Returns:
            tuple of (best_validation_value, test_validation_value)
        """
        print_iters = self._print_iters
        # Summarization variables.
        average_pl = tf.placeholder(tf.float32, name='average_pl')
        average_summary = tf.summary.scalar('average_loss', average_pl)
        if summary_log_path is not None:
            train_writer = tf.summary.FileWriter(
                os.path.join(summary_log_path, TRAIN), self._sess.graph
            )
            val_writer = tf.summary.FileWriter(
                os.path.join(summary_log_path, VALID), self._sess.graph
            )
            test_writer = tf.summary.FileWriter(
                os.path.join(summary_log_path, TEST), self._sess.graph
            )
        if save_path is not None:
            saver = tf.train.Saver()
        else:
            saver = None
        # Main train loop.
        best_loss = None
        self._sess.run(tf.global_variables_initializer())
        # Pretrain if needed.
        if self._do_pretrain:
            for i in xrange(self._pretrain_iters):
                # Run a pretraining iteration.
                train_loss, _ = self._fetchers[TRAIN].run_iter(
                    (self._loss_op, self._pretrain_op)
                )
                # Abort training if we have NaN loss
                if np.isnan(train_loss):
                    return (np.NaN, np.NaN)
                # Print to screen and save summary.
                if i % print_iters == 0:
                    print('Pretrain Iter: {} Train Loss: {}'.format(i,
                                                                    train_loss))
        train_operation = self._train_op
        for i in xrange(self._train_iters):
            # Decay the learning rate.
            if i % self._decay_interval == 0:
                if i > 0:
                    self.update_lr()
                print('Iter: {} Learning rate: {}'.format(
                    i, self._sess.run(self._lr)
                ))
            # Use a momentum operator if it is over the momentum iterations.
            if self._momentum_op is not None and i == self._momentum_iter:
                print('Using momentum.')
                train_operation = self._momentum_op
            # Training.
            # Run a training iteration.
            train_loss, _ = self._fetchers[TRAIN].run_iter(
                (self._loss_op, train_operation)
            )
            # Abort training if we have NaN loss
            # TODO: use the last saved model with a lower learning rate?
            if np.isnan(train_loss):
                return (np.NaN, np.NaN)
            # Print to screen and save summary.
            if i % print_iters == 0:
                print('Iter: {} Train Loss: {}'.format(i, train_loss))
            if summary_log_path is not None:
                train_writer.add_summary(
                    self._sess.run(average_summary,
                                   feed_dict={average_pl: train_loss}), i
                )
            # Validation.
            if i == 0 or i % self._hold_interval == 0 \
                    or i+1 == self._train_iters:
                # Get validation validation value on validation set.
                valid_loss = self.test_loss(
                    i, test_writer=val_writer, average_summary=average_summary,
                    average_pl=average_pl, set_name=VALID,
                    iters=self._valid_iters
                )
                # If this is the best validation value, record and save model.
                if best_loss is None or best_loss > valid_loss:
                    best_loss = valid_loss
                    if saver is not None:
                        saver.save(self._sess,
                                   os.path.join(save_path, 'model.ckpt'))
        # Testing.
        # Get validation value on test set.
        test_loss = self.test_loss(i, saver, save_path,
                                   test_writer, average_summary, average_pl)
        return (best_loss, test_loss)

    def test_loss(self, i, saver=None, save_path=None,
                  test_writer=None, average_summary=None, average_pl=None,
                  set_name=TEST, iters=None, return_loss_list=False):
        # TODO: write doc.
        if iters is None:
            iters = self._test_iters
        if saver is not None and save_path is not None:
            saver.restore(self._sess, os.path.join(save_path, 'model.ckpt'))
        test_loss = 0.0
        test_list = []
        try:
            for j in xrange(iters):
                loss, = self._fetchers[set_name].run_iter((self._valid_op,))
                test_loss += loss
                test_list += [loss]
            # Note: the ultimate division may be off here since the sequence
            # lengths may not have all been equal
            j = iters
        except IndexError:
            print('REACHED END @ {}'.format(j))
        test_loss /= j
        if test_writer is not None:
            test_writer.add_summary(
                self._sess.run(average_summary,
                               feed_dict={average_pl: test_loss}), i
            )
        print('{} loss: {}'.format(set_name, test_loss))
        if return_loss_list:
            return test_loss, test_list
        return test_loss

    def eval_model(self, eval_op, saver=None, save_path=None, fetcher=None,
                   set_name=TEST, iters=None):
        # TODO: write doc.
        if iters is None:
            iters = self._test_iters
        if saver is not None and save_path is not None:
            saver.restore(self._sess, os.path.join(save_path, 'model.ckpt'))
        test_list = []
        if fetcher is None:
            fetcher = self._fetchers[set_name]
        try:
            for j in xrange(iters):
                out = fetcher.run_iter(eval_op)
                test_list += [out]
            # Note: the ultimate division may be off here since the sequence
            # lengths may not have all been equal
            j = iters
        except IndexError:
            print('REACHED END @ {}'.format(j))
        return test_list
