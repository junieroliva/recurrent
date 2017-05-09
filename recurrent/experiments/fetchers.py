import tensorflow as tf
import numpy as np
import copy
import cPickle as pickle
from recurrent.model.sequence_model import TEST, TRAIN, VALID


class WindowedFetcher:
    """ Class to encapsolate iterating through a list of sequences (of
    potentially different sizes) a window at a time.
    Args:
        data: list of n x d arrays
    """
    def __init__(self, data, batch_size, window=None, random_shuffle=True,
                 state_is_tuple=False, loop_back=True, shift_sequences=True):
        # TODO: handle this case
        if not shift_sequences:
            raise NotImplementedError
        # Properties.
        self.dim = data[0].shape[1]
        self.input_dtype = data[0].dtype
        self.target_dtype = data[0].dtype
        if window is None:
            self.window = np.max([d.shape[0] for d in data])
        else:
            self.window = window
        # Internal.
        self._data = data
        self._ninstances = len(data)
        self._batch_size = batch_size
        self._random_shuffle = random_shuffle
        if self._random_shuffle:
            self._curr_instances = np.random.randint(
                self._ninstances, size=batch_size
            )
        else:
            self._curr_instances = range(batch_size)
        self._indices = np.zeros((batch_size, ), np.int64)
        self._state_is_tuple = state_is_tuple
        self._loop_back = loop_back
        # Variables for running iteration.
        self._zero_state_val = None
        self._curr_state_val = None
        self._sess = None
        self._input_data = None
        self._targets = None
        self._initial_state = None
        self._final_state_op = None
        self._sequence_length = None
        self._tensors = None

    def set_variables(self, sess, input_data, initial_state, sequence_length,
                      targets, state_op, cell, tensors={}):
        self._zero_state_val = sess.run(
            cell.zero_state(self._batch_size, tf.float32)
        )
        self._curr_state_val = self._zero_state_val
        self._sess = sess
        self._input_data = input_data
        self._targets = targets
        self._initial_state = initial_state
        self._final_state_op = state_op
        self._sequence_length = sequence_length
        self._tensors = tensors
        # Reset indices.
        if self._random_shuffle:
            self._curr_instances = np.random.randint(
                self._ninstances, size=self._batch_size
            )
        else:
            self._curr_instances = range(self._batch_size)

    def run_iter(self, eval_tuple, return_state=False):
        """Fetch input/targets based on current indices and window.
        Assumes that self._curr_state_val has the state value of the last
        window for each sequence.
        """
        tensors = copy.copy(self._tensors)
        seq_len_val = np.zeros((self._batch_size, ), np.int64)
        if self.dim is None:
            input_data_val = np.zeros((self._batch_size, self.window),
                                      self.input_dtype)
        else:
            input_data_val = np.zeros(
                (self._batch_size, self.window, self.dim), self.input_dtype
            )
        targets_val = np.zeros_like(input_data_val)
        for i, curri in enumerate(self._curr_instances):
            ind = self._indices[i]
            if self.dim is not None:
                seq = self._data[curri][ind:, :]
            else:
                seq = self._data[curri][ind:]
            # Last target mus be the last time step, hence len(seq)-1 below.
            seq_len_val[i] = np.minimum(len(seq)-1, self.window)
            if self.dim is not None:
                input_data_val[i, :seq_len_val[i], :] = seq[:seq_len_val[i]]
                targets_val[i, :seq_len_val[i], :] = seq[1:seq_len_val[i]+1]
            else:
                input_data_val[i, :seq_len_val[i]] = seq[:seq_len_val[i]]
                targets_val[i, :seq_len_val[i]] = seq[1:seq_len_val[i]+1]
            # Check if last target was the last element in sequence, and get
            # new sequence if it was.
            if seq_len_val[i]+1 == len(seq):
                if self._random_shuffle:
                    self._curr_instances[i] = \
                        np.random.randint(self._ninstances)
                else:
                    self._curr_instances[i] = (np.max(self._curr_instances)+1)
                    if self._loop_back:
                        self._curr_instances = np.mod(self._curr_instances,
                                                      self._ninstances)
                self._indices[i] = 0
            else:
                self._indices[i] += seq_len_val[i]
        # Set dictionary of placeholders.
        tensors[self._input_data] = input_data_val
        tensors[self._targets] = targets_val
        tensors[self._sequence_length] = seq_len_val
        tensors[self._initial_state] = self._curr_state_val
        result_tuple = self._sess.run(eval_tuple+(self._final_state_op,),
                                      tensors)
        state = result_tuple[-1]
        if not return_state:
            result_tuple = result_tuple[:-1]
        # Update current states/set new sequences for next iteration as needed.
        for i, is_zero in enumerate(np.equal(self._indices, 0)):
            if is_zero:
                up_state = self._zero_state_val
            else:
                up_state = state
            if not self._state_is_tuple:
                self._curr_state_val[i] = up_state[i]
            else:
                # for j in len(self._curr_state_val):
                #     self._curr_state_val[j][i] = up_state[j][i]
                self._curr_state_val[0][i] = up_state[0][i]
                self._curr_state_val[1][i] = up_state[1][i]
        return result_tuple


class WholeFetcher:
    """ Class to encapsolate iterating through a list of sequences (of
    potentially different sizes) an entire sequence at a time.
    Args:
        data: list of n x d arrays
    """
    def __init__(self, data, batch_size, window=None, random_shuffle=True,
                 state_is_tuple=False, loop_back=True, label_data=None,
                 append_labels=False):
        # Properties.
        if len(data.shape) > 2:
            self.dim = data[0].shape[1]
        else:
            self.dim = None
        self.input_dtype = data[0].dtype
        # TODO: adjust accordingly with labels.
        self.target_dtype = data[0].dtype
        if window is None:
            self.window = np.max([d.shape[0] for d in data])
        else:
            self.window = window
        # Internal.
        self._data = data
        self._ninstances = len(data)
        self._batch_size = batch_size
        self._random_shuffle = random_shuffle
        if self._random_shuffle:
            self._curr_instances = np.random.randint(
                self._ninstances, size=batch_size
            )
        else:
            self._curr_instances = range(batch_size)
        self._state_is_tuple = state_is_tuple
        self._loop_back = loop_back
        self._label_data = label_data
        self._use_labels = label_data is not None
        if self._use_labels:
            self._label_len = label_data[0].shape[-1]
        self._append_labels = append_labels
        # Variables for running iteration.
        self._zero_state_val = None
        self._curr_state_val = None
        self._sess = None
        self._input_data = None
        self._targets = None
        self._initial_state = None
        self._final_state_op = None
        self._sequence_length = None
        self._tensors = None

    def set_variables(self, sess, input_data, initial_state, sequence_length,
                      targets, state_op, cell, tensors={}):
        self._zero_state_val = sess.run(
            cell.zero_state(self._batch_size, tf.float32)
        )
        self._curr_state_val = self._zero_state_val
        self._sess = sess
        self._input_data = input_data
        self._targets = targets
        self._initial_state = initial_state
        self._final_state_op = state_op
        self._sequence_length = sequence_length
        self._tensors = tensors
        self.reset_indices()

    def reset_indices(self):
        # Reset indices
        if self._random_shuffle:
            self._curr_instances = np.random.randint(
                self._ninstances, size=self._batch_size
            )
        else:
            self._curr_instances = range(self._batch_size)

    def run_iter(self, eval_tuple, return_state=False, ):
        """Fetch input/targets based on current indices and window.
        Assumes that self._curr_state_val has the state value of the last
        window for each sequence.
        """
        if not self._random_shuffle and len(self._curr_instances) == 0:
            if self._loop_back:
                self._curr_instances = range(self._batch_size)
            else:
                # TODO: change?
                raise IndexError
        # Zeroed variables.
        tensors = copy.copy(self._tensors)
        seq_len_val = np.zeros((len(self._curr_instances), ), np.int64)
        if self.dim is not None:
            input_data_val = np.zeros(
                (len(self._curr_instances), self.window, self.dim),
                self.input_dtype
            )
        else:
            input_data_val = np.zeros(
                (len(self._curr_instances), self.window), self.input_dtype
            )
        if self._use_labels:
            label_data_val = np.zeros(
                (len(self._curr_instances), self._label_len), self.target_dtype
            )
        # Get current instances.
        for i, ci in enumerate(self._curr_instances):
            seq_len_val[i] = len(self._data[ci])
            input_data_val[i] = self._data[ci]
            if self._use_labels:
                label_data_val[i] = self._label_data[ci]
        # Set inputs/targets, sequence length, and initial state placeholders.
        if self._targets is None:
            tensors[self._input_data] = input_data_val
        else:
            tensors[self._input_data] = input_data_val[:, :-1]
            tensors[self._targets] = input_data_val[:, 1:]
            if self._use_labels:
                if not self._append_labels:
                    tensors[self._targets] = label_data_val
                else:
                    tensors[self._targets] = np.concatenate(
                        (tensors[self._targets], label_data_val), axis=1
                    )
        if self._use_labels and self._append_labels:
            tensors[self._input_data] = np.concatenate(
                (tensors[self._input_data], label_data_val), axis=1
            )
        tensors[self._sequence_length] = seq_len_val
        tensors[self._initial_state] = self._zero_state_val
        # Evaluate given the input tensors.
        if return_state:
            result_tuple = self._sess.run(eval_tuple+(self._final_state_op,),
                                          tensors)
        else:
            result_tuple = self._sess.run(eval_tuple, tensors)
        # Get next batch of instances.
        if self._random_shuffle:
            self._curr_instances = np.random.randint(
                self._ninstances, size=self._batch_size
            )
        else:
            max_ind = np.max(self._curr_instances)
            n_insts = len(self._data)
            last_ind = np.minimum(max_ind+1+self._batch_size, n_insts)
            self._curr_instances = range(max_ind+1, last_ind)
            self._zero_state_val = \
                self._zero_state_val[:len(self._curr_instances)]
        return result_tuple


def make_fetchers(data, batch_sizes,
                  fetcher_class=WholeFetcher, window=None,
                  random_shuffle=True, state_is_tuple=False,
                  data_loader=lambda x: pickle.load(open(x, 'rb'))):
    datasets = data_loader(data)
    fetchers = {TRAIN: None, VALID: None, TEST: None}
    for dset in datasets:
        if dset == TEST:
            loop_back_val = False
            random_shuffle_val = False
        else:
            loop_back_val = True
            random_shuffle_val = random_shuffle
        fetchers[dset] = fetcher_class(
            datasets[dset], batch_sizes[dset], window=window,
            random_shuffle=random_shuffle_val, state_is_tuple=state_is_tuple,
            loop_back=loop_back_val,
        )
    return fetchers
