import tensorflow as tf
import recurrent.model.losses as losses
import recurrent.experiments.fetchers as fetchers
from recurrent.model.sequence_model import TRAIN, VALID, TEST


class BaseConfig:
    def __init__(self,
                 fetcher_args=None,
                 units=64,
                 penalty=0.0,
                 loss=losses.mask_regression_loss,
                 loss_returns_valid=True,
                 state_is_tuple=False,
                 dropout_keeprate=None,
                 dropout_keeprate_val=None,
                 max_grad_norm=1,
                 train_iters=40000,
                 hold_iters=1000,
                 hold_interval=5000,
                 test_iters=1000,
                 train_batch=64,
                 valid_batch=64,
                 test_batch=64,
                 init_lr=0.1,
                 lr_decay=0.9,
                 decay_interval=10000,
                 optimizer_class=tf.train.GradientDescentOptimizer,
                 proj_size=None,
                 initializer_func=tf.random_uniform_initializer,
                 initializer_args=(-0.1, 0.1),
                 input_process=None,
                 target_process=None,
                 num_stats=64,
                 recur_dims=16,
                 alphas=[0.0, 0.25, 0.5, 0.9, 0.99],
                 num_layers=1,
                 activation=tf.nn.relu):
        if fetcher_args is not None:
            self.fetcher_args = fetcher_args
        else:
            self.featch_args = {
                'fetcher_class': fetchers.WholeFetcher,
                'window': None,
                'random_shuffle': True,
                'state_is_tuple': False,
                'loop_back': True,
                'shift_sequences': True
            }
        self.units = units
        self.penalty = penalty
        self.loss = loss
        self.loss_returns_valid = loss_returns_valid
        self.state_is_tuple = state_is_tuple
        self.dropout_keeprate = dropout_keeprate
        self.dropout_keeprate_val = dropout_keeprate_val
        self.max_grad_norm = max_grad_norm
        self.train_iters = train_iters
        self.hold_iters = hold_iters
        self.hold_interval = hold_interval
        self.test_iters = test_iters
        self.batch_sizes = {TRAIN: train_batch, VALID: valid_batch,
                            TEST: test_batch}
        self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.decay_interval = decay_interval
        self.optimizer_class = optimizer_class
        self.proj_size = proj_size
        self.num_layers = num_layers
        self._initializer = None
        self._initializer_func = initializer_func
        self._initializer_args = initializer_args
        self.input_process = input_process
        self.target_process = target_process
        # SRU config
        self.num_stats = num_stats
        self.recur_dims = recur_dims
        self.alphas = alphas
        self.activation = activation

    @property
    def initializer(self):
        if self._initializer is None:
            self._initializer = self._initializer_func(*self._initializer_args)
        return self._initializer

    def make_dropout_keeprate(self):
        if self.dropout_keeprate_val is not None:
            self.dropout_keeprate = tf.placeholder(tf.float32,
                                                   name='dropout_keeprate')
