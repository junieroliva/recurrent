import tensorflow as tf
import os
import copy
import numpy as np
import recurrent.model.sequence_model as seqmodel
import recurrent.experiments.config as configs
from recurrent.model.sequence_model import TRAIN


def main(summary_location, config, cell_class, save_location=None,
         fetchers=None, data_loader=None, data_location=None):
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session() as sess:
        if config.dropout_keeprate_val is not None:
            config.make_dropout_keeprate()
        initializer = config.initializer
        cell = cell_class(config)
        if config.proj_size is None:
            proj_size = fetchers[TRAIN].dim
        else:
            proj_size = config.proj_size
        with tf.variable_scope('model', reuse=None, initializer=initializer):
            if fetchers is None:
                fetchers = data_loader(data_location, config.batch_sizes,
                                       **config.data_args)
            model = seqmodel.SequenceModel(
                fetchers, cell(proj_size), config.loss,
                loss_returns_valid=config.loss_returns_valid,
                input_process=config.input_process,
                target_process=config.target_process,
                penalty=config.penalty,
                max_grad_norm=config.max_grad_norm,
                init_lr=config.init_lr,
                lr_decay=config.lr_decay,
                train_iters=config.train_iters,
                hold_iters=config.hold_iters,
                test_iters=config.test_iters,
                hold_interval=config.hold_interval,
                decay_interval=config.decay_interval,
                sess=sess,
                optimizer_class=config.optimizer_class,
            )
            return model.main(summary_location, save_path=save_location)


def make_main_helper(summary_base_location, save_base_location,
                     cell_class, default_config_vals={}, status_flags=None,
                     data_location=None, data_loader=None, fetchers=None,
                     config_class=configs.BaseConfig):
    def main_helper(args):
        trial_name = ''
        for attr in args:
            trial_name += '{}_{}_'.format(attr, args[attr])
        print(trial_name)
        args_full = copy.copy(args)
        for attr in default_config_vals:
            args_full[attr] = default_config_vals[attr]
        config = config_class(**args_full)
        summary_path = os.path.join(summary_base_location, trial_name)
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        save_path = os.path.join(save_base_location, trial_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        loss_val, loss_tst = main(summary_path,
                                  config,
                                  cell_class,
                                  save_path,
                                  data_location=data_location,
                                  data_loader=data_loader,
                                  fetchers=fetchers)
        result_dict = {'loss': loss_val, 'test_loss': loss_tst}
        if status_flags is not None:
            if not np.isnan(loss_val):
                result_dict['status'] = status_flags[0]
            else:
                result_dict['status'] = status_flags[1]
        return result_dict
    return main_helper
