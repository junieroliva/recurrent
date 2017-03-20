import tensorflow as tf


def mask_regression_loss(pred, targets, sequence_length=None, penalty=0.0):
    """Compute the squared norm loss.
    Args:
        pred: ? x T x d tensor of predicted values
        targets: ? x T x d tensor of target values
        sequence_length: batch length (where each element <= T) tensor
        panalty: real value indicating the ridge squared penalty
    Return:
        masked_loss, masked_xentropy tuple:
            masked_loss: the mean squared difference plus the penalty
            masked_xentropy: the mean squared difference
    """
    squared_diff = tf.reduce_sum(
        tf.squared_difference(pred, targets), 2, name='squared_diff'
    )
    if sequence_length is not None:
        masked_xentropy = tf.reduce_mean(
            tf.reduce_sum(mask_loss(squared_diff, sequence_length) /
                          tf.cast(tf.expand_dims(sequence_length, 1),
                                  tf.float32),
                          1),
            name='masked_loss'
        )
    else:
        masked_xentropy = tf.reduce_mean(squared_diff, name='masked_loss')
    if penalty >= 0.0:
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        masked_loss = masked_xentropy + penalty*sum(reg_losses)
    else:
        masked_loss = masked_xentropy
    return masked_loss, masked_xentropy


def mask_loss(losses, sequence_length):
    """ Mask the losses of a sequence model based on length.
    Args:
        losses: batch x T, or batch x T x m tensor of losses
        sequence_length: batch length (where each element <= T) tensor
    Return:
        masked_loss: Tensor with the same shape as losses, where
            masked_loss[i, j, k] is 0 if j >= sequence_length[i], or is
            loss[i, j, k] otherwise.
    """
    losses_shape = tf.shape(losses, name='losses_shape')
    seqlen_expanded = tf.expand_dims(sequence_length, -1)
    if len(losses.get_shape()) > 2:
        seqlen_expanded = tf.expand_dims(seqlen_expanded, -1)
    window_indices = tf.cast(
        tf.range(losses_shape[1]), sequence_length.dtype, name='window_indices'
    )
    if len(losses.get_shape()) > 2:
        tile_shape = [losses_shape[0], 1, losses_shape[2]]
        window_indices_exp = tf.expand_dims(
            tf.expand_dims(window_indices, 0), -1
        )
    else:
        tile_shape = [losses_shape[0], 1]
        window_indices_exp = tf.expand_dims(window_indices, 0)
    window_tiled = tf.tile(window_indices_exp, tile_shape, name='win_tiled')
    seqlen_mask = tf.cast(
        tf.less(window_tiled, seqlen_expanded), losses.dtype, name='seq_mask'
    )
    return tf.multiply(losses, seqlen_mask, name='loss_masked')
