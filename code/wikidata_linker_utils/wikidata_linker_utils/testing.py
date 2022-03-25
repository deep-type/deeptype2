import tensorflow as tf
from . import tf_logger

def dummy_activation_loss(activation, **kwargs):
    print(kwargs.keys())
    loss_val = tf.reduce_mean(tf.abs(activation))
    tf_logger.train_summary("dummy_activation_loss", loss_val)
    return -loss_val * 10.0
