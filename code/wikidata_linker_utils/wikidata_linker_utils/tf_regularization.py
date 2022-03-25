import tensorflow as tf
from .tf_operations import extract_shape
from . import tf_logger


def maybe_dropout(inputs, keep_prob, is_training):
    return tf.cond(is_training,
                   lambda: tf.nn.dropout(inputs, keep_prob),
                   lambda: inputs) if keep_prob < 1 else inputs


def add_weight_noise(x, is_training, stddev):
    return tf.cond(is_training,
                   lambda: x + tf.random_normal(
                       shape=tf.shape(x), stddev=stddev),
                   lambda: x)


def tile_like(x, shape):
    multiples = shape + [1]
    for axis in range(len(shape)):
        x = tf.expand_dims(x, axis)
    return tf.tile(x, multiples)


def maybe_replace_by_blank(inputs, keep_prob, is_training):
    inputs_shape = extract_shape(inputs)
    blank = tf.get_variable(name="Blank",
                            shape=[inputs_shape[-1]])
    def compute_blank_mask():
        keep_inputs = tf.less(tf.random.uniform(inputs_shape[:-1], dtype=tf.float32), keep_prob)
        return tf.where(keep_inputs, inputs, tile_like(blank, inputs_shape[:-1]))
    return tf.cond(is_training, compute_blank_mask, lambda: inputs)


def maybe_batch_replace_by_blank(inputs, keep_prob, is_training):
    inputs_shape = extract_shape(inputs)
    blank = tf.get_variable(name="Blank",
                            shape=[inputs_shape[-1]])
    random_var = tf.less(tf.random.uniform([], dtype=tf.float32), keep_prob)
    return tf.cond(tf.logical_or(tf.logical_not(is_training), random_var),
                   lambda: inputs,
                   lambda: tile_like(blank, inputs_shape[:-1]))


def clipped_noise(inputs, min_value, max_value, stddev, is_training):
    return tf.clip_by_value(add_weight_noise(inputs, is_training, stddev=stddev), min_value, max_value)
    # res = tf.clip_by_value(inputs, min_value, max_value)
    # idx = tf.argmax(tf.abs(inputs - res)[..., 0])
    # return tf.Print(res, [inputs[idx]], message="badness = ", summarize=100)


def activation_loss(weight, name, activation, mask, other_activations):
    # unary_scores is of form timesteps x batch x candidates
    # activation is of form timesteps x batch x candidates
    # we want delta between activation and mean activation
    mean_activ = tf.reduce_mean(tf.reduce_mean(tf.stack([activ for activ in other_activations.values()], axis=-1), axis=-1), axis=-1, keepdims=True)
    outlier_activation = tf.where(mask, activation - tf.stop_gradient(mean_activ), tf.zeros_like(activation))
    loss_val = tf.reduce_sum(tf.abs(outlier_activation)) / tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1e-6)
    tf_logger.train_summary(name, loss_val)
    return loss_val * weight
