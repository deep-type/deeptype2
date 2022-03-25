import numpy as np


def split(values, axis, num_splits, name=None):
    import tensorflow as tf
    return tf.split(values, num_splits, axis=axis, name=name)


def reverse(values, axis):
    import tensorflow as tf
    return tf.reverse(values, [axis])


def sparse_softmax_cross_entropy_with_logits(logits, labels):
    import tensorflow as tf
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)


def concat(values, axis, name=None):
    import tensorflow as tf
    if len(values) == 1:
        return values[0]
    return tf.concat(values, axis, name=name)


def sum_list(elements):
    total = elements[0]
    for el in elements[1:]:
        total += el
    return total


def silence_tensorflow_warnings():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    import tensorflow as tf
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except AttributeError:
        tf.logging.set_verbosity(tf.logging.ERROR)
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    try:
        from tensorflow.python.util import module_wrapper as deprecation
    except ImportError:
        from tensorflow.python.util import deprecation_wrapper as deprecation
    deprecation._PER_MODULE_WARNING_LIMIT = 0
    import logging
    logging.getLogger('tensorflow').disabled = True
    warnings.filterwarnings('ignore', category=UserWarning)


def count_number_of_parameters():
    import tensorflow as tf
    return int(sum([np.prod(var.get_shape().as_list())
                    for var in tf.trainable_variables()]))


def extract_shape(x):
    import tensorflow as tf
    known_shape = x.get_shape().as_list()
    symbolic_shape = tf.shape(x)
    return [known_shape[i] if known_shape[i] is not None else symbolic_shape[i]
            for i in range(len(known_shape))]


def scope_variables(scope=None):
    import tensorflow as tf
    if scope is None:
        scope = tf.get_variable_scope().name
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def recursive_shape_invariants(x, known_shape_invariants):
    import tensorflow as tf
    if isinstance(x, list):
        if known_shape_invariants is None:
            known_shape_invariants = [None for _ in x]
        return [recursive_shape_invariants(el, known) if known is None else known for el, known in zip(x, known_shape_invariants)]
    elif isinstance(x, tuple):
        if known_shape_invariants is None:
            known_shape_invariants = [None for _ in x]
        return tuple([recursive_shape_invariants(el, known) if known is None else known for el, known in zip(x, known_shape_invariants)])
    else:
        if known_shape_invariants is None:
            return x.get_shape() if hasattr(x, "get_shape") else tf.TensorShape(None)
        else:
            return known_shape_invariants


def lme_pool(inputs, segment_ids, num_segments, a=1.0, name=None):
    import tensorflow as tf
    if a != 1:
        return (1.0 / a) * tf.log(tf.math.unsorted_segment_mean(tf.exp(a * inputs), segment_ids, num_segments), name=name)
    else:
        return tf.log(tf.math.unsorted_segment_mean(tf.exp(inputs), segment_ids, num_segments), name=name)
