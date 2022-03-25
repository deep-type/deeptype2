import tensorflow as tf
from .tf_regularization import add_weight_noise, maybe_dropout
from .tf_operations import concat
from .embedding import embedding_lookup


def parse_convolutions(description):
    parsed = []
    for stage in description.split("-"):
        filters, kernel_size = stage.split(":")
        parsed.append({"filters": int(filters),
                       "strides": 1,
                       "kernel_size": int(kernel_size)})
    return parsed


def build_convolutions(inputs, is_training, stages, time_major, keep_prob, weight_noise):
    if len(stages) > 0:
        if time_major:
            # input is T, N, C -> N, T, C
            inputs = tf.transpose(inputs, (1, 0, 2))
        for layer_idx, stage in enumerate(stages):
            layer = tf.layers.Conv1D(filters=stage["filters"],
                                     strides=stage["strides"],
                                     padding="SAME",
                                     data_format="channels_last",
                                     kernel_size=stage["kernel_size"])
            layer.build(inputs.get_shape())
            if weight_noise > 0:
                layer.kernel = add_weight_noise(layer.kernel, is_training, weight_noise)
            inputs = maybe_dropout(layer.apply(inputs), keep_prob=keep_prob, is_training=is_training)
        if time_major:
            # input is N, T, C -> T, N, C
            inputs = tf.transpose(inputs, (1, 0, 2))
    return inputs


def highway(x, activation_fn=tf.nn.relu, scope=None):
    size = x.get_shape()[-1].value
    with tf.variable_scope(scope or "HighwayLayer"):
        activ = tf.contrib.layers.fully_connected(
            x, size * 2, activation_fn=None, scope="FC"
        )
        transform = tf.sigmoid(activ[..., :size], name="transform_gate")
        hidden = activation_fn(activ[..., size:])
        carry = 1.0 - transform
        return tf.add(hidden * transform, x * carry, "y")


def conv2d(inputs, output_dim, k_h, k_w,
           stddev=0.02, scope=None,
           weight_noise=0.0, is_training=True):
    with tf.variable_scope(scope or "Conv2D"):
        w = tf.get_variable('w', [k_h, k_w, inputs.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if weight_noise > 0 and not isinstance(is_training, bool):
            w = add_weight_noise(w, is_training=is_training, stddev=weight_noise)
        return tf.nn.conv2d(inputs, w, strides=[1, 1, 1, 1], padding="VALID")


def character_convolution(inputs, feature):
    inputs_2d = tf.reshape(inputs,
                           [tf.shape(inputs)[0] * tf.shape(inputs)[1], tf.shape(inputs)[2]])
    _, inputs_3d = embedding_lookup(inputs_2d,
                                    dim=feature["dimension"],
                                    # 255 different bytes (uint8)
                                    # & start and end symbol:
                                    size=257,
                                    dtype=tf.float32,
                                    mask_negative=True)
    inputs_4d = tf.expand_dims(inputs_3d, 1)
    feature_pools = []
    for idx, conv_filter in enumerate(feature["filters"]):
        width, channels = conv_filter["width"], conv_filter["channels"]
        # [batch * time x 1 x word_length x embed_dim x feature_map_dim]
        conv = tf.squeeze(conv2d(inputs_4d, channels, 1, width, scope="CharacterConvolution%d" % (idx,)), [1])
        # remove word dimension
        pool = tf.reduce_max(conv, 1)
        feature_pools.append(pool)
    activations = concat(feature_pools, axis=1)
    channels_out = sum(conv_filter["channels"] for conv_filter in feature["filters"])
    activations = tf.reshape(
        tf.tanh(activations),
        [tf.shape(inputs)[0], tf.shape(inputs)[1], channels_out],
        name="CharacterConvolutionPooled"
    )
    for idx in range(feature["highway_layers"]):
        activations = highway(activations, scope="HighwayLayer%d" % (idx,),
                              activation_fn=tf.tanh)
    return activations
