import tensorflow as tf
from .transformer import compute_qkv, multihead_attention_internal_nobias, multihead_attention_internal_nobias_weights
from .tf_regularization import maybe_dropout, maybe_replace_by_blank, maybe_batch_replace_by_blank
from .make_callable import make_callable
from .sequence_model_constants import ATTENTION_WEIGHTS


def qkv_attention(inputs, num_heads, size, combine, is_training, keep_prob, global_attention=False, log_tensorboard=False):
    with tf.variable_scope("multihead_attention"):
        batch_major_inputs = tf.transpose(inputs, [1, 0, 2])
        q, k, v = compute_qkv(batch_major_inputs, batch_major_inputs, size, size, size, size,
                              q_padding="VALID", kv_padding="VALID",
                              weight_noise=0.0, is_training=is_training)
        dropout_rate = tf.cond(is_training, lambda: 1.0 - keep_prob, lambda: 0.0)
        x = multihead_attention_internal_nobias(q, k, v, dropout_rate,
            num_heads=num_heads, total_key_depth=size, image_shapes=None)
        x = tf.transpose(x, [1, 0, 2])
        weights = multihead_attention_internal_nobias_weights(q, k, dropout_rate,
            num_heads=num_heads, total_key_depth=size, image_shapes=None)
        tf.add_to_collection(ATTENTION_WEIGHTS, weights)
        if log_tensorboard:
            for head_idx in range(num_heads):
                tf_logger.test_image_summary_bw("multihead_attention_head{}".format(head_idx), weights[0, head_idx, :, :, None])
        if global_attention:
            q_global = tf.get_variable(name="GlobalQuery", shape=[1, 1, size])
            x_global = multihead_attention_internal_nobias(q_global, k, v, dropout_rate,
                 num_heads=num_heads, total_key_depth=size, image_shapes=None)
            weights = multihead_attention_internal_nobias_weights(q_global, k, dropout_rate,
                 num_heads=num_heads, total_key_depth=size, image_shapes=None)
            tf.add_to_collection(ATTENTION_WEIGHTS, weights)
            x_global = tf.tile(tf.transpose(x_global, [1, 0, 2]), [tf.shape(x)[0], 1, 1], "tiled_global_query")
        else:
            x_global = None
        if combine == "replace":
            return x if x_global is None else tf.concat([x, x_global], axis=-1)
        elif combine == "concatenate":
            return tf.concat([inputs, x] + ([x_global] if x_global is not None else []), axis=-1)
        else:
            raise ValueError("no idea how to combine using {}".format(combine))


def build_submodel_from_spec(*, inputs, spec, is_training):
    out = inputs
    for layer_idx, layer in enumerate(spec):
        with tf.variable_scope(layer.get("variable_scope", "Layer%d" % (layer_idx,))):
            if layer["type"] == "fully_connected":
                out = tf.contrib.layers.fully_connected(out, num_outputs=layer["size"])
            elif layer["type"] == "fully_connected_tanh":
                out = tf.contrib.layers.fully_connected(out, num_outputs=layer["size"], activation_fn=tf.tanh)
            elif layer["type"] == "dropout":
                out = maybe_dropout(out, layer["keep_prob"], is_training)
            elif layer["type"] == "cast":
                out = tf.cast(out, dtype=layer["dtype"])
            elif layer["type"] == "expand_dims":
                out = tf.expand_dims(out, axis=layer["axis"])
            elif layer["type"] == "qkv_attention":
                out = qkv_attention(out, num_heads=layer["n_heads"], size=layer["size"], combine=layer["combine"],
                                    is_training=is_training, keep_prob=layer["keep_prob"], global_attention=False,
                                    log_tensorboard=layer.get("log_tensorboard", False))
            elif layer["type"] == "qkv_attention_global":
                out = qkv_attention(out, num_heads=layer["n_heads"], size=layer["size"], combine=layer["combine"],
                                    is_training=is_training, keep_prob=layer["keep_prob"], global_attention=True,
                                    log_tensorboard=layer.get("log_tensorboard", False))
            elif layer["type"] == "times_zero":
                out = out * 0
            elif layer["type"] == "replace_by_blank":
                out = maybe_replace_by_blank(out, layer["keep_prob"], is_training)
            elif layer["type"] == "batch_replace_by_blank":
                out = maybe_batch_replace_by_blank(out, layer["keep_prob"], is_training)
            elif layer["type"] == "function":
                out = make_callable(layer["function"])(inputs=out, is_training=is_training)
            else:
                raise ValueError("Unknown layer type %r." % (layer["type"],))
    return out
