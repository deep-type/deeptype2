import math
import tensorflow as tf
from tensorflow.python.framework import function


# def relu_density_logit(x, reduce_dims):
#     """logit(density(x)).
#     Useful for histograms.
#     Args:
#         x: a Tensor, typilcally the output of tf.relu
#         reduce_dims: a list of dimensions
#     Returns:
#         a Tensor
#     """
#     frac = tf.reduce_mean(tf.to_float(x > 0.0), reduce_dims)
#     scaled = tf.log(frac + math.exp(-10)) - tf.log((1.0 - frac) + math.exp(-10))
#     return scaled

def shape_list(x):
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


class recomputable(object):
    '''A wrapper that allows us to choose whether we recompute
    activations during the backward pass, or we
    just use the function as normal.
    Usage:
    @recompute_option('func')
    def func(x, y):
        k = g(x, y)
        j = h(k)
        return j
    z1 = func(x, y, recompute=True)
    z2 = func(x, y, recompute=False)
    Behavior:
        z1 will not store activations for k and j, whereas z2 will.
    NOTE: args to `func` must be tensors. kwargs must not
    be tensors. kwargs must include recompute.
    IMPORTANT: variables should *not* be declared inside of this function,
    but rather be declared externally and then passed as tensor
    arguments!
    '''

    def __init__(self, name):
        self.name = name
        self.output_shape_cache = None
        self.normal_fn = None
        self.recompute_fns = {}

    def __call__(self, f):
        self.normal_fn = f
        return self.meta_fn

    def meta_fn(self, *args, **kwargs):
        # This function decides whether to build the recompute fn,
        # apply it, or use the non-recompute function.
        # It needs to build a new function for each new set of
        # kwargs.
        name = f"{self.name}"
        for key in kwargs:
            name += f"-{key}-{kwargs[key]}"
        try:
            size_hash = str(hash(int(
                ''.join([str(len(a.shape.as_list()))
                         for a in args]))))[0:6]
        except AttributeError:
            raise ValueError('Non-tensor arguments must be keyword arguments.')
        name += size_hash
        if name not in self.recompute_fns:
            self.recompute_fns[name] = self.build_fns(name, args, kwargs)
        return self.recompute_fns[name](*args)

    def build_fns(self, name, outer_args, outer_kwargs):
        input_shapes = [x.get_shape() for x in outer_args]
        output_shape_cache = None

        @function.Defun(func_name=name + "_bwd", noinline=False)
        def bwd(*args):
            nonlocal output_shape_cache
            nonlocal input_shapes
            fwd_args = args[:-1]
            dy = args[-1]
            for i, a in enumerate(fwd_args):
                a.set_shape(input_shapes[i])
            with tf.device("/gpu:0"), tf.control_dependencies([dy]):
                y = self.normal_fn(*fwd_args, **outer_kwargs)
            gs = tf.gradients(ys=[y], xs=fwd_args, grad_ys=[dy])
            return gs

        @function.Defun(func_name=name, noinline=False, grad_func=bwd,
                        shape_func=lambda x: output_shape_cache)
        def fwd(*args):
            nonlocal output_shape_cache
            nonlocal input_shapes
            with tf.device("/gpu:0"):
                fwd_args = args
                for i, a in enumerate(args):
                    a.set_shape(input_shapes[i])
                y = self.normal_fn(*fwd_args, **outer_kwargs)
            if not output_shape_cache:
                try:
                    output_shape_cache = [o.get_shape() for o in y]
                except TypeError:
                    output_shape_cache = [y.get_shape()]
            return y

        return fwd


def conv_hidden_relu(inputs,
                     hidden_size,
                     output_size,
                     kernel_size=(1, 1),
                     second_kernel_size=(1, 1),
                     dropout=0.0,
                     **kwargs):
    """Hidden layer with RELU activation followed by linear projection."""
    name = kwargs.pop("name") if "name" in kwargs else None
    with tf.variable_scope(name, "conv_hidden_relu", [inputs]):
        if inputs.get_shape().ndims == 3:
            is_3d = True
            inputs = tf.expand_dims(inputs, 2)
        else:
            is_3d = False
        conv_f1 = conv
        h = conv_f1(inputs,
                    hidden_size,
                    kernel_size,
                    activation=tf.nn.relu,
                    name="conv1",
                    **kwargs)
        if dropout != 0.0:
            h = tf.nn.dropout(h, 1.0 - dropout)
        # if not tf.get_variable_scope().reuse:
        #     tf.summary.histogram("hidden_density_logit",
        #                          relu_density_logit(h,
        #                                             list(range(inputs.shape.ndims - 1))))
        conv_f2 = conv
        ret = conv_f2(h, output_size, second_kernel_size, name="conv2", **kwargs)
        if is_3d:
            ret = tf.squeeze(ret, 2)
        return ret


def transformer_ffn_layer(x, hparams):
    """Feed-forward layer in the transformer.
    Args:
        x: a Tensor of shape [batch_size, length, hparams.hidden_size]
        hparams: hyperparmeters for model
    Returns:
        a Tensor of shape [batch_size, length, hparams.hidden_size]
    """
    if hparams.ffn_layer == "conv_hidden_relu":
        # In simple convolution mode, use `pad_remover` to speed up processing.
        conv_output = conv_hidden_relu(x,
                                       hparams.filter_size,
                                       hparams.hidden_size,
                                       dropout=hparams.relu_dropout)
        return conv_output
    elif hparams.ffn_layer == "parameter_attention":
        return parameter_attention(x,
                                   hparams.parameter_attention_key_channels or hparams.hidden_size,
                                   hparams.parameter_attention_value_channels or hparams.hidden_size,
                                   hparams.hidden_size, hparams.filter_size, hparams.num_heads,
                                   hparams.attention_dropout)
    elif hparams.ffn_layer == "conv_hidden_relu_with_sepconv":
        return conv_hidden_relu(
            x,
            hparams.filter_size,
            hparams.hidden_size,
            kernel_size=(3, 1),
            second_kernel_size=(31, 1),
            padding="LEFT",
            dropout=hparams.relu_dropout)
    else:
        assert hparams.ffn_layer == "none"
        return x


def apply_norm(x, norm_type, depth, epsilon):
    """Apply Normalization."""
    if norm_type == "layer":
        return layer_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == "batch":
        return tf.layers.batch_normalization(x, epsilon=epsilon)
    if norm_type == "noam":
        return noam_norm(x, epsilon)
    if norm_type == "none":
        return x
    raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                     "'noam', 'none'.")


def layer_prepostprocess(previous_value, x, sequence, dropout_rate, norm_type,
                                                 depth, epsilon, name):
    """Apply a sequence of functions to the input or output of a layer.
    The sequence is specified as a string which may contain the following
    characters:
        a: add previous_value
        n: apply normalization
        d: apply dropout
    For example, if sequence=="dna", then the output is
        previous_value + normalize(dropout(x))
    Args:
        previous_value: A Tensor, to be added as a residual connection ('a')
        x: A Tensor to be transformed.
        sequence: a string.
        dropout_rate: a float
        norm_type: a string (see apply_norm())
        depth: an integer (size of last dimension of x).
        epsilon: a float (parameter for normalization)
        name: a string
    Returns:
        a Tensor
    """
    with tf.variable_scope(name):
        if sequence == "none":
            return x
        for c in sequence:
            if c == "a":
                x += previous_value
            elif c == "n":
                x = apply_norm(x, norm_type, depth, epsilon)
            else:
                assert c == "d", ("Unknown sequence step %s" % c)
                x = tf.nn.dropout(x, 1.0 - dropout_rate)
        return x


def layer_preprocess(layer_input, hparams):
    """Apply layer preprocessing.
    See layer_prepostprocess() for details.
    A hyperparemeters object is passed for convenience.  The hyperparameters
    that may be used are:
        layer_preprocess_sequence
        layer_prepostprocess_dropout
        norm_type
        hidden_size
        norm_epsilon
    Args:
        layer_input: a Tensor
        hparams: a hyperparameters object.
    Returns:
        a Tensor
    """
    assert "a" not in hparams.layer_preprocess_sequence, (
            "No residual connections allowed in hparams.layer_preprocess_sequence")
    return layer_prepostprocess(
            None,
            layer_input,
            sequence=hparams.layer_preprocess_sequence,
            dropout_rate=hparams.layer_prepostprocess_dropout,
            norm_type=hparams.norm_type,
            depth=hparams.hidden_size,
            epsilon=hparams.norm_epsilon,
            name="layer_prepostprocess")


def layer_norm_vars(filters):
    """Create Variables for layer norm."""
    scale = tf.get_variable(
            "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
            "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
    return scale, bias


def layer_norm_compute_python(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


@function.Defun(compiled=True)
def layer_norm_compute_grad(x, epsilon, scale, bias, dy):
    y = layer_norm_compute_python(x, epsilon, scale, bias)
    dx = tf.gradients(ys=[y], xs=[x, epsilon, scale, bias], grad_ys=[dy])
    return dx

allow_defun = False

@function.Defun(
        compiled=True,
        separate_compiled_gradients=True,
        grad_func=layer_norm_compute_grad)
def layer_norm_compute(x, epsilon, scale, bias):
    return layer_norm_compute_python(x, epsilon, scale, bias)


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = x.get_shape()[-1]
    with tf.variable_scope(
            name, default_name="layer_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
                "layer_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
                "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
        if allow_defun:
            result = layer_norm_compute(x, tf.constant(epsilon), scale, bias)
            result.set_shape(x.get_shape())
        else:
            result = layer_norm_compute_python(x, epsilon, scale, bias)
        return result


def noam_norm(x, epsilon=1.0, name=None):
    """One version of layer normalization."""
    with tf.name_scope(name, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) *
                        tf.sqrt(tf.to_float(shape[-1])))


def layer_postprocess(layer_input, layer_output, hparams):
    """Apply layer postprocessing.
    See layer_prepostprocess() for details.
    A hyperparemeters object is passed for convenience.  The hyperparameters
    that may be used are:
        layer_postprocess_sequence
        layer_prepostprocess_dropout
        norm_type
        hidden_size
        norm_epsilon
    Args:
        layer_input: a Tensor
        layer_output: a Tensor
        hparams: a hyperparameters object.
    Returns:
        a Tensor
    """
    return layer_prepostprocess(
            layer_input,
            layer_output,
            sequence=hparams.layer_postprocess_sequence,
            dropout_rate=hparams.layer_prepostprocess_dropout,
            norm_type=hparams.norm_type,
            depth=hparams.hidden_size,
            epsilon=hparams.norm_epsilon,
            name="layer_postprocess")


class TransformerConfig(object):
    def __init__(self, **args):
        self._data = args

    def __getattr__(self, name):
        return self._data[name]


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
    Returns:
        a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return x + signal

def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
    Args:
        x: a Tensor with shape [..., a, b]
    Returns:
        a Tensor with shape [..., ab]
    """
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)
    return ret

def combine_heads(x):
    """Inverse of split_heads.
    Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
    Returns:
        a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
    """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
        raise ValueError("Inputs to conv must have statically known rank 4. "
                                         "Shape: " + str(static_shape))
    # Add support for left padding.
    if "padding" in kwargs and kwargs["padding"] == "LEFT":
        dilation_rate = (1, 1)
        if "dilation_rate" in kwargs:
            dilation_rate = kwargs["dilation_rate"]
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
        cond_padding = tf.cond(
                tf.equal(tf.shape(inputs)[2], 1), lambda: tf.constant(0),
                lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
        inputs = tf.pad(inputs, padding)
        # Set middle two dimensions to None to prevent convolution from complaining
        inputs.set_shape([static_shape[0], None, None, static_shape[3]])
        kwargs["padding"] = "VALID"

    def conv2d_kernel(kernel_size_arg, name_suffix):
        """Call conv2d but add suffix to name."""
        if "name" in kwargs:
            original_name = kwargs["name"]
            name = kwargs.pop("name") + "_" + name_suffix
        else:
            original_name = None
            name = "conv_" + name_suffix
        original_force2d = None
        if "force2d" in kwargs:
            original_force2d = kwargs.pop("force2d")
        result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
        if original_name is not None:
            kwargs["name"] = original_name  # Restore for other calls.
        if original_force2d is not None:
            kwargs["force2d"] = original_force2d
        return result

    return conv2d_kernel(kernel_size, "single")


def conv2d_advanced(inputs, filters, kernel_size, dilation_rate,
                    weight_noise=0.0, is_training=True, name=None,
                    activation=None):
    a = conv2d(inputs, output_dim=filters, k_h=kernel_size[0], k_w=kernel_size[1],
        weight_noise=weight_noise, is_training=is_training, scope=name)
    if activation is not None:
        a = activation(a)
    return a


def conv2d_dummy(inputs, filters, kernel_size, dilation_rate,
                 weight_noise=0.0, is_training=True, name=None,
                 activation=None, padding=None):
    return tf.contrib.layers.fully_connected(inputs, filters, activation_fn=activation)


def conv(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
    return conv_internal(
            conv2d_advanced if kwargs.get("weight_noise", 0.0) > 0 else conv2d_dummy,
            inputs,
            filters,
            kernel_size,
            dilation_rate=dilation_rate,
            **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
    return tf.squeeze(
            conv(tf.expand_dims(inputs, 2),
                 filters, (kernel_size, 1),
                 dilation_rate=(dilation_rate, 1),
                 **kwargs), 2)

def parameter_attention(x,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        memory_rows,
                        num_heads,
                        dropout_rate,
                        name=None):
    """Attention over parameters.
    We use the same multi-headed attention as in the other layers, but the memory
    keys and values are model parameters.  There are no linear transformation
    on the keys or values.
    We are also a bit more careful about memory usage, since the number of
    memory positions may be very large.
    Args:
        x: a Tensor with shape [batch, length_q, channels]
        total_key_depth: an integer
        total_value_depth: an integer
        output_depth: an integer
        memory_rows: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        dropout_rate: a floating point number
        name: an optional string
    Returns:
        A Tensor.
    """
    with tf.variable_scope(name, default_name="parameter_attention", values=[x]):
        head_size_k = total_key_depth // num_heads
        head_size_v = total_value_depth // num_heads
        var_shape_k = [num_heads, memory_rows, head_size_k]
        var_shape_v = [num_heads, memory_rows, head_size_v]
        k = tf.get_variable(
                "k",
                var_shape_k,
                initializer=tf.random_normal_initializer(0, output_depth**-0.5)) * (
                        num_heads**0.5)
        v = tf.get_variable(
                "v",
                var_shape_v,
                initializer=tf.random_normal_initializer(0, output_depth**-0.5)) * (
                        output_depth**0.5)
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        q = conv1d(x, total_key_depth, 1, name="q_transform")
        if dropout_rate:
            # This is a cheaper form of attention dropout where we use to use
            # the same dropout decisions across batch elemets and query positions,
            # but different decisions across heads and memory positions.
            v = tf.nn.dropout(
                    v, 1.0 - dropout_rate, noise_shape=[num_heads, memory_rows, 1])
        # query is [batch, length, hidden_size]
        # reshape and transpose it to [heads, batch * length, head_size]
        q = tf.reshape(q, [batch_size, length, num_heads, head_size_k])
        q = tf.transpose(q, [2, 0, 1, 3])
        q = tf.reshape(q, [num_heads, batch_size * length, head_size_k])
        weights = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(weights)
        y = tf.matmul(weights, v)
        y = tf.reshape(y, [num_heads, batch_size, length, head_size_v])
        y = tf.transpose(y, [1, 2, 0, 3])
        y = tf.reshape(y, [batch_size, length, total_value_depth])
        y.set_shape([None, None, total_value_depth])
        y = conv1d(y, output_depth, 1, name="output_transform")
        return y


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
    The first of these two dimensions is n.
    Args:
        x: a Tensor with shape [..., m]
        n: an integer.
    Returns:
        a Tensor with shape [..., n, m/n]
    """
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return ret


def split_heads(x, num_heads):
    """Split channels (dimension 3) into multiple heads (becomes dimension 1).
    Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer
    Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def attention_image_summary(attn, image_shapes=None):
    """Compute color image summary.
    Args:
        attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
        image_shapes: optional tuple of integer scalars.
            If the query positions and memory positions represent the
            pixels of flattened images, then pass in their dimensions:
                (query_rows, query_cols, memory_rows, memory_cols).
            If the query positions and memory positions represent the
            pixels x channels of flattened images, then pass in their dimensions:
                (query_rows, query_cols, query_channels,
                 memory_rows, memory_cols, memory_channels).
    """
    num_heads = tf.shape(attn)[1]
    # [batch, query_length, memory_length, num_heads]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.mod(-num_heads, 3)]])
    image = split_last_dimension(image, 3)
    image = tf.reduce_max(image, 4)
    if image_shapes is not None:
        if len(image_shapes) == 4:
            q_rows, q_cols, m_rows, m_cols = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
            image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
            image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
        else:
            assert len(image_shapes) == 6
            q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3])
            image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
            image = tf.reshape(image, [-1, q_rows * m_rows * q_channnels, q_cols * m_cols * m_channels, 3])
    # tf.summary.image("attention", image, max_outputs=1)


def dot_product_attention_weights(q,
                                  k,
                                  bias,
                                  dropout_rate=0.0,
                                  image_shapes=None,
                                  name=None):
    """dot-product attention.
    Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        bias: bias Tensor (see attention_bias())
        dropout_rate: a floating point number
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        name: an optional string
    Returns:
        A Tensor.
    """
    with tf.variable_scope(
            name, default_name="dot_product_attention", values=[q, k]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        return weights


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
    Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        dropout_rate: a floating point number
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        name: an optional string
    Returns:
        A Tensor.
    """
    weights = dot_product_attention_weights(q, k, bias, dropout_rate, image_shapes, name)
    return tf.matmul(weights, v)


def compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                total_value_depth, q_filter_width=1, kv_filter_width=1,
                q_padding="VALID", kv_padding="VALID",
                weight_noise=0.0, is_training=True):
    """Computes query, key and value.
    Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        total_key_depth: an integer
        total_value_depth: and integer
        q_filter_width: An integer specifying how wide you want the query to be.
        kv_filter_width: An integer specifying how wide you want the keys and values
        to be.
        q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    Returns:
        q, k, v : [batch, length, depth] tensors
    """
    if memory_antecedent is None and q_filter_width == kv_filter_width == 1:
        # self attention with single position q, k, and v
        combined = conv1d(query_antecedent,
                          total_key_depth * 2 + total_value_depth,
                          1,
                          name="qkv_transform",
                          weight_noise=weight_noise,
                          is_training=is_training)
        q, k, v = tf.split(combined,
                           [total_key_depth, total_key_depth, total_value_depth],
                           axis=2)
        return q, k, v

    if memory_antecedent is None:
        # self attention
        q = conv1d(query_antecedent,
                   total_key_depth,
                   q_filter_width,
                   padding=q_padding,
                   name="q_transform",
                   weight_noise=weight_noise,
                   is_training=is_training)
        kv_combined = conv1d(query_antecedent,
                             total_key_depth + total_value_depth,
                             kv_filter_width,
                             padding=kv_padding,
                             name="kv_transform",
                             weight_noise=weight_noise,
                             is_training=is_training)
        k, v = tf.split(kv_combined, [total_key_depth, total_value_depth],
                        axis=2)
        return q, k, v

    # encoder-decoder attention
    q = conv1d(query_antecedent, total_key_depth, q_filter_width, padding=q_padding,
               name="q_transform",
               weight_noise=weight_noise,
               is_training=is_training)
    combined = conv1d(memory_antecedent,
                      total_key_depth + total_value_depth,
                      1,
                      padding=kv_padding,
                      name="kv_transform",
                      weight_noise=weight_noise,
                      is_training=is_training)
    k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

    return q, k, v


@recomputable('attention_impl')
def multihead_attention_internal_nobias(q, k, v, dropout_rate, num_heads, total_key_depth, image_shapes):
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    x = dot_product_attention(q, k, v, None, dropout_rate, image_shapes)
    x = combine_heads(x)
    return x


@recomputable('attention_impl')
def multihead_attention_internal_nobias_weights(q, k, dropout_rate, num_heads, total_key_depth, image_shapes):
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    weights = dot_product_attention_weights(q, k, None, dropout_rate, image_shapes)
    return weights


@recomputable('attention_impl_bias')
def multihead_attention_internal(q, k, v, dropout_rate, bias, num_heads, total_key_depth, image_shapes):
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes)
    x = combine_heads(x)
    return x


@recomputable('attention_impl_bias')
def multihead_attention_internal_weights(q, k, dropout_rate, bias, num_heads, total_key_depth, image_shapes):
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5
    weights = dot_product_attention_weights(q, k, bias, dropout_rate, image_shapes)
    return weights



def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        weight_noise=0.0,
                        is_training=True,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.
    Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        dropout_rate: a floating point number
        image_shapes: optional tuple of integer scalars.
            see comments for attention_image_summary()
        attention_type: a string, either "dot_product" or "local_mask_right" or
                                        "local_unmasked"
        block_length: an integer - relevant for "local_mask_right"
        block_width: an integer - relevant for "local_unmasked"
        q_filter_width: An integer specifying how wide you want the query to be.
        kv_filter_width: An integer specifying how wide you want the keys and values
        to be.
        q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        name: an optional string
    Returns:
        A Tensor.
    Raises:
        ValueError: if the key depth or value depth are not divisible by the
            number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))
    with tf.variable_scope(
            name,
            default_name="multihead_attention",
            values=[query_antecedent, memory_antecedent]):
        q, k, v = compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                              total_value_depth, q_filter_width, kv_filter_width,
                              q_padding, kv_padding,
                              weight_noise=weight_noise, is_training=is_training)
        if bias is None:
            x = multihead_attention_internal_nobias(q, k, v, dropout_rate,
                num_heads=num_heads, total_key_depth=total_key_depth, image_shapes=image_shapes)
        else:
            x = multihead_attention_internal(q, k, v, dropout_rate, bias,
                num_heads=num_heads, total_key_depth=total_key_depth, image_shapes=image_shapes)
        x = conv1d(x, output_depth, 1, name="output_transform")
        return x


def build_transformer(inputs, hidden_sizes, is_training, keep_prob,
                      n_heads, transformer_filter_size,
                      scope="TransformerEncoder",
                      weight_noise=0.0,
                      time_major=False):
    """A stack of transformer layers.
      Args:
        inputs: a Tensor
        hparams: hyperparameters for model
      Returns:
        y: Tensor
      """
    cond_dropout = tf.cond(is_training,
                           lambda: tf.constant(1.0 - keep_prob),
                           lambda: tf.constant(0.0))
    hparams = TransformerConfig(
        norm_type="layer",
        hidden_size=hidden_sizes[0],
        dropout=0.0,
        layer_preprocess_sequence="none",
        layer_postprocess_sequence="dan",
        layer_prepostprocess_dropout=cond_dropout,
        learning_rate_warmup_steps=4000,
        initializer_gain=1.0,
        num_hidden_layers=len(hidden_sizes),
        initializer="uniform_unit_scaling",
        # Add new ones like this.
        filter_size=transformer_filter_size,
        # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
        num_heads=n_heads,
        attention_key_channels=0,
        attention_value_channels=0,
        ffn_layer="conv_hidden_relu",
        norm_epsilon=1e-6,
        parameter_attention_key_channels=0,
        parameter_attention_value_channels=0,
        attention_dropout=cond_dropout,
        relu_dropout=cond_dropout
    )
    if time_major:
        inputs = tf.transpose(inputs, [1, 0, 2])
    if inputs.get_shape()[-1].value != hparams.hidden_size:
        inputs = tf.contrib.layers.fully_connected(inputs,
                                                   num_outputs=hparams.hidden_size,
                                                   scope="Pre-Net",
                                                   activation_fn=None)
    inputs = add_timing_signal_1d(inputs)
    encoder_self_attention_bias = None
    x = inputs
    with tf.variable_scope(scope):
        for layer in range(hparams.num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = multihead_attention(
                        layer_preprocess(
                            x, hparams), None, encoder_self_attention_bias,
                        hparams.attention_key_channels or hparams.hidden_size,
                        hparams.attention_value_channels or hparams.hidden_size,
                        hparams.hidden_size, hparams.num_heads, hparams.attention_dropout,
                        weight_noise=weight_noise, is_training=is_training)
                    x = layer_postprocess(x, y, hparams)
                if layer + 1 < hparams.num_hidden_layers:
                    with tf.variable_scope("ffn"):
                        y = transformer_ffn_layer(
                            layer_preprocess(x, hparams), hparams)
                        x = layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    out = layer_preprocess(x, hparams)
    if time_major:
        out = tf.transpose(out, [1, 0, 2])
    return out
