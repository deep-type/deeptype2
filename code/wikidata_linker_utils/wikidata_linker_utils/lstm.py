import math
import tensorflow as tf
import numpy as np
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM

from .tf_regularization import maybe_dropout, add_weight_noise
from .tf_operations import concat, reverse, split


RNNCell = tf.nn.rnn_cell.RNNCell
TFLSTMCell = tf.nn.rnn_cell.LSTMCell
MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple

# cudnn conversion to dynamic RNN:
CUDNN_LAYER_WEIGHT_ORDER = [
    "x", "x", "x", "x", "h", "h", "h", "h"
]
CUDNN_LAYER_BIAS_ORDER = [
    "bx", "bx", "bx", "bx", "bh", "bh", "bh", "bh"
]
CUDNN_TRANSPOSED = True
CUDNN_MAPPING = {"i": 0, "f": 1, "j": 2, "o": 3}


def lstm_activation(inputs, input_h, input_c, W, b, activation):
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    cell_inputs = concat([inputs, input_h], axis=1)

    lstm_matrix = tf.nn.xw_plus_b(cell_inputs, W, b)
    preactiv = split(lstm_matrix, axis=1, num_splits=4)
    # from CUDNN docs:
    # Values 0 and 4 reference the input gate.
    # Values 1 and 5 reference the forget gate.
    # Values 2 and 6 reference the new memory gate.
    # Values 3 and 7 reference the output gate
    i, f, j, o = (
        preactiv[CUDNN_MAPPING["i"]],
        preactiv[CUDNN_MAPPING["f"]],
        preactiv[CUDNN_MAPPING["j"]],
        preactiv[CUDNN_MAPPING["o"]]
    )

    c = (tf.nn.sigmoid(f) * input_c +
         tf.nn.sigmoid(i) * activation(j))

    m = tf.nn.sigmoid(o) * activation(c)
    return (c, m)


class ParametrizedLSTMCell(RNNCell):
    def __init__(self, weights, biases, hidden_size):
        self._weights = weights
        self._biases = biases
        self.hidden_size = hidden_size

    @property
    def state_size(self):
        return (self.hidden_size, self.hidden_size)

    @property
    def output_size(self):
        return self.hidden_size

    def __call__(self, inputs, state, scope=None):
        input_h, input_c = state
        c, m = lstm_activation(inputs,
                               input_h=input_h,
                               input_c=input_c,
                               b=self._biases,
                               W=self._weights,
                               activation=tf.nn.tanh)
        return m, (m, c)


class LSTMCell(TFLSTMCell):
    def __init__(self,
                 num_units,
                 keep_prob=1.0,
                 is_training=False):
        self._is_training = is_training
        self._keep_prob = keep_prob
        TFLSTMCell.__init__(
            self,
            num_units=num_units,
            state_is_tuple=True
        )

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with tf.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):  # "LSTMCell"
            concat_w = _get_concat_variable(
                "W", [input_size.value + self._num_units, 4 * self._num_units],
                dtype, 1)

            b = tf.get_variable(
                "B", shape=[4 * self._num_units],
                initializer=tf.zeros_initializer(), dtype=dtype)

        c, m = lstm_activation(inputs,
                               input_c=c_prev,
                               input_h=m_prev,
                               W=concat_w,
                               b=b,
                               activation=self._activation)
        return m, LSTMStateTuple(c, m)


def _get_sharded_variable(name, shape, dtype, num_shards):
    """Get a list of sharded variables with the given dtype."""
    if num_shards > shape[0]:
        raise ValueError("Too many shards: shape=%s, num_shards=%d" %
                         (shape, num_shards))
    unit_shard_size = int(math.floor(shape[0] / num_shards))
    remaining_rows = shape[0] - unit_shard_size * num_shards

    shards = []
    for i in range(num_shards):
        current_size = unit_shard_size
        if i < remaining_rows:
            current_size += 1
        shards.append(
            tf.get_variable(
                name + "_%d" % i,
                [current_size] + shape[1:],
                dtype=dtype
            )
        )
    return shards


def _get_concat_variable(name, shape, dtype, num_shards):
    """Get a sharded variable concatenated into one tensor."""
    sharded_variable = _get_sharded_variable(name, shape, dtype, num_shards)
    if len(sharded_variable) == 1:
        return sharded_variable[0]

    concat_name = name + "/concat"
    concat_full_name = tf.get_variable_scope().name + "/" + concat_name + ":0"
    for value in tf.get_collection(tf.GraphKeys.CONCATENATED_VARIABLES):
        if value.name == concat_full_name:
            return value

    concat_variable = tf.concat_v2(sharded_variable, 0, name=concat_name)
    tf.add_to_collection(tf.GraphKeys.CONCATENATED_VARIABLES, concat_variable)
    return concat_variable


def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights


def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))


def estimate_cudnn_parameter_size(num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            params += cudnn_lstm_parameter_size(
                isize, hidden_size
            )
        isize = hidden_size * num_directions
    return params


def consume_biases_direction(params, old_offset, hidden_size, isize):
    offset = old_offset
    layer_biases_x = []
    layer_biases_h = []

    for piece in CUDNN_LAYER_BIAS_ORDER:
        if piece == "bx":
            layer_biases_x.append(
                params[offset:offset + hidden_size]
            )
            offset += hidden_size
        elif piece == "bh":
            layer_biases_h.append(
                params[offset:offset + hidden_size]
            )
            offset += hidden_size
        else:
            raise ValueError("Unknown cudnn piece %r." % (piece,))
    b = concat(layer_biases_x, axis=0) + concat(layer_biases_h, axis=0)
    return b, offset


def consume_weights_direction(params, old_offset, hidden_size, isize):
    offset = old_offset
    layer_weights_x = []
    layer_weights_h = []
    for piece in CUDNN_LAYER_WEIGHT_ORDER:
        if piece == "x":
            layer_weights_x.append(
                tf.reshape(
                    params[offset:offset + hidden_size * isize],
                    [hidden_size, isize] if CUDNN_TRANSPOSED else [isize, hidden_size]
                )
            )
            offset += hidden_size * isize
        elif piece == "h":
            layer_weights_h.append(
                tf.reshape(
                    params[offset:offset + hidden_size * hidden_size],
                    [hidden_size, hidden_size]
                )
            )
            offset += hidden_size * hidden_size
        else:
            raise ValueError("Unknown cudnn piece %r." % (piece,))
    if CUDNN_TRANSPOSED:
        W_T = concat([concat(layer_weights_x, axis=0), concat(layer_weights_h, axis=0)], axis=1)
        W = tf.transpose(W_T)
    else:
        W = concat([concat(layer_weights_x, axis=1), concat(layer_weights_h, axis=1)], axis=0)
    return W, offset


def decompose_layer_params(params, num_layers,
                           hidden_size, cell_input_size,
                           input_mode, direction, create_fn):
    """
    This operation converts the opaque cudnn params into a set of
    usable weight matrices.
    Args:
        params : Tensor, opaque cudnn params tensor
        num_layers : int, number of stacked LSTMs.
        hidden_size : int, number of neurons in each LSTM.
        cell_input_size : int, input size for the LSTMs.
        input_mode: whether a pre-projection was used or not. Currently only
            'linear_input' is supported (e.g. CuDNN does its own projection
            internally)
        direction : str, 'unidirectional' or 'bidirectional'.
        create_fn: callback for weight creation. Receives parameter slice (op),
                   layer (int), direction (0 = fwd, 1 = bwd),
                   parameter_index (0 = W, 1 = b).
    Returns:
        weights : list of lists of Tensors in the format:
            first list is indexed layers,
            inner list is indexed by direction (fwd, bwd),
            tensors in the inner list are (Weights, biases)
    """
    if input_mode != "linear_input":
        raise ValueError("Only input_mode == linear_input supported for now.")
    num_directions = direction_to_num_directions(direction)
    offset = 0
    all_weights = [[[] for j in range(num_directions)]
                   for i in range(num_layers)]
    isize = cell_input_size
    with tf.variable_scope("DecomposeCudnnParams"):
        for layer in range(num_layers):
            with tf.variable_scope("Layer{}".format(layer)):
                for direction in range(num_directions):
                    with tf.variable_scope("fwd" if direction == 0 else "bwd"):
                        with tf.variable_scope("weights"):
                            W, offset = consume_weights_direction(
                                params,
                                old_offset=offset,
                                hidden_size=hidden_size,
                                isize=isize)
                            all_weights[layer][direction].append(
                                create_fn(W, layer, direction, 0))
            isize = hidden_size * num_directions
        isize = cell_input_size
        for layer in range(num_layers):
            with tf.variable_scope("Layer{}".format(layer)):
                for direction in range(num_directions):
                    with tf.variable_scope("fwd" if direction == 0 else "bwd"):
                        with tf.variable_scope("biases"):
                            b, offset = consume_biases_direction(
                                params,
                                old_offset=offset,
                                hidden_size=hidden_size,
                                isize=isize)
                            all_weights[layer][direction].append(
                                create_fn(b, layer, direction, 1))
            isize = hidden_size * num_directions
    return all_weights


def create_decomposed_variable(param, lidx, didx, pidx):
    with tf.device("cpu"):
        return tf.get_variable("w" if pidx == 0 else "b",
                               shape=param.get_shape().as_list(),
                               dtype=param.dtype,
                               trainable=False,
                               collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                            "excluded_variables"])


def cpu_cudnn_params(params, num_layers, hidden_size, cell_input_size, input_mode,
                     direction):
    """
    This operation converts the opaque cudnn params into a set of
    usable weight matrices, and caches the conversion.
    Args:
        params : Tensor, opaque cudnn params tensor
        num_layers : int, number of stacked LSTMs.
        hidden_size : int, number of neurons in each LSTM.
        cell_input_size : int, input size for the LSTMs.
        input_mode: whether a pre-projection was used or not. Currently only
            'linear_input' is supported (e.g. CuDNN does its own projection
            internally)
        direction : str, 'unidirectional' or 'bidirectional'.
        skip_creation : bool, whether to build variables.
    Returns:
        weights : list of lists of Tensors in the format:
            first list is indexed layers,
            inner list is indexed by direction (fwd, bwd),
            tensors in the inner list are (Weights, biases)
    """
    # create a boolean status variable that checks whether the
    # weights have been converted to cpu format:
    with tf.device("cpu"):
        cpu_conversion_status = tf.get_variable(
            name="CudnnConversionStatus", dtype=tf.float32,
            initializer=tf.zeros_initializer(), shape=[],
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    # create a fresh copy of the weights (not trainable)
    reshaped = decompose_layer_params(
        params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        cell_input_size=cell_input_size,
        input_mode=input_mode,
        direction=direction,
        create_fn=create_decomposed_variable)

    def cpu_convert():
        all_assigns = decompose_layer_params(
            params,
            num_layers=num_layers,
            hidden_size=hidden_size,
            cell_input_size=cell_input_size,
            input_mode=input_mode,
            direction=direction,
            create_fn=lambda p, lidx, didx, pidx: tf.assign(reshaped[lidx][didx][pidx], p))
        all_assigns = [assign for layer_assign in all_assigns
                       for dir_assign in layer_assign
                       for assign in dir_assign]
        all_assigns.append(tf.assign(cpu_conversion_status, tf.constant(1.0, dtype=tf.float32)))
        all_assigns.append(tf.Print(
            cpu_conversion_status, [0],
            message="Converted cudnn weights to CPU format. "))
        with tf.control_dependencies(all_assigns):
            ret = tf.identity(cpu_conversion_status)
            return ret
    # cache the reshaping/concatenating
    ensure_conversion = tf.cond(tf.greater(cpu_conversion_status, 0),
                                lambda: cpu_conversion_status,
                                cpu_convert)
    # if weights are already reshaped, go ahead:
    with tf.control_dependencies([ensure_conversion]):
        # wrap with identity to ensure there is a dependency between assignment
        # and using the weights:
        all_params = [[[tf.identity(p) for p in dir_param]
                       for dir_param in layer_param]
                      for layer_param in reshaped]
        return all_params


class CpuCudnnLSTM(object):
    def __init__(self, num_layers, hidden_size,
                 cell_input_size, input_mode, direction):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cell_input_size = cell_input_size
        self.input_mode = input_mode
        self.direction = direction

    def __call__(self,
                 inputs,
                 input_h,
                 input_c,
                 params,
                 is_training=True):
        layer_params = cpu_cudnn_params(params,
                                        num_layers=self.num_layers,
                                        hidden_size=self.hidden_size,
                                        cell_input_size=self.cell_input_size,
                                        input_mode=self.input_mode,
                                        direction=self.direction)
        REVERSED = 1
        layer_inputs = inputs
        cell_idx = 0
        for layer_param in layer_params:
            hidden_fwd_bwd = []
            final_output_c = []
            final_output_h = []
            for direction, (W, b) in enumerate(layer_param):
                if direction == REVERSED:
                    layer_inputs = reverse(layer_inputs, axis=0)
                hiddens, (output_h, output_c) = tf.nn.dynamic_rnn(
                    cell=ParametrizedLSTMCell(W, b, self.hidden_size),
                    inputs=layer_inputs,
                    dtype=inputs.dtype,
                    time_major=True,
                    initial_state=(input_h[cell_idx], input_c[cell_idx]))
                if direction == REVERSED:
                    hiddens = reverse(hiddens, axis=0)
                hidden_fwd_bwd.append(hiddens)
                final_output_c.append(tf.expand_dims(output_c, 0))
                final_output_h.append(tf.expand_dims(output_h, 0))
                cell_idx += 1
            if len(hidden_fwd_bwd) > 1:
                layer_inputs = concat(hidden_fwd_bwd, axis=2)
                final_output_c = concat(final_output_c, axis=0)
                final_output_h = concat(final_output_h, axis=0)
            else:
                layer_inputs = hidden_fwd_bwd[0]
                final_output_c = final_output_c[0]
                final_output_h = final_output_h[0]
        return layer_inputs, final_output_h, final_output_c


def bidirectional_dynamic_rnn(cell, inputs, dtype, time_major=True, swap_memory=False):
    with tf.variable_scope("forward"):
        out_fwd, final_fwd = tf.nn.dynamic_rnn(cell,
                                               inputs,
                                               time_major=time_major,
                                               dtype=dtype,
                                               swap_memory=swap_memory)

    if time_major:
        reverse_axis = 0
    else:
        reverse_axis = 1

    with tf.variable_scope("backward"):
        out_bwd, final_bwd = tf.nn.dynamic_rnn(cell,
                                               reverse(inputs, axis=reverse_axis),
                                               time_major=time_major,
                                               dtype=dtype,
                                               swap_memory=swap_memory)

    out_bwd = reverse(out_bwd, axis=reverse_axis)
    return concat([out_fwd, out_bwd], axis=2), (final_fwd, final_bwd)


def build_recurrent(inputs, cudnn, faux_cudnn, hidden_sizes, is_training,
                    keep_prob, weight_noise):
    dtype = tf.float32
    if cudnn:
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be a list of length > 1.")
        hidden_size = hidden_sizes[0]
        if any(hidden_size != hsize for hsize in hidden_sizes):
            raise ValueError("cudnn RNN requires all hidden units "
                             "to be the same size (got %r)" % (hidden_sizes,))
        num_layers = len(hidden_sizes)
        cell_input_size = inputs.get_shape()[-1].value

        est_size = estimate_cudnn_parameter_size(num_layers=num_layers,
                                                 hidden_size=hidden_size,
                                                 input_size=cell_input_size,
                                                 input_mode="linear_input",
                                                 direction="bidirectional")
        # autoswitch to GPUs based on availability of alternatives:
        cudnn_params = tf.get_variable("RNNParams",
                                       shape=[est_size],
                                       dtype=tf.float32,
                                       initializer=tf.contrib.layers.variance_scaling_initializer())
        if weight_noise > 0:
            cudnn_params = add_weight_noise(cudnn_params,
                                            stddev=weight_noise,
                                            is_training=is_training)
        if faux_cudnn:
            cudnn_cell = CpuCudnnLSTM(num_layers,
                                      hidden_size,
                                      cell_input_size,
                                      input_mode="linear_input",
                                      direction="bidirectional")
        else:
            cpu_cudnn_params(cudnn_params,
                             num_layers=num_layers,
                             hidden_size=hidden_size,
                             cell_input_size=cell_input_size,
                             input_mode="linear_input",
                             direction="bidirectional")
            cudnn_cell = CudnnLSTM(num_layers=num_layers,
                                   num_units=hidden_size,
                                   input_size=cell_input_size,
                                   input_mode="linear_input",
                                   direction="bidirectional")
        init_state = tf.fill(
            (2 * num_layers, tf.shape(inputs)[1], hidden_size),
            tf.constant(np.float32(0.0)))
        hiddens, output_h, output_c = cudnn_cell(
            inputs,
            input_h=init_state,
            input_c=init_state,
            params=cudnn_params,
            is_training=True)
        hiddens = maybe_dropout(hiddens, keep_prob, is_training)
    else:
        cell = MultiRNNCell(
            [LSTMCell(hsize, is_training=is_training, keep_prob=keep_prob)
             for hsize in hidden_sizes])
        hiddens, _ = bidirectional_dynamic_rnn(cell,
                                               inputs,
                                               time_major=True,
                                               dtype=dtype,
                                               swap_memory=True)
    return hiddens
