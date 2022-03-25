import tensorflow as tf


def concat_tensor_array(values, name=None):
    return values.stack(name=name)


def batch_gather_3d(values, indices):
    return tf.gather(tf.reshape(values, [-1, tf.shape(values)[2]]),
                     tf.range(0, tf.shape(values)[0]) * tf.shape(values)[1] +
                     indices)


def batch_gather_2d(values, indices):
    return tf.gather(tf.reshape(values, [-1]),
                     tf.range(0, tf.shape(values)[0]) * tf.shape(values)[1] +
                     indices)


def viterbi_decode(score, transition_params, sequence_lengths, back_prop=False,
                   parallel_iterations=1):
    """Decode the highest scoring sequence of tags inside of TensorFlow!!!
    This can be used anytime.
    Args:
        score: A [batch, seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        sequence_lengths: A [batch] int32 vector of the length of each score
            sequence.
    Returns:
        viterbi: A [batch, seq_len] list of integers containing the highest
            scoring tag indices.
        viterbi_score: A vector of float containing the score for the Viterbi
            sequence.
    """
    sequence_lengths = tf.convert_to_tensor(
        sequence_lengths, name="sequence_lengths")
    score = tf.convert_to_tensor(score, name="score")
    transition_params = tf.convert_to_tensor(
        transition_params, name="transition_params")

    if sequence_lengths.dtype != tf.int32:
        sequence_lengths = tf.cast(sequence_lengths, tf.int32)

    def condition(t, *args):
        """Stop when full score sequence has been read in."""
        return tf.less(t, tf.shape(score)[1])

    def body(t, trellis, backpointers, trellis_val):
        """Perform forward viterbi pass."""
        v = tf.expand_dims(trellis_val, 2) + tf.expand_dims(transition_params, 0)
        new_trellis_val = score[:, t, :] + tf.reduce_max(v, axis=1)
        new_trellis = trellis.write(t, new_trellis_val)
        new_backpointers = backpointers.write(
            t, tf.cast(tf.argmax(v, axis=1), tf.int32))
        return t + 1, new_trellis, new_backpointers, new_trellis_val

    trellis_arr = tf.TensorArray(score.dtype, size=0,
                                 dynamic_size=True, clear_after_read=False, infer_shape=False)
    first_trellis_val = score[:, 0, :]
    trellis_arr = trellis_arr.write(0, first_trellis_val)

    backpointers_arr = tf.TensorArray(tf.int32, size=0,
                                      dynamic_size=True, clear_after_read=False, infer_shape=False)
    backpointers_arr = backpointers_arr.write(0, tf.zeros_like(score[:, 0, :], dtype=tf.int32))

    _, trellis_out, backpointers_out, _ = tf.while_loop(
        condition, body,
        (tf.constant(1, name="t", dtype=tf.int32), trellis_arr, backpointers_arr, first_trellis_val),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop)

    trellis_out = concat_tensor_array(trellis_out)
    backpointers_out = concat_tensor_array(backpointers_out)
    # make batch-major:
    trellis_out = tf.transpose(trellis_out, [1, 0, 2])
    backpointers_out = tf.transpose(backpointers_out, [1, 0, 2])

    def condition(t, *args):
        return tf.less(t, tf.shape(score)[1])

    def body(t, viterbi, last_decision):
        backpointers_timestep = batch_gather_3d(
            backpointers_out, tf.maximum(sequence_lengths - t, 0))
        new_last_decision = batch_gather_2d(
            backpointers_timestep, last_decision)
        new_viterbi = viterbi.write(t, new_last_decision)
        return t + 1, new_viterbi, new_last_decision

    last_timestep = batch_gather_3d(trellis_out, sequence_lengths - 1)
    # get scores for last timestep of each batch element inside
    # trellis:
    scores = tf.reduce_max(last_timestep, axis=1)
    # get choice index for last timestep:
    last_decision = tf.cast(tf.argmax(last_timestep, axis=1), tf.int32)

    # decode backwards using backpointers:
    viterbi = tf.TensorArray(tf.int32, size=0,
                             dynamic_size=True, clear_after_read=False, infer_shape=False)
    viterbi = viterbi.write(0, last_decision)
    _, viterbi_out, _ = tf.while_loop(
        condition, body,
        (tf.constant(1, name="t", dtype=tf.int32), viterbi, last_decision),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop)
    viterbi_out = concat_tensor_array(viterbi_out)
    # make batch-major:
    viterbi_out = tf.transpose(viterbi_out, [1, 0])
    viterbi_out_fwd = tf.reverse_sequence(
        viterbi_out, sequence_lengths, seq_dim=1)
    return viterbi_out_fwd, scores
