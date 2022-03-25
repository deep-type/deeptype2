import tensorflow as tf
from .viterbi import viterbi_decode


# outputs:
DECODED = "DECODED"
DECODED_SCORES = "DECODED_SCORES"
UNARY_SCORES = "UNARY_SCORES"

# per objective metrics:
TOKEN_CORRECT = "TOKEN_CORRECT"
TOKEN_CORRECT_TOTAL = "TOKEN_CORRECT_TOTAL"
SENTENCE_CORRECT = "SENTENCE_CORRECT"
SENTENCE_CORRECT_TOTAL = "SENTENCE_CORRECT_TOTAL"

# aggregate metrics over all objectives
TOKEN_CORRECT_ALL = "TOKEN_CORRECT_ALL"
TOKEN_CORRECT_ALL_TOTAL = "TOKEN_CORRECT_ALL_TOTAL"
SENTENCE_CORRECT_ALL = "SENTENCE_CORRECT_ALL"
SENTENCE_CORRECT_ALL_TOTAL = "SENTENCE_CORRECT_ALL_TOTAL"
CONFUSION_MATRIX = "CONFUSION_MATRIX"
TRUE_POSITIVES = "TRUE_POSITIVES"
FALSE_POSITIVES = "FALSE_POSITIVES"
FALSE_NEGATIVES = "FALSE_NEGATIVES"


def masked_confusion_matrix(predicted, labels, mask, nclasses):
    one_d_length = tf.shape(predicted)[0] * tf.shape(predicted)[1]
    indices = tf.squeeze(tf.where(tf.reshape(mask, [one_d_length])), [1])
    return tf.contrib.metrics.confusion_matrix(
        tf.gather(tf.reshape(labels, [one_d_length]), indices),
        tf.gather(tf.reshape(predicted, [one_d_length]), indices),
        num_classes=nclasses
    )


def crf_metrics(unary_scores, labels, transition_params, sequence_lengths,
                mask):
    """
    Computes CRF output metrics.
    Receives:
        unary_scores : batch-major order
        labels : batch-major order
        transition_params : nclasses x nclasses matrix.
        sequence_lengths : length of each time-sequence
        mask : batch-major example mask

    Returns:
        token_correct,
        token_correct_total,
        sentence_correct,
        sentence_correct_total
    """
    classes = unary_scores.get_shape()[-1].value
    decoded, scores = viterbi_decode(unary_scores,
                                     transition_params,
                                     sequence_lengths)

    tf.add_to_collection(UNARY_SCORES, unary_scores)
    tf.add_to_collection(DECODED, decoded)
    tf.add_to_collection(DECODED_SCORES, scores)

    equals_label = tf.equal(labels, decoded)
    token_correct = tf.reduce_sum(
        tf.cast(
            tf.logical_and(equals_label, mask),
            tf.int32
        )
    )
    token_correct_total = tf.reduce_sum(tf.cast(mask, tf.int32))
    tf.add_to_collection(TOKEN_CORRECT, token_correct)
    tf.add_to_collection(TOKEN_CORRECT_TOTAL, token_correct_total)
    sentence_correct, _ = compute_sentence_correct(equals_label, mask)
    sentence_correct_total = tf.reduce_sum(tf.cast(mask[:, 0], tf.int32))

    tf.add_to_collection(SENTENCE_CORRECT, sentence_correct)
    tf.add_to_collection(SENTENCE_CORRECT_TOTAL, sentence_correct_total)

    build_true_false_positives(decoded, mask, labels,
                               classes, equals_label)

    tf.add_to_collection(
        CONFUSION_MATRIX,
        masked_confusion_matrix(
            decoded, labels, mask=mask,
            nclasses=classes
        )
    )

    return (token_correct, token_correct_total,
            sentence_correct, sentence_correct_total)


def build_true_false_positives(decoded, mask_batch_major, labels_batch_major,
                               classes, equals_label):
    # now for each class compute tp, fp, fn
    # [nclasses x batch x time]
    masked_per_class = tf.logical_and(
        tf.equal(labels_batch_major[None, :, :], tf.range(classes)[:, None, None]),
        mask_batch_major)

    # correct, and on label
    correct = tf.reduce_sum(tf.cast(tf.logical_and(masked_per_class, equals_label[None, :, :]), tf.int32),
                            axis=[1, 2])
    # predicted a particular class
    guessed = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(decoded[None, :, :],
                                                            tf.range(classes)[:, None, None]),
                                                   mask_batch_major), tf.int32),
                            axis=[1, 2])
    total = tf.reduce_sum(tf.cast(masked_per_class, tf.int32), axis=[1, 2])
    tp, fp, fn = correct, guessed - correct, total - correct

    tf.add_to_collection(TRUE_POSITIVES, tp)
    tf.add_to_collection(FALSE_POSITIVES, fp)
    tf.add_to_collection(FALSE_NEGATIVES, fn)


def compute_sentence_correct(correct, sequence_mask):
    any_label = tf.reduce_max(tf.cast(sequence_mask, tf.int32), 1)
    sentence_correct_total = tf.reduce_sum(any_label)
    # is 1 when all is correct, 0 otherwise
    sentence_correct = tf.reduce_sum(tf.reduce_prod(
        tf.cast(
            tf.logical_or(correct, tf.logical_not(sequence_mask)),
            tf.int32
        ),
        1
    ) * any_label)
    return sentence_correct, sentence_correct_total


def softmax_metrics(unary_scores, labels, mask, predictions_time_major=None, predictions_batch_major=None):
    """
    Compute softmax output stats for correct/accuracy per-token/per-sentence.
    Receive
        unary_scores : time-major
        labels : time-major
        mask : time-major
    Returns:
        token_correct,
        token_correct_total,
        sentence_correct,
        sentence_correct_total
    """
    classes = unary_scores.get_shape()[-1].value
    unary_scores_batch_major = tf.transpose(unary_scores, [1, 0, 2])
    labels_batch_major = tf.transpose(labels, [1, 0])
    mask_batch_major = tf.transpose(mask, [1, 0])

    assert not (predictions_time_major is not None and predictions_batch_major is not None), \
        "predictions_batch_major or predictions_batch_major must be None (or both)"

    if predictions_batch_major is None and predictions_time_major is not None:
        predictions_batch_major = tf.transpose(predictions_time_major)

    decoded = tf.cond(tf.greater(tf.shape(unary_scores_batch_major)[2], 0),
                      lambda: tf.cast(tf.argmax(unary_scores_batch_major, 2), labels.dtype) if predictions_batch_major is None else predictions_batch_major,
                      lambda: labels_batch_major)
    unary_probs_batch_major = tf.nn.softmax(unary_scores_batch_major)
    scores = tf.reduce_max(unary_probs_batch_major, 2)

    tf.add_to_collection(UNARY_SCORES, unary_probs_batch_major)
    tf.add_to_collection(DECODED, decoded)
    tf.add_to_collection(DECODED_SCORES, scores)

    equals_label = tf.equal(decoded, labels_batch_major)

    token_correct = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                equals_label,
                mask_batch_major
            ),
            tf.int32
        )
    )
    token_correct_total = tf.reduce_sum(tf.cast(mask, tf.int32))
    tf.add_to_collection(TOKEN_CORRECT, token_correct)
    tf.add_to_collection(TOKEN_CORRECT_TOTAL, token_correct_total)

    sentence_correct, sentence_correct_total = compute_sentence_correct(
        equals_label, mask_batch_major
    )
    tf.add_to_collection(SENTENCE_CORRECT, sentence_correct)
    tf.add_to_collection(SENTENCE_CORRECT_TOTAL, sentence_correct_total)

    # does the metrix use a fixed class dimension
    if classes is not None:
        build_true_false_positives(decoded, mask_batch_major, labels_batch_major,
                                   classes, equals_label)

        tf.add_to_collection(
            CONFUSION_MATRIX,
            masked_confusion_matrix(
                decoded, labels_batch_major, mask=mask_batch_major,
                nclasses=classes
            )
        )
    return (token_correct, token_correct_total,
            sentence_correct, sentence_correct_total)
