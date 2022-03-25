import tensorflow as tf
import numpy as np

from .lstm import LSTMCell
from .make_callable import make_callable
from . import wikidata_properties as wprop
from .tf_operations import extract_shape
from .scoped_timer import scoped_timer
from .type_interaction_constants import CONSTRAINTS_REQUIRED_PADDING, CONSTRAINTS_CREATED_PADDING, METADATA_SCORE_DIM, METADATA_INSIDE_LIST, METADATA_LIST_DIM
from .successor_mask import last_non_negative


def _beam_aware_stacked_indices(indices):
    beam_width = None
    batch_size = None
    beam_like_index = None
    for index in indices:
        batch_size = tf.shape(index)[0]
        if len(index.get_shape()) > 1:
            beam_width = tf.shape(index)[1]
            beam_like_index = index
            break
    if beam_width is not None:
        out = [tf.reshape(index[:, None] + tf.zeros_like(beam_like_index), [-1]) if len(index.get_shape()) == 1 else tf.reshape(index, [-1])
               for index in indices]
        stacked = tf.stack(out, axis=-1)
    else:
        stacked = tf.stack(indices, axis=-1)
    return stacked, batch_size, beam_width


def beam_aware_gather_nd(x, indices):
    stacked, batch_size, beam_width = _beam_aware_stacked_indices(indices)
    x_shape = extract_shape(x)
    gathered = tf.gather_nd(x, stacked)
    if beam_width is not None:
        return tf.reshape(gathered, [batch_size, beam_width] + x_shape[2:])
    return gathered


def last_prediction_bit_initial_state(batch_size, num_entities, input_size, candidates_constraints, **kwargs):
    res = -tf.ones((batch_size, tf.shape(candidates_constraints)[3]), dtype=tf.int32)
    shape = tf.TensorShape([None, None])
    return (res, shape)


def past_prediction_bit_initial_state(batch_size, num_entities, input_size, legacy_consistency_check=False, **kwargs):
    res = tf.zeros((batch_size, 0), dtype=tf.int32)
    shape = tf.TensorShape([None, None])
    if legacy_consistency_check:
        res = (res, tf.zeros((batch_size, num_entities), dtype=tf.bool))
        shape = (shape,  tf.TensorShape([None, None]))
    return (res, shape)


def lstm_initial_state(batch_size, num_entities, kernel_size, **kwargs):
    return ((tf.zeros((batch_size, kernel_size), dtype=tf.float32), tf.zeros((batch_size, kernel_size), dtype=tf.float32)),
            (tf.TensorShape([None, kernel_size]), tf.TensorShape([None, kernel_size])))


def _gate_update(num_predictions, valid_tstep, x, prev_x):
    return tf.where(tf.logical_and(tf.equal(num_predictions, 1), valid_tstep),
                    # if no predictions have occured, just grab the new embeddings,
                    x,
                    # else max-pool over previous state info.
                    tf.where(valid_tstep,
                             x,
                             prev_x))


def _make_2d(x):
    if isinstance(x, tuple):
        return tuple([_make_2d(v) for v in x])
    if len(x.get_shape()) > 2:
        return tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1], x.get_shape()[-1].value])
    return x


def lstm_update(prev_state, valid_tstep,
                predicted_tstep_ids, candidate_embeddings, candidate_embeddings_featurizations,
                num_predictions, beam_search, is_training, ignored_featurizations=None, **kwargs):
    # index into Batchsize x nexamples per step
    # using class ids
    batch_size = tf.shape(prev_state[0])[0]
    batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
    cell = LSTMCell(prev_state[0].get_shape()[-1].value, is_training=is_training)
    if ignored_featurizations is not None and len(ignored_featurizations) > 0:
        candidate_embeddings = candidate_embeddings_with_ignores(
            candidate_embeddings_featurizations, ignored_featurizations, candidate_embeddings=candidate_embeddings)

    _, (m, c) = cell(state=_make_2d(prev_state),
                     inputs=_make_2d(beam_aware_gather_nd(candidate_embeddings, indices=[batch_indices, predicted_tstep_ids])))
    
    if beam_search:
        beam_width = tf.shape(prev_state[0])[1]
        m = tf.reshape(m, [batch_size, beam_width, m.get_shape()[-1].value])
        c = tf.reshape(c, [batch_size, beam_width, m.get_shape()[-1].value])
    
    return (_gate_update(num_predictions=num_predictions, valid_tstep=valid_tstep, x=m, prev_x=prev_state[0]),
            _gate_update(num_predictions=num_predictions, valid_tstep=valid_tstep, x=c, prev_x=prev_state[1]))


def lstm_to_features(state, candidate_embeddings, beam_search, normalize=False, **kwargs):
    m, c = state
    # Compare candidate embeddings to their state representation
    # (Batch x n-scenarios x h) * (Batch x h) *  -> Batch x n-scenarios x 1
    candidates_proj = tf.contrib.layers.fully_connected(candidate_embeddings, m.get_shape()[-1].value, scope="CandidateFC", activation_fn=None)
    if normalize:
        candidates_proj, _ = tf.linalg.normalize(candidates_proj, axis=-1)
    # add a candidates dimension to the state:
    if beam_search:
        # add a beam dimension to the candidates
        candidates_proj = tf.expand_dims(candidates_proj, 1)    
    return tf.matmul(candidates_proj, tf.expand_dims(m, -1), name="Coherence")


def max_entities_initial_state(batch_size, num_entities, kernel_size, **kwargs):
    return (tf.zeros((batch_size, kernel_size), dtype=tf.float32), tf.TensorShape([None, kernel_size]))


def candidate_embeddings_with_ignores(candidate_embeddings_featurizations, ignored_featurizations, candidate_embeddings):
    segments = []
    so_far = 0
    for featurization, h in candidate_embeddings_featurizations:
        if featurization["name"] not in ignored_featurizations:
            segments.append((so_far, so_far + h))
        so_far += h
    assert len(segments) > 0, \
        ("At least one featurizations should be used by `max_entities_update` but none were left. "
         "(ignored_featurizations={}).".format(ignored_featurizations))
    slices = []
    start, end = segments[0]
    for i in range(1, len(segments)):
        if segments[i][0] == end:
            # extend
            end = segments[i][1]
        else:
            slices.append(candidate_embeddings[..., start:end])
            # slice and create new
            start, end = segments[i]
    slices.append(candidate_embeddings[..., start:end])
    # bring back slices together
    return tf.concat(slices, axis=-1)


def max_entities_update(prev_state, valid_tstep,
                        predicted_tstep_ids, candidate_embeddings, candidate_embeddings_featurizations,
                        num_predictions, beam_search, ignored_featurizations=None, **kwargs):
    # index into Batchsize x nexamples per step
    # using class ids
    batch_size = tf.shape(prev_state)[0]
    batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
    if ignored_featurizations is not None and len(ignored_featurizations) > 0:
        candidate_embeddings = candidate_embeddings_with_ignores(
            candidate_embeddings_featurizations, ignored_featurizations, candidate_embeddings=candidate_embeddings)

    new_embeddings = tf.contrib.layers.fully_connected(beam_aware_gather_nd(candidate_embeddings, indices=[batch_indices, predicted_tstep_ids]),
                                                       prev_state.get_shape()[-1].value,
                                                       scope="KeyFC")
    return tf.where(tf.logical_and(tf.equal(num_predictions, 1), valid_tstep),
                    # if no predictions have occured, just grab the new embeddings,
                    new_embeddings,
                    # else max-pool over previous state info.
                    tf.where(valid_tstep,
                             tf.maximum(prev_state, new_embeddings),
                             prev_state))


def max_entities_to_features(state, candidate_embeddings, beam_search, normalize=False, **kwargs):
    # Compare candidate embeddings to their state representation
    # (Batch x n-scenarios x h) * (Batch x h) *  -> Batch x n-scenarios x 1
    candidates_proj = tf.contrib.layers.fully_connected(candidate_embeddings, state.get_shape()[-1].value, scope="CandidateFC", activation_fn=None)
    if normalize:
        candidates_proj, _ = tf.linalg.normalize(candidates_proj, axis=-1)
    # add a candidates dimension to the state:
    if beam_search:
        # add a beam dimension to the candidates
        candidates_proj = tf.expand_dims(candidates_proj, 1)    
    return tf.matmul(candidates_proj, tf.expand_dims(state, -1), name="Coherence")


def max_entities_post_agg_initial_state(batch_size, num_entities, kernel_size, **kwargs):
    return (tf.zeros((batch_size, 1, kernel_size), dtype=tf.float32), tf.TensorShape([None, None, kernel_size]))


def max_entities_post_agg_update(prev_state, valid_tstep,
                                 predicted_tstep_ids, candidate_embeddings, candidate_embeddings_featurizations,
                                 num_predictions, beam_search, ignored_featurizations=None, **kwargs):
    # index into Batchsize x nexamples per step
    # using class ids
    batch_size = tf.shape(prev_state)[0]
    batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
    if ignored_featurizations is not None and len(ignored_featurizations) > 0:
        candidate_embeddings = candidate_embeddings_with_ignores(
            candidate_embeddings_featurizations, ignored_featurizations, candidate_embeddings=candidate_embeddings)

    new_embeddings = tf.contrib.layers.fully_connected(beam_aware_gather_nd(candidate_embeddings, indices=[batch_indices, predicted_tstep_ids]),
                                                       prev_state.get_shape()[-1].value,
                                                       scope="KeyFC")
    return tf.concat([prev_state,
                      tf.expand_dims(tf.where(valid_tstep, new_embeddings, tf.zeros_like(new_embeddings)), -2)], axis=-2)


def max_entities_post_agg_to_features(state, candidate_embeddings, beam_search, normalize=False, **kwargs):
    # Compare candidate embeddings to their state representation
    # (Batch x n-scenarios x h) * (Batch x h) *  -> Batch x n-scenarios x 1
    candidates_proj = tf.contrib.layers.fully_connected(candidate_embeddings, state.get_shape()[-1].value, scope="CandidateFC", activation_fn=None)
    if normalize:
        candidates_proj, _ = tf.linalg.normalize(candidates_proj, axis=-1)
    # add a candidates dimension to the state:
    if beam_search:
        # add a beam dimension to the candidates
        candidates_proj = tf.expand_dims(candidates_proj, 1)

    # (Batch x n-scenarios x Kernel Size) * (Batch x Beam * Past x Kernel Size x 1)
    if beam_search:
        res = tf.reduce_max(tf.reshape(tf.matmul(candidates_proj,
                                                  tf.reshape(state, [tf.shape(state)[0], tf.shape(state)[1] * tf.shape(state)[2], state.get_shape()[-1].value, 1]),
                                                  name="Coherence"),
                                        [tf.shape(state)[0], tf.shape(state)[1], tf.shape(state)[2], tf.shape(candidates_proj)[1], 1]), axis=-3)
    else:
        # res = tf.reduce_max(tf.reshape(tf.matmul(candidates_proj,
        #                                           tf.reshape(state, [tf.shape(state)[0], tf.shape(state)[1], state.get_shape()[-1].value, 1]),
        #                                           name="Coherence"),  [tf.shape(state)[0], tf.shape(state)[1], tf.shape(candidates_proj)[1], 1]), axis=-3)
        # (Batch x Candidates x 1 x Kernel Size) * (Batch x 1 x Timesteps x Kernel Size)
        # => (Batch x Candidates x Timesteps x Kernel Size)
        # => sum => (Batch x Candidates x Timesteps)
        # => max => Batch x Candidates
        res = tf.reduce_max(tf.reduce_sum(tf.expand_dims(candidates_proj, 2) * tf.expand_dims(state, 1), axis=-1), axis=-1, keepdims=True)
    return res


def last_prediction_bit_update(tstep, prev_state, valid_tstep, candidates_constraints,
                               predicted_tstep_ids, beam_search, **kwargs):
    def perform_update():
        batch_size = tf.shape(prev_state)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
        # convert back into global ids:
        predicted_candidates_constraints = beam_aware_gather_nd(candidates_constraints[tstep, :], indices=[batch_indices, predicted_tstep_ids])
        if beam_search:
            # in beam search situations, predicted_tstep_ids is a 2D array shape=(Batch x Beams)
            assert len(extract_shape(prev_state)) == 3, "expected prev_state to be 3D during beam_search but got {}.".format(prev_state)
            # valid_tstep has shape (Batch)
            # however prev_state has shape (Batch, Beams, single decision so far)
            # So we expand valid_tstep into (Batch, 1, 1)
            updated_state = tf.where(tf.tile(valid_tstep[..., None, None],
                                            [1, tf.shape(predicted_candidates_constraints)[1], tf.shape(predicted_candidates_constraints)[2]]),
                                    predicted_candidates_constraints,
                                    -tf.ones_like(predicted_candidates_constraints))
        else:
            assert len(extract_shape(prev_state)) == 2, "expected prev_state to be 2D but got {}.".format(prev_state)
            updated_state = tf.where(tf.tile(valid_tstep[..., None], [1, tf.shape(predicted_candidates_constraints)[1]]),
                                    predicted_candidates_constraints,
                                    -tf.ones_like(predicted_candidates_constraints))
        return updated_state
    return tf.cond(tf.greater(tf.shape(candidates_constraints)[3], 0), perform_update, lambda: prev_state)
    

def past_prediction_bit_update(tstep, prev_state, valid_tstep, candidates_ids,
                               candidates_ids_flat, predicted_tstep_ids, beam_search,
                               legacy_consistency_check=False, **kwargs):
    if legacy_consistency_check:
        prev_state, legacy_prev_state = prev_state
    batch_size = tf.shape(prev_state)[0]
    batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
    # convert back into global ids:
    predicted_candidate_ids = beam_aware_gather_nd(candidates_ids[tstep, :], indices=[batch_indices, predicted_tstep_ids])
    if beam_search:
        # in beam search situations, predicted_tstep_ids is a 2D array shape=(Batch x Beams)
        assert len(extract_shape(prev_state)) == 3, "expected prev_state to be 3D during beam_search but got {}.".format(prev_state)
        # valid_tstep has shape (Batch)
        # however prev_state has shape (Batch, Beams, decisions so far)
        # So we expand valid_tstep into (Batch, 1, 1)
        updated_state = tf.concat([prev_state, tf.where(tf.tile(valid_tstep[..., None, None], [1, tf.shape(predicted_candidate_ids)[1], 1]),
                                                        predicted_candidate_ids[..., None],
                                                        -tf.ones_like(predicted_candidate_ids[..., None]))], axis=-1)
    else:
        assert len(extract_shape(prev_state)) == 2, "expected prev_state to be 2D but got {}.".format(prev_state)
        updated_state = tf.concat([prev_state, tf.where(valid_tstep[..., None],
                                                        predicted_candidate_ids[..., None],
                                                        -tf.ones_like(predicted_candidate_ids[..., None]))], axis=-1)
    if legacy_consistency_check:
        # use global ids to update database:
        if beam_search:
            legacy_update = tf.logical_and(tf.equal(predicted_candidate_ids[..., None], candidates_ids_flat[None, None, :]),
                                           valid_tstep[:, None, None])
        else:
            legacy_update = tf.logical_and(tf.equal(predicted_candidate_ids[:, None], candidates_ids_flat[None, :]),
                                           valid_tstep[:, None])
        return updated_state, legacy_update
    else:
        return updated_state


def is_list_consistent(list_constraint_applies, candidates_ids, state):
    if not np.any(list_constraint_applies):
        return np.zeros(list(state.shape[:-1]) + [candidates_ids.shape[-1]], dtype=np.int32)
    same_instance = _get_relation("INSTANCE_OF", False).batch_is_related(candidates_ids, state, direct=False, indirect=True)
    same_state = _get_relation("us_state", True).batch_is_related(candidates_ids, state, direct=False, indirect=True)
    return same_instance.astype(np.int32) + same_state.astype(np.int32)
    

def list_matches_state_instance_of_to_features(tstep, tstep_labels, state, candidates_ids, candidates_metadata, beam_search, **kwargs):
    list_constraint_applies = tf.equal(candidates_metadata[tstep, :, 0, METADATA_LIST_DIM], METADATA_INSIDE_LIST)[:, None]
    candidates_ids = candidates_ids[tstep]
    if beam_search:
        # add beam dimension
        list_constraint_applies = list_constraint_applies[:, None]
        candidates_ids = candidates_ids[:, None]
    is_list_consistent_tf = tf.py_func(is_list_consistent, (list_constraint_applies, candidates_ids, state[..., -1:]), tf.int32)
    is_list_consistent_tf.set_shape(tuple([None, None, None] if beam_search else [None, None]))
    
    return tf.expand_dims(tf.cast(is_list_consistent_tf, tf.float32), [-1])


def last_prediction_bit_to_features(tstep, tstep_labels, state, candidates_ids, candidates_constraints_required, candidates_metadata, beam_search, **kwargs):
    # check if no constraint required or matches past constraints
    unconstrained = tf.cond(tf.greater(tf.shape(candidates_constraints_required)[3], 0),
                            # if the first constraint is padding, then this item is unconstrained.
                            lambda: tf.equal(candidates_constraints_required[tstep, :, :, 0], CONSTRAINTS_REQUIRED_PADDING),
                            # if the list of requirements is 0, then this item is also unconstrained.
                            lambda: tf.ones((tf.shape(candidates_constraints_required)[1], tf.shape(candidates_constraints_required)[2]), dtype=tf.bool))
    if beam_search:
        # matching will look for equality between created constraints at previous steps and the current step's requirements. If the current step's requirements
        # are empty or padding, then using the unconstrained boolean array above we can also determine that no computation is needed to be checked.
        #
        # Broadcasting:
        # ------------
        # broadcast candidates_constraints_required shape from Batch x Entities x Requirements into Batch x Beams (1) x Entities x 1 x Requirements
        # broadcast state shape from Batch x Beams x active constraints into Batch x Beams x 1 x active constraints x Requirements (1)
        matches_constraints = tf.reduce_any(tf.reduce_any(
            tf.equal(
                candidates_constraints_required[tstep, :, None, :, None, :],
                state[:, :, None, :, None]),
            [-1]), [-1])
        # after reduction we are left with Batch x Beams x Entities x Entities
        #
        # broadcast unconstrained to add a beam dimension (1) on second axis:
        unconstrained = unconstrained[:, None]
    else:
        matches_constraints = tf.reduce_any(tf.reduce_any(
            tf.equal(
                candidates_constraints_required[tstep, :, :, None, :],
                state[:, None, :, None]),
            [-1]), [-1])
    candidates_metadata_score = tf.cast(candidates_metadata[tstep, :, :, METADATA_SCORE_DIM], tf.float32)
    return tf.expand_dims(-1000 * tf.cast(tf.logical_not(tf.logical_or(unconstrained, matches_constraints)), tf.float32) +
                          candidates_metadata_score[:, None] if beam_search else candidates_metadata_score, [-1])


def incompatible_constraints_penalty_score(inputs, features, is_training, beam_search):
    return tf.squeeze(features, -1)


def past_prediction_bit_to_features_bool(tstep, tstep_labels, state, candidates_ids, beam_search, legacy_consistency_check=False, **kwargs):
    if legacy_consistency_check:
        state, legacy_state = state
        legacy_was_previously_predicted_bool = reshape_first_dimension(
            batch_gather_multi_timestep(legacy_state, tstep_labels, reshape=False), (tf.shape(tstep_labels)[0], tf.shape(tstep_labels)[1]))
    # Batch x choices x 1 === Batch x 1 x past_predictions
    if beam_search:
        # broadcast candidates_ids shape from Batch x Entities into Batch x Beams x Entities x 1
        # broadcast state shape from Batch x Beams x past_predictions into Batch x Beams x 1 x past_predictions
        with tf.device("cpu"):
            was_previously_predicted_bool = tf.reduce_any(tf.equal(candidates_ids[tstep, :, None, :, None], state[:, :, None, :]), axis=-1)
    else:
        with tf.device("cpu"):
            was_previously_predicted_bool = tf.reduce_any(tf.equal(candidates_ids[tstep, :, :, None], state[:, None, :]), axis=-1)

    if legacy_consistency_check:
        with tf.control_dependencies([tf.debugging.assert_equal(legacy_was_previously_predicted_bool, was_previously_predicted_bool)]):
            was_previously_predicted_bool = tf.identity(was_previously_predicted_bool)
    return was_previously_predicted_bool


def past_prediction_bit_to_features(*args, **kwargs):
    return tf.expand_dims(tf.cast(past_prediction_bit_to_features_bool(*args, **kwargs), tf.float32), [-1])


def candidate_callable_bit_initial_state(batch_size, num_entities, input_size, candidates_ids, candidate_ids2initial, **kwargs):
    transposed_candidate_ids = tf.transpose(candidates_ids, (1, 0, 2))
    initial_condition = tf.py_func(make_callable(candidate_ids2initial),
        [transposed_candidate_ids], tf.bool)
    initial_condition.set_shape(transposed_candidate_ids.shape)
    return (initial_condition, tf.TensorShape([None, None, None]))


def candidate_callable_bit_update(prev_state, **kwargs):
    return prev_state


def candidate_callable_bit_to_features(tstep, tstep_labels, state, candidates_ids, beam_search, **kwargs):
    # if beam_search broadcast candidates_ids shape from Batch x Entities into Batch x Beams x Entities x 1
    # broadcast state shape from Batch x Beams x past_predictions into Batch x Beams x 1 x past_predictions
    return tf.expand_dims(tf.cast(state[:, :, tstep] if beam_search else state[:, tstep], tf.float32), [-1])


def _get_relation(relation_name, classifier):
    if classifier:
        handler = make_callable("wikidata_linker_utils.sequence_model:default_classification_handler")()
        # is a candidate connected to any state item
        return handler.get_classifier(relation_name, variable_length=True).classification
    else:
        c = make_callable("wikidata_linker_utils.sequence_model:default_type_collection")()
        return c.relation(getattr(wprop, relation_name))



find_by_relation_conditions = {}

is_human = None

def _fill_state_min_max_dob(*, past_dob, past_dod, has_dod, relevant, state_max_dod, state_min_dob):
    if past_dob.ndim > 2:
        past_dob = past_dob.reshape(-1, past_dob.shape[-1])
        past_dod = past_dod.reshape(-1, past_dod.shape[-1])
        relevant = relevant.reshape(-1, relevant.shape[-1])
        has_dod = has_dod.reshape(-1, has_dod.shape[-1])
        state_max_dod = state_max_dod.reshape(-1, state_max_dod.shape[-1])
        state_min_dob = state_min_dob.reshape(-1, state_min_dob.shape[-1])

    # state has shape Batch x Beams x History or Batch x History
    any_relevant = np.any(relevant, axis=-1)

    for i in range(len(past_dob)):
        if any_relevant[i]:
            state_min_dob[i] = past_dob[i, relevant[i]].min()
            state_max_dod[i] = past_dod[i, relevant[i]].max()
            # still alive?:
            if np.any(np.logical_not(has_dod[i, relevant[i]])):
                state_max_dod[i] = 2100


def detect_contemporary(candidates_ids, state,):
    with scoped_timer("detect_contemporary"):
        if state.size == 0 or candidates_ids.size == 0:
            return np.zeros(list(state.shape[:-1]) + [candidates_ids.shape[-1]], dtype=np.bool)
        c = make_callable("wikidata_linker_utils.sequence_model:default_type_collection")()
        global is_human
        if is_human is None:
            is_human = c.satisfy([wprop.INSTANCE_OF], [c.name2index["Q5"]])
        dob = c.attribute(wprop.DATE_OF_BIRTH)
        dod = c.attribute(wprop.DATE_OF_DEATH)
        
        candidates_dob = dob.dense[candidates_ids]
        candidates_dod = dod.dense[candidates_ids]
        candidates_has_dod = dob.mask[candidates_ids]
        candidates_has_dob = dod.mask[candidates_ids]
        candidates_is_not_neg_1 = candidates_ids >= 0
        candidates_relevant = np.logical_and(np.logical_and(is_human[candidates_ids], candidates_has_dob), candidates_is_not_neg_1)

        if not np.any(candidates_relevant):
            return np.zeros(list(state.shape[:-1]) + [candidates_ids.shape[-1]], dtype=np.bool)
        past_dob = dob.dense[state]
        past_dod = dob.dense[state]
        has_dob = dob.mask[state]
        has_dod = dob.mask[state]
        is_not_neg_1 = state >= 0
        relevant = np.logical_and(np.logical_and(is_human[state], has_dob), is_not_neg_1)
        state_min_dob = np.zeros(list(state.shape[:-1]) + [1], dtype=np.int32)
        state_max_dod = np.zeros(list(state.shape[:-1]) + [1], dtype=np.int32)
        _fill_state_min_max_dob(past_dob=past_dob,
                                past_dod=past_dod,
                                has_dod=has_dod,
                                relevant=relevant,
                                state_max_dod=state_max_dod,
                                state_min_dob=state_min_dob)
        
        return np.logical_or(
            # is born before the last person in the state died, and is alive or died after the first birth
            np.logical_and(
                candidates_dob <= state_max_dod,
                np.logical_or(np.logical_not(candidates_has_dod), candidates_dod >= state_min_dob)
            ),
            np.logical_not(candidates_relevant)
        )

def find_by_relation(candidates_ids, state, relation_name, classifier, direct, indirect, max_relatable, related_or_empty, single_step_history):
    with scoped_timer("find_by_relation_" + relation_name):
        if single_step_history:
            state = last_non_negative(state)[..., None]
        condition = None
        rel = _get_relation(relation_name, classifier)
        if direct and related_or_empty:
            # to avoid treating empty sets as equally good when unrelatedness is not meaningful (e.g. if the target set contains items that are unreachable)
            # we construct a membership condition set.
            global find_by_relation_conditions
            if relation_name not in find_by_relation_conditions:
                condition = np.zeros(len(rel), dtype=np.bool)
                # mark these items as reachable.
                condition[rel.values] = True
                find_by_relation_conditions[relation_name] = condition
            else:
                condition = find_by_relation_conditions[relation_name]
        return rel.batch_is_related(
            candidates_ids, state, direct=direct, indirect=indirect, max_relatable=max_relatable,
            related_or_empty=related_or_empty, condition=condition)


def candidate_related_by(candidates_ids, relation_name, classifier, direct, indirect):
    with scoped_timer("candidate_related_by_" + relation_name):
        # (Batch x Candidates) -> Is a candidate connected to any other candidate for that batch row?
        return _get_relation(relation_name, classifier).batch_is_related(candidates_ids, direct=direct, indirect=indirect)


def past_callable_bit_to_features(tstep, tstep_labels, state, candidates_ids, beam_search, ids2ids, **kwargs):
    res = tf.py_func(make_callable(ids2ids), [candidates_ids[tstep, :, None] if beam_search else candidates_ids[tstep], state], tf.bool)
    res.set_shape(tuple([None, None, None] if beam_search else [None, None]))
    return res[..., None]


def past_callable_bit_to_score(inputs, features, is_training, beam_search, weight=2.0):
    return tf.squeeze(tf.cast(features, tf.float32) * weight, -1)

