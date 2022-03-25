import tensorflow as tf
import json
from .tf_operations import concat, extract_shape, recursive_shape_invariants, scope_variables
from .make_callable import make_callable
from .sequence_model_constants import FEATURE_INTERACTIONS, SCENARIO_FEATURE_SCORES, FEATURE_INTERACTIONS
from .submodel_spec_builder import build_submodel_from_spec


def scenario_add_interaction_terms(*, objective, featurization_embeddings, feature_interactions_hiddens, names):
    for interaction, feature_embed in zip(objective.get(FEATURE_INTERACTIONS, []), feature_interactions_hiddens):
        name = interaction.get("name", "_".join(interaction["features"]))
        featurization_embeddings[name] = feature_embed
        names.append(name)


def scenario_dynamic_fully_connected_combine(*, hiddens, objective, is_training, input_size,
                                             featurization_embeddings):
    with tf.variable_scope("VariableAssignments"):
        with tf.variable_scope("Combine"):
            hiddens = build_submodel_from_spec(inputs=hiddens, spec=objective["model"], is_training=is_training)
            interaction_vars = scope_variables()
        with tf.variable_scope("CrossInteractions"):
            feature_interactions_hiddens = []
            for interaction in objective.get(FEATURE_INTERACTIONS, []):
                assert len(interaction["features"]) > 1, \
                    "expected interaction's features to be between more than 1 term but got {}.".format(interaction["features"])
                with tf.variable_scope(interaction.get("name", "_".join(interaction["features"]))):
                    combined_features = concat([featurization_embeddings[feat_name] for feat_name in interaction["features"]], axis=-1)
                    combined_features = build_submodel_from_spec(inputs=combined_features,
                                                                 spec=interaction["model"],
                                                                 is_training=is_training)
                    feature_interactions_hiddens.append(combined_features)
            if len(feature_interactions_hiddens) > 0:
                hiddens = concat([hiddens] + feature_interactions_hiddens, axis=-1)
        with tf.variable_scope("LinearProject"):
            hiddens = tf.contrib.layers.fully_connected(hiddens,
                                                        num_outputs=input_size,
                                                        activation_fn=None)
            linear_vars = scope_variables()
    return hiddens, interaction_vars, linear_vars, feature_interactions_hiddens



def autoregressive_scenario_decoding(*, objective, inputs, scenario_embeddings_uncombined,
                                     labels, mask, is_training, candidates_ids, candidates_ids_flat, candidates_constraints,
                                     candidates_constraints_required, candidates_metadata, greedy_decoding, beam_width, packed_sequence,
                                     feature_name2ph, reorder_decoding=False):
    non_autoreg_classifications = [featurization for featurization in objective["classifications"]
                                   if featurization["type"] != "predicted"]
    autoreg_classifications = [featurization for featurization in objective["classifications"]
                               if featurization["type"] == "predicted"]
    # now do a gather on the mask in this order, returns with shape (Time x Batch x n-examples)
    time_major_candidate_ids = tf.transpose(candidates_ids, (1, 0, 2))
    time_major_candidates_constraints = tf.transpose(candidates_constraints, (1, 0, 2, 3))
    time_major_candidates_constraints_required = tf.transpose(candidates_constraints_required, (1, 0, 2, 3))
    time_major_candidates_metadata = tf.transpose(candidates_metadata, (1, 0, 2, 3))

    if reorder_decoding:
        # reordering will change the access order and thus requires a special scatter operation to undo
        packed_sequence = False
        # order by number of candidates at every timestep (mask is Time x Batch x Examples)
        num_candidates_per_step = tf.math.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        # timesteps ordered in increasing number of candidates (shape is Time x Batch)
        # reordered_timesteps = tf.argsort(num_candidates_per_step, axis=0, direction='ASCENDING', stable=False, name="IncreasingCandidateEvents")
        reordered_timesteps = tf.argsort(num_candidates_per_step, axis=0, direction='DESCENDING', stable=False, name="IncreasingCandidateEvents")

        reordered_mask = batch_gather_multi_timestep(mask, reordered_timesteps, reshape=True, time_major=True)
        reordered_labels = batch_gather_multi_timestep(labels, reordered_timesteps, reshape=True, time_major=True)
        reordered_candidate_ids = batch_gather_multi_timestep(time_major_candidate_ids, reordered_timesteps, reshape=True, time_major=True)
        reordered_candidates_constraints = batch_gather_multi_timestep(time_major_candidates_constraints, reordered_timesteps, reshape=True, time_major=True)
        reordered_candidates_constraints_required = batch_gather_multi_timestep(time_major_candidates_constraints_required, reordered_timesteps, reshape=True, time_major=True)
        reordered_candidates_metadata = batch_gather_multi_timestep(time_major_candidates_metadata, reordered_timesteps, reshape=True, time_major=True)
        reordered_inputs = batch_gather_multi_timestep(inputs, reordered_timesteps, reshape=True, time_major=True)
    else:
        reordered_mask, reordered_labels, reordered_inputs, reordered_candidate_ids, reordered_candidates_constraints, reordered_candidates_constraints_required, reordered_candidates_metadata = (
            mask, labels, inputs, time_major_candidate_ids, time_major_candidates_constraints, time_major_candidates_constraints_required, time_major_candidates_metadata)
    has_interaction_vars = [False]
    autoregressive_scenario_decoding_internal_kwargs = dict(
        mask=reordered_mask,
        labels=reordered_labels,
        inputs=reordered_inputs,
        scenario_embeddings_uncombined=scenario_embeddings_uncombined,
        candidates_ids_flat=candidates_ids_flat,
        candidates_ids=reordered_candidate_ids,
        candidates_constraints=reordered_candidates_constraints,
        candidates_constraints_required=reordered_candidates_constraints_required,
        candidates_metadata=reordered_candidates_metadata,
        feature_name2ph=feature_name2ph,
        autoreg_classifications=autoreg_classifications,
        non_autoreg_classifications=non_autoreg_classifications,
        has_interaction_vars=has_interaction_vars,
        objective=objective,
        is_training=is_training,
        greedy_decoding=greedy_decoding,
        beam_width=beam_width,
        packed_sequence=packed_sequence,
    )
    unary_scores_arr, feature_scores_arr, useful_steps, predictions = tf.cond(
        tf.logical_or(is_training, tf.equal(beam_width, 1)),
        lambda: autoregressive_scenario_decoding_internal(beam_search=False, **autoregressive_scenario_decoding_internal_kwargs),
        lambda: autoregressive_scenario_decoding_internal(beam_search=True, **autoregressive_scenario_decoding_internal_kwargs),
        name="beam_search"
    )
    has_interaction_vars = has_interaction_vars[0]
    # Useful steps gives us the positions with the reordered timesteps, so assuming our data was as follows:
    # mask:
    # [x x x 0 0 0]
    # [x 0 0 0 0 0]
    # [0 0 0 0 0 0]

    # Then we reorder in increasing number of events 'x':
    # reordered_timesteps = [2, 1, 0]

    # reordered_mask:
    # [0 0 0 0 0 0]
    # [x 0 0 0 0 0]
    # [x x x 0 0 0]

    # Useful steps only contains step that have at least 1 event:
    # useful_steps = [1, 2]

    # To put back output data to where it belongs, e.g.
    # useful_steps[0] -> 1 -> 1
    # useful_steps[1] -> 2 -> 0
    # We need to reverse the lookups:

    # The data coming in is of dimensions useful_steps x batch x scenarios
    # We can put this back into original_timesteps x batch x scenarios by figuring out what were the right timesteps for each:
    # We now have original timesteps (of len useful steps) x batch
    batch_size = tf.shape(inputs)[1]
    timesteps = tf.shape(labels)[0]
    if reorder_decoding:
        reverse_reordered_timesteps = tf.gather(reordered_timesteps, useful_steps)
        unary_scores = batch_scatter_multi_timestep(unary_scores_arr,
                                                    input_shape=[tf.shape(useful_steps)[0], batch_size] + extract_shape(reordered_labels)[2:],
                                                    indices=reverse_reordered_timesteps,
                                                    name="UnaryScores",
                                                    time_major=True,
                                                    timesteps=timesteps)
        predictions = batch_scatter_multi_timestep(predictions,
                                                   input_shape=[tf.shape(useful_steps)[0], batch_size],
                                                   indices=reverse_reordered_timesteps,
                                                   time_major=True,
                                                   timesteps=timesteps)
    elif packed_sequence:
        unary_scores = unary_scores_arr
    else:
        unary_scores = tf.cond(tf.greater(tf.shape(useful_steps)[0], 0),
                               lambda: tf.scatter_nd(useful_steps[:, None], unary_scores_arr,
                                     shape=[timesteps, tf.shape(labels)[1], tf.shape(labels)[2]],
                                     name="UnaryScores"),
                               lambda: tf.zeros([timesteps, tf.shape(labels)[1], tf.shape(labels)[2]], dtype=tf.float32))
        predictions = tf.scatter_nd(useful_steps[:, None], predictions,
                                    shape=[timesteps, batch_size])
    extra_loss = None
    if not has_interaction_vars:
        if reorder_decoding:
            feature_scores = [batch_scatter_multi_timestep(fscore,
                                                           input_shape=[tf.shape(useful_steps)[0], batch_size] + extract_shape(reordered_labels)[2:],
                                                           indices=reverse_reordered_timesteps,
                                                           timesteps=tf.shape(labels)[0],
                                                           time_major=True)
                              for fscore in feature_scores_arr]
        elif packed_sequence:
            feature_scores = feature_scores_arr
        else:
            feature_scores = [tf.scatter_nd(useful_steps[:, None], fscore,
                                            shape=[timesteps, tf.shape(labels)[1], tf.shape(labels)[2]])
                              for fscore in feature_scores_arr]
        feature_scores_non_autoreg = feature_scores[:len(non_autoreg_classifications)]
        feature_scores_autoreg = feature_scores[len(non_autoreg_classifications):len(non_autoreg_classifications) + len(autoreg_classifications)]
        feature_scores_interaction = feature_scores[len(non_autoreg_classifications) + len(autoreg_classifications):]
        non_autoreg_classifications_idx = 0
        autoreg_classifications_idx = 0
        feature_activations = []
        for featurization in objective["classifications"]:
            if featurization["type"] == "predicted":
                assert autoreg_classifications[autoreg_classifications_idx]["type"] == featurization["type"]
                assert autoreg_classifications[autoreg_classifications_idx]["name"] == featurization["name"]
                fscore = feature_scores_autoreg[autoreg_classifications_idx]
                tf.add_to_collection(SCENARIO_FEATURE_SCORES, tf.transpose(fscore, (1, 0, 2)))
                autoreg_classifications_idx += 1
            else:
                assert non_autoreg_classifications[non_autoreg_classifications_idx]["type"] == featurization["type"]
                assert non_autoreg_classifications[non_autoreg_classifications_idx]["name"] == featurization["name"]
                fscore = feature_scores_non_autoreg[non_autoreg_classifications_idx]
                tf.add_to_collection(SCENARIO_FEATURE_SCORES, tf.transpose(fscore, (1, 0, 2)))
                non_autoreg_classifications_idx += 1
            feature_activations.append(fscore)

        named_feature_activations = {featurization["name"]: fscore
                                     for fscore, featurization in zip(feature_activations, objective["classifications"])}

        for fscore, featurization in zip(feature_activations, objective["classifications"]):
            if featurization.get("activation_loss") is not None:
                new_loss = make_callable(featurization["activation_loss"])(
                    activation=fscore,
                    mask=mask,
                    other_activations=named_feature_activations)
                if extra_loss is None:
                    extra_loss = new_loss
                else:
                    extra_loss += new_loss
        for feature_score in feature_scores_interaction:
            tf.add_to_collection(SCENARIO_FEATURE_SCORES, tf.transpose(feature_score, (1, 0, 2)))
    return unary_scores, extra_loss, predictions


def _recursive_expand_dims(xs, shapes):
    if isinstance(xs, tf.Tensor):
        new_xs = tf.expand_dims(xs, 1)
        shape = shapes.as_list()
        new_shape = tf.TensorShape(shape[:1] + [None] + shape[1:])
        return new_xs, new_shape
    elif isinstance(xs, (tuple, list)):
        new_xs_and_shapes = [_recursive_expand_dims(x, s) for x, s in zip(xs, shapes)]
        out_xs = [x for x, _ in new_xs_and_shapes]
        out_shapes = [s for _, s in new_xs_and_shapes]
        return type(xs)(out_xs), type(xs)(out_shapes)
    else:
        raise ValueError("Expected tf.Tensor or list/tuple but got unexpected datatype in _recursive_expand_dims: {}.".format(xs))


def batch_gather_multi_timestep(x, indices, reshape=True, time_major=False, name=None):
    batch_axis = 1 if time_major else 0
    batch_size = tf.shape(x)[batch_axis]
    batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
    if time_major:
        # time major ordering, gather along 1st axis
        gather_indices = tf.stack([tf.reshape(indices, [-1]),
                                   tf.reshape(batch_indices[None, :] + tf.zeros_like(indices), [-1])], axis=-1)
        # Return with shape: (Timesteps x Batch x x.shape[2:])
        output_shape = [extract_shape(indices)[0], batch_size] + extract_shape(x)[2:]
    else:
        # batch-major ordering, gather along 2nd axis
        gather_indices = tf.stack([tf.reshape(batch_indices[:, None] + tf.zeros_like(indices), [-1]),
                                   tf.reshape(indices, [-1])], axis=-1)
        # Return with shape: (Batch x Timesteps x x.shape[2:])
        output_shape = [batch_size, extract_shape(indices)[1]] + extract_shape(x)[2:]
    res = tf.gather_nd(x, gather_indices, name=name if not reshape else None)
    if reshape:
        return tf.reshape(res, output_shape, name=name)
    return res


def safe_batch_gather_multi_timestep(x, indices, reshape=True, time_major=False, name=None):
    if isinstance(x, tuple):
        return tuple([safe_batch_gather_multi_timestep(v, indices, reshape=reshape, time_major=time_major, name=name) for v in x])
    return tf.cond(tf.greater(tf.size(x), 0),
                   lambda: batch_gather_multi_timestep(x, indices, reshape=reshape, time_major=time_major, name=name),
                   lambda: tf.zeros(extract_shape(indices) + extract_shape(x)[2:], dtype=x.dtype))
    

def batch_scatter_multi_timestep(x, indices, timesteps, input_shape, time_major=False, name=None):
    batch_axis = 1 if time_major else 0
    batch_size = tf.shape(x)[batch_axis]
    batch_indices = tf.range(batch_size, dtype=tf.int32, name="batch_index")
    x_shape = input_shape
    if time_major:
        scatter_indices = tf.stack([tf.reshape(indices, [-1]),
                                    tf.reshape(tf.zeros_like(indices) + tf.range(batch_size, dtype=tf.int32)[None, :], [-1])],
                                   axis=-1)
        x = tf.reshape(x, [batch_size * tf.shape(indices)[0]] + x_shape[2:])
        output_shape = [timesteps, batch_size] + x_shape[2:]
    else:
        scatter_indices = tf.stack([tf.reshape(tf.zeros_like(indices) + tf.range(batch_size, dtype=tf.int32)[:, None], [-1]),
                                    tf.reshape(indices, [-1])],
                                   axis=-1)
        x = tf.reshape(x, [batch_size * tf.shape(indices)[1]] + x_shape[2:])
        output_shape = [batch_size, timesteps] + x_shape[2:]
    res = tf.scatter_nd(scatter_indices, x, name=name, shape=output_shape)
    return res



def autoregressive_scenario_decoding_internal(*, mask, labels, inputs, candidates_ids_flat, candidates_ids, candidates_constraints,
                                              candidates_constraints_required, candidates_metadata, scenario_embeddings_uncombined,
                                              feature_name2ph, autoreg_classifications, non_autoreg_classifications, objective, is_training,
                                              greedy_decoding, beam_width, beam_search, has_interaction_vars, packed_sequence):

    batch_size = tf.shape(inputs)[1]
    num_entities = tf.shape(candidates_ids_flat)[0]
    initial_states = []
    initial_states_shapes = []
    feat_name_2_state_idx = {}
    feat_name_2_is_responsible_for_update = {}
    initial_update_fn_2_state_idx = {}
    for featurization in autoreg_classifications:
        if featurization["initial_state"] is None:
            continue
        initial_update_fn_key = json.dumps((featurization["initial_state"], featurization["update"]), separators=(',', ':'))
        if initial_update_fn_key in initial_update_fn_2_state_idx:
            # Repeating an autoregressive state
            feat_name_2_state_idx[featurization["name"]] = initial_update_fn_2_state_idx[initial_update_fn_key]
            feat_name_2_is_responsible_for_update[featurization["name"]] = False
        else:
            with tf.variable_scope(featurization["name"]):
                init_state, shape = make_callable(featurization["initial_state"])(
                    batch_size=batch_size, num_entities=num_entities, input_size=inputs.get_shape()[-1].value,
                    candidates_ids=candidates_ids,
                    candidates_constraints=candidates_constraints,
                    candidates_constraints_required=candidates_constraints_required,
                    candidates_metadata=candidates_metadata,
                    feature_name2ph=feature_name2ph)
                if beam_search:
                    # insert beam dimension here:
                    init_state, shape = _recursive_expand_dims(init_state, shape)
                state_idx = len(initial_states)
                initial_states.append(init_state)
                initial_states_shapes.append(shape)
                feat_name_2_state_idx[featurization["name"]] = state_idx
                feat_name_2_is_responsible_for_update[featurization["name"]] = True
                initial_update_fn_2_state_idx[initial_update_fn_key] = state_idx
    
    candidate_embeddings_featurizations = []
    if len(scenario_embeddings_uncombined) == 0:
        non_autoreg_scenario_embeddings = None
    else:
        candidate_embedding_ignore = set(objective.get("candidate_embedding_ignore", []))
        non_autoreg_scenario_embeddings = concat([
            h for featurization, h in zip(non_autoreg_classifications, scenario_embeddings_uncombined)
            if featurization["name"] not in candidate_embedding_ignore
        ], -1)
        for featurization, h in zip(non_autoreg_classifications, scenario_embeddings_uncombined):
            candidate_embeddings_featurizations.append((featurization, extract_shape(h)[-1]))
    
    # shape=timesteps, indices where something is happening
    if packed_sequence:
        useful_steps = tf.constant(0, dtype=tf.int32)
    else:
        useful_steps = tf.cast(tf.squeeze(tf.where(tf.math.reduce_any(mask, axis=(1, 2))), axis=-1), tf.int32)
    
    ground_truth_decode = tf.logical_not(greedy_decoding)
    train_autoregressive_decode_fn = make_callable(objective["train_autoregressive_decode_fn"])(batch_size=batch_size) if objective.get("train_autoregressive_decode_fn") else None

    def condition(event_tstep, *args):
        return tf.less(event_tstep, tf.shape(inputs)[0]) if packed_sequence else tf.less(event_tstep, tf.shape(useful_steps)[0])

    def body(event_tstep, num_predictions, unary_scores_arr, feature_scores_arr, states, predictions_so_far):
        if beam_search:
            predictions_so_far, beam_scores = predictions_so_far
        if packed_sequence:
            tstep = event_tstep
        else:
            tstep = useful_steps[event_tstep]
        batch_indices = tf.range(tf.shape(candidates_ids)[1], dtype=tf.int32, name="batch_index")
        tstep_labels = labels[tstep]
        # grab per batch and scenario items
        # add the information about past predictions here
        candidate_embeddings = tf.nn.embedding_lookup(non_autoreg_scenario_embeddings, tstep_labels) if non_autoreg_scenario_embeddings is not None else None

        with tf.variable_scope("VariableAssignmentsAutoregressive", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("Embed"):
                autoreg_features_uncombined = []
                initial_update_feat_cache = {}
                for featurization in autoreg_classifications:
                    with tf.variable_scope(featurization["name"]):
                        with tf.variable_scope("Combine"):
                            initial_update_feat_cache_key = None if featurization["type"] != "predicted" else json.dumps((featurization["initial_state"], featurization["update"], featurization["state_to_features"]))
                            if initial_update_feat_cache_key is not None and initial_update_feat_cache_key in initial_update_feat_cache:
                                hiddens_float = initial_update_feat_cache[initial_update_feat_cache_key]
                            else:
                                has_state = featurization["initial_state"] is not None and featurization["update"] is not None
                                hiddens_float = make_callable(featurization["state_to_features"])(
                                    tstep=tstep,
                                    tstep_labels=tstep_labels,
                                    candidates_ids=candidates_ids,
                                    candidate_embeddings=candidate_embeddings,
                                    candidates_constraints=candidates_constraints,
                                    candidates_constraints_required=candidates_constraints_required,
                                    candidates_metadata=candidates_metadata,
                                    feature_name2ph=feature_name2ph,
                                    is_training=is_training,
                                    beam_search=beam_search,
                                    **({"state": states[feat_name_2_state_idx[featurization["name"]]]} if has_state else {})
                                )
                                if initial_update_feat_cache_key is not None:
                                    initial_update_feat_cache[initial_update_feat_cache_key] = hiddens_float
                            hiddens_float = build_submodel_from_spec(
                                inputs=hiddens_float, spec=featurization.get("model", []), is_training=is_training)

                        autoreg_features_uncombined.append(hiddens_float)

            featurization_embeddings = {}
            if beam_search:
                current_beam_width = tf.shape(autoreg_features_uncombined[0])[1]
            # we linearly combine scenario features, so we can extract their contribution to the final decision
            # more easily.
            assert len(autoreg_features_uncombined) == len(autoreg_classifications), \
                "expected same number of scenario feature embeds as classifications."
            for featurization, feature_embed in zip(non_autoreg_classifications + autoreg_classifications,
                                                    scenario_embeddings_uncombined + autoreg_features_uncombined):
                # multiply 
                if not featurization["type"] == "predicted":
                    # do an embedding lookup first
                    feature_embed = tf.nn.embedding_lookup(feature_embed, tstep_labels)
                    if beam_search:
                        feature_embed = tf.tile(tf.expand_dims(feature_embed, 1, name=featurization["name"] + "_Beam"), [1, current_beam_width, 1, 1],
                                                name=featurization["name"] + "_BeamTile")
                else:
                    # lookup was already done because this is done autoregressively
                    pass
                featurization_embeddings[featurization["name"]] = feature_embed


            autoreg_features_uncombined_concatenative = [feature_embed for featurization, feature_embed in
                                                         zip(autoreg_classifications, autoreg_features_uncombined)
                                                         if featurization.get("combine", "concatenate") == "concatenate"]
            autoreg_features_uncombined_additive = [(featurization, feature_embed) for featurization, feature_embed in
                                                    zip(autoreg_classifications, autoreg_features_uncombined)
                                                    if featurization.get("combine", "concatenate") == "add"]
            non_autoreg_features_uncombined_additive = [(featurization, feature_embed) for featurization, feature_embed in
                                                        zip(non_autoreg_classifications, scenario_embeddings_uncombined)
                                                        if featurization.get("combine", "concatenate") == "add"]


            # batch x per-example-scenarios
            if beam_search:
                # in beam search candidate_embeddings have shape (Batch x candidates x feats)  and autoreg_features_uncombined have shape (Batch x Beams x candidates x feats)
                # thus we need to tile candidate_embeddings to accomodate the other features during concatenation
                
                scenario_embeddings_uncombined_tstep = (concat([tf.tile(tf.expand_dims(candidate_embeddings, 1), [1, current_beam_width, 1, 1])] + autoreg_features_uncombined_concatenative, axis=-1)
                                                        if non_autoreg_scenario_embeddings is not None else
                                                        concat(autoreg_features_uncombined_concatenative, -1))
            else:
                scenario_embeddings_uncombined_tstep = (concat([candidate_embeddings] + autoreg_features_uncombined_concatenative, axis=-1)
                                                        if non_autoreg_scenario_embeddings is not None else
                                                        concat(autoreg_features_uncombined_concatenative, -1))
            # now integrate into full scheme.
            scenario_embeddings, interaction_vars, (linear_combiner_w, linear_combiner_b), feature_interactions_hiddens = scenario_dynamic_fully_connected_combine(
                hiddens=scenario_embeddings_uncombined_tstep,
                featurization_embeddings=featurization_embeddings,
                objective=objective,
                is_training=is_training,
                input_size=inputs.get_shape()[-1].value)

            inputs_tstep = inputs[tstep]
            featurization_additive_scores = {}
            if beam_search:
                # (batch x beams x per-example-scenarios x h) * (batch x h x 1)
                scenario_embeddings_shape = extract_shape(scenario_embeddings)
                # during matmul collapse beams and per-example-scenarios into one dimension
                unary_scores = tf.squeeze(tf.matmul(
                    tf.reshape(scenario_embeddings,
                               [scenario_embeddings_shape[0],
                                scenario_embeddings_shape[1] * scenario_embeddings_shape[2],
                                scenario_embeddings_shape[3]]),
                    tf.expand_dims(inputs_tstep, 2)), axis=-1)
                unary_scores = tf.reshape(unary_scores, [scenario_embeddings_shape[0], scenario_embeddings_shape[1], scenario_embeddings_shape[2]])
            else:
                # (batch x time x per-example-scenarios x h) * (batch x time x h x 1)
                unary_scores = tf.squeeze(tf.matmul(
                    scenario_embeddings,
                    tf.expand_dims(inputs_tstep, 2)), axis=-1)
            post_norm_unary_scores = None

            for featurization, feature_embed in (autoreg_features_uncombined_additive + [(featurization, featurization_embeddings[featurization["name"]])
                                                                                         for featurization, _ in non_autoreg_features_uncombined_additive]):
                score = make_callable(featurization["features_to_score_additive"])(
                    features=feature_embed,
                    inputs=inputs_tstep,
                    is_training=is_training,
                    beam_search=beam_search)
                if featurization.get("normalize", True):
                    featurization_additive_scores[featurization["name"]] = score
                    unary_scores += score
                else:
                    if post_norm_unary_scores is None:
                        post_norm_unary_scores = tf.zeros_like(unary_scores)
                    featurization_additive_scores[featurization["name"]] = tf.nn.log_softmax(score, axis=-1)
                    post_norm_unary_scores += score

            if beam_search:
                # use a beam-width here to do decoding
                labels_mask_casted = tf.cast(mask[tstep][:, None, :], inputs.dtype)
                # Batch x beams + Batch x beams x candidates
                masked_unary_scores = labels_mask_casted * unary_scores - (1 - labels_mask_casted) * 50
                masked_beam_unary_scores = beam_scores[:, :, None] + tf.nn.log_softmax(masked_unary_scores, axis=-1)
                if post_norm_unary_scores is not None:
                    masked_beam_unary_scores = masked_beam_unary_scores + post_norm_unary_scores
                # collapse beam and candidate dimension
                flat_masked_beam_unary_scores = tf.reshape(masked_beam_unary_scores, [batch_size, tf.shape(masked_beam_unary_scores)[1] * tf.shape(masked_beam_unary_scores)[2]])
                # prediction for each batch element for this timestep
                k = tf.minimum(tf.shape(flat_masked_beam_unary_scores)[1], beam_width)
                surviving_beam_scores_premask, predicted_beam_branches_and_ids = tf.math.top_k(flat_masked_beam_unary_scores, k=k,
                                                                                               name="BeamSearchRanking")
                predicted_beam_branches_and_ids = tf.cast(predicted_beam_branches_and_ids, tf.int32)
                # Now recover the id of the previous beam that came in, shape Batch x new_beams (probably k)
                parent_beam_id_premask = predicted_beam_branches_and_ids // tf.shape(masked_beam_unary_scores)[2]
                # if this is a prediction step then use the new beam ids, else keep the beam ids the same
                pad_k = tf.cast(tf.ceil(k / tf.shape(beam_scores)[1]), tf.int32)
                parent_beam_id = tf.where(mask[tstep, :, 0], parent_beam_id_premask,
                                          tf.tile(tf.range(tf.shape(beam_scores)[1])[None, :], [batch_size, pad_k])[:, :k])

                surviving_beam_scores = tf.where(mask[tstep, :, 0],
                                                 surviving_beam_scores_premask,
                                                 tf.tile(beam_scores, [1, pad_k])[:, :k])
                # Which decisions were taken (regardless of the beam chosen), shape Batch x new_beams (probably k)
                predicted_tstep_ids = predicted_beam_branches_and_ids % tf.shape(masked_beam_unary_scores)[2]
                # now we need to update the states according to the beams that survived
                states = [safe_batch_gather_multi_timestep(state, parent_beam_id, time_major=False, reshape=True)
                          for state in states]
                predictions_so_far = tf.concat(
                    [safe_batch_gather_multi_timestep(predictions_so_far, parent_beam_id, time_major=False, reshape=True),
                     predicted_tstep_ids[:, :, None]], axis=-1, name="BatchBeamMajorPredictions")
            else:
                if post_norm_unary_scores is not None:
                    unary_scores = unary_scores + post_norm_unary_scores
                
                def maybe_greedy_decode():
                    labels_mask_casted = tf.cast(mask[tstep], inputs.dtype)
                    masked_unary_scores = labels_mask_casted * unary_scores - (1 - labels_mask_casted) * 50
                    return tf.cast(tf.argmax(masked_unary_scores, axis=-1), tf.int32)

                predicted = tf.cond(tf.greater(tf.shape(unary_scores)[1], 0),
                                    maybe_greedy_decode,
                                    lambda: tf.zeros((tf.shape(unary_scores)[0],), tf.int32))
                
                predicted_tstep_ids = tf.cond(
                    tf.logical_or(is_training, ground_truth_decode),
                    lambda: tf.zeros(batch_size, dtype=tf.int32) if train_autoregressive_decode_fn is None else train_autoregressive_decode_fn(mask=mask[tstep], unary_scores=unary_scores),
                    lambda: predicted)
                # write down actual decoded predictions:
                predictions_so_far = predictions_so_far.write(event_tstep, predicted, name="TimeMajorGreeedyPredictions")
            
            # write in the order of events, not at the true temporal location
            if beam_search:
                unary_scores_arr = tf.concat([
                    safe_batch_gather_multi_timestep(unary_scores_arr, parent_beam_id, time_major=False, reshape=True),
                    batch_gather_multi_timestep(
                        # Batch x beams x candidates
                        unary_scores,
                        parent_beam_id,
                        reshape=True, time_major=False
                    )[:, :, :, None]
                ], axis=-1, name="BatchBeamMajorUnaryScores")
            else:
                unary_scores_arr = unary_scores_arr.write(event_tstep, unary_scores)

            if len(interaction_vars) == 0:
                # linear_combiner_w is (scenario_features x model_hidden_size)
                # inputs[tstep] is (batch x model_hidden_size)
                # to get batch x scenario_features
                linear_weights = tf.matmul(inputs_tstep, linear_combiner_w, transpose_b=True)
                # we linearly combine scenario features, so we can extract their contribution to the final decision
                # more easily.
                so_far = 0
                new_feature_scores = []
                assert len(autoreg_features_uncombined) == len(autoreg_classifications), \
                    "expected same number of scenario feature embeds as classifications."

                names = [featurization["name"] for featurization
                         in non_autoreg_classifications + autoreg_classifications]

                scenario_add_interaction_terms(names=names,
                                               featurization_embeddings=featurization_embeddings,
                                               objective=objective,
                                               feature_interactions_hiddens=feature_interactions_hiddens)
                for name, featurization in zip(names,
                                               non_autoreg_classifications + autoreg_classifications + objective.get(FEATURE_INTERACTIONS, [])):
                    if name in featurization_additive_scores:
                        feature_score = featurization_additive_scores[name]
                    else:
                        feature_embed = featurization_embeddings[name]
                        sliced_linear_weights = linear_weights[:, so_far:so_far + feature_embed.get_shape()[-1].value, None]
                        if beam_search:
                            # to get batch x 1 x scenario_features
                            sliced_linear_weights = sliced_linear_weights[:, None]
                        feature_score = tf.squeeze(tf.matmul(feature_embed, sliced_linear_weights), -1, name=name + "_score")
                        so_far += feature_embed.get_shape()[-1].value

                        if "interactive_input" in featurization:
                            res = tf.py_func(make_callable(featurization["interactive_input"]), [feature_score, sliced_linear_weights], tf.float32)
                            with tf.control_dependencies([res]):
                                feature_score = tf.identity(feature_score)
                    new_feature_scores.append(feature_score)
                    
                if beam_search:
                    feature_scores_arr = [tf.concat([
                        safe_batch_gather_multi_timestep(arr, parent_beam_id, time_major=False, reshape=True, name="arr_parent"),
                        batch_gather_multi_timestep(
                            # Batch x beams x candidates
                            fscore,
                            parent_beam_id,
                            reshape=True, time_major=False, name="new_fscore"
                        )[:, :, :, None]
                    ], axis=-1, name="BatchBeamMajorFeatureScores") for fscore, arr in zip(new_feature_scores, feature_scores_arr)]
                else:
                    feature_scores_arr = [arr.write(event_tstep, fscore)
                                          for fscore, arr in zip(new_feature_scores, feature_scores_arr)]
            else:
                has_interaction_vars[0] = True

            updated_states = []
            valid_tstep = mask[tstep, :, 0]
            num_predictions = num_predictions + tf.cast(valid_tstep, tf.int32)

            with tf.variable_scope("Update"):
                for axis, featurization in enumerate(objective["classifications"]):
                    with tf.variable_scope(featurization["name"], reuse=tf.AUTO_REUSE):
                        if (featurization["type"] == "predicted" and
                            featurization["initial_state"] is not None and
                            featurization["update"] is not None and
                            feat_name_2_is_responsible_for_update[featurization["name"]]):
                            updated_states.append(make_callable(featurization["update"])(
                                valid_tstep=valid_tstep,
                                tstep=tstep,
                                candidate_embeddings=candidate_embeddings,
                                candidate_embeddings_featurizations=candidate_embeddings_featurizations,
                                candidates_ids=candidates_ids,
                                candidates_constraints=candidates_constraints,
                                candidates_constraints_required=candidates_constraints_required,
                                candidates_metadata=candidates_metadata,
                                feature_name2ph=feature_name2ph,
                                candidates_ids_flat=candidates_ids_flat,
                                predicted_tstep_ids=predicted_tstep_ids,
                                num_predictions=num_predictions,
                                is_training=is_training,
                                prev_state=states[feat_name_2_state_idx[featurization["name"]]],
                                beam_search=beam_search))
        while_loop_output = [event_tstep + 1, num_predictions, unary_scores_arr, list(feature_scores_arr), updated_states]
        if beam_search:
            # add the beam scores
            while_loop_output.append((predictions_so_far, surviving_beam_scores))
        else:
            while_loop_output.append(predictions_so_far)
        # now we can use these predictions to inform future predictions
        return while_loop_output


    if beam_search:
        unary_scores_arr = tf.zeros((batch_size, 1, tf.shape(labels)[2], 0), name="BeamUnaryScores", dtype=tf.float32)
        unary_scores_shape = tf.TensorShape([None, None, None, None])
        feature_scores_arr = [tf.zeros((batch_size, 1, tf.shape(labels)[2], 0), name=f"BeamFeatureScores{featurization['name']}", dtype=tf.float32)
                              for featurization in objective["classifications"] + objective.get(FEATURE_INTERACTIONS, [])]
        feature_scores_arr_shape = [tf.TensorShape([None, None, None, None]) for _ in feature_scores_arr]
    else:
        unary_scores_arr = tf.TensorArray(inputs.dtype,
                                          size=0,
                                          dynamic_size=True,
                                          clear_after_read=False,
                                          infer_shape=False)
        unary_scores_shape = None
        feature_scores_arr = [tf.TensorArray(inputs.dtype,
                                             size=0,
                                             dynamic_size=True,
                                             clear_after_read=False,
                                             infer_shape=False) for _ in objective["classifications"] + objective.get(FEATURE_INTERACTIONS, [])]
        feature_scores_arr_shape = None
    while_loop_inputs = [tf.constant(0, name="event_tstep", dtype=tf.int32),
                         tf.zeros(batch_size, name="num_predictions", dtype=tf.int32), unary_scores_arr, feature_scores_arr, initial_states]
    shape_invariants = [None, None, unary_scores_shape, feature_scores_arr_shape, initial_states_shapes]
    shape_invariants = recursive_shape_invariants(while_loop_inputs, known_shape_invariants=shape_invariants)
    if beam_search:
        # add beam scores as a final component of the while loop state:
        # contains either beam log likelihood or score function for sequence of decisions
        while_loop_inputs.append((tf.zeros((batch_size, 1, 0), dtype=tf.int32), tf.zeros((batch_size, 1), name="BeamScores")))
        shape_invariants.append((tf.TensorShape([None, None, None]), tf.TensorShape([None, None])))
    else:
        while_loop_inputs.append(tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False, infer_shape=False))
        shape_invariants.append(tf.TensorShape(None))

    event_tstep_final, npred, unary_scores_arr, feature_scores_arr, states, predictions_so_far = tf.while_loop(
        condition, body,
        while_loop_inputs,
        back_prop=True,
        shape_invariants=shape_invariants)
    if beam_search:
        predictions_so_far, surviving_beam_scores = predictions_so_far
        # make unary scores time-major and grab top-beam (Batch x Beams x Candidates x Time)
        unary_scores_arr = tf.transpose(unary_scores_arr[:, 0], (2, 0, 1), name="BeamUnaryScores")
        feature_scores_arr = [tf.transpose(fscore[:, 0], (2, 0, 1), name=f"BeamFeatureScores{featurization['name']}")
                              for fscore, featurization in zip(feature_scores_arr,
                                                               objective["classifications"] + objective.get(FEATURE_INTERACTIONS, []))]
        predictions = tf.transpose(predictions_so_far[:, 0], (1, 0), name="BeamSearchPredictions")
    else:
        predictions = predictions_so_far.stack(name="GreedyPredictions")
        predictions.set_shape(tf.TensorShape([None, None]))
        unary_scores_arr = unary_scores_arr.stack()
        feature_scores_arr = [fscore.stack() for fscore in feature_scores_arr]
    return unary_scores_arr, feature_scores_arr, useful_steps, predictions
