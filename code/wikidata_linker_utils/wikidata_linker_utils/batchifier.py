import numpy as np
import string
import pickle
from .dataset import TSVDataset, H5Dataset, CombinedDataset
from .generator import prefetch_generator
from .type_interaction_constants import CONSTRAINTS_REQUIRED_PADDING, CONSTRAINTS_CREATED_PADDING, METADATA_DIMS
from .hashing import nested_hash_vals
from os.path import join, dirname, exists
from os import makedirs
import hashlib
import base64


class BatchifierStatsCollector(object):
    def __init__(self):
        self.reset()

    def log(self, key, value):
        self._collected += 1
        self._stats[key] = value

    def log_mean(self, key, value):
        if key in self._stats:
            self._stats_sum[key] += value
            self._stats_tot[key] += 1
        else:
            self._stats_sum[key] = value
            self._stats_tot[key] = 1
        self._stats[key] = self._stats_sum[key] / self._stats_tot[key]

    def has_stats(self):
        return self._collected > 0

    def dump(self):
        stats = self._stats
        self.reset()
        return stats

    def reset(self):
        self._collected = 0
        self._stats = {}
        self._stats_sum = {}
        self._stats_tot = {}



def word_dropout(inputs, rng, keep_prob):
    inputs_ndim = inputs.ndim
    mask_shape = [len(inputs)] + [1] * (inputs_ndim - 1)
    return (
        inputs *
        (
            rng.random_sample(size=mask_shape) <
            keep_prob
        )
    ).astype(inputs.dtype)


def extract_feat(feat):
    if feat["type"] == "word":
        if feat.get("lowercase", False):
            return lambda x: x.lower()
        return lambda x: x
    elif feat["type"] == "suffix":
        length = feat["length"]
        if feat.get("lowercase", False):
            return lambda x: x[-length:].lower()
        return lambda x: x[-length:]
    elif feat["type"] == "prefix":
        length = feat["length"]
        if feat.get("lowercase", False):
            return lambda x: x[:length].lower()
        return lambda x: x[:length]
    elif feat["type"] == "digit":
        return lambda x: x.isdigit()
    elif feat["type"] == "punctuation_count":
        return lambda x: sum(c in string.punctuation for c in x)
    elif feat["type"] == "uppercase":
        return lambda x: len(x) > 0 and x[0].isupper()
    elif feat["type"] == "character-conv":
        max_size = feat["max_word_length"]

        def extract(x):
            x_bytes = x.encode("utf-8")
            if len(x_bytes) > max_size:
                return np.concatenate(
                    [
                        [255],
                        list(x_bytes[:max_size]),
                        [256]
                    ]
                )
            else:
                return np.concatenate(
                    [
                        [255],
                        list(x_bytes),
                        [256],
                        -np.ones(max_size - len(x_bytes), dtype=np.int32),
                    ]
                )
        return extract
    elif feat["type"] == "bio":
        return lambda x, y: -1 if y[0] is None else y[0].anchor_idx
    else:
        raise ValueError("unknown feature %r." % (feat,))


def extract_word_keep_prob(feat):
    return feat.get("word_keep_prob", 0.85)


def extract_case_keep_prob(feat):
    return feat.get("case_keep_prob", 0.95)


def extract_s_keep_prob(feat):
    return feat.get("s_keep_prob", 0.95)


def apply_case_s_keep_prob(feat, rng, keep_case, keep_s):
    if len(feat) == 0:
        return feat
    if keep_case < 1 and feat[0].isupper() and rng.random_sample() >= keep_case:
        feat = feat.lower()
    if keep_s < 1 and feat.endswith("s") and rng.random_sample() >= keep_s:
        feat = feat[:-1]
    return feat


def requires_character_convolution(feat):
    return feat["type"] in {"character-conv"}


def requires_vocab(feat):
    return feat["type"] in {"word", "suffix", "prefix"}


def requires_label(feat):
    return feat["type"] == "bio"


def feature_npdtype(feat):
    if requires_vocab(feat):
        return np.int32
    elif feat["type"] in {"digit", "punctuation_count", "uppercase"}:
        return np.float32
    elif requires_character_convolution(feat):
        return np.int32
    elif feat["type"] == "bio":
        return np.int32
    else:
        raise ValueError("unknown feature %r." % (feat,))


def get_vocabs(dataset, extractors, max_vocabs, extra_words=None):
    word_vocab = dataset.get_word_vocab()
    index2words = [[] for i in range(len(max_vocabs))]
    occurrences = [{} for i in range(len(max_vocabs))]
    for extractor, max_vocab, index2word, occurrence in zip(extractors, max_vocabs,
                                                            index2words, occurrences):
        for word, count in word_vocab.items():
            el = extractor(word)
            if el not in occurrence:
                index2word.append(el)
                occurrence[el] = count
            else:
                occurrence[el] += count
    if extra_words is not None:
        for word in extra_words:
            for occurrence in occurrences:
                if word in occurrence:
                    del occurrence[word]
    index2words = [
        sorted(index2word, key=lambda x: occurrence[x], reverse=True)
        for index2word, occurrence in zip(index2words, occurrences)
    ]
    index2words = [
        index2word[:max_vocab] if max_vocab > 0 else index2word
        for index2word, max_vocab in zip(index2words, max_vocabs)
    ]
    if extra_words is not None:
        index2words = [
            extra_words + index2word for index2word in index2words
        ]
    return index2words


def unsorted_unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


# def projection_augmentation(featurization, classifier, n):
#     return np.random.choice(len(classifier.classes), size=n)


# def variable_length_projection_augmentation(featurization, classifier, n):
#     counts = np.random.randint(0, 3, size=n)
#     return [np.random.choice(len(classifier.classes), size=count) for count in counts]


# def float_augmentation(featurization, classifier, n):
#     raise NotImplementedError("not ready yet")


# SCENARIO_FEATURIZATION_TYPE_2_AUGMENTATION = {
#     "projection": projection_augmentation,
#     "variable_length_projection": variable_length_projection_augmentation,
#     "float": float_augmentation,
# }


# def fake_var_assignments(classifiers, objective_classifications, n):
#     new_var_assignments = []
#     for classifier, featurization in zip(classifiers, objective_classifications):
#         new_var_assignments.append(SCENARIO_FEATURIZATION_TYPE_2_AUGMENTATION[featurization["type"]](
#             featurization, classifier, n))
#     return new_var_assignments


def flatten(x):
    return [v for group in x for v in group]


def unique_examples(examples):
    # grab all candidates:
    all_indices = [idx for ex in examples
                   for ex_step in ex if ex_step is not None
                   for idx in ex_step.indices]
    all_counts = [count for ex in examples
                  for ex_step in ex if ex_step is not None
                  for count in ex_step.counts]
    all_denominator_counts = np.zeros(len(all_counts), dtype=np.int32)
    all_repeat_counts = np.zeros(len(all_counts), dtype=np.int32)
    all_pos_denominator_count = None
    idx = 0
    for ex in examples:
        anchor_visited = set()
        candidate_counts = {}
        for ex_step in ex:
            if ex_step is not None and ex_step.anchor_idx not in anchor_visited:
                anchor_visited.add(ex_step.anchor_idx)
                for step_candidate in ex_step.indices:
                    if step_candidate not in candidate_counts:
                        candidate_counts[step_candidate] = 1
                    else:
                        candidate_counts[step_candidate] += 1
        for ex_step in ex:
            if ex_step is not None:
                all_denominator_counts[idx:idx + len(ex_step.counts)] = ex_step.counts.sum()
                all_repeat_counts[idx:idx + len(ex_step.counts)] = [candidate_counts.get(step_candidate, 0) for step_candidate in ex_step.indices]
                if ex_step.pos_counts is not None and all_pos_denominator_count is None:
                    # pos_counts is candidates x counts (e.g. anchor, noun -> entity A, anchor, noun -> entity B in first column,
                    # and anchor, adj -> entity A in 2nd column)
                    all_pos_denominator_count = np.zeros((len(all_counts), ex_step.pos_counts.shape[1]), dtype=np.int32)
                if ex_step.pos_counts is not None:
                    # get pos denominators for each pos group (e.g. total noun uses, or total adj uses, etc.)
                    all_pos_denominator_count[idx:idx + len(ex_step.counts)] = ex_step.pos_counts.sum(axis=0)
                idx += len(ex_step.counts)

    all_entity_counts = [count for ex in examples
                         for ex_step in ex if ex_step is not None
                         for count in ex_step.entity_counts]
    # obtain all possible entity ids:
    example_2_data = np.stack([all_indices,
                               all_counts,
                               all_denominator_counts,
                               all_entity_counts,
                               all_repeat_counts], axis=-1)
    pos_start_dim = example_2_data.shape[1]
    if all_pos_denominator_count is not None:
        # add all pos groups to this:
        all_pos_counts = [count for ex in examples
                          for ex_step in ex if ex_step is not None
                          for count in ex_step.pos_counts]
        all_pos_entity_counts = [count for ex in examples
                                 for ex_step in ex if ex_step is not None
                                 for count in ex_step.pos_entity_counts]
        pos_length = all_pos_denominator_count.shape[1]
        example_2_data = np.concatenate([
            example_2_data,
            all_pos_counts,
            all_pos_denominator_count,
            np.array(all_pos_entity_counts)[:, None]], axis=-1)
    else:
        pos_length = 4
        example_2_data = np.concatenate([
            example_2_data,
            np.zeros((len(example_2_data), pos_length * 2 + 1), dtype=np.int32)], axis=-1)

    if example_2_data.dtype != np.int32:
        example_2_data = example_2_data.astype(np.int32)
    # use the merged reps instead of the original reps
    merged_rep = np.zeros(len(example_2_data), dtype=np.int32)
    # reduce the list to only unique cases:
    if len(example_2_data) > 0:
        db = {}
        unique_elements = np.unique(example_2_data, axis=0)
        for index, el in enumerate(unique_elements):
            db[tuple(el)] = index
        for index, el in enumerate(example_2_data):
            merged_rep[index] = db[tuple(el)]
    else:
        unique_elements = np.zeros((0, example_2_data.shape[1]), dtype=np.int32)
    return (# ND (candidates x count measurements + indices
            unique_elements,
            # ?
            merged_rep,
            # ND (candidates x pos tags)
            unique_elements[:, pos_start_dim:pos_start_dim + pos_length] if pos_length > 0 else None,
            # ND (candidates x pos tags)
            unique_elements[:, pos_start_dim + pos_length:pos_start_dim + 2 * pos_length] if pos_length > 0 else None,
            # 1D
            unique_elements[:, pos_start_dim + 2 * pos_length] if pos_length > 0 else None)


def scenario_build_labels(examples, label2index, train, timesteps, training_scenarios, negative_sample_missing,
                          objective_classifications, multiword_pool, raw_examples, frequency_weights,
                          batchifier_stats_collector):
    """
    Construct the dynamic softmax for scenario classification.
    """
    if multiword_pool is not None:
        # combine labels across steps if the successive anchor_idx are the same
        new_examples = []
        segment_id = 0
        old_timesteps = timesteps
        segment_ids = np.zeros((old_timesteps, len(examples)), dtype=np.int32)
        batch_segment_locations = [[] for _ in range(len(examples))]
        merge_disconnected = False

        for ex_idx in range(len(examples)):
            ex = examples[ex_idx]
            new_ex = []
            prev_anchor_idx = None
            current_segment_id = None
            previous_segments = {}

            # compute connected triggers
            for step_idx, ex_step in enumerate(ex):
                if ex_step is None:
                    segment_ids[step_idx, ex_idx] = 0
                    prev_anchor_idx = None
                else:
                    if prev_anchor_idx is None:
                        if merge_disconnected and ex_step.anchor_idx in previous_segments:
                            current_segment_id = previous_segments[ex_step.anchor_idx]
                            assert current_segment_id is not None
                            new_ex.append(ex_step)
                            batch_segment_locations[ex_idx].append(current_segment_id)
                        else:
                            segment_id += 1
                            current_segment_id = segment_id
                            new_ex.append(ex_step)
                            batch_segment_locations[ex_idx].append(current_segment_id)
                        # for idx in ex_step.indices:
                        #     previous_segments[idx] = current_segment_id
                        previous_segments[ex_step.anchor_idx] = current_segment_id
                        
                        prev_anchor_idx = ex_step.anchor_idx
                    elif ex_step.anchor_idx == prev_anchor_idx:
                        pass
                    else:
                        # change in trigger
                        if merge_disconnected and ex_step.anchor_idx in previous_segments:
                            current_segment_id = previous_segments[ex_step.anchor_idx]
                            assert current_segment_id is not None
                            new_ex.append(ex_step)
                            batch_segment_locations[ex_idx].append(current_segment_id)
                        else:
                            segment_id += 1
                            current_segment_id = segment_id
                            new_ex.append(ex_step)
                            batch_segment_locations[ex_idx].append(current_segment_id)
                        previous_segments[ex_step.anchor_idx] = current_segment_id
                        prev_anchor_idx = ex_step.anchor_idx
                        
                    segment_ids[step_idx, ex_idx] = current_segment_id
            new_examples.append(new_ex)
        timesteps = max(map(len, new_examples))
        examples = new_examples
        segment_locations = np.zeros((timesteps, len(examples)), dtype=np.int32)
        for ex_idx in range(len(examples)):
            segment_locations[:len(batch_segment_locations[ex_idx]), ex_idx] = batch_segment_locations[ex_idx]
    else:
        segment_ids = None
        segment_locations = None

    classifiers = label2index["classifiers"]
    num_entities = label2index["num_entities"]
    num_scenarios = 0
    max_constraints = 0
    max_constraints_required = 0
    weights = np.zeros((timesteps, len(examples)), dtype=np.float32)
    for ex_idx in range(len(examples)):
        ex = examples[ex_idx]
        for step_idx, ex_step in enumerate(ex):
            if ex_step is not None:
                if len(ex_step.indices) == 0:
                    # no candidates left here.
                    examples[ex_idx][step_idx] = None
                    continue
                default_meaning = False
                if ex_step.label is not None:
                    # put ground truth label first.
                    try:
                        true_label_index = np.where(ex_step.indices == ex_step.label)[0][0]
                    except IndexError as e:
                        true_label_index = -1
                    if true_label_index == -1:
                        # remove this example
                        examples[ex_idx][step_idx] = None
                    else:
                        default_meaning = np.argmax(ex_step.counts) == true_label_index
                        ex_step.place_index_at_start(true_label_index)
                if train:
                    assert ex_step.label is not None, "a label is needed for training."
                    if len(ex_step.indices) < training_scenarios and negative_sample_missing:
                        # add some fake examples to this trigger...
                        if ex_step.uniform_sample:
                            new_indices = np.random.randint(num_entities, size=training_scenarios - len(ex_step.indices)).astype(np.int32)
                            new_counts = np.random.randint(min(ex_step.counts), max(ex_step.counts) + 1, size=training_scenarios - len(ex_step.indices))
                            new_entity_counts = np.maximum(new_counts, np.random.randint(min(ex_step.entity_counts), max(ex_step.entity_counts) + 1,
                                                                                         size=training_scenarios - len(ex_step.indices)).astype(np.int32))
                        else:
                            new_indices = np.searchsorted(
                                ex_step.index2incoming_cumprob,
                                np.random.random(size=training_scenarios - len(ex_step.indices))).astype(np.int32)
                            new_entity_counts = ex_step.index2incoming_count[new_indices]
                            proportion = np.random.uniform(0.1, 1.0, size=training_scenarios - len(ex_step.indices))
                            new_counts = (proportion * new_entity_counts).astype(np.int32)
                        if ex_step.pos_counts is not None:
                            new_pos_counts = np.random.randint(ex_step.pos_counts.min(axis=0), np.ceil((ex_step.pos_counts.max(axis=0) * 1.05 + 1)),
                                                               size=(training_scenarios - len(ex_step.indices), ex_step.pos_counts.shape[1]))
                            new_pos_entity_counts = np.maximum(new_pos_counts.sum(axis=-1), np.random.randint(ex_step.pos_entity_counts.min(), ex_step.pos_entity_counts.max() + 1,
                                                                                                              size=training_scenarios - len(ex_step.indices)).astype(np.int32))
                        else:
                            new_pos_counts = None
                            new_pos_entity_counts = None
                        ex_step.extend_indices(
                            indices=new_indices,
                            counts=new_counts,
                            entity_counts=new_entity_counts,
                            pos_counts=new_pos_counts,
                            pos_entity_counts=new_pos_entity_counts,
                            created_constraints=None,
                            constraint_required=None,
                            candidate_metadata=None
                        )
                    elif len(ex_step.indices) > training_scenarios:
                        # remove some examples:
                        transition_probs = np.array(ex_step.counts[1:]) / np.sum(ex_step.counts[1:])
                        alternatives = np.concatenate([[0], np.random.choice(len(ex_step.indices) - 1,
                                                                             replace=False,
                                                                             p=transition_probs,
                                                                             size=training_scenarios - 1) + 1])
                        ex_step.reorder_indices(alternatives)
                num_scenarios = max(num_scenarios, len(ex_step.indices))
                if ex_step.created_constraints is not None:
                    max_constraints = max(max_constraints, ex_step.created_constraints.shape[1])
                if ex_step.constraint_required is not None:
                    max_constraints_required = max(max_constraints_required, ex_step.constraint_required.shape[1])
                weights[step_idx, ex_idx] = frequency_weights[0 if default_meaning else 1]

    unique_elements, example2unique, pos_numerators, pos_denominators, pos_incoming_counts = unique_examples(examples)
    # featurize the unique entities
    featurized_entities = []
    for featurization, classifier in zip(objective_classifications, classifiers):
        if featurization["type"] == "projection":
            try:
                featurized_entities.append(classifier.classify(unique_elements[:, 0]))
            except IndexError as e:
                print(unique_elements)
                print(type(unique_elements))
                raise e

        elif featurization["type"] == "variable_length_projection":
            classifications = classifier.classify(unique_elements[:, 0])
            featurized_entities.append(flatten(classifications))
            featurized_entities.append([len(c) for c in classifications])
        elif featurization["type"] == "float":
            classifications = classifier.classify(unique_elements[:, 0])
            featurized_entities.append(classifications)
        elif featurization["type"] == "wikipedia_probs":
            classifications = np.stack([
                # compute prob of trigger -> entity / Sum trigger -> entities
                unique_elements[:, 1] / np.maximum(unique_elements[:, 2], 1),
                # compute prob of entity -> trigger / Sum entity -> triggers
                unique_elements[:, 1] / np.maximum(unique_elements[:, 3], 1)
            ], axis=-1)

            power = featurization.get("power", 1.0)
            if power != 1:
                classifications = classifications ** power

            if featurization.get("post_process", None):
                classifications = np.minimum(featurization["post_process"]["maxval"], classifications)
                classifications = np.maximum(featurization["post_process"]["minval"], classifications)
            featurized_entities.append(classifications)
        elif featurization["type"] == "wikipedia_pos_probs":
            # sum across all tags possible for the denominator (e.g. total entities pointed at by this trigger)
            # this is num(trigger)
            num_trigger = np.maximum(pos_denominators[:, :].sum(axis=-1), 1)
            try:
                to_stack = (
                    # prob(entity|trigger, pos_tag) = num(entity, trigger, pos_tag) / num(trigger, pos_tag)
                    [pos_numerators[:, i] / np.maximum(pos_denominators[:, i], 1) for i in range(pos_numerators.shape[1])] +
                    # prob(trigger,pos_tag|entity) = num(trigger,pos_tag) / num(entity)
                    [pos_numerators[:, i] / np.maximum(pos_incoming_counts, 1) for i in range(pos_numerators.shape[1])] +
                    # pros(pos_tag|trigger) = num(trigger, pos_tag) / num(trigger)
                    [pos_denominators[:, i] / num_trigger for i in range(pos_numerators.shape[1])] +
                    [
                        # compute entity popularity:
                        np.log1p(unique_elements[:, 3] ** 0.2),
                    ]
                )
                classifications = np.stack(to_stack, axis=-1)
            except:
                import ipdb; ipdb.set_trace()
            featurized_entities.append(classifications)
        elif featurization["type"] == "repeat_candidate":
            classifications = (unique_elements[:, 4:5] > 1).astype(np.float32)
            featurized_entities.append(classifications)
        elif featurization["type"] == "word_predict":
            window_size = featurization["window_size"]
            candidate_words = np.zeros((len(unique_elements), 2 * window_size + 1), dtype=np.int32)
            candidate_words.fill(-1)
            word_window_labels = np.zeros((len(unique_elements),), dtype=np.bool)
            candidate_idx = 0
            for example_idx, ex in enumerate(examples):
                for tstep, ex_step in enumerate(ex):
                    if ex_step is not None:
                        tstep_candidates = example2unique[candidate_idx:candidate_idx + len(ex_step.indices)]
                        # index into flattened input ids (assuming batch-major)
                        window_words = example_idx * timesteps + np.arange(max(0, tstep - window_size), min(timesteps, tstep + window_size + 1))
                        candidate_words[tstep_candidates, :len(window_words)] = window_words
                        if train:
                            # the first candidate has a positive word window, everybody else is the wrong candidate...
                            word_window_labels[tstep_candidates[0]] = True
                        candidate_idx += len(ex_step.indices)
            featurized_entities.append(candidate_words)
            featurized_entities.append(word_window_labels)
        elif featurization["type"] == "predicted":
            pass
        else:
            raise ValueError("unknown scenario featurization type \"{}\".".format(featurization["type"]))

    if batchifier_stats_collector is not None:
        batchifier_stats_collector.log_mean("training/scenario_featurized_entities", len(unique_elements))
    # for each timestep we will grab the new index of the featurized entity after
    # it was reduced by uniqueness
    labels = np.zeros((len(examples), timesteps, num_scenarios), dtype=np.int32)
    labels_mask = np.zeros((len(examples), timesteps, num_scenarios), dtype=np.bool)
    candidate_ids = np.zeros((len(examples), timesteps, num_scenarios), dtype=np.int32)
    supervised_time_major = np.zeros((timesteps, len(examples)), dtype=np.bool)
    candidate_ids.fill(-1)
    candidate_ids_flat = unique_elements[:, 0]
    candidates_constraints = np.zeros((len(examples), timesteps, num_scenarios, max_constraints), dtype=np.int32)
    candidates_constraints.fill(CONSTRAINTS_CREATED_PADDING)
    candidates_constraints_required = np.zeros((len(examples), timesteps, num_scenarios, max_constraints_required), dtype=np.int32)
    candidates_constraints_required.fill(CONSTRAINTS_REQUIRED_PADDING)
    candidates_metadata = np.zeros((len(examples), timesteps, num_scenarios, METADATA_DIMS), dtype=np.int32)
    candidate_idx = 0
    for example_idx, ex in enumerate(examples):
        for tstep, ex_step in enumerate(ex):
            if ex_step is not None:
                tstep_candidates = example2unique[candidate_idx:candidate_idx + len(ex_step.indices)]
                try:
                    labels[example_idx, tstep, :len(tstep_candidates)] = tstep_candidates
                except Exception as e:
                    print(labels.shape)
                    print(tstep_candidates)
                    raise
                labels_mask[example_idx, tstep, :len(tstep_candidates)] = True
                candidate_ids[example_idx, tstep, :len(tstep_candidates)] = unique_elements[tstep_candidates, 0]
                if ex_step.constraint_required is not None:
                    candidates_constraints_required[example_idx, tstep, :len(tstep_candidates), :ex_step.constraint_required.shape[1]] = ex_step.constraint_required
                if ex_step.created_constraints is not None:
                    candidates_constraints[example_idx, tstep, :len(tstep_candidates), :ex_step.created_constraints.shape[1]] = ex_step.created_constraints
                if ex_step.candidate_metadata is not None:
                    candidates_metadata[example_idx, tstep, :len(tstep_candidates), :] = ex_step.candidate_metadata
                candidate_idx += len(ex_step.indices)
                supervised_time_major[tstep, example_idx] = ex_step.label is not None
    return labels, labels_mask, [supervised_time_major, candidate_ids, candidate_ids_flat, weights, candidates_constraints, candidates_constraints_required, candidates_metadata] + ([segment_ids, segment_locations] if segment_ids is not None else []) + featurized_entities


HASH_IGNORED_KEYS = {"word_keep_prob", "case_keep_prob", "s_keep_prob"}


def obtain_feature_vocabs_hash(features, dataset, extra_words):
    feat_hash = []
    m = hashlib.sha256()
    nested_hash_vals(m, features, HASH_IGNORED_KEYS)
    dataset.hash(m)
    for word in extra_words:
        m.update(word.encode("utf-8"))
    return base64.b64encode(m.digest()).decode("ascii")


def get_feature_vocabs(features, dataset, extra_words=None, cache=True):
    feature_vocabs_hash = obtain_feature_vocabs_hash(features, dataset, extra_words).replace("/", "")
    hash_file = join("/tmp", "deeptype2", "feature_vocabs", feature_vocabs_hash + ".pkl")
    if cache and exists(hash_file):
        print("Found saved feature set, loading...")
        with open(hash_file, "rb") as fin:
            return pickle.load(fin)
    else:
        print("No saved feature set under {}, building...".format(hash_file))
        out, feats_needing_vocab, feats_with_vocabs, vocabs = [], [], [], []
        if hasattr(dataset, "set_ignore_y"):
            dataset.set_ignore_y(True)
        try:
            for feat in features:
                if requires_vocab(feat):
                    if feat.get("path") is not None:
                        with open(feat["path"], "rt") as fin:
                            index2word = fin.read().splitlines()
                        if extra_words is not None:
                            for word in extra_words:
                                try:
                                    index2word.remove(word)
                                except ValueError:
                                    pass
                        if feat.get("max_vocab", -1) > 0:
                            index2word = index2word[:feat["max_vocab"]]
                        if extra_words is not None:
                            index2word = extra_words + index2word
                        feats_with_vocabs.append(index2word)
                    else:
                        feats_needing_vocab.append(feat)
            if len(feats_needing_vocab) > 0:
                extractors = tuple(
                    [extract_feat(feat) for feat in feats_needing_vocab]
                )
                vocabs = get_vocabs(
                    dataset, extractors,
                    max_vocabs=[feat.get("max_vocab", -1) for feat in feats_needing_vocab],
                    extra_words=extra_words
                )
            vocab_feature_idx = 0
            preexisting_vocab_feature_idx = 0
            for feat in features:
                if requires_vocab(feat):
                    if feat.get("path") is not None:
                        out.append(feats_with_vocabs[preexisting_vocab_feature_idx])
                        preexisting_vocab_feature_idx += 1
                    else:
                        out.append(vocabs[vocab_feature_idx])
                        vocab_feature_idx += 1
                else:
                    out.append(None)
        finally:
            if hasattr(dataset, "set_ignore_y"):
                dataset.set_ignore_y(False)
        print("Saving feature set")
        makedirs(dirname(hash_file), exist_ok=True)
        with open(hash_file, "wb") as fout:
            pickle.dump(out, fout)
    return out


def pad_arrays(arrays, padding, make_time_major=False):
    out_ndim = arrays[0].ndim + 1
    out_shape = [0] * out_ndim
    out_shape[0] = len(arrays)
    for arr in arrays:
        for dim_idx in range(arr.ndim):
            out_shape[1 + dim_idx] = max(out_shape[1 + dim_idx], arr.shape[dim_idx])
    out = np.empty(out_shape, dtype=arrays[0].dtype)
    out.fill(padding)
    for arr_idx, array in enumerate(arrays):
        arr_slice = [arr_idx]
        for dim_idx in range(arr.ndim):
            arr_slice.append(slice(0, array.shape[dim_idx]))
        arr_slice = tuple(arr_slice)
        out[arr_slice] = array

    if make_time_major and out.ndim > 1:
        out = out.swapaxes(0, 1)
    return out


def build_objective_mask(label_sequence, objective_idx, objective_type):
    if objective_type == 'crf':
        if len(label_sequence) == 0 or label_sequence[0][objective_idx] is None:
            return np.array(False, dtype=np.bool)
        else:
            return np.array(True, dtype=np.bool)
    elif objective_type == 'softmax':
        return np.array(
            [w[objective_idx] is not None for w in label_sequence], dtype=np.bool
        )
    else:
        raise ValueError(
            "unknown objective type %r." % (objective_type,)
        )


def allocate_shrunk_batches(max_length, batch_size, lengths):
    typical_indices = max_length * batch_size
    i = 0
    ranges = []
    while i < len(lengths):
        j = i + 1
        current_batch_size = 1
        longest_ex = lengths[j - 1]
        while j < len(lengths) and j - i < batch_size:
            # can grow?
            new_batch_size = current_batch_size + 1
            new_j = j + 1
            if max(longest_ex, lengths[new_j - 1]) * new_batch_size < typical_indices:
                j = new_j
                longest_ex = max(longest_ex, lengths[new_j - 1])
                current_batch_size = new_batch_size
            else:
                break
        ranges.append((i, j))
        i = j
    return ranges


def convert_label_to_index(label, label2index):
    if label is None:
        return 0
    if isinstance(label, str):
        return label2index[label]
    return label


class Batchifier(object):
    def __init__(self, *, rng, feature_word2index, objective_types, objective_classifications,
                 multiword_pool, training_scenarios, negative_sample_missing, frequency_weights, label2index,
                 fused, sequence_lengths, labels, labels_mask,
                 input_placeholders, features, dataset, batch_size, train,
                 autoresize=True, max_length=100, batchifier_stats_collector=None, blank_triggers=False):
        assert(batch_size > 0), "batch size must be strictly positive (got %r)." % (batch_size,)
        # dictionaries, strings defined by model:
        self.objective_types = objective_types
        self.objective_is_scenario = [objtype == "scenario" for objtype in objective_types]
        self.objective_classifications = objective_classifications
        self.multiword_pool = multiword_pool
        self.training_scenarios = training_scenarios
        self.negative_sample_missing = negative_sample_missing
        self.frequency_weights = frequency_weights
        self.label2index = label2index
        self.feature_word2index = feature_word2index
        self.rng = rng
        self.fused = fused

        # tf placeholders:
        self.sequence_lengths = sequence_lengths
        self.labels = labels
        self.labels_mask = labels_mask
        self.input_placeholders = input_placeholders

        self.dataset = dataset
        self.batch_size = batch_size
        self.train = train
        self.blank_triggers = blank_triggers

        self.dataset_is_lazy = isinstance(dataset, (TSVDataset, H5Dataset, CombinedDataset))
        self.autoresize = autoresize
        self.max_length = max_length
        self.batchifier_stats_collector = batchifier_stats_collector

        indices = np.arange(len(dataset))

        if train:
            if self.dataset_is_lazy:
                dataset.set_rng(rng)
                dataset.set_randomize(True)
            elif isinstance(dataset, list):
                rng.shuffle(indices)
        self.batch_indices = []
        if self.autoresize and not self.dataset_is_lazy:
            ranges = allocate_shrunk_batches(
                max_length=self.max_length,
                batch_size=self.batch_size,
                lengths=[len(dataset[indices[i]][0]) for i in range(len(indices))]
            )
            for i, j in ranges:
                self.batch_indices.append(indices[i:j])
        else:
            for i in range(0, len(indices), self.batch_size):
                self.batch_indices.append(indices[i:i + self.batch_size])
        self.extractors = [
            (extract_feat(feat), requires_vocab(feat), requires_label(feat), feature_npdtype(feat),
             extract_word_keep_prob(feat), extract_case_keep_prob(feat), extract_s_keep_prob(feat))
            for feat in features
        ]

    def generate_batch(self, examples):
        X = [[] for i in range(len(self.extractors))]
        Y = []
        Y_mask = []
        # regular per-timestep objectives
        for ex, label in examples:
            for idx, (extractor, uses_vocab, uses_label, dtype, keep_word, keep_case, keep_s) in enumerate(self.extractors):
                if self.train and (keep_case < 1 or keep_s < 1):
                    ex = [apply_case_s_keep_prob(w, self.rng, keep_case, keep_s) for w in ex]
                if uses_vocab:
                    word_feats = np.array(
                        [self.feature_word2index[idx].get(extractor(w), 0) for w in ex],
                        dtype=dtype)
                elif uses_label:
                    word_feats = np.array([extractor(w, w_l) for w, w_l in zip(ex, label)], dtype=dtype)
                else:
                    word_feats = np.array([extractor(w) for w in ex], dtype=dtype)
                if self.train and keep_word < 1:
                    word_feats = word_dropout(word_feats, self.rng, keep_word)
                if self.blank_triggers:
                    word_feats = np.array([f * (l[0] is None) for l, f in zip(label, word_feats)]).astype(word_feats.dtype)
                X[idx].append(word_feats)
            all_objectives = []
            all_objectives_mask = []
            for objective_idx, label2index in enumerate(self.label2index):
                if self.objective_is_scenario[objective_idx]:
                    all_objectives.append(None)
                    all_objectives_mask.append(None)
                else:
                    all_objectives.append(np.array([convert_label_to_index(w[objective_idx], label2index)
                                                    for w in label], dtype=np.int32))
                    all_objectives_mask.append(build_objective_mask(label, objective_idx,
                                                                    self.objective_types[objective_idx]))
            Y.append(all_objectives)
            Y_mask.append(all_objectives_mask)

        Y = [pad_arrays([row[objective_idx] for row in Y], 0, make_time_major=True)
             if not self.objective_is_scenario[objective_idx] else None
             for objective_idx in range(len(self.objective_types))]
        Y_mask = [pad_arrays([row[objective_idx] for row in Y_mask], 0.0, make_time_major=True)
                  if not self.objective_is_scenario[objective_idx] else None
                  for objective_idx, objective_type in enumerate(self.objective_types)]
        sequence_lengths = np.array([len(x) for x in X[0]], dtype=np.int32)
        X = [pad_arrays(x, -1, make_time_major=True) for x in X]
        num_default_inputs = len(X)
        num_label_inputs = 0
        # support scenario objectives
        for objective_idx, label2index in enumerate(self.label2index):
            if self.objective_is_scenario[objective_idx]:
                targets, mask, label_input = scenario_build_labels(
                    [[l[objective_idx] for l in label] for _, label in examples],
                    label2index,
                    train=self.train,
                    timesteps=X[0].shape[0],
                    objective_classifications=self.objective_classifications[objective_idx],
                    multiword_pool=self.multiword_pool[objective_idx],
                    training_scenarios=self.training_scenarios[objective_idx],
                    negative_sample_missing=self.negative_sample_missing[objective_idx],
                    raw_examples=examples,
                    frequency_weights=self.frequency_weights[objective_idx],
                    batchifier_stats_collector=self.batchifier_stats_collector)
                Y[objective_idx] = targets.swapaxes(0, 1)
                Y_mask[objective_idx] = mask.swapaxes(0, 1)
                X.extend(label_input)
                num_label_inputs += len(label_input)
        feed_dict = {self.sequence_lengths: sequence_lengths}
        if self.fused:
            feed_dict[self.labels[0]] = np.stack(Y, axis=-1)
            feed_dict[self.labels_mask[0]] = np.stack(Y_mask, axis=-1)
        else:
            for y, placeholder in zip(Y, self.labels):
                feed_dict[placeholder] = y
            for y_mask, placeholder in zip(Y_mask, self.labels_mask):
                feed_dict[placeholder] = y_mask
        assert len(self.input_placeholders) == len(X), \
            f"expected same number of input_placeholders ({len(self.input_placeholders)}) as inputs ({len(X)}: features {num_default_inputs}, label features {num_label_inputs})."
        for idx, x in enumerate(X):
            feed_dict[self.input_placeholders[idx]] = x
        return feed_dict

    def as_list(self):
        return list(self.iter_batches())

    def iter_batches(self, pbar=None):
        gen = range(len(self.batch_indices))
        if pbar is not None:
            pbar.max_value = len(self.batch_indices)
            pbar.value = 0
            gen = pbar(gen)
        if self.autoresize and self.dataset_is_lazy:
            for batch_idx, idx in enumerate(gen):
                examples = [self.dataset[ex] for ex in self.batch_indices[idx]]
                ranges = allocate_shrunk_batches(
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    lengths=[len(ex[0]) for ex in examples]
                )
                for i, j in ranges:
                    yield self.generate_batch(examples[i:j])
        else:
            for idx in gen:
                yield self.generate_batch(
                    [self.dataset[ex] for ex in self.batch_indices[idx]]
                )


def batch_worker(rng,
                 features,
                 feature_word2index,
                 objective_types,
                 objective_classifications,
                 multiword_pool,
                 training_scenarios,
                 negative_sample_missing,
                 frequency_weights,
                 label2index,
                 fused,
                 sequence_lengths,
                 labels,
                 labels_mask,
                 input_placeholders,
                 autoresize,
                 train,
                 batch_size,
                 max_length,
                 dataset,
                 pbar,
                 batch_queue,
                 death_event):
    batchifier = Batchifier(rng=rng,
                            features=features,
                            feature_word2index=feature_word2index,
                            training_scenarios=training_scenarios,
                            negative_sample_missing=negative_sample_missing,
                            frequency_weights=frequency_weights,
                            objective_types=objective_types,
                            objective_classifications=objective_classifications,
                            multiword_pool=multiword_pool,
                            label2index=label2index,
                            fused=fused,
                            sequence_lengths=sequence_lengths,
                            labels=labels,
                            labels_mask=labels_mask,
                            input_placeholders=input_placeholders,
                            autoresize=autoresize,
                            train=train,
                            batch_size=batch_size,
                            max_length=max_length,
                            dataset=dataset)
    for batch in batchifier.iter_batches(pbar=pbar):
        if death_event.is_set():
            break
        batch_queue.put(batch)
    if not death_event.is_set():
        batch_queue.put(None)


def range_size(start, size):
    return [i for i in range(start, start + size)]


class ProcessHolder(object):
    def __init__(self, process, death_event, batch_queue):
        self.process = process
        self.batch_queue = batch_queue
        self.death_event = death_event

    def close(self):
        self.death_event.set()
        try:
            self.batch_queue.close()
            while True:
                self.batch_queue.get_nowait()
        except Exception:
            pass
        self.process.terminate()
        self.process.join()

    def __del__(self):
        self.close()


def batch_2_feed_dict(model, tensorflow_placeholders, batch):
    assert len(batch) == len(tensorflow_placeholders), "missing batch items to feed placeholders."
    feed_dict = {}
    for idx, key in enumerate(tensorflow_placeholders):
        feed_dict[key] = batch[idx]
    model.postprocess_feed_dict(feed_dict)
    return feed_dict


def iter_batches_single_threaded(model,
                                 dataset,
                                 batch_size,
                                 train,
                                 blank_triggers=False,
                                 autoresize=True,
                                 max_length=100,
                                 batchifier_stats_collector=None,
                                 pbar=None):
    tensorflow_placeholders = [model.sequence_lengths] + model.labels + model.labels_mask + model.input_placeholders
    labels_start = 1
    labels_mask_start = labels_start + len(model.labels)
    placeholder_start = labels_mask_start + len(model.labels_mask)
    batchifier = Batchifier(rng=model.rng,
                            features=model.features,
                            feature_word2index=model.feature_word2index,
                            objective_types=[obj["type"] for obj in model.objectives],
                            objective_classifications=[obj.get("classifications") for obj in model.objectives],
                            training_scenarios=[obj["training_scenarios"] if obj["type"] == "scenario" else None
                                                for obj in model.objectives],
                            negative_sample_missing=[obj.get("negative_sample_missing", True)
                                                     if obj["type"] == "scenario" else None
                                                     for obj in model.objectives],
                            frequency_weights=[obj.get("frequency_weights", [1, 1]) if obj["type"] == "scenario" else None
                                                for obj in model.objectives],
                            label2index=model.label2index,
                            fused=model.fused,
                            sequence_lengths=0,
                            labels=range_size(labels_start, len(model.labels)),
                            labels_mask=range_size(labels_mask_start, len(model.labels_mask)),
                            input_placeholders=range_size(placeholder_start,
                                                          len(model.input_placeholders)),
                            multiword_pool=[obj.get("multiword_pool", None) for obj in model.objectives],
                            autoresize=autoresize,
                            blank_triggers=blank_triggers,
                            train=train,
                            batch_size=batch_size,
                            max_length=max_length,
                            dataset=dataset,
                            batchifier_stats_collector=batchifier_stats_collector)
    with prefetch_generator(batchifier.iter_batches(pbar=pbar), to_fetch=100) as gen:
        for batch_idx, batch in enumerate(gen):
            yield batch_2_feed_dict(model, tensorflow_placeholders, batch)


def iter_batches(model,
                 dataset,
                 batch_size,
                 train,
                 autoresize=True,
                 max_length=100,
                 pbar=None):
    import multiprocessing
    batch_queue = multiprocessing.Queue(maxsize=10)
    tensorflow_placeholders = [model.sequence_lengths] + model.labels + model.labels_mask + model.input_placeholders
    labels_start = 1
    labels_mask_start = labels_start + len(model.labels)
    placeholder_start = labels_mask_start + len(model.labels_mask)
    death_event = multiprocessing.Event()
    batch_process = ProcessHolder(multiprocessing.Process(
        target=batch_worker,
        daemon=True,
        args=(
            model.rng,
            model.features,
            model.feature_word2index,
            [obj["type"] for obj in model.objectives],
            [obj.get("classifications") for obj in model.objectives],
            [obj.get("multiword_pool", None) for obj in model.objectives],
            [obj.get("training_scenarios", 10)
             if obj["type"] == "scenario" else None
             for obj in model.objectives],
            [obj.get("negative_sample_missing", True)
             if obj["type"] == "scenario" else None
             for obj in model.objectives],
            [obj["frequency_weights"] if obj["type"] == "scenario" else None
             for obj in model.objectives],
            [obj["frequency_weights"] if obj["type"] == "scenario" else None
             for obj in model.objectives],
            model.label2index,
            model.fused,
            0,
            range_size(labels_start, len(model.labels)),
            range_size(labels_mask_start, len(model.labels_mask)),
            range_size(placeholder_start, len(model.input_placeholders)),
            autoresize,
            train,
            batch_size,
            max_length,
            dataset,
            pbar,
            batch_queue,
            death_event
        )
    ), death_event, batch_queue)
    batch_process.process.name = "iter_batches"
    batch_process.process.start()
    while True:
        batch = batch_queue.get()
        if batch is None:
            break
        else:
            yield batch_2_feed_dict(model, tensorflow_placeholders, batch)
        del batch


def get_vocab(dataset, max_vocab=-1, extra_words=None):
    index2word = []
    occurrence = {}
    for el in dataset:
        if el not in occurrence:
            index2word.append(el)
            occurrence[el] = 1
        else:
            occurrence[el] += 1
    index2word = sorted(index2word, key=lambda x: occurrence[x], reverse=True)
    if max_vocab > 0:
        index2word = index2word[:max_vocab]
    if extra_words is not None:
        index2word = extra_words + index2word
    return index2word


def get_objectives(objectives, dataset):
    out = []
    for obj_idx, objective in enumerate(objectives):
        if objective["type"] == "scenario":
            copied = objective.copy()
            for key in ["name", "type", "classifications", "model",
                        "max_scenarios", "training_scenarios", "frequency_weights"]:
                assert key in copied, "expected key {} in objective.".format(key)
            out.append(copied)
        else:
            if "vocab" in objective:
                with open(objective["vocab"], "rt") as fin:
                    vocab = fin.read().splitlines()
            else:
                vocab = get_vocab((w[obj_idx] for _, y in dataset for w in y if w[obj_idx] is not None), -1) if dataset is not None else None

            out.append({"vocab": vocab,
                        "type": objective["type"],
                        "name": objective["name"]})
    return out
