import sys
import pickle
import json
import tensorflow as tf
import numpy as np

from os import makedirs
from os.path import join, exists
from contextlib import contextmanager

from . import tf_saver, tf_metrics, tf_logger
from .embedding import EMBEDDING_CPU_DEVICE, embedding_lookup
from .lstm import build_recurrent
from .convolution import build_convolutions, parse_convolutions, character_convolution
from .transformer import build_transformer, compute_qkv, multihead_attention_internal, multihead_attention_internal_weights
from .batchifier import requires_vocab, requires_character_convolution
from .tf_operations import concat, sparse_softmax_cross_entropy_with_logits, sum_list, extract_shape, lme_pool
from .tf_regularization import maybe_dropout, maybe_replace_by_blank, maybe_batch_replace_by_blank, tile_like
from .lazy_adam_optimizer import LazyAdamOptimizer
from .make_callable import make_callable
from .autoregressive_scenario_decoding import autoregressive_scenario_decoding, scenario_add_interaction_terms, scenario_dynamic_fully_connected_combine
from .submodel_spec_builder import build_submodel_from_spec
from .sequence_model_constants import *
DEFAULT_TYPE_COLLECTION = None
DEFAULT_CLASSIFICATION_HANDLER = None


class StaticPlaceholder(object):
    def __init__(self, name, dtype, collection_name):
        self.name = name
        self.dtype = dtype
        self.collection_name = collection_name

    def declare_and_store(self):
        self.placeholder = tf.placeholder(self.dtype, [], name=self.name)
        tf.add_to_collection(self.collection_name, self.placeholder)

    def retrieve(self):
        return tf.get_collection(self.collection_name)[0] if len(tf.get_collection(self.collection_name)) > 0 else None


@contextmanager
def as_default_type_collection(type_collection):
    global DEFAULT_TYPE_COLLECTION
    DEFAULT_TYPE_COLLECTION = type_collection
    yield


def default_type_collection():
    global DEFAULT_TYPE_COLLECTION
    return DEFAULT_TYPE_COLLECTION


@contextmanager
def as_default_classification_handler(classification_handler):
    global DEFAULT_CLASSIFICATION_HANDLER
    DEFAULT_CLASSIFICATION_HANDLER = classification_handler
    yield


def default_classification_handler():
    global DEFAULT_CLASSIFICATION_HANDLER
    return DEFAULT_CLASSIFICATION_HANDLER


STATIC_PLACEHOLDERS = [
    StaticPlaceholder("is_training", tf.bool, "IS_TRAINING"),
    StaticPlaceholder("greedy_decoding", tf.bool, "GREEDY_DECODING"),
    StaticPlaceholder("beam_width", tf.int32, "BEAM_WIDTH"),
]


def load_scenario_label2index(objective, classifications):
    """
    Hold onto multiple oracle classifiers specifically
    for a scenario objective
    """
    label2index = {"classifiers": [], "vocab": [], "num_entities": classifications.num_entities()}
    for obj in objective["classifications"]:
        assert isinstance(obj, dict), "expected scenario featurization to be a dict, but got a {} (\"{}\").".format(
            type(obj), obj)
        if obj["type"] in ("projection", "variable_length_projection", "float"):
            classifier = classifications.get_classifier(
                obj.get("classification_name", obj["name"]),
                variable_length=obj["type"] == "variable_length_projection")
            label2index["classifiers"].append(classifier)
            label2index["vocab"].append(classifier.classes)
        elif obj["type"] in ("wikipedia_probs", "wikipedia_pos_probs", "word_predict", "predicted", "repeat_candidate"):
            label2index["classifiers"].append(None)
            label2index["vocab"].append(None)
        else:
            raise ValueError("unknown scenario featurization type \"{}\".".format(featurization["type"]))
    return label2index


def explicitly_set_fields():
    received = set()
    for argument in sys.argv:
        if argument.startswith("--"):
            received.add(argument[2:])
            if argument[2:].startswith("no"):
                received.add(argument[4:])
    return received


def feature_dtype(feat):
    if requires_vocab(feat):
        return tf.int32
    elif feat["type"] in {"digit", "punctuation_count", "uppercase"}:
        return tf.float32
    elif requires_character_convolution(feat):
        return tf.int32
    elif feat["type"] == "bio":
        return tf.int32
    else:
        raise ValueError("unknown feature %r." % (feat,))


def feature_shape(feature):
    if requires_vocab(feature) or feature["type"] in {'digit', 'punctuation_count', 'uppercase', 'bio'}:
        return [None, None]
    elif requires_character_convolution(feature):
        return [None, None, None]
    else:
        raise ValueError("unknown feature %r." % (feature,))


def label_shape(objective):
    if objective["type"] in ("crf", "softmax"):
        return [None, None]
    elif objective["type"] == "scenario":
        return [None, None, None]
    else:
        raise ValueError("unknown objective %r." % (objective,))


def label_mask_shape(objective):
    if objective["type"] == "crf":
        return [None]
    elif objective["type"] == "softmax":
        return [None, None]
    elif objective["type"] == "scenario":
        return [None, None, None]
    else:
        raise ValueError("unknown objective %r." % (objective,))


def inexpensive_to_decode(objective):
    return objective["type"] in ("softmax", "scenario")


def gradient_barrier(inputs, freeze_rate, freeze_rate_anneal, global_step):
    if freeze_rate < 1.0:
        if freeze_rate_anneal < 1.0:
            freeze_rate = 1.0 - tf.train.exponential_decay(1.0 - freeze_rate, global_step,
                                                           1000, freeze_rate_anneal, staircase=False)
        return tf.stop_gradient(inputs) * (1.0 - freeze_rate) + freeze_rate * inputs
    else:
        return inputs


def ranking_loss(unary_scores, non_zero_weights, num_predictions, margin, **kwargs):
    """
    Tries to achieve at least 'margin' loss between desired target and alternatives.
    """
    loss = tf.reduce_sum(tf.nn.relu(unary_scores[:, 1:] + margin - unary_scores[:, 0:1]),
                         axis=-1)
    return loss * non_zero_weights / num_predictions


def build_embed(inputs, features, index2words, keep_prob, is_training, create_embedding_lookup=True):
    embeddings = []
    word_embeddings = None
    for idx, (values, feature, index2word) in enumerate(zip(inputs, features, index2words)):
        if requires_vocab(feature):
            with tf.variable_scope("embedding_%d" % (idx,)):
                if create_embedding_lookup:
                    W, embedding = embedding_lookup(values,
                                                    dim=feature["dimension"],
                                                    size=len(index2word),
                                                    dtype=tf.float32,
                                                    mask_negative=True)
                    if feature["type"] == "word":
                        word_embeddings = (W, embedding, values)
                else:
                    embedding = tf.placeholder(tf.float32,
                                               feature_shape(feature) + [feature["dimension"]])
                tf.add_to_collection(EMBEDDED_INDICES, embedding)
                embeddings.append(embedding)
        elif requires_character_convolution(feature):
            embeddings.append(character_convolution(values, feature))
        elif feature["type"] == "bio":
            # b if different from previous tag
            start_is_b = tf.greater(values[:1], -1)
            b = tf.logical_and(tf.greater(values, -1), tf.concat([start_is_b, tf.not_equal(values[:-1], values[1:])], axis=0))
            i = tf.logical_and(tf.greater(values, -1), tf.concat([tf.zeros_like(start_is_b), tf.equal(values[:-1], values[1:])], axis=0))
            # O-tag is -1
            o = tf.less(values, 0)
            embeddings.append(tf.cast(tf.stack([b, i, o], axis=-1, name="bool_BIO"), tf.float32, name="BIO"))
        else:
            embeddings.append(tf.expand_dims(values, 2))
    return maybe_dropout(concat(embeddings, axis=2), keep_prob, is_training), word_embeddings


def build_inputs(features, objectives, fused, class_weights,
                 class_weights_clipval):
    input_placeholders = []
    labels = []
    labels_mask = []
    labels_class_weights = []

    with tf.variable_scope("Inputs"):
        static_phs = {}
        for static_placeholder in STATIC_PLACEHOLDERS:
            static_placeholder.declare_and_store()
            static_phs[static_placeholder.name] = static_placeholder.placeholder

        is_training = static_phs["is_training"]
        greedy_decoding = static_phs["greedy_decoding"]
        beam_width = static_phs["beam_width"]

        for idx, feat in enumerate(features):
            input_placeholder = tf.placeholder(
                feature_dtype(feat), feature_shape(feat),
                name="input_placeholders_%d" % (idx,)
            )
            input_placeholders.append(input_placeholder)
            tf.add_to_collection(INPUT_PLACEHOLDERS, input_placeholder)

        if fused:
            max_output_vocab = max(len(obj["vocab"]) for obj in objectives)
            label_placeholder = tf.placeholder(tf.int32,
                                               [None, None, len(objectives)])
            labels_mask_placeholder = tf.placeholder(tf.bool,
                                                     [None, None, len(objectives)],
                                                     name="labels_mask")
            labels.append(label_placeholder)
            labels_mask.append(labels_mask_placeholder)
            tf.add_to_collection(LABEL_PLACEHOLDERS, label_placeholder)
            tf.add_to_collection(LABEL_MASK_PLACEHOLDERS, labels_mask_placeholder)

            if class_weights:
                with tf.variable_scope("FusedClassWeights"):
                    init_class_weights = tf.get_variable(
                        name="class_weights",
                        shape=[len(objectives) * max_output_vocab],
                        initializer=tf.constant_initializer(1),
                        dtype=tf.int64,
                        trainable=False)
                    init_class_count = tf.get_variable(
                        name="class_weights_denominator",
                        shape=[len(objectives)],
                        initializer=tf.constant_initializer(1),
                        dtype=tf.int64,
                        trainable=False)

                    def update_class_weights():
                        mask_as_ints = tf.cast(tf.reshape(labels_mask_placeholder, [-1, len(objectives)]), tf.int64)
                        updated_cls_weights = tf.scatter_add(
                            init_class_weights,
                            tf.reshape(label_placeholder + tf.reshape(tf.range(len(objectives)) * max_output_vocab,
                                                                      [1, 1, len(objectives)]),
                                       [-1]),
                            tf.reshape(mask_as_ints, [-1])
                        )
                        updated_class_count = tf.assign_add(init_class_count, tf.reduce_sum(mask_as_ints, 0))

                        # class weight: weight_i = total / class_i
                        weights = tf.clip_by_value(tf.expand_dims(updated_class_count, 1) /
                                                   tf.reshape(updated_cls_weights, [len(objectives), max_output_vocab]),
                                                   1e-6, class_weights_clipval)
                        return tf.cast(weights, tf.float32)

                    def return_class_weights():
                        # class weight: weight_i = total / class_i
                        return tf.cast(
                            tf.clip_by_value(tf.expand_dims(init_class_count, 1) /
                                             tf.reshape(init_class_weights, [len(objectives), max_output_vocab]),
                                             1e-6, class_weights_clipval), tf.float32)

                    labels_class_weights.append(
                        tf.cond(is_training,
                                update_class_weights,
                                return_class_weights))
            else:
                labels_class_weights.append(None)
        else:
            for objective in objectives:
                with tf.variable_scope(objective["name"]):
                    label_placeholder = tf.placeholder(
                        tf.int32, label_shape(objective), name="labels")
                    labels.append(label_placeholder)
                    labels_mask_placeholder = tf.placeholder(
                        tf.bool, label_mask_shape(objective), name="labels_mask")
                    labels_mask.append(labels_mask_placeholder)
                    tf.add_to_collection(LABEL_PLACEHOLDERS, label_placeholder)
                    tf.add_to_collection(LABEL_MASK_PLACEHOLDERS, labels_mask_placeholder)

                    if objective["type"] == "softmax" and class_weights:
                        init_class_weights = tf.get_variable(
                            name="class_weights",
                            shape=len(objective["vocab"]),
                            initializer=tf.constant_initializer(1),
                            dtype=tf.int64,
                            trainable=False)
                        init_class_count = tf.get_variable(
                            name="class_weights_denominator",
                            shape=[],
                            initializer=tf.constant_initializer(1),
                            dtype=tf.int64,
                            trainable=False)

                        def update_class_weights():
                            mask_as_ints = tf.cast(tf.reshape(labels_mask_placeholder, [-1]), tf.int64)
                            updated_cls_weights = tf.scatter_add(
                                init_class_weights,
                                tf.reshape(label_placeholder, [-1]),
                                mask_as_ints
                            )
                            updated_class_count = tf.assign_add(init_class_count, tf.reduce_sum(mask_as_ints))

                            # class weight: weight_i = total / class_i
                            weights = tf.clip_by_value(updated_class_count / updated_cls_weights,
                                                       1e-6, class_weights_clipval)
                            return tf.cast(weights, tf.float32)

                        def return_class_weights():
                            # class weight: weight_i = total / class_i
                            return tf.cast(
                                tf.clip_by_value(init_class_count / init_class_weights,
                                                 1e-6, class_weights_clipval), tf.float32)

                        labels_class_weights.append(tf.cond(is_training,
                                                            update_class_weights,
                                                            return_class_weights))
                    else:
                        labels_class_weights.append(None)
        sequence_lengths = tf.placeholder(tf.int32, [None],
                                          name="sequence_lengths")
        tf.add_to_collection(SEQUENCE_LENGTHS, sequence_lengths)
    return (input_placeholders,
            labels,
            labels_mask,
            labels_class_weights,
            sequence_lengths,
            is_training,
            greedy_decoding,
            beam_width)


def _update_objectives(old, new):
    assert len(old) == len(new), "number of objectives has changed between old ({}) and new config ({}) (delta={}).".format(
        len(old), len(new), set([o["name"] for o in new]).difference(set([o["name"] for o in old])))
    for obj_idx, objective in enumerate(new):
        assert old[obj_idx]["type"] == new[obj_idx]["type"], "objective {} changed type ({} -> {}).".format(
            obj_idx, old[obj_idx]["type"], new[obj_idx]["type"])
        assert old[obj_idx]["name"] == new[obj_idx]["name"], "objective {} changed name ({} -> {}).".format(
            obj_idx, old[obj_idx]["name"], new[obj_idx]["name"])
        if objective["type"] == "scenario":
            for key in objective:
                old[obj_idx][key] = objective[key]


def _collection_first_or_none(collection_name):
    collection = tf.get_collection(collection_name)
    return collection[0] if len(collection) > 0 else None


class SequenceModel(object):
    def __init__(self,
                 objectives,
                 features,
                 feature_index2words,
                 hidden_sizes,
                 keep_prob,
                 lr,
                 solver,
                 seed=1234,
                 input_keep_prob=0.7,
                 clip_norm=-1,
                 name="SequenceTagger",
                 cudnn=False,
                 anneal_rate=0.99,
                 anneal_every=33000,
                 macro_anneal_rate=1.0,
                 macro_anneal_every=1000000,
                 trainable=True,
                 weight_noise=0.0,
                 class_weights_normalize=False,
                 faux_cudnn=False,
                 class_weights=False,
                 class_weights_clipval=1000.0,
                 freeze_rate=1.0,
                 fused=False,
                 freeze_rate_anneal=0.8,
                 n_transformer_heads=4,
                 transformer_hidden_sizes=None,
                 transformer_filter_size=512,
                 post_process_spec=None,
                 convolutions=None,
                 create_variables=True,
                 classifications=None,
                 create_embedding_lookup=True,
                 memmap_embedding_variables_path=None,
                 model_load_path=None,
                 gradient_accumulation_steps=1):
        if memmap_embedding_variables_path is None:
            memmap_embedding_variables_path = MEMMAP_EMBEDDING_VARIABLES_PATH
        if fused and objectives[0]["type"] in ("crf", "scenario"):
            fused = False
        if transformer_hidden_sizes is None:
            transformer_hidden_sizes = []
        if post_process_spec is None:
            post_process_spec = []
        remainder = {}
        for key, value in locals().items():
            if key in ("self", "remainder"):
                continue
            if key in FIELDS_TO_SAVE:
                setattr(self, key, value)
            elif key in EPHEMERAL_FIELDS:
                remainder[key] = value
            else:
                raise ValueError("argument \"{}\" is not being saved and is not marked as ephemeral.".format(key))
        self.setup(**remainder)

    def setup(self, *, create_variables, trainable, model_load_path, classifications):
        self.rng = np.random.RandomState(self.seed)
        self._model_load_path = model_load_path
        self._embedding_lookup_with_memmap_loaded = False
        self.feature_word2index = [
            {w: k for k, w in enumerate(index2word)} if index2word is not None else None
            for index2word in self.feature_index2words
        ]
        self.label2index = [
            {w: k for k, w in enumerate(objective["vocab"])}
            if objective["type"] != "scenario" else load_scenario_label2index(objective, classifications)
            for objective in self.objectives
        ]
        with as_default_type_collection(classifications.type_collection if classifications is not None else None):
            if create_variables:
                # 1) build graph here (TF functional code pattern)
                self.create_variables(trainable=trainable)
            # 2) and use meta graph to recover these fields:
            self.recover_graph_variables()

    def create_variables(self, trainable):
        build_model(name=self.name,
                    trainable=trainable,
                    objectives=self.objectives,
                    features=self.features,
                    feature_index2words=self.feature_index2words,
                    hidden_sizes=self.hidden_sizes,
                    keep_prob=self.keep_prob,
                    solver=self.solver,
                    freeze_rate=self.freeze_rate,
                    class_weights_normalize=self.class_weights_normalize,
                    class_weights=self.class_weights,
                    class_weights_clipval=self.class_weights_clipval,
                    freeze_rate_anneal=self.freeze_rate_anneal,
                    cudnn=self.cudnn,
                    lr=self.lr,
                    n_transformer_heads=self.n_transformer_heads,
                    transformer_hidden_sizes=self.transformer_hidden_sizes,
                    transformer_filter_size=self.transformer_filter_size,
                    post_process_spec=self.post_process_spec,
                    fused=self.fused,
                    weight_noise=self.weight_noise,
                    anneal_rate=self.anneal_rate,
                    anneal_every=self.anneal_every,
                    macro_anneal_every=self.macro_anneal_every,
                    macro_anneal_rate=self.macro_anneal_rate,
                    input_keep_prob=self.input_keep_prob,
                    faux_cudnn=self.faux_cudnn,
                    convolutions=self.convolutions,
                    clip_norm=self.clip_norm,
                    label2index=self.label2index,
                    create_embedding_lookup=self.create_embedding_lookup,
                    gradient_accumulation_steps=self.gradient_accumulation_steps)

    def recover_graph_variables(self):
        """Use TF meta graph to obtain key metrics
        and outputs from model."""
        self.labels = tf.get_collection(LABEL_PLACEHOLDERS)
        self.labels_mask = tf.get_collection(LABEL_MASK_PLACEHOLDERS)
        self.input_placeholders = tf.get_collection(INPUT_PLACEHOLDERS)
        self.sequence_lengths = tf.get_collection(SEQUENCE_LENGTHS)[0]

        self.decoded = tf.get_collection(tf_metrics.DECODED)
        self.decoded_scores = tf.get_collection(tf_metrics.DECODED_SCORES)
        self.unary_scores = tf.get_collection(tf_metrics.UNARY_SCORES)
        self.scenario_feature_scores = tf.get_collection(SCENARIO_FEATURE_SCORES)

        self.token_correct = tf.get_collection(tf_metrics.TOKEN_CORRECT)
        self.token_correct_total = tf.get_collection(tf_metrics.TOKEN_CORRECT_TOTAL)

        self.sentence_correct = tf.get_collection(tf_metrics.SENTENCE_CORRECT)
        self.sentence_correct_total = tf.get_collection(tf_metrics.SENTENCE_CORRECT_TOTAL)

        self.token_correct_all = tf.get_collection(tf_metrics.TOKEN_CORRECT_ALL)[0]
        self.token_correct_all_total = tf.get_collection(tf_metrics.TOKEN_CORRECT_ALL_TOTAL)[0]
        self.sentence_correct_all = tf.get_collection(tf_metrics.SENTENCE_CORRECT_ALL)[0]
        self.sentence_correct_all_total = tf.get_collection(tf_metrics.SENTENCE_CORRECT_ALL_TOTAL)[0]

        self.confusion_matrices = tf.get_collection(tf_metrics.CONFUSION_MATRIX)
        self.true_positives = tf.get_collection(tf_metrics.TRUE_POSITIVES)
        self.false_positives = tf.get_collection(tf_metrics.FALSE_POSITIVES)
        self.false_negatives = tf.get_collection(tf_metrics.FALSE_NEGATIVES)

        self.attention_weights = tf.get_collection(ATTENTION_WEIGHTS)

        if len(self.true_positives) == 0 and len(self.token_correct) != 0:
            self.true_positives = [None for _ in self.token_correct]
            self.false_positives = [None for _ in self.token_correct]
            self.false_negatives = [None for _ in self.token_correct]

        if len(tf.get_collection(GLOBAL_STEP)) > 0:
            self.global_step = tf.get_collection(GLOBAL_STEP)[0]
        else:
            try:
                self.global_step = tf.get_default_graph().get_tensor_by_name(
                    self.name + "/" + "global_step:0")
            except KeyError:
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
            tf.add_to_collection(GLOBAL_STEP, self.global_step)

        for static_placeholder in STATIC_PLACEHOLDERS:
            setattr(self, static_placeholder.name, static_placeholder.retrieve())

        self.noop = tf.no_op()
        self.train_op = tf.get_collection(TRAIN_OP)[0]
        self.train_accumulate_grad_op = _collection_first_or_none(TRAIN_ACCUMULATE_GRAD_OP)
        self.train_zero_accumulator_op = _collection_first_or_none(TRAIN_ZERO_ACCUMULATOR_OP)
        self.train_accumulate_op = _collection_first_or_none(TRAIN_ACCUMULATE_OP)
        self.train_summaries = tf.get_collection(tf_logger.TRAIN_SUMMARIES)
        self.train_summaries_names = tf.get_collection(tf_logger.TRAIN_SUMMARIES_NAMES)
        assert len(self.train_summaries) == len(self.train_summaries_names), "expected as many train_summaries as train_summaries_names"

        self.test_image_summaries_bw = tf.get_collection(tf_logger.TEST_IMAGE_SUMMARIES_BW)
        self.test_image_summaries_bw_names = tf.get_collection(tf_logger.TEST_IMAGE_SUMMARIES_BW_NAMES)
        self.nll = tf.get_collection(NLL)[0]
        self.nll_total = tf.get_collection(NLL_TOTAL)[0]
        self.saver = tf.train.Saver()
        self.candidate_word_distances = tf.get_collection(CANDIDATE_WORD_DISTANCES)

    def _load_memmap_embeddings(self):
        if self._model_load_path is None:
            raise ValueError("model was not loaded. Cannot use _embedding_lookup_with_memmap.")
        self._memmap_input_indices = []
        for placeholder, feature in zip(self.input_placeholders, self.features):
            if requires_vocab(feature):
                self._memmap_input_indices.append(placeholder)
        self._memmap_embedded_indices = tf.get_collection(EMBEDDED_INDICES)
        if len(self._memmap_embedded_indices) != len(self._memmap_input_indices):
            raise ValueError("when obtaining placeholders and embeddings for the model, "
                             "a different number of indices ({}) and embeddings ({}) were "
                             "found".format(len(self._memmap_input_indices),
                                            len(self._memmap_embedded_indices)))
        self._memmap_lookups = []
        for idx in range(len(self._memmap_embedded_indices)):
            self._memmap_lookups.append(np.load(join(self._model_load_path,
                                                     self.memmap_embedding_variables_path,
                                                     "embedding_{}.npy".format(idx)), mmap_mode="r"))

    def postprocess_feed_dict(self, feed_dict):
        if not self.create_embedding_lookup:
            self._embedding_lookup_with_memmap(feed_dict)

    def _embedding_lookup_with_memmap(self, feed_dict):
        """Inplace do a lookup for indices in memory-mapped
        numpy arrays containing the embeddings."""
        if not self._embedding_lookup_with_memmap_loaded:
            self._load_memmap_embeddings()
            self._embedding_lookup_with_memmap_loaded = True
        for placeholder, indices, embedding in zip(self._memmap_embedded_indices,
                                                   self._memmap_input_indices,
                                                   self._memmap_lookups):
            # do a lookup using numpy:
            feed_dict[placeholder] = embedding[feed_dict[indices]]
            # no longer need to give the indices to tensorflow feed-dict:
            del feed_dict[indices]

    def predict(self, session, feed_dict):
        feed_dict[self.is_training] = False
        self.postprocess_feed_dict(feed_dict)
        outputs, outputs_probs = session.run(
            (self.decoded, self.decoded_scores), feed_dict
        )
        predictions_out = {}
        for value, val_prob, objective in zip(outputs, outputs_probs, self.objectives):
            predictions_out[objective["name"]] = (value, val_prob)
        return predictions_out

    def predict_proba(self, session, feed_dict):
        feed_dict[self.is_training] = False
        self.postprocess_feed_dict(feed_dict)
        outputs = session.run(self.unary_scores, feed_dict)
        predictions_out = {}
        for value, objective in zip(outputs, self.objectives):
            predictions_out[objective["name"]] = value
        return predictions_out

    def save(self, session, path):
        makedirs(path, exist_ok=True)
        with open(join(path, "model.json"), "wt") as fout:
            save_dict = {}
            for field in FIELDS_TO_SAVE:
                save_dict[field] = getattr(self, field)
            json.dump(save_dict, fout)

        with open(join(path, "rng.pkl"), "wb") as fout:
            pickle.dump(self.rng, fout)

        tf_saver.save_session(session, self.saver, path, verbose=True)

    @classmethod
    def load(cls, session, path, args=None, verbose=True, trainable=True,
             legacy=False, faux_cudnn=False, replace_to=None, replace_from=None,
             classifications=None, objectives=None, **kwargs):
        """Convenience method for using a tensorflow session to reload
        a previously saved + serialized model from disk."""
        with open(join(path, "model.json"), "rt") as fin:
            model_props = json.load(fin)

        if objectives is not None:
            _update_objectives(model_props["objectives"], objectives)


        # update fields based on CLI:
        if args is not None:
            ex_fields = explicitly_set_fields()
            for field in OVERRIDEABLE_FIELDS:
                if field in ex_fields:
                    model_props[field] = getattr(args, field)

        # prune old fields based on changes to saveable fields:
        relevant_props = {}
        for field in FIELDS_TO_SAVE:
            if field in model_props:
                relevant_props[field] = model_props[field]

        for key, value in kwargs.items():
            if key in OVERRIDEABLE_FIELDS:
                relevant_props[key] = value
            else:
                raise ValueError("trying to set field {} with {} during model load, "
                                 "but it is not overrideable.".format(key, value))

        relevant_props["trainable"] = trainable
        relevant_props["faux_cudnn"] = faux_cudnn
        relevant_props["classifications"] = classifications
        relevant_props["model_load_path"] = path

        if legacy:
            print("Using legacy mode: creating a new graph.", flush=True)
            relevant_props["create_variables"] = True
            model = cls(**relevant_props)
            print("graph rebuilt, restoring session...", flush=True)
            tf_saver.restore_session(
                session, path,
                replace_to=replace_to,
                replace_from=replace_from,
                verbose=verbose,
                use_metagraph=False
            )
            print("Done restoring session.", flush=True)
        else:
            if model_props.get("cudnn", False):
                import tensorflow.contrib.cudnn_rnn # noqa
            relevant_props["create_variables"] = False
            tf_saver.restore_session(
                session, path,
                verbose=verbose,
                use_metagraph=True
            )
            model = cls(**relevant_props)

        rng_path = join(path, "rng.pkl")
        if exists(rng_path):
            # apply the saved random number generator to this
            # model:
            with open(rng_path, "rb") as fin:
                model.rng = pickle.load(fin)
        return model


def add_objective_names_types(objectives):
    for objective in objectives:
        with tf.variable_scope(objective["name"]):
            # store objective names in graph:
            tf.add_to_collection(OBJECTIVE_NAMES, tf.constant(objective["name"],
                                                              name="objective_name"))
            tf.add_to_collection(OBJECTIVE_TYPES, tf.constant(objective["type"],
                                                              name="objective_type"))


def multiply_by(inputs, is_training, multiplier):
    return inputs * multiplier


def make_sequence_ids(lens):
    # Note: causes segfaults in some rase cases on GPU due to work_element something...
    c = tf.cumsum(lens)
    def build_seq_ids():
        return tf.searchsorted(c, tf.range(c[-1]), side='right')

    def build_empty_seq_ids():
        return tf.zeros((0), dtype=tf.int32)
    return tf.cond(tf.greater(c[-1], 0), build_seq_ids, build_empty_seq_ids)


def random_greedy_decode(prob_greedy, batch_size):
    noise = tf.less(tf.random.uniform(shape=[batch_size], dtype=tf.float32), prob_greedy)
    correct_answers = tf.zeros(batch_size, dtype=tf.int32)
    # prediction for each batch element for this timestep
    def apply_fn(unary_scores, mask):
        labels_mask_casted = tf.cast(mask, unary_scores.dtype)
        masked_unary_scores = labels_mask_casted * unary_scores - (1 - labels_mask_casted) * 50
        greedy_answers = tf.cast(tf.argmax(masked_unary_scores, axis=-1), tf.int32)
        return tf.where(noise,
                        greedy_answers,
                        correct_answers)
    return apply_fn


# def make_sequence_ids(lens):
#     # Get accumulated sums (e.g. [2, 3, 1] -> [2, 5, 6])
#     c = tf.cumsum(lens)
#     def build_seq_ids():
#         # Take all but the last accumulated sum value as indices
#         idx = c[:-1]
#         # Put ones on every index
#         s = tf.scatter_nd(tf.expand_dims(idx, 1), tf.ones_like(idx), [c[-1]])
#         # Use accumulated sums to generate ids for every segment
#         return tf.cumsum(s)

#     def build_empty_seq_ids():
#          return tf.zeros((0), dtype=tf.int32)
#     return tf.cond(tf.greater(c[-1], 0), build_seq_ids, build_empty_seq_ids)

def reshape_first_dimension(x, shape):
    existing_shape = extract_shape(x)
    new_shape = list(shape) + existing_shape[1:]
    return tf.reshape(x, new_shape)


def scenario_dynamic_fully_connected(*, objective, objective_label2index, 
                                     is_training, word_embeddings):
    feature_name2ph = {}
    
    def create_named_placeholder(feat_name, *args, **kwargs):
        ph = tf.placeholder(*args, **kwargs)
        tf.add_to_collection(INPUT_PLACEHOLDERS, ph)
        if feat_name not in feature_name2ph:
            feature_name2ph[feat_name] = {}
        feature_name2ph[feat_name][kwargs["name"]] = ph
        return ph

    extra_loss = None
    with tf.variable_scope("VariableAssignments"):
        with tf.variable_scope("Embed"):
            hiddens = []
            assert len(objective_label2index["vocab"]) == len(objective["classifications"])
            for axis, (featurization, index2word) in enumerate(zip(objective["classifications"],
                                                                   objective_label2index["vocab"])):
                def create_placeholder(*args, **kwargs):
                    return create_named_placeholder(featurization["name"], *args, **kwargs)
                with tf.variable_scope(featurization["name"]):
                    if featurization["type"] in ("variable_length_projection", "projection"):
                        unique_assignments = create_placeholder(tf.int32, [None], name="unique_assignments")
                        # unique_assignments x dim
                        _, embedding = embedding_lookup(
                            unique_assignments,
                            dim=featurization["dimension"],
                            size=len(index2word),
                            dtype=tf.float32,
                            mask_negative=True)
                        if featurization["type"] == "variable_length_projection":
                            # do some additional pooling here:
                            segment_lengths = create_placeholder(tf.int32, [None], name="segment_lengths")
                            num_segments = tf.shape(segment_lengths)[0]
                            segment_ids = tf.cond(
                                tf.greater(num_segments, 0), 
                                lambda: make_sequence_ids(segment_lengths),
                                lambda: segment_lengths)
                            if featurization["pool"] in ("max", "max_with_empty"):
                                pooled_embedding = tf.math.unsorted_segment_max(
                                    embedding,
                                    segment_ids,
                                    num_segments)
                            elif featurization["pool"] in ("lme", "lme_with_empty"):
                                pooled_embedding = lme_pool(embedding, segment_ids, num_segments)
                            else:
                                raise ValueError(
                                    "unknown pooling method \"{}\" for scenario featurization \"{}\".".format(
                                        featurization["pool"], featurization["name"]))
                            # if nothing was pooled replace by zeros
                            present = tf.not_equal(segment_lengths, 0)
                            if featurization["pool"].endswith("_with_empty"):
                                # learnable missing value embedding:
                                blank = tf.get_variable(name="MissingValue", shape=[extract_shape(pooled_embedding)[-1]])
                                pooled_embedding = tf.where(present,
                                    pooled_embedding,
                                    tile_like(blank, extract_shape(pooled_embedding)[:-1]))
                            else:
                                pooled_embedding = (tf.cast(present[:, None], tf.float32) * pooled_embedding)
                        else:
                            pooled_embedding = embedding
                        with tf.variable_scope("Combine"):
                            pooled_embedding = build_submodel_from_spec(
                                inputs=pooled_embedding, spec=featurization.get("model", []), is_training=is_training)
                        hiddens.append(pooled_embedding)
                    elif featurization["type"] in ("float", "wikipedia_probs", "wikipedia_pos_probs", "repeat_candidate"):
                        assert isinstance(featurization["shape"], (list, tuple))
                        dtype = featurization.get("dtype", "float32")
                        if dtype == "int32":
                            dtype = tf.int32
                        elif dtype == "float32":
                            dtype = tf.float32
                        else:
                            raise ValueError(f"unknown dtype {dtype}")
                        float_input_ph = create_placeholder(dtype, [None] + featurization["shape"],
                                                            name="float_input")
                        hiddens_float = float_input_ph
                        with tf.variable_scope("Combine"):
                            hiddens_float = build_submodel_from_spec(
                                inputs=hiddens_float, spec=featurization.get("model", []), is_training=is_training)
                        hiddens.append(hiddens_float)
                    elif featurization["type"] == "word_predict":
                        num_negatives = featurization["negative_samples"]
                        # scenarios x word window (indexing into input ids in batch-major order)
                        word_window_ph = create_placeholder(tf.int32, [None, None], name="word_window")
                        # input words.. lookup using batch idx per scenario idx
                        assert word_embeddings is not None
                        # grab words x dim, and time x batch x dim word embeddings
                        word_embeddings_mat, word_embeddings_inputs, word_ids = word_embeddings
                            
                        # scenario x dim word embedding
                        hconcat = concat(hiddens, -1)
                        hconcat_temp = tf.contrib.layers.fully_connected(
                            hconcat,
                            num_outputs=word_embeddings_mat.shape[1].value,
                            activation_fn=None)

                        all_word_logits = tf.matmul(word_embeddings_mat, hconcat_temp, transpose_b=True)
                        tf.add_to_collection(CANDIDATE_WORD_DISTANCES, all_word_logits)

                        # make batch-major, and express as flat array
                        word_ids_batch_major = tf.reshape(tf.transpose(word_ids, [1, 0]), [-1])
                        # index into word_ids_batch_major, obtaining scenarios x word_window
                        word_window_ids = tf.nn.embedding_lookup(word_ids_batch_major, tf.maximum(word_window_ph, 0))

                        # scenario x word_window (indicates present/absent)
                        word_present = tf.cast(tf.logical_and(tf.not_equal(word_window_ph, -1), tf.not_equal(word_window_ids, -1)), tf.float32)
                        # grab scenario x time x dim word embedding
                        scenario_word_embeddings = tf.nn.embedding_lookup(word_embeddings_mat, tf.maximum(word_window_ids, 0))
                        # scenario x word_window
                        window_logits = tf.squeeze(
                            tf.matmul(tf.expand_dims(hconcat_temp, 1),
                                      scenario_word_embeddings,
                                      transpose_b=True), 1)
                        # now do classification for these:
                        timesteps_per_scenario = tf.reduce_sum(word_present, axis=-1, keepdims=True)

                        pos_scores = tf.reduce_sum(tf.nn.sigmoid(window_logits) * word_present, axis=-1, keepdims=True) / timesteps_per_scenario
                        hiddens.append(tf.stop_gradient(pos_scores))


                        # LOGIC used for training the word prediction model (not used at inference):
                        word_window_labels = create_placeholder(tf.bool, [None], name="word_window_labels")

                        # obtain scenario dim loss (basically logprob(context))
                        window_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=window_logits,
                            labels=tf.expand_dims(tf.cast(word_window_labels, tf.float32), 1)) * word_present, axis=-1) / timesteps_per_scenario)
                        tf_logger.train_summary("word_predict/window_loss", window_loss)
                        # output dim: scalar
                        # inputs: scenario x h, negatives x h
                        # then negative samples... num_negatives x dim word embedding
                        scenario_negative_word_embeddings = tf.nn.embedding_lookup(
                            word_embeddings_mat,
                            tf.random.uniform(
                                shape=[num_negatives], dtype=tf.int32,
                                minval=0, maxval=word_embeddings_mat.shape[0].value,
                                name="negative_word_samples"))
                        neg_logits = tf.matmul(hconcat_temp, scenario_negative_word_embeddings, transpose_b=True)
                        neg_scores = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.zeros([tf.shape(word_window_ph)[0], num_negatives], dtype=tf.float32),
                            logits=neg_logits,
                            name="word_predict_negative_scores"))
                        tf_logger.train_summary("word_predict/neg_score", neg_scores)
                        local_loss = window_loss + neg_scores
                        if extra_loss is None:
                            extra_loss = local_loss
                        else:
                            extra_loss += local_loss
                    elif featurization["type"] == "predicted":
                        pass
                    else:
                        raise ValueError("unknown scenario featurization type \"{}\".".format(featurization["type"]))
    return hiddens, extra_loss, feature_name2ph


def known_shape(inputs):
    shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    return [val if val is not None else shape[axis] for axis, val in enumerate(static_shape)]


def flat_lookup(inputs, indices, shape=None):
    if shape is None:
        shape = known_shape(inputs)
    # collapse the first 2 dimensions:
    shape_out = [shape[0] * shape[1]] + shape[2:]
    # keep the rest the same:
    return tf.nn.embedding_lookup(tf.reshape(inputs, shape_out), indices)


def multiword_lmepool_qkv_attention_global(*args, **kwargs):
    return multiword_pool_qkv_attention_global(*args, **kwargs, pool_fn=multiword_lme)


def multiword_maxpool_qkv_attention_global(*args, **kwargs):
    return multiword_pool_qkv_attention_global(*args, **kwargs, pool_fn=multiword_maxpool)


def multiword_pool_qkv_attention_global(inputs, embed, segment_ids, segment_locations, n_heads, keep_prob, size, is_training, pool_fn,
                                        combine="concatenate",
                                        attend_over_embed=False, log_tensorboard=False):
    # obtain Time(unique) x Batch x Hidden
    pooled = pool_fn(inputs, segment_ids, segment_locations, is_training=is_training, embed=embed)
    # now also do attention over the original inputs, but only from the point of
    # view of the pooled locations
    batch_major_inputs = tf.transpose(tf.concat([inputs, embed], axis=-1) if attend_over_embed else inputs, [1, 0, 2])
    batch_major_pooled = tf.transpose(pooled, [1, 0, 2])
    q, k, v = compute_qkv(batch_major_pooled, batch_major_inputs, size, size, size, size,
                          q_padding="VALID", kv_padding="VALID",
                          weight_noise=0.0, is_training=is_training)
    dropout_rate = tf.cond(is_training, lambda: 1.0 - keep_prob, lambda: 0.0)
    seqlen = tf.get_collection(SEQUENCE_LENGTHS)[0]

    # Convert # Batch x K-Time
    # into
    # Batch x Heads x Q-time x K-Time
    bias = -50.0 * tf.cast(tf.expand_dims(tf.expand_dims(tf.logical_not(tf.sequence_mask(seqlen)), 1), 1), tf.float32)
    x = multihead_attention_internal(q, k, v, dropout_rate, bias,
        num_heads=n_heads, total_key_depth=size, image_shapes=None)
    weights = multihead_attention_internal_weights(q, k, dropout_rate, bias,
        num_heads=n_heads, total_key_depth=size, image_shapes=None)
    tf.add_to_collection(ATTENTION_WEIGHTS, weights)
    if log_tensorboard:
        for head_idx in range(n_heads):
            tf_logger.test_image_summary_bw("multihead_attention_head{}".format(head_idx), weights[0, head_idx, :, :, None])
    q_global = tf.get_variable(name="GlobalQuery", shape=[1, 1, size])
    x_global = multihead_attention_internal(q_global, k, v, dropout_rate, bias,
         num_heads=n_heads, total_key_depth=size, image_shapes=None)
    weights = multihead_attention_internal_weights(q_global, k, dropout_rate, bias,
        num_heads=n_heads, total_key_depth=size, image_shapes=None)
    tf.add_to_collection(ATTENTION_WEIGHTS, weights)
    x = tf.transpose(x, [1, 0, 2])
    x_global = tf.tile(tf.transpose(x_global, [1, 0, 2]), [tf.shape(pooled)[0], 1, 1])
    if combine == "concatenate":
        return tf.concat([pooled, x, x_global], axis=-1)
    elif combine == "replace":
        return tf.concat([x, x_global], axis=-1)
    else:
        raise ValueError()


def multiword_maxpool(inputs, segment_ids, segment_locations, is_training, embed):
    # inputs is in the format Time x Batch x hidden
    num_segments = tf.reduce_max(segment_ids) + 1
    # Convert Time x Batch x hidden into Time * Batch x Hidden
    # then aggregate according to the segment ids (also flattened)

    inputs = tf.exp(inputs)
    pooled = tf.log(tf.math.unsorted_segment_mean(tf.reshape(inputs, [tf.shape(inputs)[0] * tf.shape(inputs)[1], inputs.get_shape()[-1].value]),
                                                  tf.reshape(segment_ids, [tf.shape(inputs)[0] * tf.shape(inputs)[1]]), num_segments=num_segments,
                                                  name="GroupByTrigger"))
    # now do a gather to recover the proper batch dimensions (note: padding is 0, all padded inputs are maxpooled into position 0)
    # this output now has dimensions Time(unique) x Batch x Hidden
    return tf.nn.embedding_lookup(pooled, segment_locations)



def multiword_lme(inputs, segment_ids, segment_locations, is_training, embed):
    # inputs is in the format Time x Batch x hidden
    num_segments = tf.reduce_max(segment_ids) + 1
    # Convert Time x Batch x hidden into Time * Batch x Hidden
    # then aggregate according to the segment ids (also flattened)
    pooled = lme_pool(tf.reshape(inputs, [tf.shape(inputs)[0] * tf.shape(inputs)[1], inputs.get_shape()[-1].value]),
                      tf.reshape(segment_ids, [tf.shape(inputs)[0] * tf.shape(inputs)[1]]),
                      num_segments=num_segments, name="GroupByTrigger")
    # now do a gather to recover the proper batch dimensions (note: padding is 0, all padded inputs are maxpooled into position 0)
    # this output now has dimensions Time(unique) x Batch x Hidden
    return tf.nn.embedding_lookup(pooled, segment_locations)



def multiword_firstpool(inputs, segment_ids, segment_locations, is_training, embed):
    # inputs is in the format Time x Batch x hidden

    flat_segment_ids = tf.reshape(segment_ids, [tf.shape(inputs)[0] * tf.shape(inputs)[1]])

    def index1d(t):
        return tf.reduce_min(tf.where(tf.equal(t, flat_segment_ids)))
    idx = tf.map_fn(index1d, tf.range(0, tf.reduce_max(flat_segment_ids) + 1), dtype=tf.int64)
    # Convert Time x Batch x hidden into Time * Batch x Hidden
    # then aggregate according to the segment ids (also flattened)
    pooled = tf.nn.embedding_lookup(tf.reshape(inputs, [tf.shape(inputs)[0] * tf.shape(inputs)[1], inputs.get_shape()[-1].value]),
                                    idx, name="GroupByTrigger")
    # now do a gather to recover the proper batch dimensions (note: padding is 0, all padded inputs are maxpooled into position 0)
    # this output now has dimensions Time(unique) x Batch x Hidden
    return tf.nn.embedding_lookup(pooled, segment_locations)


def multiword_meanpool(inputs, segment_ids, segment_locations, is_training, embed):
    # inputs is in the format Time x Batch x hidden
    num_segments = tf.reduce_max(segment_ids) + 1
    # Convert Time x Batch x hidden into Time * Batch x Hidden
    # then aggregate according to the segment ids (also flattened)
    pooled = tf.math.unsorted_segment_mean(tf.reshape(inputs, [tf.shape(inputs)[0] * tf.shape(inputs)[1], inputs.get_shape()[-1].value]),
                                           tf.reshape(segment_ids, [tf.shape(inputs)[0] * tf.shape(inputs)[1]]), num_segments=num_segments,
                                           name="GroupByTrigger")
    # now do a gather to recover the proper batch dimensions (note: padding is 0, all padded inputs are maxpooled into position 0)
    # this output now has dimensions Time(unique) x Batch x Hidden
    return tf.nn.embedding_lookup(pooled, segment_locations)


def interactive_mlp(feature_score, w):
    import ipdb; ipdb.set_trace()
    return feature_score


def build_scenario_loss(*, objective, inputs, embed, objective_labels, mask,
                        objective_class_weights, objective_label2index,
                        word_embeddings, beam_width, greedy_decoding,
                        is_training, sequence_lengths, losses, negative_log_likelihoods):
    # here are the per timestep scores pre-masking
    # time-major, answers does the label have supervision? 
    supervised_time_major = tf.placeholder(tf.bool, [None, None], name="supervised_time_major")
    tf.add_to_collection(INPUT_PLACEHOLDERS, supervised_time_major)
    # batch x time x var assignments
    candidates_ids = tf.placeholder(tf.int32, [None, None, None], name="candidates_ids")
    tf.add_to_collection(INPUT_PLACEHOLDERS, candidates_ids)
    candidates_ids_flat = tf.placeholder(tf.int32, [None], name="candidates_ids_flat")
    tf.add_to_collection(INPUT_PLACEHOLDERS, candidates_ids_flat)
    weights = tf.placeholder(tf.float32, [None, None], name="weights")
    tf.add_to_collection(INPUT_PLACEHOLDERS, weights)
    candidates_constraints = tf.placeholder(tf.int32, [None, None, None, None], name="candidates_constraints")
    tf.add_to_collection(INPUT_PLACEHOLDERS, candidates_constraints)
    candidates_constraints_required = tf.placeholder(tf.int32, [None, None, None, None], name="candidates_constraints_required")
    tf.add_to_collection(INPUT_PLACEHOLDERS, candidates_constraints_required)
    candidates_metadata = tf.placeholder(tf.int32, [None, None, None, None], name="candidates_metadata")
    tf.add_to_collection(INPUT_PLACEHOLDERS, candidates_metadata)
    packed_sequence = False
    if objective.get("multiword_pool", None) is not None:
        with tf.variable_scope("MultiwordPool"):
            # time-major segment locations
            segment_ids = tf.placeholder(tf.int32, [None, None], name="segment_ids")
            tf.add_to_collection(INPUT_PLACEHOLDERS, segment_ids)
            segment_locations = tf.placeholder(tf.int32, [None, None], name="segment_locations")
            tf.add_to_collection(INPUT_PLACEHOLDERS, segment_locations)
            inputs = make_callable(objective["multiword_pool"])(inputs=inputs, embed=embed, segment_ids=segment_ids, segment_locations=segment_locations, is_training=is_training)
        packed_sequence = True
    if "max_scenarios" not in objective:
        objective["max_scenarios"] = 200
    if objective["max_scenarios"] > 0:
        objective_labels = objective_labels[:, :, :objective["max_scenarios"]]
        mask = mask[:, :, :objective["max_scenarios"]]
        # for each candidate placeholder.
        candidates_ids = candidates_ids[:, :, :objective["max_scenarios"]]
        candidates_constraints = candidates_constraints[:, :, :objective["max_scenarios"]]
        candidates_constraints_required = candidates_constraints_required[:, :, :objective["max_scenarios"]]
        candidates_metadata = candidates_metadata[:, :, :objective["max_scenarios"]]

    unique_feature_names = set()
    for el in objective["classifications"]:
        assert el["name"] not in unique_feature_names, f"duplicate feature in objective {el['name']}."
        unique_feature_names.add(el["name"])
    non_empty_batch = tf.greater(tf.shape(mask)[2], 0)
    autoregressive = any(featurization["type"] == "predicted" for featurization in objective["classifications"])
    scenario_placeholders = []
    scenario_embeddings_uncombined, extra_loss, feature_name2ph = scenario_dynamic_fully_connected(
        objective=objective, objective_label2index=objective_label2index,
        is_training=is_training,
        word_embeddings=word_embeddings)
    if extra_loss is not None:
        losses.append(extra_loss)
    # nonzero indices into first 2 dimensions expressed with those dimensions flattened: e.g. batch * time
    non_zero_indices = tf.squeeze(tf.where(tf.reshape(mask[:, :, 0:1], [-1])), axis=1)
    def flat2batch(x, name):
        if len(x.get_shape()) > 1:
            return tf.reshape(tf.scatter_nd(tf.expand_dims(non_zero_indices, axis=1), x,
                                            shape=[tf.shape(inputs)[0] * tf.shape(inputs)[1], tf.shape(x)[1]]),
                              [tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(x)[1]],
                              name=name)
        else:
            return tf.reshape(tf.scatter_nd(tf.expand_dims(non_zero_indices, axis=1), x,
                                            shape=[tf.shape(inputs)[0] * tf.shape(inputs)[1]]),
                              [tf.shape(inputs)[0], tf.shape(inputs)[1]],
                              name=name)
    features_extra_loss = None
    if autoregressive:
        # do teacher forcing if training, else decode for validation...
        unary_scores_autoreg, features_extra_loss, predictions_time_major = autoregressive_scenario_decoding(
            inputs=inputs,
            objective=objective,
            scenario_embeddings_uncombined=scenario_embeddings_uncombined,
            mask=mask,
            labels=objective_labels,
            candidates_ids=candidates_ids,
            candidates_ids_flat=candidates_ids_flat,
            candidates_constraints=candidates_constraints,
            candidates_constraints_required=candidates_constraints_required,
            candidates_metadata=candidates_metadata,
            feature_name2ph=feature_name2ph,
            is_training=is_training,
            greedy_decoding=greedy_decoding,
            beam_width=beam_width,
            packed_sequence=packed_sequence,
            reorder_decoding=objective.get("reorder_decoding", False))
        nonzero_unary_scores = flat_lookup(unary_scores_autoreg, non_zero_indices, shape=extract_shape(mask))
        unary_scores = unary_scores_autoreg
    else:
        featurization_embeddings = {featurization["name"]: feature_embed
                                    for featurization, feature_embed in
                                    zip(objective["classifications"], scenario_embeddings_uncombined)}

        scenario_embeddings, interaction_vars, (linear_combiner_w, linear_combiner_b), feature_interactions_hiddens = scenario_dynamic_fully_connected_combine(
            hiddens=concat([v for featurization, v in
                            zip(objective["classifications"], scenario_embeddings_uncombined)
                            if featurization.get("combine", "concatenate") == "concatenate"], -1),
            featurization_embeddings=featurization_embeddings,
            objective=objective, is_training=is_training,
            input_size=inputs.get_shape()[-1].value)
        non_zero_inputs_2d = flat_lookup(inputs, non_zero_indices)
        non_zero_objective_labels = flat_lookup(objective_labels, non_zero_indices)
        non_zero_inputs = tf.expand_dims(non_zero_inputs_2d, -1)
        # (batch x time x per-example-scenarios x h) * (batch x time x h x 1)
        nonzero_unary_scores = tf.squeeze(tf.matmul(
            tf.nn.embedding_lookup(scenario_embeddings, non_zero_objective_labels),
            non_zero_inputs), -1)

        featurization_additive_scores = {featurization["name"]: make_callable(featurization["features_to_score_additive"])(
                                            features=tf.nn.embedding_lookup(feature_embed, non_zero_objective_labels),
                                            inputs=non_zero_inputs,
                                            is_training=is_training)
                                         for featurization, feature_embed in zip(objective["classifications"], scenario_embeddings_uncombined)
                                         if featurization.get("combine", "add") == "add"}
        for featurization in objective["classifications"]:
            if featurization.get("combine", "add") == "add":
                nonzero_unary_scores = nonzero_unary_scores + featurization_additive_scores[featurization["name"]]

        if len(interaction_vars) == 0:
            # linear_combiner_w is (scenario_features x model_hidden_size)
            # non_zero_inputs is (batch x model_hidden_size)
            # to get batch x scenario_features
            non_zero_linear_weights = tf.matmul(non_zero_inputs_2d, linear_combiner_w, transpose_b=True)
            

            # we linearly combine scenario features, so we can extract their contribution to the final decision
            # more easily.
            so_far = 0
            feature_scores = []
            assert len(scenario_embeddings_uncombined) == len(objective["classifications"]), \
                "expected same number of scenario feature embeds as classifications."

            names = [featurization["name"] for featurization in objective["classifications"]]
            # add interaction terms:
            scenario_add_interaction_terms(objective=objective,
                                           feature_interactions_hiddens=feature_interactions_hiddens,
                                           featurization_embeddings=featurization_embeddings,
                                           names=names)

            for featurization in objective["classifications"]:
                combine = featurization.get("combine", "concatenate")
                name = featurization["name"]
                if combine == "concatenate":
                    feature_embed = featurization_embeddings[name]
                    featurization_w = non_zero_linear_weights[:, so_far:so_far + feature_embed.get_shape()[-1].value, None]
                    feature_score = tf.squeeze(tf.matmul(
                        tf.nn.embedding_lookup(feature_embed, non_zero_objective_labels),
                        featurization_w), -1,
                        name=name + "_score")
                    tf.add_to_collection(SCENARIO_FEATURE_SCORES,
                                         tf.transpose(flat2batch(feature_score,
                                                                 name=name + "_batch_score"),
                                                      (1, 0, 2)))
                    
                    so_far += feature_embed.get_shape()[-1].value
                elif combine == "add":
                    feature_score = featurization_additive_scores[name]
                    tf.add_to_collection(SCENARIO_FEATURE_SCORES,
                                         tf.transpose(flat2batch(feature_score,
                                                                 name=name + "_batch_score"),
                                                      (1, 0, 2)))
                else:
                    raise ValueError("unknown combination method")
        predictions_time_major = flat2batch(tf.cast(tf.argmax(nonzero_unary_scores, axis=-1), tf.int32), name="predictions_time_major")
        unary_scores = flat2batch(nonzero_unary_scores, name="UnaryScores")
    if features_extra_loss is not None:
        losses.append(features_extra_loss)
    non_zero_labels_mask_casted = tf.cast(flat_lookup(mask, non_zero_indices), nonzero_unary_scores.dtype)
    masked_nonzero_unary_scores = non_zero_labels_mask_casted * nonzero_unary_scores - (1 - non_zero_labels_mask_casted) * 50
    non_zero_weights = flat_lookup(weights, non_zero_indices)
    num_predictions = tf.maximum(tf.reduce_sum(tf.cast(supervised_time_major, tf.float32)), 1e-6)
    batch_acc = tf.cond(
        non_empty_batch,
        lambda: tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(predictions_time_major, 0), supervised_time_major), tf.float32)) / num_predictions,
        lambda: tf.constant(1.0, dtype=tf.float32))
    def compute_sparse_softmax_cross_entropy():
        return sparse_softmax_cross_entropy_with_logits(
            logits=masked_nonzero_unary_scores,
            # labels for scenario are the first value of the scenario dimension:
            labels=tf.zeros((tf.shape(non_zero_indices)[0],), tf.int32))

    negative_log_likelihood = tf.cond(non_empty_batch,
                                      compute_sparse_softmax_cross_entropy,
                                      lambda: tf.zeros([tf.shape(masked_nonzero_unary_scores)[0]], dtype=tf.float32))
    
    
    if "loss_fn" in objective:
        normed_loss = make_callable(objective["loss_fn"])(unary_scores=masked_nonzero_unary_scores,
                                                          negative_log_likelihood=negative_log_likelihood,
                                                          non_zero_weights=non_zero_weights,
                                                          num_predictions=num_predictions)
    else:
        masked_negative_log_likelihood = negative_log_likelihood * non_zero_weights
        normed_loss = tf.reduce_sum(masked_negative_log_likelihood) / num_predictions
    losses.append(normed_loss)
    masked_negative_log_likelihood_sum = tf.reduce_sum(negative_log_likelihood * tf.cast(tf.greater(non_zero_weights, 0.0), tf.float32))
    negative_log_likelihoods.append(masked_negative_log_likelihood_sum)
    labels_mask_casted = tf.cast(mask, inputs.dtype)
    masked_unary_scores = labels_mask_casted * unary_scores - (1 - labels_mask_casted) * 50
    tf_logger.train_summary(
        "{}_batch_acc".format(objective["name"]),
        batch_acc)
    (token_correct,
     token_correct_total,
     sentence_correct,
     sentence_correct_total) = tf_metrics.softmax_metrics(masked_unary_scores,
                                                          labels=tf.zeros((tf.shape(inputs)[0], tf.shape(inputs)[1]), tf.int32),
                                                          predictions_time_major=predictions_time_major,
                                                          mask=tf.cond(non_empty_batch,
                                                                       lambda: supervised_time_major,
                                                                       lambda: tf.zeros([tf.shape(mask)[0], tf.shape(mask)[1]], dtype=tf.bool)))
    return token_correct, token_correct_total, sentence_correct, sentence_correct_total


def build_loss(inputs, embed, objectives, labels, labels_mask,
               labels_class_weights, fused, sequence_lengths,
               class_weights_normalize, label2index, is_training,
               greedy_decoding, beam_width, word_embeddings):
    """
    Compute loss function given the objectives.
    Assumes inputs are of the form [time, batch, features].

    Arguments:
    ----------
        inputs : tf.Tensor
        objectives : list<dict>, objective specs
        labels : list<tf.Tensor>
        labels_mask : list<tf.Tensor>
        labels_class_weights : list<tf.Tensor>
        sequence_lengths : tf.Tensor

    Returns:
        loss : tf.Tensor (scalar)
    """
    losses = []
    negative_log_likelihoods = []
    sentence_corrects = []
    sentence_corrects_total = []
    token_corrects = []
    token_corrects_total = []
    add_objective_names_types(objectives)

    if fused:
        max_output_vocab = max(len(obj["vocab"]) for obj in objectives)
        total_output_size = len(objectives) * max_output_vocab
        with tf.variable_scope("FusedOutputs"):
            objective_labels = labels[0]
            mask = labels_mask[0]
            objective_class_weights = labels_class_weights[0]
            # perform all classifications at once:
            unary_scores = tf.contrib.layers.fully_connected(
                inputs, total_output_size,
                activation_fn=None
            )

            unary_scores = tf.reshape(unary_scores,
                                      [tf.shape(unary_scores)[0],
                                       tf.shape(unary_scores)[1],
                                       len(objectives),
                                       max_output_vocab])
            negative_log_likelihood = sparse_softmax_cross_entropy_with_logits(
                logits=unary_scores,
                labels=objective_labels
            )
            labels_mask_casted = tf.cast(mask, negative_log_likelihood.dtype)
            masked_negative_log_likelihood = negative_log_likelihood * labels_mask_casted
            if objective_class_weights is not None:
                class_weights_mask = tf.gather(
                    tf.reshape(objective_class_weights, [-1]),
                    objective_labels +
                    tf.reshape(tf.range(len(objectives)) * max_output_vocab, [1, 1, len(objectives)]))
                if class_weights_normalize:
                    masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood * class_weights_mask
                    num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted * class_weights_mask), 1e-6)
                    normed_loss = masked_weighed_negative_log_likelihood_sum / (num_predictions / len(objectives))
                else:
                    masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood * class_weights_mask
                    num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                    normed_loss = masked_weighed_negative_log_likelihood_sum / (num_predictions / len(objectives))
            else:
                masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood
                num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                normed_loss = masked_weighed_negative_log_likelihood_sum / (num_predictions / len(objectives))

            masked_negative_log_likelihood_sum = tf.reduce_sum(masked_negative_log_likelihood)
            losses.append(normed_loss)
            negative_log_likelihoods.append(masked_negative_log_likelihood_sum)

            for idx, objective in enumerate(objectives):
                with tf.variable_scope(objective["name"]):
                    (token_correct,
                     token_correct_total,
                     sentence_correct,
                     sentence_correct_total) = tf_metrics.softmax_metrics(
                        unary_scores[:, :, idx, :len(objective["vocab"])],
                        labels=objective_labels[:, :, idx],
                        mask=mask[:, :, idx])
                    token_corrects.append(token_correct)
                    token_corrects_total.append(token_correct_total)
                    sentence_corrects.append(sentence_correct)
                    sentence_corrects_total.append(sentence_correct_total)
    else:
        for obj_group in zip(objectives, labels, labels_mask, labels_class_weights, label2index):
            (objective, objective_labels, mask, objective_class_weights, objective_label2index) = obj_group
            with tf.variable_scope(objective["name"]):
                if objective["type"] == "crf":
                    unary_scores = tf.contrib.layers.fully_connected(
                        inputs,
                        len(objective["vocab"]),
                        activation_fn=None
                    )
                    unary_scores_batch_major = tf.transpose(unary_scores, [1, 0, 2])
                    labels_batch_major = tf.transpose(objective_labels, [1, 0])

                    padded_unary_scores_batch_major = tf.cond(
                        tf.greater(tf.shape(unary_scores_batch_major)[1], 1),
                        lambda: unary_scores_batch_major,
                        lambda: tf.pad(unary_scores_batch_major, [[0, 0], [0, 1], [0, 0]]))
                    padded_labels_batch_major = tf.cond(
                        tf.greater(tf.shape(labels_batch_major)[1], 1),
                        lambda: labels_batch_major,
                        lambda: tf.pad(labels_batch_major, [[0, 0], [0, 1]]))

                    log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                        padded_unary_scores_batch_major, padded_labels_batch_major, sequence_lengths
                    )
                    labels_mask_casted = tf.cast(mask, log_likelihood.dtype)
                    masked_log_likelihood = (
                        log_likelihood * labels_mask_casted
                    )
                    masked_negative_log_likelihood_sum = -tf.reduce_sum(masked_log_likelihood)
                    num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                    losses.append(masked_negative_log_likelihood_sum / num_predictions)
                    negative_log_likelihoods.append(masked_negative_log_likelihood_sum)
                    sequence_mask = tf.logical_and(
                        tf.sequence_mask(sequence_lengths),
                        # pad the time dimension:
                        tf.expand_dims(mask, 1))

                    (token_correct,
                     token_correct_total,
                     sentence_correct,
                     sentence_correct_total) = tf_metrics.crf_metrics(unary_scores_batch_major,
                                                                      labels=labels_batch_major,
                                                                      mask=sequence_mask,
                                                                      transition_params=transition_params,
                                                                      sequence_lengths=sequence_lengths)
                elif objective["type"] == "softmax":
                    unary_scores = tf.contrib.layers.fully_connected(
                        inputs,
                        len(objective["vocab"]),
                        activation_fn=None
                    )

                    negative_log_likelihood = sparse_softmax_cross_entropy_with_logits(
                        logits=unary_scores,
                        labels=objective_labels
                    )
                    labels_mask_casted = tf.cast(mask, negative_log_likelihood.dtype)
                    masked_negative_log_likelihood = (
                        negative_log_likelihood * labels_mask_casted
                    )
                    if objective_class_weights is not None:
                        class_weights_mask = tf.gather(objective_class_weights, objective_labels)
                        masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood * class_weights_mask
                        masked_negative_log_likelihood_sum = tf.reduce_sum(masked_negative_log_likelihood)

                        if class_weights_normalize:
                            num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted * class_weights_mask), 1e-6)
                            normed_loss = masked_weighed_negative_log_likelihood_sum / num_predictions
                        else:
                            num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                            normed_loss = masked_weighed_negative_log_likelihood_sum / num_predictions
                    else:
                        masked_weighed_negative_log_likelihood_sum = masked_negative_log_likelihood
                        masked_negative_log_likelihood_sum = tf.reduce_sum(masked_negative_log_likelihood)
                        num_predictions = tf.maximum(tf.reduce_sum(labels_mask_casted), 1e-6)
                        normed_loss = masked_weighed_negative_log_likelihood_sum / num_predictions

                    losses.append(normed_loss)
                    negative_log_likelihoods.append(masked_negative_log_likelihood_sum)

                    (token_correct,
                     token_correct_total,
                     sentence_correct,
                     sentence_correct_total) = tf_metrics.softmax_metrics(unary_scores,
                                                                          labels=objective_labels,
                                                                          mask=mask)
                elif objective["type"] == "scenario":
                    token_correct, token_correct_total, sentence_correct, sentence_correct_total = build_scenario_loss(
                        objective=objective, inputs=inputs, objective_labels=objective_labels, mask=mask,
                        objective_class_weights=objective_class_weights, objective_label2index=objective_label2index,
                        word_embeddings=word_embeddings, beam_width=beam_width, greedy_decoding=greedy_decoding,
                        is_training=is_training, sequence_lengths=sequence_lengths, losses=losses,
                        negative_log_likelihoods=negative_log_likelihoods, embed=embed)
                else:
                    raise ValueError(
                        "unknown objective type %r" % (objective["type"],)
                    )
                token_corrects.append(token_correct)
                token_corrects_total.append(token_correct_total)
                sentence_corrects.append(sentence_correct)
                sentence_corrects_total.append(sentence_correct_total)
    # aggregate metrics for all objectives:
    total_loss = tf.reduce_sum(sum_list(losses))
    neg_log_likelihood_total = sum_list(negative_log_likelihoods)
    tf.add_to_collection(NLL, neg_log_likelihood_total)
    tf.add_to_collection(NLL_TOTAL, tf.shape(inputs)[1])

    sentence_corrects_total = sum_list(sentence_corrects_total)
    sentence_corrects = sum_list(sentence_corrects)
    tf.add_to_collection(tf_metrics.SENTENCE_CORRECT_ALL, sentence_corrects)
    tf.add_to_collection(tf_metrics.SENTENCE_CORRECT_ALL_TOTAL, sentence_corrects_total)

    token_corrects_total = sum_list(token_corrects_total)
    token_corrects = sum_list(token_corrects)
    tf.add_to_collection(tf_metrics.TOKEN_CORRECT_ALL, token_corrects)
    tf.add_to_collection(tf_metrics.TOKEN_CORRECT_ALL_TOTAL, token_corrects_total)
    return total_loss


def grad_var_not_none(grad, var):
    assert grad is not None, "None gradient not supported for variable {}".format(var.name)
    return True


def build_model(name,
                trainable,
                features,
                feature_index2words,
                objectives,
                keep_prob,
                input_keep_prob,
                hidden_sizes,
                freeze_rate,
                freeze_rate_anneal,
                solver,
                cudnn,
                fused,
                faux_cudnn,
                transformer_hidden_sizes,
                transformer_filter_size,
                post_process_spec,
                n_transformer_heads,
                class_weights,
                class_weights_normalize,
                class_weights_clipval,
                lr,
                weight_noise,
                anneal_rate,
                anneal_every,
                macro_anneal_every,
                macro_anneal_rate,
                clip_norm,
                convolutions,
                label2index,
                create_embedding_lookup,
                gradient_accumulation_steps):
    # mixed output fusing is currently unsupported
    if fused and any(obj["type"] != "softmax" for obj in objectives):
        raise ValueError("cannot fuse outputs and use non-softmax output.")
    # clear all existing collections to ensure every new collection is
    # is created fresh
    graph = tf.get_default_graph()
    for collection_name in graph.get_all_collection_keys():
        graph.clear_collection(collection_name)

    # build a model under the model's name to prevent collisions
    # when multiple models are restored simultaneously
    with tf.variable_scope(name):
        global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.add_to_collection(GLOBAL_STEP, global_step)
        # model placeholders:
        (input_placeholders,
         labels,
         labels_mask,
         labels_class_weights,
         sequence_lengths,
         is_training,
         greedy_decoding,
         beam_width) = build_inputs(features,
                                    objectives=objectives,
                                    fused=fused,
                                    class_weights=class_weights,
                                    class_weights_clipval=class_weights_clipval)
        embed, word_embeddings = build_embed(input_placeholders,
                                             features=features,
                                             index2words=feature_index2words,
                                             is_training=is_training,
                                             keep_prob=input_keep_prob,
                                             create_embedding_lookup=create_embedding_lookup)
        hiddens = embed
        if convolutions is not None:
            hiddens = build_convolutions(hiddens,
                                         stages=parse_convolutions(convolutions),
                                         is_training=is_training,
                                         keep_prob=keep_prob,
                                         weight_noise=weight_noise,
                                         time_major=True)

        if len(hidden_sizes) > 0:
            hiddens = build_recurrent(hiddens,
                                      cudnn=cudnn,
                                      faux_cudnn=faux_cudnn,
                                      hidden_sizes=hidden_sizes,
                                      keep_prob=keep_prob,
                                      weight_noise=weight_noise,
                                      is_training=is_training)
        if len(transformer_hidden_sizes) > 0:
            hiddens = build_transformer(hiddens,
                                        hidden_sizes=transformer_hidden_sizes,
                                        n_heads=n_transformer_heads,
                                        transformer_filter_size=transformer_filter_size,
                                        keep_prob=keep_prob,
                                        weight_noise=weight_noise,
                                        is_training=is_training,
                                        time_major=True)

        with tf.variable_scope("PostProcess"):
            hiddens = build_submodel_from_spec(inputs=hiddens,
                                               spec=post_process_spec,
                                               is_training=is_training)

        # when loading pre-trained weights we can cut-off
        # gradients or reduce their scale on the "encoder"
        # part of the network.
        hiddens = gradient_barrier(hiddens,
                                   global_step=global_step,
                                   freeze_rate=freeze_rate,
                                   freeze_rate_anneal=freeze_rate_anneal)

        loss = build_loss(hiddens,
                          embed=embed,
                          objectives=objectives,
                          fused=fused,
                          labels=labels,
                          labels_mask=labels_mask,
                          labels_class_weights=labels_class_weights,
                          class_weights_normalize=class_weights_normalize,
                          sequence_lengths=sequence_lengths,
                          label2index=label2index,
                          is_training=is_training,
                          greedy_decoding=greedy_decoding,
                          beam_width=beam_width,
                          word_embeddings=word_embeddings)
        if trainable:
            lr_anneal = tf.train.exponential_decay(lr, global_step, macro_anneal_every, macro_anneal_rate, staircase=True)
            learning_rate = tf.train.exponential_decay(lr_anneal, global_step,
                                                       anneal_every, anneal_rate, staircase=True)

            if solver == "adam":
                optimizer = LazyAdamOptimizer(learning_rate)
            elif solver == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                raise ValueError("Unknown solver %r." % (solver))

            grad_vars = optimizer.compute_gradients(loss if gradient_accumulation_steps == 1 else loss / gradient_accumulation_steps)
            if clip_norm > 0:
                grad_vars = [(tf.clip_by_norm(grad, clip_norm), var)
                             for grad, var in grad_vars if grad_var_not_none(grad, var)]
            if gradient_accumulation_steps > 1:
                # ccumulate gradients step
                accum_vars = [(tf.get_variable(name=f"GradAccumulator{var.name.split(':')[0]}",
                                               initializer=tf.zeros_initializer(),
                                               shape=var.shape, dtype=var.dtype, trainable=False), grad, var)
                              for grad, var in grad_vars if grad is not None]
                zero_accumulator_op = tf.group(*[accumulator.assign(tf.zeros_like(accumulator)) for accumulator, _, _ in accum_vars])
                tf.add_to_collection(TRAIN_ZERO_ACCUMULATOR_OP, zero_accumulator_op)
                accumulate_grad_op = tf.group(*[accumulator.assign_add(grad) for accumulator, grad, _ in accum_vars])
                tf.add_to_collection(TRAIN_ACCUMULATE_GRAD_OP, accumulate_grad_op)
                accumulate_train_op = optimizer.apply_gradients([(accumulator, var) for accumulator, _, var in accum_vars],
                                                                global_step=global_step)
                tf.add_to_collection(TRAIN_ACCUMULATE_OP, accumulate_train_op)

            # normal gradient step
            cpu_vars = tf.get_collection(EMBEDDING_CPU_DEVICE)
            gpu_grad_vars = [(g, v) for g, v in grad_vars if v not in cpu_vars]
            tf_logger.train_summary("grad_norm", tf.global_norm([grad for grad, _ in gpu_grad_vars]))
            tf_logger.train_summary("batch_loss", loss)

            train_op = optimizer.apply_gradients(gpu_grad_vars, global_step=global_step)
            if len(cpu_vars) > 0:
                with tf.device("cpu"):
                    cpu_grad_vars = [(g, v) for g, v in grad_vars if v in cpu_vars]
                    cpu_train_op = optimizer.apply_gradients(cpu_grad_vars)
                train_op = tf.group(train_op, cpu_train_op)
        else:
            train_op = tf.no_op()
        tf.add_to_collection(TRAIN_OP, train_op)
