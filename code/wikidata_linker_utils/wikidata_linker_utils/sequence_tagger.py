import warnings
import tensorflow as tf
import numpy as np

from .batchifier import iter_batches_single_threaded
from .sequence_model import SequenceModel


class SequenceTagger(object):
    def __init__(self, path, device="gpu", faux_cudnn=False, legacy=False, max_threads=-1, **kwargs):
        tf.reset_default_graph()
        thread_conf = {}
        if max_threads != -1:
            thread_conf["inter_op_parallelism_threads"] = max_threads
            thread_conf["intra_op_parallelism_threads"] = max_threads
            thread_conf["device_count"] = {"CPU": 1}
        session_conf = tf.ConfigProto(
            allow_soft_placement=True, **thread_conf
        )
        self.session = tf.InteractiveSession(config=session_conf)
        with tf.device(device):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self._model = SequenceModel.load(
                    self.session,
                    path,
                    args=None,
                    verbose=False,
                    trainable=False,
                    legacy=legacy,
                    faux_cudnn=faux_cudnn,
                    **kwargs)

    @property
    def objectives(self):
        return self._model.objectives

    def predict_proba(self, tokens):
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = list(iter_batches_single_threaded(
            model=self._model,
            dataset=[
                (tokens, [blank_labels for t in tokens])
            ],
            batch_size=1,
            train=False,
            autoresize=False
        ))
        batches[0][self._model.is_training] = False
        probs_out = self._model.predict_proba(
            self.session, batches[0]
        )
        return probs_out

    def predict_proba_sentences(self, sentences):
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = iter_batches_single_threaded(
            model=self._model,
            dataset=[
                (sentence, [blank_labels for t in sentence])
                for sentence in sentences
            ],
            batch_size=min(256, len(sentences)),
            train=False,
            autoresize=False
        )
        for batch in batches:
            batch[self._model.is_training] = False
            yield self._model.predict_proba(
                self.session, batch
            )

    def predict_topk_sentences(self, sentences, k=5):
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = iter_batches_single_threaded(
            model=self._model,
            dataset=[
                (sentence, [blank_labels for t in sentence])
                for sentence in sentences
            ],
            batch_size=min(256, len(sentences)),
            train=False,
            autoresize=False
        )
        for batch in batches:
            outputs = self._model.predict_proba(
                self.session, batch
            )
            named_outputs = {}
            for objective in self._model.objectives:
                obj_name = objective["name"]
                tags, scores = outputs[obj_name]
                if objective["type"] == "crf":
                    named_outputs[obj_name] = [
                        [(token, [objective["vocab"][tag]], [score]) for token, tag in zip(tokens, tags)]
                        for tokens, tags, score in zip(sentences, tags, scores)
                    ]
                elif objective["type"] == 'softmax':
                    all_sent_scores = []

                    for tokens, scores in zip(sentences, scores):
                        sent_scores = []
                        for token, token_scores in zip(tokens, scores):
                            topk = np.argsort(token_scores)[::-1][:k]
                            sent_scores.append(
                                (
                                    token,
                                    [objective["vocab"][idx] for idx in topk],
                                    [token_scores[idx] for idx in topk]
                                )
                            )
                        all_sent_scores.append(sent_scores)
                    named_outputs[obj_name] = all_sent_scores
                else:
                    raise ValueError("unknown objective type %r." % (objective["type"],))
            yield named_outputs

    def tag_sentences(self, sentences):
        if len(sentences) == 0:
            return {
                objective["name"]: []
                for objective in self._model.objectives
            }
        blank_labels = tuple(None for _ in self._model.objectives)
        batches = list(iter_batches_single_threaded(
            self._model,
            [
                (sentence, [blank_labels for t in sentence])
                for sentence in sentences
            ],
            batch_size=min(256, len(sentences)),
            train=False,
            autoresize=False
        ))

        named_outputs = {}
        sentence_idx = 0

        for batch in batches:
            outputs = self._model.predict(self.session, batch)
            for objective in self._model.objectives:
                obj_name = objective["name"]
                if obj_name not in named_outputs:
                    named_outputs[obj_name] = []
                tags, scores = outputs[obj_name]
                nsentences = len(tags)
                if objective["type"] == "crf":
                    named_outputs[obj_name].extend([
                        [(token, objective["vocab"][tag], score) for token, tag in zip(tokens, tags)]
                        for tokens, tags, score in zip(sentences[sentence_idx:sentence_idx + nsentences], tags, scores)
                    ])
                elif objective["type"] == 'softmax':
                    named_outputs[obj_name].extend([
                        [(token, objective["vocab"][tag], score)
                         for token, tag, score in zip(tokens, tags, scores)]
                        for tokens, tags, scores in zip(sentences[sentence_idx:sentence_idx + nsentences], tags, scores)
                    ])
                else:
                    raise ValueError("unknown objective type %r." % (objective["type"],))
            sentence_idx += nsentences

        return named_outputs


TAGGER = None
TAGGER_PATH = None


def get_tf_tagger(path, **kwargs):
    global TAGGER
    global TAGGER_PATH
    if TAGGER is None or TAGGER_PATH != path:
        TAGGER = SequenceTagger(path, **kwargs)
        TAGGER_PATH = path
    return TAGGER
