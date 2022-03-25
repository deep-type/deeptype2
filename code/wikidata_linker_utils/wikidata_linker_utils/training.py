import sys
import numpy as np
import tensorflow as tf
import pickle
import time

from .batchifier import iter_batches_single_threaded, BatchifierStatsCollector
from .progressbar import get_progress_bar
from .sequence_model import inexpensive_to_decode
from .tf_logger import Logger



def log_outcome(logger, outcome, step, name):
    for k, v in sorted(outcome.items()):
        if "total" in k or "confusion_matrix" in k:
            continue
        else:
            total = outcome[k + "_total"]
            if total == 0:
                continue
            logger.log(k, v / total)
    logger.dump(step=step)


def present_outcome(outcome, epoch, name):
    string_rows = []
    for k, v in sorted(outcome.items()):
        if "total" in k or "confusion_matrix" in k:
            continue
        else:
            total = outcome[k + "_total"]
            if total == 0:
                continue
            if "correct" in k:
                string_rows.append(
                    [
                        k,
                        "%.2f%%" % (100.0 * v / total),
                        "(%d correct / %d)" % (v, total)
                    ]
                )
            else:
                string_rows.append(
                    [
                        k,
                        "%.3f" % (v / total),
                        ""
                    ]
                )
    max_len_cols = [
        max(len(row[colidx]) for row in string_rows)
        for colidx in range(len(string_rows[0]))
    ] if len(string_rows) > 0 else []
    rows = []
    for row in string_rows:
        rows.append(
            " ".join(
                [col + " " * (max_len_cols[colidx] - len(col))
                 for colidx, col in enumerate(row)]
            )
        )
    return "\n".join(["Epoch {epoch}: {name}".format(epoch=epoch, name=name)] + rows)


def print_outcome(outcome, objectives, epoch, step, name, log_dir,
                  logger=None):
    outcome_report = present_outcome(outcome, epoch, name)
    if logger is not None:
        log_outcome(logger, outcome, step, name)
    print(outcome_report)


def maybe_decode(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def _merge_all_metrics(metrics):
    out = {}
    for key, metric in metrics.items():
        for subkey, submetric in metric.items():
            if len(key) > 0:
                out[key + "_" + subkey] = submetric
                if subkey not in out:
                    out[subkey] = submetric
                else:
                    out[subkey] += submetric
            else:
                out[subkey] = submetric
    return out


def compute_f1(metrics, objectives, report_class_f1):
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total = 0
    for objective in objectives:
        name = objective["name"]
        key = "%s_true_positives" % (name,)
        if key not in metrics:
            continue
        tp = metrics[key]
        fp = metrics["%s_false_positives" % (name,)]
        fn = metrics["%s_false_negatives" % (name,)]
        del metrics[key]
        del metrics["%s_false_positives" % (name,)]
        del metrics["%s_false_negatives" % (name,)]

        precision = 1. * tp / np.maximum((tp + fp), 1e-6)
        recall = 1. * tp / np.maximum((tp + fn), 1e-6)
        f1 = 2.0 * precision * recall / np.maximum((precision + recall), 1e-6)

        support = tp + fn

        full_f1 = np.average(f1, weights=support) * 100.0
        full_recall = np.average(recall, weights=support) * 100.0
        full_precision = np.average(precision, weights=support) * 100.0

        total_f1 += full_f1
        total_recall += full_recall
        total_precision += full_precision
        total += 1
        if report_class_f1:
            print("F1 %s: %r" % (name, full_f1))
            print("Name\tF1\tTP\tFP\tFN")
            rows = zip([label for label, has_support in zip(objective["vocab"],
                                                            support > 0)
                        if has_support],
                       f1, tp, fp, fn)
            for val, f1_val, val_tp, val_fp, val_fn in rows:
                print("%s\t%r\t%d\t%d\t%d" % (
                    val, f1_val, val_tp, val_fp, val_fn))
            print("")
    if total > 0:
        metrics["F1"] = total_f1
        metrics["recall"] = total_recall
        metrics["precision"] = total_precision
        metrics["F1_total"] = total
        metrics["recall_total"] = total
        metrics["precision_total"] = total


def _outputs_into_metrics(outputs, metrics_names):
    return dict(zip(metrics_names, outputs))


def _perform_session_runs(session, feed_dicts, outputs_val, model, train, do_profile, logger):
    if not train or len(feed_dicts) == 1:
        if do_profile:
            _, step, summary_out, summary_img_out, new_metrics = logger.profile_session_run(session, outputs_val, feed_dicts[0])
        else:
            _, step, summary_out, summary_img_out, new_metrics = session.run(outputs_val, feed_dicts[0])
    else:
        # zero-out gradients
        session.run(model.train_zero_accumulator_op)
        new_metrics = None
        for idx, feed_dict in enumerate(feed_dicts):
            if idx + 1 == len(feed_dicts):
                _, summary_out, summary_img_out, new_metrics_temp = session.run([model.train_accumulate_grad_op] + outputs_val[2:], feed_dict)
            else:
                _, new_metrics_temp = session.run([model.train_accumulate_grad_op, outputs_val[4]], feed_dict)
            if new_metrics is None:
                new_metrics = new_metrics_temp
            else:
                for key, value in new_metrics_temp.items():
                    if key not in new_metrics:
                        new_metrics[key] = value
                    else:
                        new_metrics[key] += value
        # apply gradients
        _, step = session.run((model.train_accumulate_op, model.global_step))
    return step, summary_out, summary_img_out, new_metrics
    

def _take_n(n, iterable):
    for idx, v in enumerate(iterable):
        yield v
        if idx + 1 == n:
            break


def accuracy(model, session, datasets, batch_size, train,
             report_metrics_per_axis, report_class_f1,
             callback=None, callback_period=None, logger=None,
             callback_nan=None, beam_width=8, profile=True):
    pbar = get_progress_bar("train" if train else "validation", item="batches")
    if not isinstance(datasets, dict):
        datasets = {"": datasets}
    all_metrics_agg = {}

    if callback is not None:
        if callback_period is None:
            raise ValueError("callback_period cannot be None if "
                             "callback is used.")
    else:
        callback_period = None

    if train:
        if model.gradient_accumulation_steps == 1:
            train_op = model.train_op
            train_init_op = None
            train_finalize_op = None
        else:
            train_op = model.train_accumulate_grad_op
            train_init_op = model.train_zero_accumulator_op
            train_finalize_op = model.train_accumulate_op
    else:
        train_op = model.noop
        train_init_op = None
        train_finalize_op = None
    is_training = model.is_training
    greedy_decoding = model.greedy_decoding
    beam_width_ph = model.beam_width
    metrics = {"nll": model.nll, "nll_total": model.nll_total}
    summaries = []
    summaries_names = []
    image_summaries = []
    image_summaries_names = []

    if not train or all([inexpensive_to_decode(obj) for obj in model.objectives]):
        metric_iter = zip(model.objectives,
                          model.token_correct,
                          model.token_correct_total,
                          model.sentence_correct,
                          model.sentence_correct_total,
                          model.true_positives,
                          model.false_positives,
                          model.false_negatives)
        for metric_vars in metric_iter:
            (
                objective,
                token_correct,
                token_correct_total,
                sentence_correct,
                sentence_correct_total,
                true_positives,
                false_positives,
                false_negatives
            ) = metric_vars
            name = objective["name"]
            if report_metrics_per_axis:
                metrics["%s_token_correct" % (name,)] = token_correct
                metrics["%s_token_correct_total" % (name,)] = token_correct_total
                metrics["%s_sentence_correct" % (name,)] = sentence_correct
                metrics["%s_sentence_correct_total" % (name,)] = sentence_correct_total
            if true_positives is not None:
                metrics["%s_true_positives" % (name,)] = true_positives
                metrics["%s_false_positives" % (name,)] = false_positives
                metrics["%s_false_negatives" % (name,)] = false_negatives
        metrics["token_correct"] = model.token_correct_all
        metrics["token_correct_total"] = model.token_correct_all_total
        metrics["sentence_correct"] = model.sentence_correct_all
        metrics["sentence_correct_total"] = model.sentence_correct_all_total

        if len(model.confusion_matrices) > 0:
            for objective, conf_mat_var in zip(model.objectives, model.confusion_matrices):
                metrics["%s_confusion_matrix" % (objective["name"],)] = conf_mat_var

    if train and logger is not None:
        summaries = model.train_summaries
        summaries_names = [maybe_decode(el) for el in session.run(model.train_summaries_names)]

    if not train and logger is not None:
        image_summaries = model.test_image_summaries_bw
        image_summaries_names = [maybe_decode(el) for el in session.run(model.test_image_summaries_bw_names)]

    outputs_val = [train_op, model.global_step, summaries, image_summaries, metrics]
    feed_dict_updates = {}
    feed_dict_updates[is_training] = train
    if greedy_decoding is not None:
        feed_dict_updates[greedy_decoding] = True
    if beam_width_ph is not None:
        feed_dict_updates[beam_width_ph] = beam_width

    for title, dataset in datasets.items():
        batchifier_stats_collector = BatchifierStatsCollector() if (train and logger is not None) else None
        batches = (feed_dict for feed_dict in iter_batches_single_threaded(model=model,
                                                                           dataset=dataset,
                                                                           batch_size=batch_size,
                                                                           train=train,
                                                                           pbar=pbar,
                                                                           batchifier_stats_collector=batchifier_stats_collector))
        metrics_agg = {}
        iteration = 0
        while True:
            do_profile = train and logger is not None and iteration == 1 and profile
            try:
                t0 = time.time()
                feed_dicts = list(_take_n(model.gradient_accumulation_steps if train else 1, batches))
                for feed_dict in feed_dicts:
                    feed_dict.update(feed_dict_updates)
                t1 = time.time()
                if len(feed_dicts) == 0:
                    break
                step, summary_out, summary_img_out, new_metrics = _perform_session_runs(
                    session, feed_dicts, outputs_val,
                    model=model, train=train, do_profile=do_profile, logger=logger)
                train_end_t0 = time.time()
                if logger is not None:
                    for name, val in zip(summaries_names, summary_out):
                        logger.log(name, val)
                    for name, val in zip(image_summaries_names, summary_img_out):
                        logger.log_image(name, val)
                    if train and iteration > 0:
                        # if batch waiting is slow:
                        if (t1 - t0) > (0.1 * (train_end_t0 - t1)):
                            logger.log("training/batch_wait_time", t1 - t0)
                        if not do_profile:
                            # ignore profiling runs for train timing
                            logger.log("training/train_time", train_end_t0 - t1)
                        for key, v in batchifier_stats_collector.dump().items():
                            logger.log(key, v)
                    if train or len(summary_out) > 0:
                        logger.dump(step=step)
                if callback_nan is not None and np.isnan(new_metrics['nll']):
                    callback_nan()
                else:
                    for key, value in new_metrics.items():
                        if key not in metrics_agg:
                            metrics_agg[key] = value
                        else:
                            metrics_agg[key] += value
                iteration += 1
                if callback_period is not None and iteration % callback_period == 0:
                    callback(iteration)
                t0 = time.time()
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                if train:
                    if len(model.input_placeholders) > 0:
                        print("OOM running example, input shape: {}".format(feed_dict[model.input_placeholders[0]].shape))
                    else:
                        print("OOM running example")
                    continue
                else:
                    raise e
        compute_f1(metrics_agg, model.objectives, report_class_f1)
        all_metrics_agg[title] = metrics_agg
        del batches
    return _merge_all_metrics(all_metrics_agg)


class ImprovementTracker(object):
    def __init__(self, improvement_key, improvement_callback):
        self.improvement_key = improvement_key
        self.improvement_callback = improvement_callback
        self.best_epoch = 0
        self.best_score = None
        self.best_outcome = None

    def is_improvement(self, original, new):
        if original is None:
            return True
        if self.improvement_key in ("loss", "nll") or self.improvement_key.endswith("_nll"):
            # some scores are better if they decrease, others are better if they increase
            return new < original
        return new > original

    def observe(self, outcome, epoch, callback=True):
        if outcome is not None and self.improvement_key in outcome and self.is_improvement(self.best_score, outcome[self.improvement_key]):
            if "nll" in outcome and np.isnan(outcome["nll"]):
                print("Loss is NaN, skipping result.")
            else:
                print("Improvement in {} from {} to {}".format(self.improvement_key, self.best_score, outcome[self.improvement_key]))
                self.best_score = outcome[self.improvement_key]
                self.best_epoch = epoch
                self.best_outcome = outcome
                if callback:
                    self.improvement_callback()


class TestCallback(object):
    def __init__(self, model, session, dataset, epoch, args, logger, batch_size, improvement_tracker):
        self.model = model
        self.session = session
        self.dataset = dataset
        self.epoch = epoch
        self.args = args
        self.batch_size = batch_size
        self.logger = logger
        self.report_metrics_per_axis = args.report_metrics_per_axis
        self.report_class_f1 = args.report_class_f1
        self.improvement_tracker = improvement_tracker

    def test(self, iteration):
        if len(self.dataset) > 0:
            dev_outcome = accuracy(self.model, self.session, self.dataset, self.batch_size,
                                   train=False,
                                   report_metrics_per_axis=self.report_metrics_per_axis,
                                   report_class_f1=self.report_class_f1,
                                   beam_width=self.args.beam_width)
            print_outcome(dev_outcome, self.model.objectives,
                          epoch="{}-{}".format(self.epoch, iteration),
                          step=self.session.run(self.model.global_step),
                          name="validation",
                          log_dir=self.args.log_dir,
                          logger=self.logger)
            if self.args.save_dir is not None:
                self.model.save(self.session, self.args.save_dir)
            self.improvement_tracker.observe(dev_outcome, epoch=self.epoch)
