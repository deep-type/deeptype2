
from os.path import join

from wikidata_linker_utils.batchifier import get_feature_vocabs, get_objectives

import tensorflow as tf
import numpy as np

from wikidata_linker_utils.training_config import Config, multigetattr
from wikidata_linker_utils.sequence_model import SequenceModel, FIELDS_TO_SAVE, MEMMAP_FIELDS
import wikidata_linker_utils.tf_logger as tf_logger
import wikidata_linker_utils.training as training
import wikidata_linker_utils.tf_saver as tf_saver
from wikidata_linker_utils.tf_operations import silence_tensorflow_warnings, count_number_of_parameters
from wikidata_linker_utils.nan_rewind import NanRewind
from wikidata_linker_utils.entrypoint import parse_args
from wikidata_linker_utils.snapshot import snapshot_code_and_commands


def compute_epoch(session, model, train_set,
                  validation_set, test_callback, epoch,
                  train_logger, test_logger,
                  nan_rewind, args):

    if args.nan_rewind_every == -1 or nan_rewind is None:
        def callback_nan(iteration):
            raise ValueError("loss is NaN.")
        callback_period = args.test_every
        callback = test_callback.test
    else:
        callback_nan = nan_rewind.rewind
        callback_period = 1
        def callback(iteration):
            if iteration % args.test_every == 0:
                test_callback.test(iteration)
            if iteration % args.test_every == 0 or iteration % args.nan_rewind_every == 0:
                nan_rewind.backup()

    test_callback.epoch = epoch
    train_outcome = training.accuracy(
        model,
        session,
        train_set,
        args.batch_size,
        train=True,
        callback_period=callback_period,
        logger=train_logger,
        report_metrics_per_axis=args.report_metrics_per_axis,
        report_class_f1=args.report_class_f1,
        callback=callback,
        callback_nan=callback_nan,
        profile=args.tensorflow_profile
    )
    global_step = session.run(model.global_step)
    training.print_outcome(train_outcome,
                           model.objectives,
                           epoch=epoch,
                           name="train",
                           step=global_step,
                           log_dir=args.log_dir,
                           logger=train_logger)
    if len(validation_set) > 0:
        validation_batch_size = args.batch_size if args.validation_batch_size is None else args.validation_batch_size
        dev_outcome = training.accuracy(
            model, session, validation_set, validation_batch_size,
            train=False,
            report_metrics_per_axis=args.report_metrics_per_axis,
            report_class_f1=args.report_class_f1,
            beam_width=args.beam_width,
            profile=args.tensorflow_profile,
            logger=test_logger)
        training.print_outcome(dev_outcome,
                               model.objectives,
                               epoch=epoch,
                               step=global_step,
                               name="validation",
                               log_dir=args.log_dir,
                               logger=test_logger)
    else:
        dev_outcome = {}
    if args.save_dir is not None:
        model.save(session, args.save_dir)
        if nan_rewind is not None:
            nan_rewind.backup()
    return dev_outcome

def notify_missing(args, field):
    if hasattr(args, field):
        return True
    else:
        print("Cannot find field {}".format(field))
        return False


def main():
    args = parse_args()
    config = Config.load(args.config, cwd=args.cwd)
    validation_set = config.load_dataset("dev", merge=False)
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    validation_batch_size = args.batch_size if args.validation_batch_size is None else args.validation_batch_size
    import faulthandler
    faulthandler.enable()
    if args.save_dir is not None:
        snapshot_code_and_commands(args.save_dir)

    silence_tensorflow_warnings()
    with tf.Session(config=session_conf) as session, tf.device(args.device):
        if args.load_dir is not None:
            model = SequenceModel.load(session, args.load_dir,
                                       args=args, legacy=args.legacy, faux_cudnn=args.faux_cudnn,
                                       replace_to=args.name,
                                       replace_from=args.old_name,
                                       objectives=get_objectives(config.objectives, None),
                                       classifications=config.classifications)
            dev_outcome = training.accuracy(model, session, validation_set, train=False,
                                            report_metrics_per_axis=args.report_metrics_per_axis,
                                            report_class_f1=args.report_class_f1,
                                            batch_size=validation_batch_size,
                                            beam_width=args.beam_width,
                                            profile=args.tensorflow_profile)
            training.print_outcome(dev_outcome,
                                   model.objectives, 0,
                                   name="loaded validation",
                                   step=session.run(model.global_step),
                                   log_dir=args.log_dir,
                                   logger=None)
            # dev_outcome = None
            if args.legacy and args.save_dir is not None:
                model.save(session, args.save_dir)
            train_set = config.load_dataset("train")
        else:
            # load classes and index2word from a file.
            dev_outcome = None
            train_set = config.load_dataset("train")
            model_kwargs = dict(
                objectives=get_objectives(config.objectives, train_set),
                features=config.features,
                feature_index2words=get_feature_vocabs(config.features, train_set, ["<UNK>"], cache=args.cache_features),
                classifications=config.classifications,
                create_variables=True,
            )
            model_kwargs.update({field: multigetattr((config, args), field) for field in FIELDS_TO_SAVE
                                 if field not in model_kwargs and field not in MEMMAP_FIELDS})
            model = SequenceModel(**model_kwargs)
            session.run(tf.global_variables_initializer())
            if args.restore_input_features is not None:
                tf_saver.restore_session(session, args.restore_input_features,
                                         verbose=True,
                                         use_metagraph=False,
                                         only_features=True)

        print("Model has %d trainable parameters." % (count_number_of_parameters(),), flush=True)
        def save_best():
            if args.save_dir is not None:
                model.save(session, join(args.save_dir, "best"))
        
        improvement_tracker = training.ImprovementTracker(
            improvement_key=args.improvement_key,
            improvement_callback=save_best)
        if dev_outcome is not None:
            improvement_tracker.observe(dev_outcome, epoch=-1, callback=False)
        patience = 0
        best_score = improvement_tracker.best_score
        
        if args.save_dir is not None:
            train_logger = tf_logger.Logger(tf.summary.FileWriter(join(args.save_dir, "train")))
            test_logger = tf_logger.Logger(tf.summary.FileWriter(join(args.save_dir, "test")))
            embedding_writer = tf.summary.FileWriter(args.save_dir)
            tf_logger.add_embeddings_to_logger(session, model, embedding_writer, args.save_dir)
        else:
            train_logger, test_logger = None, None

        test_callback = training.TestCallback(
            model, session, validation_set, -1, args, logger=test_logger,
            batch_size=validation_batch_size,
            improvement_tracker=improvement_tracker)
        if len(train_set) > 0:
            train_set.set_randomize(True)
            train_set.set_rng(model.rng)
            nan_rewind = NanRewind(session=session, model=model) if args.nan_rewind_every != -1 else None

            for epoch in range(args.max_epochs):
                dev_outcome = compute_epoch(session, model,
                                            train_set=train_set,
                                            validation_set=validation_set,
                                            epoch=epoch,
                                            test_callback=test_callback,
                                            train_logger=train_logger,
                                            test_logger=test_logger,
                                            nan_rewind=nan_rewind,
                                            args=args)

                improvement_tracker.observe(dev_outcome, epoch=epoch)
                if args.improvement_key in dev_outcome:
                    if improvement_tracker.is_improvement(best_score, improvement_tracker.best_score):
                        patience = 0
                    else:
                        patience += 1
                        if patience >= args.max_patience:
                            print("No improvements for %d epochs. Stopping." % (args.max_patience,))
                            break
                best_score = improvement_tracker.best_score
        if improvement_tracker.best_outcome is not None:
            training.print_outcome(improvement_tracker.best_outcome,
                                   model.objectives,
                                   epoch=improvement_tracker.best_epoch,
                                   name="validation-best",
                                   step=session.run(model.global_step),
                                   log_dir=args.log_dir, logger=None)


if __name__ == "__main__":
    main()
