import argparse
import time
from contextlib import contextmanager


def boolean_argument(parser, name, default):
    parser.add_argument("--" + name, action="store_true", default=default)
    parser.add_argument("--no" + name, action="store_false", dest=name)


def parse_args(parser=None, args=None, load_dir_required=False):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--anneal_rate", type=float, default=0.99)
    parser.add_argument("--anneal_every", type=float, default=33000)
    parser.add_argument("--macro_anneal_rate", type=float, default=1.0)
    parser.add_argument("--macro_anneal_every", type=float, default=1000000)
    parser.add_argument("--clip_norm", type=float, default=-1)
    parser.add_argument("--weight_noise", type=float, default=0.0)
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[200, 200])
    parser.add_argument("--transformer_hidden_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--transformer_filter_size", type=int, default=512)
    parser.add_argument("--convolutions", type=str, default=None)
    parser.add_argument("--load_dir", type=str, default=None, required=load_dir_required)
    parser.add_argument("--restore_input_features", type=str, default=None)
    parser.add_argument("--improvement_key", type=str, default="token_correct")
    parser.add_argument("--freeze_rate", type=float, default=1.0)
    parser.add_argument("--freeze_rate_anneal", type=float, default=0.8)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--test_every", type=int, default=10000,
                        help="Number of training iterations after which testing should occur.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--validation_batch_size", type=int, default=128)
    parser.add_argument("--max_patience", type=int, default=10)
    parser.add_argument("--class_weights_clipval", type=float, default=1000.0)
    parser.add_argument("--device", type=str, default="gpu:0")
    parser.add_argument("--keep_prob", type=float, default=0.5)
    parser.add_argument("--input_keep_prob", type=float, default=0.7)
    parser.add_argument("--solver", type=str, default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--name", type=str, default="SequenceTagger")
    parser.add_argument("--old_name", type=str, default=None)
    parser.add_argument("--n_transformer_heads", type=int, default=4)
    parser.add_argument("--nan_rewind_every", type=int, default=100)
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beam_width", type=int, default=1)

    boolean_argument(parser, "transformer", False)
    boolean_argument(parser, "cudnn", True)
    boolean_argument(parser, "faux_cudnn", False)
    boolean_argument(parser, "class_weights", False)
    boolean_argument(parser, "legacy", False)
    boolean_argument(parser, "class_weights_normalize", False)
    boolean_argument(parser, "fused", True)
    boolean_argument(parser, "report_metrics_per_axis", True)
    boolean_argument(parser, "report_class_f1", False)
    boolean_argument(parser, "cache_features", True)
    boolean_argument(parser, "tensorflow_profile", False)
    return parser.parse_args(args=args)


@contextmanager
def load_model(parser=None, args=None):
    from .tf_operations import silence_tensorflow_warnings
    silence_tensorflow_warnings()
    from wikidata_linker_utils.training_config import Config
    from wikidata_linker_utils.sequence_model import SequenceModel
    from wikidata_linker_utils.batchifier import get_objectives
    from . import tf_logger as tf_logger
    from . import training as training
    from . import tf_saver as tf_saver
    args = parse_args(parser=parser, load_dir_required=True, args=args)
    config = Config.load(args.config, cwd=args.cwd)
    config.setup_classification_handler()
    import tensorflow as tf
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    
    with tf.Session(config=session_conf) as session:
        model = SequenceModel.load(session, args.load_dir,
                                args=args, legacy=args.legacy, faux_cudnn=args.faux_cudnn,
                                objectives=get_objectives(config.objectives, None),
                                replace_to=args.name,
                                replace_from=args.old_name,
                                classifications=config.classifications)
        yield session, model, config, args
