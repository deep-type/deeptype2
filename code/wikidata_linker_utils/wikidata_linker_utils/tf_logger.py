import time
import tensorflow as tf
from .batchifier import requires_vocab
from tensorflow.contrib.tensorboard.plugins import projector

from os.path import realpath, join

TRAIN_SUMMARIES_NAMES = "TRAIN_SUMMARIES_NAMES"
TRAIN_SUMMARIES = "TRAIN_SUMMARIES"

TEST_IMAGE_SUMMARIES_BW = "TEST_IMAGE_SUMMARIES_BW"
TEST_IMAGE_SUMMARIES_BW_NAMES = "TEST_IMAGE_SUMMARIES_BW_NAMES"


def train_summary(key, value):
    tf.add_to_collection(TRAIN_SUMMARIES_NAMES, tf.constant(key))
    tf.add_to_collection(TRAIN_SUMMARIES, value)


def test_image_summary_bw(key, value):
    tf.add_to_collection(TEST_IMAGE_SUMMARIES_BW_NAMES, tf.constant(key))
    tf.add_to_collection(TEST_IMAGE_SUMMARIES_BW, value)


def _make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    # from PIL import Image
    import matplotlib.pyplot as plt
    from io import BytesIO
    height, width, channel = tensor.shape

    s = BytesIO()
    plt.imsave(s, tensor[..., 0], format='png')
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=s.getvalue())



class Logger(object):
    def __init__(self, writer, profile_every=1500):
        self._writer = writer
        self._scalars = {}
        self._images = {}
        self._last_profile = None
        self._profile_every = profile_every

    def log(self, name, value):
        self._scalars[name] = value

    def log_image(self, name, value):
        self._images[name] = value

    def dump(self, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=float(value))
                                    for key, value in self._scalars.items()] +
                                   [tf.Summary.Value(tag=key, image=_make_image(img)) for key, img in self._images.items()])
        self._writer.add_summary(summary.SerializeToString(),
                                 global_step=step)
        self._writer.flush()
        self._scalars.clear()
        self._images.clear()

    def profile_session_run(self, session, outputs, feed_dict):
        if self._last_profile is None or (time.time() - self._last_profile) > self._profile_every:
            from tensorflow.core.protobuf import config_pb2
            from tensorflow.python.profiler import option_builder, model_analyzer
            run_meta = config_pb2.RunMetadata()
            results = session.run(outputs, feed_dict,
                options=config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE),
                run_metadata=run_meta)
            opts = (option_builder.ProfileOptionBuilder(option_builder.ProfileOptionBuilder.trainable_variables_parameter())
                .with_timeline_output(join(self._writer.get_logdir(), "timeline.json"))
                .with_accounted_types(['.*'])
                .select(['params', 'float_ops', 'micros', 'bytes',
                         'device', 'op_types', 'occurrence']).build())
            profiler = model_analyzer.Profiler(session.graph)
            profiler.add_step(0, run_meta)
            profiler.profile_graph(opts)
            advice_pb = profiler.advise(model_analyzer.ALL_ADVICE)
            for report in advice_pb.checkers['AcceleratorUtilizationChecker'].reports:
                print(report)
            for report in advice_pb.checkers['ExpensiveOperationChecker'].reports:
                print(report)
            self._last_profile = time.time()
            return results
        else:
            return session.run(outputs, feed_dict)


def add_embeddings_to_logger(session, model, writer, save_dir):
    config = projector.ProjectorConfig()
    for idx, (feature, index2word) in enumerate(zip(model.features, model.feature_index2words)):
        if requires_vocab(feature):
            embedding_var = tf.get_default_graph().get_operation_by_name(
                "%s/embedding_%d/embedding" % (model.name, idx))
            metadata_path = realpath(join(save_dir, "embedding-metadata-%d.tsv" % (idx,)))
            with open(metadata_path, "wt") as fout:
                for word in index2word:
                    fout.write(word + "\n")
            # You can add multiple embeddings. Here we add only one.
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = metadata_path

    config.model_checkpoint_path = realpath(join(save_dir, "model.ckpt"))
    model.save(session, save_dir)
    # Saves a configuration file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)
    writer.flush()
