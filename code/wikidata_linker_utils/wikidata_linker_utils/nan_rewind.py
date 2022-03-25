import tempfile
import tensorflow as tf
from . import tf_saver


class NanRewind(object):
    def __init__(self, model, session):
        self.model = model
        self.session = session
        self.rewind_dir = tempfile.TemporaryDirectory()
        self.assign_global_step_ph = tf.placeholder(shape=[], dtype=self.model.global_step.dtype,
            name="GlobalStepPlaceholder")
        self.assign_global_step = tf.assign(self.model.global_step, self.assign_global_step_ph)
        self.backup()

    def backup(self):
        tf_saver.save_session(self.session, self.model.saver, self.rewind_dir.name, verbose=True)

    def rewind(self):
        current_global_step = self.session.run(self.model.global_step)
        tf_saver.restore_session(self.session,
                                 self.rewind_dir.name,
                                 verbose=True,
                                 use_metagraph=False,
                                 only_features=False)
        # bringing back actual global step pre-rewind
        self.session.run(self.assign_global_step, {self.assign_global_step_ph: current_global_step})
