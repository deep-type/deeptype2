import tensorflow as tf
from os import makedirs
from os.path import basename, join

from tensorflow.python.client import device_lib


def _get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def save_session(session, saver, path, verbose=False):
    """
    Call save on tf.train.Saver on a specific path to store all the variables
    of the current tensorflow session to a file for later restoring.

    Arguments:
        session : tf.Session
        path : str, place to save session
    """
    makedirs(path, exist_ok=True)
    if not path.endswith("/"):
        path = path + "/"

    path = join(path, "model.ckpt")
    if verbose:
        print("Saving session under %r" % (path,), flush=True)
    saver.save(session, path)
    print("Saved", flush=True)


def restore_session(session,
                    path,
                    replace_to=None,
                    replace_from=None,
                    verbose=False,
                    use_metagraph=True,
                    only_features=False):
    """
    Call restore on tf.train.Saver on a specific path to store all the
    variables of the current tensorflow session to a file for later restoring.

    Arguments:
        session : tf.Session
        path : str, place containing the session data to restore
        verbose : bool, print status messages.
        use_metagraph : bool, restore by re-creating saved metagraph.

    Returns:
        bool : success or failure of the restoration
    """
    makedirs(path, exist_ok=True)
    if not path.endswith("/"):
        path = path + "/"
    checkpoint = tf.train.get_checkpoint_state(path)
    if verbose:
        print("Looking for saved session under %r" % (path,), flush=True)
    if checkpoint is None or checkpoint.model_checkpoint_path is None:
        if verbose:
            print("No saved session found", flush=True)
        return False
    fname = basename(checkpoint.model_checkpoint_path)
    if verbose:
        print("Restoring saved session from %r" % (join(path, fname),), flush=True)

    if use_metagraph:
        param_saver = tf.train.import_meta_graph(join(path, fname + ".meta"),
                                                 clear_devices=len(_get_available_gpus()) == 0)
        missing_vars = []
    else:
        if only_features:
            to_restore = {}
            whitelist = ["embedding", "/RNN/", "/RNNParams", "CharacterConvolution", "HighwayLayer"]
            for var in tf.global_variables():
                if any(keyword in var.name for keyword in whitelist):
                    if "/type/" not in var.name:
                        to_restore[var.name[:-2]] = var
            param_saver = tf.train.Saver(to_restore)
            missing_vars = []
        else:
            if replace_to is not None and replace_from is not None:
                to_restore = {}
                for var in tf.global_variables():
                    var_name = var.name[:var.name.rfind(":")]
                    old_name = var_name.replace(replace_to, replace_from)
                    to_restore[old_name] = var
                param_saver = tf.train.Saver(to_restore)
                missing_vars = []
            else:
                reader = tf.train.NewCheckpointReader(join(path, fname))
                saved_shapes = reader.get_variable_to_shape_map()
                found_vars = [var for var in tf.global_variables()
                              if var.name.split(':')[0] in saved_shapes]
                missing_vars = [var for var in tf.global_variables()
                                if var.name.split(':')[0] not in saved_shapes]
                param_saver = tf.train.Saver(found_vars)
    param_saver.restore(session, join(path, fname))
    session.run([var.initializer for var in missing_vars])
    return True
