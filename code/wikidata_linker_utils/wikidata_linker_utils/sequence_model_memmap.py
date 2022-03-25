"""
Load a trained SequenceModel, strip out its embeddings, and save them to a
directory along with a model that can load them at runtime using
memmaps.

Usage
-----

```
python3 -m wikidata_linker_utils.sequence_model_memmap
    load_path save_path [--legacy LEGACY] [--config CONFIG]
```

"""
import tensorflow as tf
import numpy as np
import argparse
import os

from os.path import join

from .sequence_model import SequenceModel, MEMMAP_EMBEDDING_VARIABLES_PATH
from .training_config import Config
from .embedding import EMBEDDING_VARIABLES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory", type=str)
    parser.add_argument("output_directory", type=str)
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--legacy", action="store_true", default=False)
    parser.add_argument("--nolegacy", action="store_false", dest="legacy")
    args = parser.parse_args()
    if args.config is not None:
        config = Config.load(args.config)
    else:
        config = None
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf) as session:
        SequenceModel.load(session, args.model_directory,
                           args=None, legacy=args.legacy, faux_cudnn=True,
                           classifications=config.classifications if config is not None else None)

        embeddings = tf.get_collection(EMBEDDING_VARIABLES)
        if len(embeddings) == 0 and not args.legacy:
            raise ValueError("Loaded model does not contain any embeddings. Try running with --legacy")
        os.makedirs(args.output_directory, exist_ok=True)
        os.makedirs(join(args.output_directory, "embedding_variables"), exist_ok=True)

        for idx, embedding in enumerate(embeddings):
            embedding_np = session.run(embedding)
            np.save(join(args.output_directory,
                         MEMMAP_EMBEDDING_VARIABLES_PATH,
                         "embedding_{}.npy".format(idx)),
                    embedding_np)

    tf.reset_default_graph()
    with tf.Session(config=session_conf) as session:
        model = SequenceModel.load(session, args.model_directory,
                                   args=None, legacy=True, faux_cudnn=True,
                                   classifications=config.classifications if config is not None else None,
                                   create_embedding_lookup=False,
                                   memmap_embedding_variables_path=MEMMAP_EMBEDDING_VARIABLES_PATH)
        model.save(session, args.output_directory)


if __name__ == "__main__":
    main()
