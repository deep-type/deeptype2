import json
import logging
from os.path import join, dirname, splitext

from .dataset import (TSVDataset,
                      CombinedDataset,
                      H5Dataset,
                      H5ScenarioDataset,
                      StandardClassificationDataset,
                      StandardScenarioDataset,
                      Oversample,
                      ClassificationHandler,
                      TrieHandler)
from .make_callable import make_callable


logger = logging.getLogger(__name__)


def make_dict_path_absolute(obj, basepath):
    copied = obj.copy()
    for key in ["path", "vocab"]:
        if key in copied:
            copied[key] = join(basepath, copied[key])
    return copied


def make_path_absolute(path, basepath):
    if path is not None:
        return join(basepath, path)
    return path


def multigetattr(sources, key):
    for source in sources:
        if hasattr(source, key):
            return getattr(source, key)
    raise AttributeError("Cannot find {} in {}".format(key, sources))


class Config(object):
    def __init__(self, datasets, features, objectives,
                 wikidata_path, classification_path, num_names_to_load, trie_path, post_process_spec):
        assert(len(features) > 0)
        self.datasets = datasets
        self.features = features
        self.objectives = objectives
        self.classifications = None
        self.tries = None
        self.wikidata_path = wikidata_path
        self.classification_path = classification_path
        self.num_names_to_load = num_names_to_load
        self.trie_path = trie_path
        self.post_process_spec = post_process_spec

        # build the objective names:
        self._named_objectives = [obj["name"] for obj in self.objectives]

    @classmethod
    def load(cls, path, cwd=None):
        try:
            import _jsonnet
            config = json.loads(_jsonnet.evaluate_file(path))
        except ImportError:
            logger.warning("Could not load jsonnet (`pip3 install jsonnet`), falling back to json config.")
            with open(path, "rt") as fin:
                config = json.load(fin)
        if cwd is None:
            cwd = dirname(path)
        return cls(
            datasets=[make_dict_path_absolute(dataset, cwd) for dataset in config['datasets']],
            features=[make_dict_path_absolute(feat, cwd) for feat in config['features']],
            objectives=[make_dict_path_absolute(objective, cwd) for objective in config['objectives']],
            wikidata_path=make_path_absolute(config.get("wikidata_path", None), cwd),
            trie_path=make_path_absolute(config.get("trie_path", None), cwd),
            classification_path=make_path_absolute(config.get("classification_path", None), cwd),
            post_process_spec=config.get("post_process_spec", []),
            num_names_to_load=config.get("num_names_to_load", 0),
        )

    def setup_classification_handler(self):
        if self.classifications is None:
            if self.wikidata_path is None or self.classification_path is None:
                raise ValueError("missing wikidata_path and "
                                 "classification_path, cannot "
                                 "construct H5Dataset.")
            self.classifications = ClassificationHandler(
                wikidata_path=self.wikidata_path,
                classification_path=self.classification_path,
                num_names_to_load=self.num_names_to_load)

    def setup_trie_handler(self):
        if self.tries is None:
            if self.trie_path is None:
                raise ValueError("missing trie_path, cannot "
                                 "construct Scenario dataset.")
            self.tries = TrieHandler(self.trie_path)

    def load_dataset_separate(self, dataset_type):
        paths = [dataset for dataset in self.datasets if dataset["type"] == dataset_type]
        all_examples = {}
        for dataset in paths:
            _, extension = splitext(dataset["path"])
            if extension == ".h5" or extension == ".hdf5":
                self.setup_classification_handler()
                if dataset.get("scenario", False):
                    self.setup_trie_handler()
                    examples = H5ScenarioDataset(
                        dataset["path"],
                        dataset['x'],
                        dataset['y'],
                        self._named_objectives,
                        ignore_value=dataset.get('ignore', None),
                        tries=self.tries,
                        classifications=self.classifications,
                        kwargs=dataset
                    )
                else:
                    examples = H5Dataset(
                        dataset["path"],
                        dataset['x'],
                        dataset['y'],
                        self._named_objectives,
                        ignore_value=dataset.get('ignore', None),
                        classifications=self.classifications,
                        kwargs=dataset
                    )
            else:
                standard_dataset_loader = dataset.get("standard_dataset_loader")
                if standard_dataset_loader is not None:
                    self.setup_classification_handler()
                    if dataset.get("scenario", False):
                        self.setup_trie_handler()
                        examples = StandardScenarioDataset(dataset["path"], dataset['x'], dataset['y'], self._named_objectives,
                                                           ignore_value=dataset.get('ignore', None),
                                                           tries=self.tries,
                                                           corpus_loader=make_callable(standard_dataset_loader),
                                                           classifications=self.classifications,
                                                           kwargs=dataset)
                    else:
                        examples = StandardClassificationDataset(dataset["path"], dataset['y'], self._named_objectives,
                                                                 classifications=self.classifications, kwargs=dataset,
                                                                 corpus_loader=make_callable(standard_dataset_loader))
                else:
                    examples = TSVDataset(
                        dataset["path"],
                        dataset['x'],
                        dataset['y'],
                        self._named_objectives,
                        comment=dataset.get('comment', '#'),
                        ignore_value=dataset.get('ignore', None),
                        retokenize=dataset.get('retokenize', False),
                        kwargs=dataset
                    )
            filename = dataset["path"].split('/')[-1].split(".")[0]
            dataset_name = dataset.get("name", filename)
            name = dataset_name
            if "oversample" in dataset and dataset["oversample"] is not None and dataset["oversample"] > 1:
                examples = Oversample(examples, dataset["oversample"])
            iteration = 1
            while name in all_examples:
                name = dataset_name + "-%d" % (iteration,)
                iteration += 1
            all_examples[name] = examples
        return all_examples

    def load_dataset(self, dataset_type, merge=True):
        datasets = self.load_dataset_separate(dataset_type)
        if merge:
            return CombinedDataset(list(datasets.values()))
        return datasets
