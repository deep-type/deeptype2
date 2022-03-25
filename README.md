DeepType 2
----------

This repository contains the code for the AAAI 2022 paper [DeepType 2: Superhuman Entity Linking, All You Need is Type Interactions](https://www.aaai.org/AAAI22Papers/AAAI-2612.RaimanJ.pdf).

This repository contains the code for training the model, the data densification, and the [human benchmark responses](human_benchmark), as well as the [paper appendix](DeepType2_appendix.pdf) containing supplementary figures and information.


# Training Data


## Get wikiarticle -> wikidata mapping (all languages) + Get anchor tags, redirections, category links, statistics (per language):

To store all wikidata ids, their key properties (`instance of`, `part of`, etc..), and
a mapping from all wikipedia article names to a wikidata id do as follows,
along with wikipedia anchor tags and links, in three languages: English (en), French (fr), and Spanish (es):

```
export DATA_DIR=data/
./extraction/full_preprocess.sh ${DATA_DIR} en fr es
```

## Graph projection

To build a graph projection using a set of rules inside `type_projection.py`
(or any Python file containing a `classify` method), and a set of nodes
that should not be traversed in `blacklist.json`:
To save a graph projection as a numpy array along with a list of classes to a
directory stored in `CLASSIFICATION_DIR`:

```
export LANGUAGE=fr
export DATA_DIR=data/
export CLASSIFICATION_DIR=data/type_classification
python3 extraction/project_graph.py ${DATA_DIR}wikidata/ extraction/blacklist.json extraction/classifiers/type_classifier.py  --export_classification ${CLASSIFICATION_DIR}
```

## Installation

### Mac OSX

```
pip3 install -r requirements
pip3 install wikidata_linker_utils_src/
```

### Fedora 25

```
sudo dnf install redhat-rpm-config
sudo dnf install gcc-c++
sudo pip3 install marisa-trie==0.7.5
sudo pip3 install -r requirements.txt.linux
pip3 install wikidata_linker_utils_src/
```


## Training

For each language create a training file:

```
export LANGUAGE=en
python3 extraction/produce_wikidata_tsv.py extraction/disambiguator_configs/${LANGUAGE}_disambiguator_config_export.json /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.tsv  --relative_to /Volumes/Samsung_T3/tahiti/2017-12/
```

Then create an H5 file from each language:

```
export LANGUAGE=en
python3 extraction/produce_windowed_h5_tsv.py  /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.tsv /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_train.h5 /Volumes/Samsung_T3/tahiti/2017-12/${LANGUAGE}_dev.h5 --window_size 10  --validation_start 1000000 --total_size 200500000
```

Create a training config with all languages, `my_config.json`. Paths to the datasets is relative to config file (e.g. you can place it in the same directory as the dataset h5 files):

```
{
    "datasets": [
        {
            "type": "train",
            "path": "en_train.h5",
            "x": 0,
            "ignore": "other",
            "y": [
                {
                    "column": 1,
                    "objective": "type",
                    "classification": "type_classification"
                },...
            ],
            "comment": "#//#"
        },
        {
            "type": "dev",
            "path": "en_dev.h5",
            "x": 0,
            "ignore": "other",
            "y": [
                {
                    "column": 1,
                    "objective": "type",
                    "classification": "type_classification"
                },...
            ],
            "comment": "#//#"
        }, ...
    ],
    "features": [
        {
            "type": "word",
            "dimension": 200,
            "max_vocab": 1000000
        },...
    ],
    "objectives": [
        {
            "name": "type",
            "type": "softmax",
            "vocab": "type_classes.txt"
        }, ...
    ],
    "wikidata_path": "/Volumes/Samsung_T3/tahiti/2017-02/wikidata",
    "classification_path": "/Volumes/Samsung_T3/tahiti/2017-02"
}
```

Launch training on a single gpu:

```
CUDA_VISIBLE_DEVICES=0 python3 learning/train_type.py my_config.json --cudnn --fused --hidden_sizes 200 200 --batch_size 256 --max_epochs 10000  --name TypeClassifier --weight_noise 1e-6  --save_dir my_great_model  --anneal_rate 0.9999
```

Several key parameters:

- `name`: main scope for model variables, avoids name clashing when multiple classifiers are loaded
- `batch_size`: how many examples are used for training simultaneously, can cause out of memory issues
- `max_epochs`: length of training before auto-stopping. In practice this number should be larger than needed.
- `fused`: glue all output layers into one, and do a single matrix multiply (recommended).
- `hidden_sizes`: how many stacks of LSTMs are used, and their sizes (here 2, each with 200 dimensions).
- `cudnn`: use faster CuDNN kernels for training
- `anneal_rate`: shrink the learning rate by this amount every 33000 training steps
- `weight_noise`: sprinkle Gaussian noise with this standard deviation on the weights of the LSTM (regularizer, recommended).


## Deep Type 2

To create dummy training data:

```
python3 test_train/build_test_scenario_dataset.py
```

To train a model on this dataset (use --nocudnn if on CPU):

```
python3 learning/train_type.py  test_train/scenario_config_test.json --nofused --nocudnn
```

### Config

The root of the configuration changes to support scenarios by adding a source of tries:
- `trie_path` : relative directory for tries.
- `objectives`: new objective type `scenario`.

See `test_train/scenario_config_test.json` for a working example.

#### Scenario Objective

- `dimensions`: list<int>, embedding dimensions for each label type
- `model`: list<dict>, list of layers stacked ontop of the embedding.
  Preferably combine layers together to ensure the loss can reflect
  the cross-interaction of features. Currently only a single layer
  is supported: `fully_connected`. Specify its dimensionality with `size`.
- `classifications`: list of type neighborhoods to use

Example:

```
{
    "name": "type",
    "type": "scenario",
    "classifications": [
        "a_classification",
        "b_classification"
    ],
    "dimensions": [10, 10],
    "model": [
        {
            "type": "fully_connected",
            "size": 20
        }
    ]
}
```

#### Scenario Dataset
The configuration of the H5 dataset to support scenario ranking:

- `scenario`: each dataset item in the datasets list must now state `"scenario": true`
- `y`: the y column can only have length 1 (e.g. only one source of labels per dataset object).
- `language_path`: You can specify which trie you will be using to derive
  labels using the `language_path` value. It should be a path relative
  to the `trie_path` key in the json root.

Example:

```
{
    "type": "train",
    "path": "scenario_data/scenario_train.h5",
    "x": 0,
    "scenario": true,
    "ignore": "other",
    "y": [
        {
            "column": 1,
            "anchor_column": 2,
            "objective": "type",
            "language_path": "fake_trie"
        }
    ]
},
```

#### Creating projections

```
python3 extraction/neighborhood_graph.py  ${DATA_DIR}wikidata/ --language_path ${DATA_DIR}${LANGUAGE}_trie/ --export_classification /path/to/exported/projections/
```

### Custom Test Set

```
cp data/custom_en_test_cases.xml /Volumes/Samsung_T3/tahiti/2017-12/ && python3 extraction/produce_wikidata_tsv.py extraction/disambiguator_configs/custom_en_test_disambiguator_config_export.json /Volumes/Samsung_T3/tahiti/2017-12/training_data/custom_en_test.tsv  --relative_to /Volumes/Samsung_T3/tahiti/2017-12/
python3 extraction/produce_windowed_h5_tsv.py  /Volumes/Samsung_T3/tahiti/2017-12/training_data/custom_en_test.tsv /Volumes/Samsung_T3/tahiti/2017-12/training_data/custom_en_test.h5 /dev/null --window_size 10  --validation_start 1000000 --total_size 200500000 --validation_size 0
```

### Autoregressive decoding

Autoregressive encoding enables entities to have autoregressive features during decoding (e.g. how well do you match with other predicted entities).
This can be tested using the `test_train/fake_standard_datasets/test_autoreg.tsv` dataset as follows:

Train a dummy model that has autoregressive features (higher `max_patience` is needed because many gradient steps are needed to make this feature work on dummy data):
```
python3 learning/train_type.py test_train/softautoreg_scenario_config_test.json --nocache_features --nocudnn --max_patience 50 --improvement_key token_correct --save_dir ~/Desktop/thing
```

From our save directory we can now use this exported model to see what features are being used:
```python3 learning/test_report.py test_train/softautoreg_scenario_config_test.json  --load_dir ~/Desktop/thing/ --report_dir ~/Desktop/myreport
```

You can view a webpage of this model's output here:
```
python3 learning/report_viewer.py test_train/softautoreg_scenario_config_test.json ~/Desktop/thing ~/Desktop/myreport/test_autoreg_report.pkl  --num_examples 2
```
