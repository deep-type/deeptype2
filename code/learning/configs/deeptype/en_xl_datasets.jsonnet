{
    datasets(train=true, test=true, train_suffix="densified")::
    (if train then [{
        "type": "train",
        "path": "en_train_" + train_suffix + ".h5",
        "x": 0,
        "ignore": "other",
        "scenario": true,
        "y": [
        	{
                "column": 1,
                "anchor_column": 3,
                "objective": "type",
                "language_path": "en_trie"
            }
        ],
        "kwargs": {
            "uniform_sample": true
        }
    }] else []) +
    (if test then [
        {
            "type": "dev",
            "path": "en_dev.h5",
            "x": 0,
            "ignore": "other",
            "scenario": true,
            "y": [
            	{
                    "column": 1,
                    "anchor_column": 3,
                    "objective": "type",
                    "language_path": "en_trie"
                }
            ],
            "comment": "#//#"
        },
        {
            "type": "dev",
            "path": "en_dev_densified.h5",
            "x": 0,
            "ignore": "other",
            "scenario": true,
            "y": [
                {
                    "column": 1,
                    "anchor_column": 3,
                    "objective": "type",
                    "language_path": "en_trie"
                }
            ],
            "comment": "#//#",
            "filter_labels": {
                "function": "wikidata_linker_utils.dataset:must_be_ambiguous",
                "args": {}
            }
        },
        {
            "type": "dev",
            "path": "custom_en_test.h5",
            "name": "custom_dev_v0",
            "x": 0,
            "scenario": true,
            "ignore": "other",
            "y": [
                {
                    "column": 1,
                    "anchor_column": 3,
                    "objective": "type",
                    "language_path": "en_trie"
                }
            ],
            "comment": "#//#"
        },
    ] else [])
}