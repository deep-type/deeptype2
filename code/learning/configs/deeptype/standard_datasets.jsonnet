{
    datasets(train, test, ignore_train_hash=true, oversample=null, merge_triggers=false, min_trigger_count=null, min_trigger_percent=null, min_reverse_trigger_percent=null, force_place_comma_place=false, base_path=null, disallow_venue_inside_place=true, string_match_min_trigger=false, tac=false, aida=true,
             force_city_of_at_in_place=true, detect_acronyms=true, force_rank_person=true, detect_lists=true, paper_data=false)::
    local internal_base_path = if base_path == null then "/Volumes/Samsung_T3/tahiti" else base_path;
    [] +
    (if tac then [{
            "type": "dev",
            "path": internal_base_path + "/test_datasets/tac_kbp10.json",
            "name": "tac_kbp10",
            "scenario": true,
            "x": 0,
            "y": [
                {
                    "objective": "type",
                    "language_path": "en_trie_stop"
                }
            ],
            "max_inception_date": 2010,
            "densify": true,
            "standard_dataset_loader": "wikidata_linker_utils.tac_kbp:load_tac_docs",
            "string_match_min_trigger": false,
        }] else []) +
        (if aida then [{
            "type": "dev",
            "path": internal_base_path + "/test_datasets/AIDA-YAGO2-testb.tsv",
            "name": "aida-testb",
            "scenario": true,
            "string_match_min_trigger": true,
            "x": 0,
            "y": [
                {
                    "objective": "type",
                    "language_path": "en_trie"
                }
            ],
            "densify": false,
            "max_inception_date": 1996,
            "aida_means": internal_base_path + "/2017-12/en_trie/aida_means",
            "standard_dataset_loader": "wikidata_linker_utils.aida:load_aida_docs",
        }
    ] else []) + (if aida && train then 
    [{
        "type": "train",
        "path": internal_base_path + "/test_datasets/AIDA-YAGO2-dataset.tsv",
        "name": "aida",
        "scenario": true,
        "merge_triggers": false,
        "string_match_min_trigger": true,
        "x": 0,
        "y": [
            {
                "objective": "type",
                "language_path": "en_trie"
            }
        ],
        "ignore_hash": ignore_train_hash,
        [if oversample != null then 'oversample']: oversample,
        "standard_dataset_loader": {
            "function": "wikidata_linker_utils.aida:load_aida_docs",
            "args": {
                "ignore": ["testb"]
            }
        }
    }] else [])
}
