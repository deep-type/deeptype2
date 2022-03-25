{
    features(candidate_graph_features=false, candidate_bit_features=false,
             country=true,
             employer=true,
             league=true,
             list=true,
             place_comma_place=true,
             sibling=true,
             sport=true,
             sports_team_member=true,
             spouse=true,
             us_state=true,
             educated_at=true,
             contemporary=true,
             political_party=true,
             global_normalization=true)::
    [] + [
        {
            // child sees parent in past predictions
            "name": "past_" + std.asciiLower(relation["relation_name"]) + (if relation["indirect"] then "_indirect" else "") + (if relation["normalize"] then "" else "_unnorm") + "_bit",
            "type": "predicted",
            "initial_state": "wikidata_linker_utils.autoregressive_type_interactions:past_prediction_bit_initial_state",
            "update": "wikidata_linker_utils.autoregressive_type_interactions:past_prediction_bit_update",
            "state_to_features": {
                "function": "wikidata_linker_utils.autoregressive_type_interactions:past_callable_bit_to_features",
                "args": {
                    "ids2ids": {
                        "function": "wikidata_linker_utils.autoregressive_type_interactions:find_by_relation",
                        "args": {"relation_name": relation["relation_name"], "classifier": relation["classifier"], "direct": relation["direct"], "indirect": relation["indirect"],
                                 "max_relatable": relation["max_relatable"], "related_or_empty": relation["related_or_empty"],
                                 "single_step_history": relation["single_step_history"]}
                    }
                }
            },
            "direction": "ltr",
            "normalize": true
        } for relation in ([] +
        (if employer then [{"relation_name": "EMPLOYER", "classifier": false, "direct": true, "indirect": false, "max_relatable": -1, "related_or_empty": false, "single_step_history": false}] else []) +
        (if sports_team_member then [{"relation_name": "MEMBER_OF_SPORTS_TEAM", "classifier": false, "direct": true, "indirect": false, "max_relatable": -1, "related_or_empty": false, "single_step_history": false}] else []) +
        (if educated_at then [{"relation_name": "EDUCATED_AT", "classifier": false, "direct": true, "indirect": false, "max_relatable": -1, "related_or_empty": false, "single_step_history": false}] else []) +
        (if sibling then [{"relation_name": "SIBLING", "classifier": false, "direct": true, "indirect": false, "max_relatable": -1, "related_or_empty": false, "single_step_history": false}] else []) +
        (if spouse then [{"relation_name": "SPOUSE", "classifier": false, "direct": true, "indirect": false, "max_relatable": -1, "related_or_empty": false, "single_step_history": false}] else []) +
        (if us_state then [{"relation_name": "us_state", "classifier": true, "direct": true, "indirect": false, "max_relatable": -1, "related_or_empty": false, "single_step_history": false}] else []) +
                          // Normalize and don't normalize (e.g. forward & backwards features)
        (if sport then [{"relation_name": "sport_occupation", "classifier": true, "direct": false, "indirect": true, "max_relatable": -1, "related_or_empty": true, "single_step_history": false}] else []) +
        (if sport then [{"relation_name": "sport_occupation", "classifier": true, "direct": false, "indirect": true, "max_relatable": -1, "related_or_empty": true, "single_step_history": false}] else []) +

        (if league then [{"relation_name": "league_part_of", "classifier": true, "direct": true, "indirect": true, "max_relatable": -1, "related_or_empty": true, "single_step_history": false}] else []) +

                            // Helps in rare cases (e.g. lists of players)
        (if employer then [{"relation_name": "MEMBER_OF_SPORTS_TEAM", "classifier": false, "direct": true, "indirect": true, "max_relatable": -1, "related_or_empty": false, "single_step_history": true}] else []) +
        (if country then [{"relation_name": "COUNTRY", "classifier": false, "direct": true, "indirect": true, "max_relatable": -1, "related_or_empty": true, "single_step_history": false}] else []) +
        (if political_party then [{"relation_name": "MEMBER_OF_POLITICAL_PARTY", "classifier": false, "direct": true, "indirect": true, "max_relatable": -1, "related_or_empty": true, "single_step_history": false}] else [])
    )
    ] +
        // this stage takes about 200ms per example:
    (if place_comma_place then [{
            "name": "place_comma_place",
            "type": "predicted",
            "initial_state": "wikidata_linker_utils.autoregressive_type_interactions:last_prediction_bit_initial_state",
            "update": "wikidata_linker_utils.autoregressive_type_interactions:last_prediction_bit_update",
            "state_to_features": "wikidata_linker_utils.autoregressive_type_interactions:last_prediction_bit_to_features",
            "direction": "ltr",
            "normalize": true
        }] else []) +
        // favor sequences of items with same instance of as previous item if they are inside a 'list'.
    (if contemporary then [{
            "name": "contemporary",
            "type": "predicted",
            "initial_state": "wikidata_linker_utils.autoregressive_type_interactions:past_prediction_bit_initial_state",
            "update": "wikidata_linker_utils.autoregressive_type_interactions:past_prediction_bit_update",
            "state_to_features": {
                "function": "wikidata_linker_utils.autoregressive_type_interactions:past_callable_bit_to_features",
                "args": {
                    "ids2ids": "wikidata_linker_utils.autoregressive_type_interactions:detect_contemporary"
                }
            },
            "direction": "ltr",
            "normalize": true
        }] else []) +
    (if list then [{
            "name": "list_matches_state_instance_of",
            "type": "predicted",
            "initial_state": "wikidata_linker_utils.autoregressive_type_interactions:past_prediction_bit_initial_state",
            "update": "wikidata_linker_utils.autoregressive_type_interactions:past_prediction_bit_update",
            "state_to_features": "wikidata_linker_utils.autoregressive_type_interactions:list_matches_state_instance_of_to_features",
            "direction": "ltr",
        }] else [])
}