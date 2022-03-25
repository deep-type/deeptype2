import argparse
import numpy as np
from os.path import join, dirname, realpath
from os import makedirs
from wikidata_linker_utils.type_collection import TypeCollection
import wikidata_linker_utils.wikidata_properties as wprop
from wikidata_linker_utils.successor_mask import offset_values_mask, multi_step_neighborhood
from wikidata_linker_utils.offset_array import OffsetArray, save_values_offsets
from wikidata_linker_utils.logic import logical_ors
import logging
import time

logger = logging.getLogger(__name__)

SCRIPT_DIR = dirname(realpath(__file__))


def find_dictionary(collection, original_reachable_set, steps, relation_names):
    reachable_set = np.zeros(len(collection.ids), dtype=np.bool)
    reachable_set[original_reachable_set] = True
    total_counts = np.zeros(len(collection.ids), dtype=np.int32)
    relations = [collection.relation(relation_name) for relation_name in relation_names]
    for step in range(steps):
        round_counts = np.zeros(len(collection.ids), dtype=np.int32)
        for relation in relations:
            mask = offset_values_mask(relation.values, relation.offsets, np.where(reachable_set)[0].astype(np.int32))
            round_counts += np.bincount(relation.values[mask], minlength=len(collection.ids))
        reachable_set = round_counts > 0
        total_counts += round_counts
    return total_counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('wikidata', type=str,
                        help="Location of wikidata properties.")
    parser.add_argument('--language_path', type=str, required=True,
                        help="path to a specific language's wikipedia trie export.")
    parser.add_argument('--export_classification', type=str, default=None,
                        help="Location to save the result of the entity classification.")
    return parser.parse_args()


def _minimums(xs):
    x = np.zeros_like(xs[0].dense)
    x.fill((np.iinfo(np.int32).max))
    x[xs[0].mask] = xs[0].dense[xs[0].mask]
    for v in xs[1:]:
        x[v.mask] = np.minimum(x[v.mask], v.dense[v.mask])
    return x


def main():
    args = parse_args()
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    collection = TypeCollection(
        args.wikidata,
        num_names_to_load=0,
        cache=False
    )
    export_continuous = True
    export_en_reachable = True

    # grab just the indices you care about for building a limited set of useful classes
    original_reachable_set = np.unique(OffsetArray.load(
        join(args.language_path, "trie_index2indices")
    ).values)

    if export_en_reachable and args.export_classification is not None:
        vals = np.zeros(len(collection.ids), dtype=np.int32)
        vals.fill(len(original_reachable_set))
        vals[original_reachable_set] = np.arange(0, len(original_reachable_set))

        makedirs(join(args.export_classification, "en_reachable"), exist_ok=True)
        np.save(join(args.export_classification, "en_reachable", "classification.npy"), vals)
        with open(join(args.export_classification, "en_reachable", "classes.txt"), "wt") as fout:
            for idx in original_reachable_set:
                fout.write(collection.ids[idx] + "\n")
            fout.write("other" + "\n")

    projections = [
        {
            "relations": [wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
            "max_steps": 2,
            "dictionary_min_count": 5,
            "name": "instance_subclass"
        },
        {
            "relations": [wprop.OCCUPATION],
            "max_steps": 1,
            "dictionary_min_count": 10,
            "name": "occupation"
        },
        {
            "relations": [wprop.COUNTRY, wprop.COUNTRY_OF_CITIZENSHIP, wprop.COUNTRY_OF_ORIGIN],
            "max_steps": 1,
            "dictionary_min_count": 3,
            "name": "country"
        },
        {
            "relations": [wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY],
            "max_steps": 2,
            "dictionary_min_count": 10,
            "name": "admin_territorial_entity"
        },
        {
            "relations": [wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY],
            "max_steps": 4,
            "dictionary_condition": collection.satisfy([wprop.INSTANCE_OF], [collection.name2index["Q35657"]]),
            "dictionary_min_count": -1,
            "name": "us_state",
            # store original wikidata ids as neighbor ids instead of rebuilding a limited set from the discovered neighbors
            "remap_dictionary": True
        },
        {
            "relations": [wprop.SPORT, wprop.OCCUPATION, wprop.FIELD_OF_THIS_OCCUPATION],
            "max_steps": 4,
            "dictionary_condition": collection.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [collection.name2index["Q31629"]]),
            "dictionary_min_count": -1,
            "name": "sport_occupation",
            # store original wikidata ids as neighbor ids instead of rebuilding a limited set from the discovered neighbors
            "remap_dictionary": True
        },
        {
            "relations": [wprop.CATEGORY_LINK],
            "max_steps": 1,
            "dictionary_min_count": 53,
            "name": "en_category_link"
        },
        {
            "relations": [wprop.FIELD_OF_WORK,
                          wprop.FIELD_OF_THIS_OCCUPATION,
                          wprop.MEDICAL_SPECIALITY,
                          wprop.SPORT,
                          wprop.STUDIES,
                          wprop.INDUSTRY],
            "max_steps": 1,
            "dictionary_min_count": 10,
            "name": "sport_industry_work"
        },
        {
            "relations": [wprop.LEAGUE, wprop.PART_OF],
            "max_steps": 3,
            "dictionary_condition": collection.relation(wprop.LEAGUE + ".inv").edges() > 0,
            "dictionary_min_count": -1,
            "name": "league_part_of",
            # store original wikidata ids as neighbor ids instead of rebuilding a limited set from the discovered neighbors
            "remap_dictionary": True
        },
    ]

    for projection in projections:
        # > 10 or other
        if args.export_classification is not None:
            makedirs(join(args.export_classification, projection["name"]), exist_ok=True)
        if projection["dictionary_min_count"] == -1:
            dictionary = (np.where(projection["dictionary_condition"])[0] if "dictionary_condition" in projection else
                          np.arange(len(collection.ids))).astype(np.int32)
            print(len(dictionary))
        else:
            logger.info("Building dictionary for {}.".format(projection["name"]))
            counts = find_dictionary(collection,
                                     original_reachable_set,
                                     projection["max_steps"], projection["relations"])
            condition = counts > projection["dictionary_min_count"]
            if projection.get("dictionary_condition") is not None:
                condition = np.logical_and(condition, projection["dictionary_condition"])
            dictionary = np.where(condition)[0].astype(np.int32)
            del condition
            logger.info("{} has a dictionary with {} items.".format(projection["name"], len(dictionary)))
            if args.export_classification is not None:
                with open(join(args.export_classification, projection["name"], "classes.txt"), "wt") as fout:
                    for val in dictionary:
                        fout.write(collection.ids[val] + "\n")
        # now we need to create a local neighborhood for each entity
        relations = [collection.relation(relation_name) for relation_name in projection["relations"]]
        logger.info("Finding neighbors through {} steps using relations {}.".format(
            projection["max_steps"], projection["relations"]))
        t0 = time.time()
        values, offsets = multi_step_neighborhood(relations=relations,
                                                  max_steps=projection["max_steps"],
                                                  bad_node_array=collection._bad_node_array,
                                                  bad_node_pair_right=collection._bad_node_pair_right,
                                                  dictionary=dictionary)
        if projection.get("remap_dictionary", False):
            values = dictionary[values]
        t1 = time.time()
        logger.info("Done in {}s. Found {} edges.".format(t1 - t0, len(values)))
        if args.export_classification is not None:
            save_values_offsets(
                values, offsets,
                join(args.export_classification, projection["name"], "classification"))

    if export_continuous:
        lats = collection.attribute(wprop.COORDINATE_LOCATION_LATITUDE)
        longs = collection.attribute(wprop.COORDINATE_LOCATION_LONGITUDE)

        lats_np = np.radians(lats.dense / 1e7).astype(np.float32)
        longs_np = np.radians(longs.dense / 1e7).astype(np.float32)

        x = np.cos(lats_np) * np.cos(longs_np)
        y = np.cos(lats_np) * np.sin(longs_np)
        z = np.sin(lats_np)
        res = np.stack([x, y, z, lats.mask.astype(np.float32)], axis=-1)
        res[np.logical_not(lats.mask), :3] = 0
        if args.export_classification is not None:
            makedirs(join(args.export_classification, "latlong"), exist_ok=True)
            np.save(join(args.export_classification, "latlong", "classification.npy"), res)
        # does the item have a dissolved date?
        res = logical_ors([collection.attribute(wprop.DATE_OF_DEATH).mask,
                        collection.attribute(wprop.DISSOLVED_OR_ABOLISHED).mask,
                        collection.attribute(wprop.END_TIME).mask]).astype(np.float32)[:, None]
        if args.export_classification is not None:
            makedirs(join(args.export_classification, "was_dissolved"), exist_ok=True)
            np.save(join(args.export_classification, "was_dissolved", "classification.npy"), res)

        res = logical_ors([collection.attribute(wprop.DATE_OF_BIRTH).mask,
                        collection.attribute(wprop.INCEPTION).mask,
                        collection.attribute(wprop.POINT_IN_TIME).mask,
                        collection.attribute(wprop.START_TIME).mask,
                        collection.attribute(wprop.PUBLICATION_DATE).mask])
        vals = np.zeros((len(res), 2), dtype=np.int32)
        vals[:, 1] = res
        vals[:, 0] = _minimums([collection.attribute(wprop.DATE_OF_BIRTH),
                                collection.attribute(wprop.INCEPTION),
                                collection.attribute(wprop.POINT_IN_TIME),
                                collection.attribute(wprop.START_TIME),
                                collection.attribute(wprop.PUBLICATION_DATE)])
        vals[np.logical_not(res), 0] = 0
        if args.export_classification is not None:
            makedirs(join(args.export_classification, "inception_date_fixed"), exist_ok=True)
            np.save(join(args.export_classification, "inception_date_fixed", "classification.npy"), vals)


if __name__ == "__main__":
    main()
