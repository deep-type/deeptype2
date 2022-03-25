import argparse
import time
import marisa_trie
import numpy as np
import pandas

from os.path import join, realpath, dirname
from os import makedirs

from wikidata_linker_utils.wikidata_iterator import open_wikidata_file
from wikidata_linker_utils.file import true_exists
from wikidata_linker_utils.bash import count_lines
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.offset_array import save_record_with_offset
from wikidata_linker_utils.wikidata_ids import (
    WIKIDATA_IDS_NAME, WIKITILE_2_WIKIDATA_TRIE_NAME, WIKITILE_2_WIKIDATA_TSV_NAME,
    load_wikidata_ids, property_names, temporal_property_names, latlong_property_names
)
import wikidata_linker_utils.wikidata_properties as wikidata_properties

SCRIPT_DIR = dirname(realpath(__file__))
PROJECT_DIR = dirname(SCRIPT_DIR)

STORAGE_TYPE_ID = 0
STORAGE_TYPE_TIME = 1
STORAGE_TYPE_LAT = 2
STORAGE_TYPE_LONG = 3


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wikidata_dump", type=str,
        help="Path to wikidata dump file.")
    parser.add_argument(
        "wikidata", type=str,
        help="Path to save location for wikidata properties.")
    parser.add_argument("--batch_size", type=int, default=1000)
    return parser.parse_args(args=args)


def get_related_nested_field(doc_claims, nested_field):
    out = []
    for claim in doc_claims:
        mainsnak = claim.get("mainsnak", None)
        if mainsnak is None:
            continue
        datavalue = mainsnak.get("datavalue", None)
        if datavalue is None:
            continue
        value = datavalue.get("value", None)
        if value is None:
            continue
        value_id = value.get(nested_field, None)
        if value_id is None:
            continue
        out.append(value_id)
    return out


def name2field_name(name, storage_type):
    if storage_type == STORAGE_TYPE_LONG:
        return name[:-len("_long")]
    elif storage_type == STORAGE_TYPE_LAT:
        return name[:-len("_lat")]
    return name


storage_type2getter = {
    STORAGE_TYPE_ID: lambda doc_claims: get_related_nested_field(doc_claims, "id"),
    STORAGE_TYPE_LAT: lambda doc_claims: [str(x) for x in get_related_nested_field(doc_claims, "latitude")[:1]],
    STORAGE_TYPE_LONG: lambda doc_claims: [str(x) for x in get_related_nested_field(doc_claims, "longitude")[:1]],
    STORAGE_TYPE_TIME: lambda doc_claims: get_related_nested_field(doc_claims, "time"),
}


def get_wikidata_mapping(name2id_path,
                         wikidata_ids_path,
                         jsons,
                         relation_names,
                         verbose=False):
    approx_max_quantity = 91381092  # December 25th 2020 amount of articles
    if verbose:
        pbar = None
        from IPython.display import clear_output
    else:
        pbar = get_progress_bar("collect wikilinks", max_value=approx_max_quantity)
        pbar.start()
        clear_output = None
    seen = 0

    relations = {
        name: (open(outfile, "wt", encoding="utf-8"),
               name2field_name(name, storage_type),
               storage_type2getter[storage_type])
        for name, outfile, storage_type in relation_names
    }
    fout_name2id = None if true_exists(name2id_path) else open(name2id_path, "wt", encoding="utf-8")
    fout_wikidata_ids = None if true_exists(wikidata_ids_path) else open(wikidata_ids_path, "wt", encoding="utf-8")
    try:
        t_then = time.time()
        seen_last = 0
        speed = None
        index = 0
        for doc in jsons:
            seen += 1
            if seen % 2000 == 0:
                if verbose:
                    t_now = time.time()
                    new_speed = (seen - seen_last) / (t_now - t_then)
                    if speed is None:
                        speed = new_speed
                    else:
                        speed = 0.9 * speed + 0.1 * new_speed
                    clear_output(wait=True)
                    print("%.3f%% done (%d seen, %.3f docs/s, ETA: %ds)" % (
                        100.0 * seen / approx_max_quantity,
                        seen,
                        speed,
                        int((approx_max_quantity - seen) / speed)
                    ), flush=True)
                    seen_last = seen
                    t_then = t_now
                else:
                    if seen < approx_max_quantity:
                        pbar.update(seen)
            if fout_name2id is not None:
                if "sitelinks" in doc:
                    for key, value in doc["sitelinks"].items():
                        if key.endswith("wiki"):
                            fout_name2id.write(key + "/" + value["title"] + "\t" + str(index) + "\n")
            index += 1
            if fout_wikidata_ids is not None:
                fout_wikidata_ids.write(doc["id"] + "\n")
            for name, (outfile, field_name, getter) in relations.items():
                outfile.write("\t".join(getter(doc["claims"].get(field_name, []))) + "\n")
        if pbar is not None:
            pbar.finish()
    finally:
        for (outfile, _, _) in relations.values():
            outfile.close()
        if fout_name2id is not None:
            fout_name2id.close()
        if fout_wikidata_ids is not None:
            fout_wikidata_ids.close()


def convert_wikidata_ids_to_ids(id2index, wikidata_ids):
    return [[id2index.get(wikidata_id, -1) for wikidata_id in propgroup] for propgroup in wikidata_ids]


def parse_year(text):
    pos = text[1:].find("-")
    return int(text[:pos + 1])


def parse_latlong_into_int(text):
    # storing into int32 a float64 that ranges from -90 to +90
    return int(float(text) * 1e7)


def values_exist(path):
    return (
        true_exists(path + "_values.npy") or
        true_exists(path + "_values.sparse.npy")
    )


def line2indices(id2index, line):
    if len(line) == 0:
        return []
    out = []
    for el in line.split("\t"):
        idx = id2index.get(el, None)
        if idx is None:
            continue
        else:
            out.append(idx)
    return out


def fixed_point_name_alternates(name):
    if name.endswith(")"):
        pos_closing = name.rfind("(")
        return (name, name[:pos_closing].strip())
    if name.endswith("ses"):
        return (name, name[:-2] + "is")
    if name.endswith("ies"):
        return (name, name[:-3] + "y")
    if name.endswith("s"):
        return (name, name[:-1])
    return (name,)


def build_fixed_point(out, prefix):
    wiki_fixed_point_save = join(out, "wikidata_%s_fixed_points_values.npy" % (prefix,))
    if not true_exists(wiki_fixed_point_save):
        print("building %s fixed point property." % (prefix,))
        trie = marisa_trie.RecordTrie('i').load(join(out, WIKITILE_2_WIKIDATA_TRIE_NAME))
        num_items = count_lines(join(out, WIKIDATA_IDS_NAME))
        fixed_point_relation = {}

        category_prefix = "%s/Category:" % (prefix,)
        article_prefix = "%s/" % (prefix,)
        relevant_items = trie.iteritems(category_prefix)

        for name, category_idx in relevant_items:
            article_name = article_prefix + name[len(category_prefix):]
            for fixed_point_name_alternate in fixed_point_name_alternates(article_name):
                matches = trie.get(fixed_point_name_alternate, None)
                if matches is not None and len(matches) > 0:
                    fixed_point_relation[category_idx] = [matches[0][0]]
                    break
        print("Found %d fixed point relations for %s" % (len(fixed_point_relation), prefix,))
        save_record_with_offset(
            join(out, "wikidata_%s_fixed_points" % (prefix,)),
            fixed_point_relation,
            num_items
        )


def sparse_store_attribute(num_items, output_path, input_buffer, storage_type):
    if storage_type == STORAGE_TYPE_TIME:
        parser = parse_year
    elif storage_type == STORAGE_TYPE_LAT or storage_type == STORAGE_TYPE_LONG:
        parser = parse_latlong_into_int
    else:
        raise ValueError("No storage method implemented for storage_type {}.".format(storage_type))
    value = np.zeros(num_items * 2 + 1, dtype=np.int32)
    position = 1
    seen = 0
    for idx, line in enumerate(input_buffer):
        for raw_attribute in line.split('\t'):
            if len(raw_attribute) > 0:
                value[position] = idx
                value[position + 1] = parser(raw_attribute)
                position += 2
                seen += 1
                break
    value[0] = num_items
    value = value[:position]
    np.save(output_path, value)


def main():
    args = parse_args()
    makedirs(args.wikidata, exist_ok=True)
    wikidata_names2prop_names = property_names(  # noqa
        join(PROJECT_DIR, "data", "wikidata", 'wikidata_property_names.json')
    )
    wikidata_names2temporal_prop_names = temporal_property_names(
        join(PROJECT_DIR, "data", "wikidata", 'wikidata_time_property_names.json')
    )
    wikidata_names2latlong_prop_names = latlong_property_names(
        join(PROJECT_DIR, "data", "wikidata", 'wikidata_latlong_property_names.json')
    )

    def name2storage_type(name):
        if name in wikidata_names2temporal_prop_names:
            return STORAGE_TYPE_TIME
        if name in wikidata_names2latlong_prop_names:
            if name.endswith("_lat"):
                return STORAGE_TYPE_LAT
            elif name.endswith("_long"):
                return STORAGE_TYPE_LONG
            else:
                raise ValueError("No storage type for property named: '{}'.".format(name))
        assert name in wikidata_names2prop_names
        return STORAGE_TYPE_ID

    # fields to make easily accessible:
    wikidata_important_properties = [
        wikidata_properties.INSTANCE_OF,
        wikidata_properties.SUBCLASS_OF,
        wikidata_properties.PART_OF,
        wikidata_properties.OCCUPATION,

        wikidata_properties.EMPLOYER,
        wikidata_properties.MEMBER_OF_SPORTS_TEAM,

        wikidata_properties.SIBLING,
        wikidata_properties.SPOUSE,
        wikidata_properties.CHILD,
        wikidata_properties.UNMARRIED_PARTNER,
        wikidata_properties.OFFICIAL_RESIDENCE,
        wikidata_properties.OFFICEHOLDER,
        wikidata_properties.OWNED_BY,
        wikidata_properties.PARENT_ORGANIZATION,
        wikidata_properties.MILITARY_RANK,
        wikidata_properties.EDUCATED_AT,
        wikidata_properties.LOCATION_OF_FORMATION,
        
        wikidata_properties.MEMBER_OF_POLITICAL_PARTY,
        wikidata_properties.MEMBER_OF,
        wikidata_properties.LEAGUE,

        wikidata_properties.FIELD_OF_WORK,
        wikidata_properties.FIELD_OF_THIS_OCCUPATION,
        wikidata_properties.MEDICAL_SPECIALITY,
        wikidata_properties.GENRE,
        wikidata_properties.SEX_OR_GENDER,
        wikidata_properties.COUNTRY_OF_CITIZENSHIP,
        wikidata_properties.COUNTRY,
        wikidata_properties.CONTINENT,
        wikidata_properties.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY,
        wikidata_properties.SPORT,
        wikidata_properties.STUDIES,
        wikidata_properties.SERIES,
        wikidata_properties.USE,
        wikidata_properties.LOCATION,
        wikidata_properties.FACET_OF,
        wikidata_properties.IS_A_LIST_OF,
        wikidata_properties.COUNTRY_OF_ORIGIN,
        wikidata_properties.PRODUCT_OR_MATERIAL_PRODUCED,
        wikidata_properties.INDUSTRY,
        wikidata_properties.PARENT_TAXON,
        wikidata_properties.APPLIES_TO_TERRITORIAL_JURISDICTION,
        wikidata_properties.POSITION_HELD,
        wikidata_properties.CATEGORYS_MAIN_TOPIC,
        # temporal properties
        wikidata_properties.PUBLICATION_DATE,
        wikidata_properties.DATE_OF_BIRTH,
        wikidata_properties.DATE_OF_DEATH,
        wikidata_properties.INCEPTION,
        wikidata_properties.DISSOLVED_OR_ABOLISHED,
        wikidata_properties.POINT_IN_TIME,
        wikidata_properties.START_TIME,
        wikidata_properties.END_TIME,
        # lat long property
        wikidata_properties.COORDINATE_LOCATION_LATITUDE,
        wikidata_properties.COORDINATE_LOCATION_LONGITUDE,
    ]
    wikidata_important_properties_fnames = [
        (name, join(args.wikidata, "wikidata_%s.txt" % (name,)), name2storage_type(name))
        for name in wikidata_important_properties
    ]

    missing_wikidata_important_properties_fnames = [
        (name, outfile, storage_type)
        for name, outfile, storage_type in wikidata_important_properties_fnames
        if not true_exists(outfile)
    ]

    wikidata_ids_path = join(args.wikidata, WIKIDATA_IDS_NAME)
    wikititle2wikidata_path = join(args.wikidata, WIKITILE_2_WIKIDATA_TSV_NAME)

    work_to_be_done = (
        not true_exists(wikidata_ids_path) or
        not true_exists(wikititle2wikidata_path) or
        len(missing_wikidata_important_properties_fnames) > 0
    )

    if work_to_be_done:
        print("Collecting wikidata properties:")
        for name, _, _ in missing_wikidata_important_properties_fnames:
            print("    {}".format(name))
        get_wikidata_mapping(
            wikititle2wikidata_path,
            wikidata_ids_path,
            open_wikidata_file(args.wikidata_dump, args.batch_size),
            missing_wikidata_important_properties_fnames
        )

    numpy_wikidata_important_properties_fnames = [
        (name, outfile, storage_type)
        for name, outfile, storage_type in wikidata_important_properties_fnames
        if not values_exist(join(args.wikidata, "wikidata_%s" % (name,)))
    ]

    # obtain a mapping from id -> number
    if len(numpy_wikidata_important_properties_fnames) > 0:
        _, id2index = load_wikidata_ids(args.wikidata)
        # make relations numerical:
        for relname, outfile, storage_type in numpy_wikidata_important_properties_fnames:
            with open(outfile, "rt", encoding="utf-8") as fin:
                lines = fin.read().splitlines()
            fin_pbar = get_progress_bar("loading relation %r" % (relname,))(lines)
            if storage_type in (STORAGE_TYPE_TIME, STORAGE_TYPE_LAT, STORAGE_TYPE_LONG):
                sparse_store_attribute(num_items=len(lines),
                                       output_path=join(args.wikidata, "wikidata_%s_values.sparse.npy" % (relname,)),
                                       input_buffer=fin_pbar,
                                       storage_type=storage_type)
            else:
                relation = [
                    line2indices(id2index, line) for line in fin_pbar
                ]
                save_record_with_offset(
                    join(args.wikidata, "wikidata_%s" % (relname,)),
                    relation
                )
        del id2index

    # convert the mapping from wikinames to integer values:
    trie_save_path = join(args.wikidata, WIKITILE_2_WIKIDATA_TRIE_NAME)
    if not true_exists(trie_save_path):
        print("loading wikipedia name -> wikidata")
        name2id = pandas.read_csv(wikititle2wikidata_path, sep="\t", encoding='utf-8')
        print("loaded")
        trie = marisa_trie.RecordTrie(
            'i',
            get_progress_bar("convert to trie", max_value=name2id.shape[0])(
                (key, (value,)) for _, key, value in name2id.itertuples()
            )
        )
        trie.save(trie_save_path)

    build_fixed_point(args.wikidata, "enwiki")


if __name__ == '__main__':
    main()
