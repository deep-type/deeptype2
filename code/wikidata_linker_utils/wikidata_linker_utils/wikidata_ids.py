from os.path import exists, join, dirname
import marisa_trie
import json
import pickle
from .file import true_exists
from os import makedirs
WIKIDATA_IDS_NAME = "wikidata_ids.txt"
WIKITILE_2_WIKIDATA_TRIE_NAME = "wikititle2wikidata.marisa"
WIKITILE_2_WIKIDATA_TSV_NAME = "wikidata_wikititle2wikidata.tsv"
DEFAULT_NAME_SERVER_PORT = 7878


def sparql_query(query):
    import requests
    wikidata_url = "https://query.wikidata.org/sparql"
    response = requests.get(wikidata_url,
                            params={"format": "json", "query": query}).json()
    return response


def _extract_sparql_dict(response, key_field, value_field):
    out = {}
    for el in response["results"]['bindings']:
        value = el[value_field]['value']
        key = el[key_field]['value']
        if key.startswith("http://www.wikidata.org/entity/"):
            key = key[len("http://www.wikidata.org/entity/"):]
        if value.startswith("http://www.wikidata.org/entity/"):
            value = value[len("http://www.wikidata.org/entity/"):]
        out[key] = value
    return out


def load_wikidata_redirection(savename):
    return saved_sparql_query(savename,
                              'SELECT ?origin ?destination {?origin owl:sameAs ?destination.}',
                              key_field="origin",
                              value_field="destination")


class MarisaAsDict(object):
    def __init__(self, marisa):
        self.marisa = marisa

    def get(self, key, fallback):
        value = self.marisa.get(key, None)
        if value is None:
            return fallback
        else:
            return value[0][0]

    def __getitem__(self, key):
        value = self.marisa[key]
        return value[0][0]

    def __contains__(self, key):
        return key in self.marisa


def load_wikidata_ids(path, verbose=True, rebuild_trie=False):
    wikidata_ids_inverted_path = join(path, 'wikidata_ids_inverted.marisa')
    with open(join(path, WIKIDATA_IDS_NAME), "rt", encoding="utf-8") as fin:
        ids = fin.read().splitlines()
    if exists(wikidata_ids_inverted_path) and not rebuild_trie:
        if verbose:
            print("loading wikidata id -> index")
        name2index = MarisaAsDict(marisa_trie.RecordTrie('i').load(wikidata_ids_inverted_path))
        if verbose:
            print("done")
    else:
        if verbose:
            print("building trie")
        redirections = load_wikidata_redirection(join(path, "wikidata_ids_redirections.json"))
        if verbose:
            print("Got {} redirections.".format(len(redirections)))
        name2index = {name: k for k, name in enumerate(ids)}
        if verbose:
            print("Built index from non-redirected ids.")
        additions = 0
        for origin, destination in redirections.items():
            if origin not in name2index and destination in name2index:
                name2index[origin] = name2index[destination]
                additions += 1
        if verbose:
            print("Added {} redirections to the index.".format(additions))
        name2index = MarisaAsDict(
            marisa_trie.RecordTrie('i', [(name, (k,)) for name, k in name2index.items()])
        )
        name2index.marisa.save(wikidata_ids_inverted_path)
        if verbose:
            print("done")
    return (ids, name2index)


def load_names(path, num, prefix):
    names = {}
    errors = 0  # debug
    if num > 0:
        with open(path, "rt", encoding="utf-8") as fin:
            for line in fin:
                try:
                    name, number = line.rstrip('\n').split('\t')
                except ValueError:
                    errors += 1
                number = int(number)
                if number >= num:
                    break
                else:
                    if name.startswith(prefix):
                        names[number] = name[7:]
        if errors > 0:
            print(errors)  # debug
    return names


def load_all_names(path, prefix):
    names = {}
    errors = 0
    wtitle2wdata_fname = join(path, WIKITILE_2_WIKIDATA_TSV_NAME)
    if true_exists(wtitle2wdata_fname):
        fname = join(path, prefix + "_known_names.pkl")
        if not true_exists(fname):
            print("Extracting known_names...")
            with open(wtitle2wdata_fname, "rt", encoding="UTF-8") as fin:
                for line in fin:
                    try:
                        name, number = line.rstrip("\n").split("\t")
                    except ValueError:
                        errors += 1
                    number = int(number)
                    if name.startswith(prefix):
                        names[number] = name[7:]
            print(len(names), "names extracted for", prefix)
            if errors > 0:
                print(errors, "errors found during extraction.")
            with open(fname, "wb") as fout:
                pickle.dump(names, fout)
        else:
            with open(fname, "rb") as fin:
                names = pickle.load(fin)
            print(len(names), "names loaded for", prefix)

    else:
        print(f"File '{WIKITILE_2_WIKIDATA_TSV_NAME}' missing. Known names extraction aborted.")
    return names


def saved_sparql_query(savename, query, key_field="property", value_field="propertyLabel"):
    directory = dirname(savename)
    makedirs(directory, exist_ok=True)
    if true_exists(savename):
        with open(savename, "rt", encoding="utf-8") as fin:
            out = json.load(fin)
        return out
    else:
        out = _extract_sparql_dict(sparql_query(query), key_field=key_field, value_field=value_field)
        with open(savename, "wt", encoding="utf-8") as fout:
            json.dump(out, fout)
        return out


def property_names(prop_save_path):
    """"
    Retrieve the mapping between wikidata properties ids (e.g. "P531") and
    their human-readable names (e.g. "diplomatic mission sent").

    Returns:
        dict<str, str> : mapping from property id to property descriptor.
    """
    return saved_sparql_query(
        prop_save_path,
        """
        SELECT DISTINCT ?property ?propertyLabel
        WHERE
        {
            ?property a wikibase:Property .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        """
    )


def temporal_property_names(prop_save_path):
    """"
    Retrieve the mapping between wikidata properties ids (e.g. "P531") and
    their human-readable names (e.g. "diplomatic mission sent") only
    for fields that are time-based.

    Returns:
        dict<str, str> : mapping from property id to property descriptor.
    """
    return saved_sparql_query(
        prop_save_path,
        """
        SELECT DISTINCT ?property ?propertyLabel
        WHERE
        {
            ?property a wikibase:Property .
            {?property wdt:P31 wd:Q18636219} UNION {?property wdt:P31 wd:Q22661913} .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        """
    )


def latlong_property_names(prop_save_path):
    """"
    Retrieve the mapping between wikidata properties ids (e.g. "P531") and
    their human-readable names (e.g. "diplomatic mission sent") only
    for fields that are latitude longitude based.

    Returns:
        dict<str, str> : mapping from property id to property descriptor.
    """
    q = saved_sparql_query(
        prop_save_path,
        """
        SELECT DISTINCT ?property ?propertyLabel
        WHERE
        {
            ?property a wikibase:Property .
            {?property wdt:P1629 wd:Q22664} .
            SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        """
    )
    out = {}
    for k, v in q.items():
        out[k + "_lat"] = v
        out[k + "_long"] = v
    return out
