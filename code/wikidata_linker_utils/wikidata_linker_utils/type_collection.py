import json
import warnings

from os.path import join, exists, dirname, realpath
from functools import lru_cache

import marisa_trie
import requests
import numpy as np

from .successor_mask import (
    successor_mask, invert_relation, offset_values_mask, multi_step_successor_mask, distance, inside_polygon, is_member_with_path
)
from .offset_array import OffsetArray, SparseAttribute
from .wikidata_ids import (
    WIKITILE_2_WIKIDATA_TRIE_NAME, load_wikidata_ids, property_names, temporal_property_names, latlong_property_names
)
from .name_server import load_names_or_name_server
from . import wikidata_properties as wprop

WIKIDATA_LINKER_UTILS_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(dirname(WIKIDATA_LINKER_UTILS_DIR)), "data")


class CachedRelation(object):
    def __init__(self, use, state):
        self.use = use
        self.state = state


@lru_cache(maxsize=None)
def get_name(wikidata_id):
    res = requests.get("https://www.wikidata.org/wiki/" + wikidata_id)
    el = res.text.find('<title>')
    el_end = res.text.find('</title>')
    return res.text[el + len('<title>'):el_end]


def default_printer(*args, **kwargs):
    return print(*args, **kwargs)


class Path(object):
    __slots__ = ["prev_node", "field", "node"]
    def __init__(self, prev_node, field, node):
        self.field = field
        self.node = node
        self.prev_node = prev_node
        
    def decompose_path(self):
        if self.field is None:
            return [self.node]
        return self.prev_node.decompose_path() + [self.field, self.node]


class TypeCollection(object):
    def __init__(self, path, num_names_to_load=100000, language_path=None, prefix="enwiki", verbose=True,
                 cache=True):
        self.cache = cache
        self.path = path
        self.verbose = verbose
        self.wikidata_names2prop_names = property_names(
            join(path, 'wikidata_property_names.json')
        )
        self.wikidata_names2temporal_prop_names = temporal_property_names(
            join(path, 'wikidata_time_property_names.json')
        )
        self.wikidata_names2latlong_prop_names = latlong_property_names(
            join(path, 'wikidata_latlong_property_names.json')
        )
        for key, val in self.wikidata_names2latlong_prop_names.items():
            if key.endswith("_lat"):
                self.wikidata_names2prop_names[key] = "latitude"
            elif key.endswith("_long"):
                self.wikidata_names2prop_names[key] = "longitude"
        # add wikipedia english category links:
        self.wikidata_names2prop_names[wprop.CATEGORY_LINK] = "category_link"
        self.wikidata_names2prop_names[wprop.FIXED_POINTS] = "fixed_points"
        self.known_names, self.num_names_to_load, self.name_server = load_names_or_name_server(
            path,
            num_names_to_load,
            prefix=prefix
        )
        self.ids, self.name2index = load_wikidata_ids(path, verbose=self.verbose)
        self._relations = {}
        self._attributes = {}
        self._inverted_relations = {}
        self._article2id = None
        self._web_get_name = True
        self._state_abbreviations = None
        self._satisfy_cache = {}

        # empty blacklist:
        self.set_bad_node(
            set(), set()
        )
        if language_path is not None:
            article_links = np.load(join(language_path, "trie_index2indices_values.npy"))
            article_links_counts = np.load(join(language_path, "trie_index2indices_counts.npy"))
            self._weighted_articles = np.bincount(article_links, weights=article_links_counts).astype(np.int32)
            if len(self._weighted_articles) != len(self.ids):
                self._weighted_articles = np.concatenate(
                    [
                        self._weighted_articles,
                        np.zeros(len(self.ids) - len(self._weighted_articles), dtype=np.int32)
                    ]
                )
        else:
            self._weighted_articles = None

    def state_abbreviations(self):
        if self._state_abbreviations is None:
            with open(join(DATA_DIR, "state_abbreviations.json"), "rt") as fin:
                self._state_abbreviations = json.load(fin)
        return self._state_abbreviations

    def attribute(self, name):
        if name not in self._attributes:
            is_temporal = name in self.wikidata_names2temporal_prop_names
            is_latlong = name in self.wikidata_names2latlong_prop_names
            assert(is_temporal or is_latlong), "load relations using `relation` method."
            if self.verbose:
                print('load %r (%r)' % (name, self.wikidata_names2prop_names[name],))
            self._attributes[name] = SparseAttribute.load(
                join(self.path, "wikidata_%s" % (name,))
            )
        return self._attributes[name]

    @property
    def article2id(self):
        if self._article2id is None:
            if self.verbose:
                print('load %r' % ("article2id",))
            self._article2id = marisa_trie.RecordTrie('i').load(
                join(self.path, WIKITILE_2_WIKIDATA_TRIE_NAME)
            )
            if self.verbose:
                print("done.")
        return self._article2id

    def distance(self, latitude, longitude, mask=None):
        latitudes = self.attribute(wprop.COORDINATE_LOCATION_LATITUDE)
        longitudes = self.attribute(wprop.COORDINATE_LOCATION_LONGITUDE)
        if mask is None:
            mask = latitudes.mask
        else:
            mask = np.logical_and(mask, latitudes.mask)
        return distance(latitudes.dense, longitudes.dense, mask, latitude, longitude)

    def inside_polygon(self, polygon, mask=None):
        latitudes = self.attribute(wprop.COORDINATE_LOCATION_LATITUDE)
        longitudes = self.attribute(wprop.COORDINATE_LOCATION_LONGITUDE)
        if mask is None:
            mask = latitudes.mask
        else:
            mask = np.logical_and(mask, latitudes.mask)
        polygon = np.array(polygon, dtype=np.float32)
        assert polygon.ndim == 2, "expected a 2D array of coordinates of the form [(lat1, long1), (lat2, long2), ...]."
        return inside_polygon(latitudes.dense, longitudes.dense, mask, polygon)

    def relation(self, name):
        if name.endswith(".inv"):
            return self.get_inverted_relation(name[:-4])
        if name not in self._relations:
            is_temporal = name in self.wikidata_names2temporal_prop_names
            is_latlong = name in self.wikidata_names2latlong_prop_names
            assert(not is_temporal and not is_latlong), "load attributes using `attribute` method."
            if self.verbose:
                print('load %r (%r)' % (name, self.wikidata_names2prop_names[name],))
            self._relations[name] = OffsetArray.load(
                join(self.path, "wikidata_%s" % (name,)),
                compress=True
            )
            if len(self._relations[name]) != len(self.ids):
                raise ValueError(("wrong version of relation %r, has different "
                                  "number of members (%r) than known ids (%r)") % (
                                  name, len(self._relations[name]), len(self.ids)))
        return self._relations[name]

    def set_bad_node(self, bad_node, bad_node_pair):
        changed = False
        if hasattr(self, "_bad_node") and self._bad_node != bad_node:
            changed = True
        if hasattr(self, "_bad_node_pair") and self._bad_node_pair != bad_node_pair:
            changed = True

        self._bad_node = bad_node
        self._bad_node_pair = bad_node_pair
        self._bad_node_array = np.array(list(bad_node), dtype=np.int32)

        bad_node_pair_right = {}
        for node_left, node_right in self._bad_node_pair:
            if node_right not in bad_node_pair_right:
                bad_node_pair_right[node_right] = [node_left]
            else:
                bad_node_pair_right[node_right].append(node_left)
        bad_node_pair_right = {
            node_right: np.array(node_lefts, dtype=np.int32)
            for node_right, node_lefts in bad_node_pair_right.items()
        }
        self._bad_node_pair_right = bad_node_pair_right

        if changed:
            self.reset_cache()

    def get_name(self, identifier, web=False, add_id=True):
        if self.name_server:
            name = self.known_names.get(self.ids[identifier])
        elif identifier >= self.num_names_to_load and (web or self._web_get_name):
            try:
                res = get_name(self.ids[identifier])
                if add_id:
                    res = res + " (" + self.ids[identifier] + ")"
                return res
            except requests.exceptions.ConnectionError:
                self._web_get_name = False
        else:
            name = self.known_names.get(identifier, None)
        if name is None:
            return self.ids[identifier]
        else:
            res = name
            if add_id:
                res = res + " (" + self.ids[identifier] + ")"
            return res

    def describe_connection(self, source, destination, allowed_edges, bad_nodes=None, max_steps=None):
        if bad_nodes is None:
            bad_nodes = self._bad_node
        else:
            bad_nodes = self._bad_node | set(bad_nodes)
        if isinstance(source, str):
            if source in self.name2index:
                source_index = self.name2index[source]
            else:
                source_index = self.article2id["enwiki/" + source][0][0]
        else:
            source_index = source

        if isinstance(destination, str):
            if destination in self.name2index:
                dest_index = self.name2index[destination]
            else:
                dest_index = self.article2id["enwiki/" + destination][0][0]
        else:
            dest_index = destination

        found_path = self.is_member_with_path(
            source_index,
            allowed_edges,
            [dest_index],
            bad_nodes=bad_nodes,
            max_steps=max_steps
        )
        if found_path is not None:
            _, path = found_path
            for el in path:
                if isinstance(el, str):
                    if el.endswith(".inv"):
                        print("    {} (inverted {})".format(el, self.wikidata_names2prop_names[el[:-4]]))
                    else:
                        print("    {} ({})".format(el, self.wikidata_names2prop_names[el]))
                else:
                    print(self.get_name(el), el)
        else:
            print('%r and %r are not connected' % (source, destination))

    def is_member_with_path(self, root, fields, member_fields, bad_nodes, max_steps=None):
        if max_steps is None:
            max_steps = float("inf")

        if len(member_fields) == 1:
            res = is_member_with_path([self.relation(field) for field in fields],
                                      int(root), fields, member_fields, bad_nodes, max_steps)
            if len(res) == 0:
                return None
            for i in range(1, len(res), 2):
                res[i] = fields[res[i]]
            return True, res
        else:
            visited = set()
            candidates = [Path(None, None, root)]
            steps = 0
            while len(candidates) > 0 and steps < max_steps:
                new_candidates = []
                for candidate in candidates:
                    for field in fields:
                        field_parents = self.relation(field)[candidate.node]
                        for el in field_parents:
                            if el not in bad_nodes and (root, el) not in self._bad_node_pair and el not in visited:
                                if el in member_fields:
                                    return True, candidate.decompose_path() + [field, el]
                                visited.add(el)
                                new_candidates.append(
                                    Path(candidate, field, el)
                                )
                candidates = new_candidates
                steps += 1
            return None

    def get_inverted_relation(self, relation_name):
        if relation_name.endswith(".inv"):
            return self.relation(relation_name[:-4])
        if relation_name not in self._inverted_relations:
            new_values_path = join(self.path, "wikidata_inverted_%s_values.npy" % (relation_name,))
            new_offsets_path = join(self.path, "wikidata_inverted_%s_offsets.npy" % (relation_name,))

            if not exists(new_values_path):
                relation = self.relation(relation_name)
                if self.verbose:
                    print("inverting relation %r (%r)" % (relation_name, self.wikidata_names2prop_names[relation_name],))
                new_values, new_offsets = invert_relation(
                    relation.values,
                    relation.offsets
                )
                np.save(new_values_path, new_values)
                np.save(new_offsets_path, new_offsets)
            if self.verbose:
                print("load inverted %r (%r)" % (relation_name, self.wikidata_names2prop_names[relation_name]))
            self._inverted_relations[relation_name] = OffsetArray.load(
                join(self.path, "wikidata_inverted_%s" % (relation_name,)),
                compress=True
            )
            if len(self._inverted_relations[relation_name]) != len(self.ids):
                raise ValueError(("wrong version of inverted relation %r, has different "
                                  "number of members (%r) than known ids (%r)") % (
                                  relation_name, len(self._inverted_relations[relation_name]), len(self.ids)))
        return self._inverted_relations[relation_name]

    def successor_mask(self, relation, active_nodes):
        if isinstance(active_nodes, list):
            active_nodes = np.array(active_nodes, dtype=np.int32)
        if active_nodes.dtype != np.int32:
            active_nodes = active_nodes.astype(np.int32)
        return successor_mask(
            relation.values, relation.offsets, self._bad_node_pair_right, active_nodes
        )

    def multi_step_successor_mask(self, relations, active_nodes, bad_nodes, max_steps=None, legacy=False):
        if isinstance(active_nodes, list):
            active_nodes = np.array(active_nodes, dtype=np.int32)
        if active_nodes.dtype != np.int32:
            active_nodes = active_nodes.astype(np.int32)
        if max_steps is None:
            max_steps = -1

        bad_node_array = self._bad_node_array
        if len(bad_nodes) > 0:
            bad_node_array = np.concatenate([bad_node_array, bad_nodes]).astype(bad_node_array.dtype)

        state = np.zeros(relations[0].size(), dtype=np.bool)
        state[active_nodes] = True
        if legacy:
            # Python logic:
            step = 0
            while len(active_nodes) > 0:
                succ = None
                for relation in relations:
                    if succ is None:
                        succ = self.successor_mask(relation, active_nodes)
                    else:
                        succ = succ | self.successor_mask(relation, active_nodes)
                new_state = state | succ
                new_state[bad_node_array] = False
                (active_nodes,) = np.where(state != new_state)
                active_nodes = active_nodes.astype(np.int32)
                state = new_state
                step += 1
                if step == max_steps:
                    break
        else:
            # Faster cython path:
            multi_step_successor_mask(state, relations, active_nodes,
                                      bad_node_array,
                                      self._bad_node_pair_right,
                                      max_steps)
        return state

    def satisfy(self, relation_names, active_nodes, max_steps=None, legacy=False, bad_nodes=None):
        assert(len(relation_names) > 0), (
            "relation_names cannot be empty."
        )
        if bad_nodes is None:
            bad_nodes = tuple()
        if self.cache and isinstance(active_nodes, (list, tuple)) and len(active_nodes) < 100 and len(bad_nodes) < 100:
            satisfy_key = (tuple(sorted(relation_names)), tuple(sorted(active_nodes)),
                           max_steps, tuple(sorted(bad_nodes)))
            if satisfy_key in self._satisfy_cache:
                cached = self._satisfy_cache[satisfy_key]
                cached.use += 1
                return cached.state
        else:
            satisfy_key = None
        inverted_relations = [self.get_inverted_relation(relation_name) for relation_name in relation_names]
        state = self.multi_step_successor_mask(inverted_relations, active_nodes,
                                               max_steps=max_steps, bad_nodes=bad_nodes, legacy=legacy)
        if satisfy_key is not None:
            self._satisfy_cache[satisfy_key] = CachedRelation(1, state)
        return state

    def reset_cache(self):
        cache_keys = list(self._satisfy_cache.keys())
        for key in cache_keys:
            if self._satisfy_cache[key].use == 0:
                del self._satisfy_cache[key]
            else:
                self._satisfy_cache[key].use = 0

    def print_top_class_members(self, truth_table, name="Other", topn=20, printer=None):
        if printer is None:
            printer = default_printer
        if self._weighted_articles is not None:
            printer("%s category, highly linked articles in wikipedia:" % (name,))
            sort_weight = self._weighted_articles * truth_table
            linked_articles = int((sort_weight > 0).sum())
            printer("%s category, %d articles linked in wikipedia:" % (name, linked_articles))
            top_articles = np.argsort(sort_weight)[::-1]
            for art in top_articles[:topn]:
                if not truth_table[art]:
                    break
                printer("%r (%d)" % (self.get_name(art), self._weighted_articles[art]))
            printer("")
        else:
            printer("%s category, sample of members:" % (name,))
            top_articles = np.where(truth_table)[0]
            for art in top_articles[:topn]:
                printer("%r" % (self.get_name(art),))
            printer("")

    def class_report(self, relation_names, truth_table, name="Other", topn=20, printer=None):
        if printer is None:
            printer = default_printer
        active_nodes = np.where(truth_table)[0].astype(np.int32)
        num_active_nodes = len(active_nodes)
        printer("%s category contains %d unique items." % (name, num_active_nodes,))
        relations = [self.relation(relation_name) for relation_name in relation_names]
        for relation, relation_name in zip(relations, relation_names):
            mask = offset_values_mask(relation.values, relation.offsets, active_nodes)
            counts = np.bincount(relation.values[mask])
            topfields = np.argsort(counts)[::-1]
            printer("%s category, most common %r:" % (name, relation_name,))
            for field in topfields[:topn]:
                if counts[field] == 0:
                    break
                printer("%.3f%% (%d): %r" % (
                        100.0 * counts[field] / num_active_nodes,
                        counts[field],
                        self.get_name(field)
                        ))
            printer("")

        is_fp = np.logical_and(
            np.logical_or(
                self.relation(wprop.FIXED_POINTS + ".inv").edges() > 0,
                self.relation(wprop.FIXED_POINTS).edges() > 0
            ),
            truth_table
        )
        self.print_top_class_members(
            is_fp, topn=topn, name=name + " (fixed points)", printer=printer
        )
        if self._weighted_articles is not None:
            self.print_top_class_members(truth_table, topn=topn, name=name, printer=printer)

    def load_blacklist(self, path):
        with open(path, "rt") as fin:
            blacklist = json.load(fin)
        filtered_bad_node = []
        for el in blacklist["bad_node"]:
            if el not in self.name2index:
                warnings.warn("Node %r under `bad_node` is not a known wikidata id." % (
                    el
                ))
                continue
            filtered_bad_node.append(el)
        bad_node = set(self.name2index[el] for el in filtered_bad_node)

        filtered_bad_node_pair = []

        for el, oel in blacklist["bad_node_pair"]:
            if el not in self.name2index:
                warnings.warn("Node %r under `bad_node_pair` is not a known wikidata id." % (
                    el
                ))
                continue
            if oel not in self.name2index:
                warnings.warn("Node %r under `bad_node_pair` is not a known wikidata id." % (
                    oel
                ))
                continue
            filtered_bad_node_pair.append((el, oel))
        bad_node_pair = set(
            [(self.name2index[el], self.name2index[oel])
            for el, oel in filtered_bad_node_pair]
        )
        self.set_bad_node(bad_node, bad_node_pair)
