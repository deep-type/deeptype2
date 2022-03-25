cimport cython

from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from third_party.libcpp11.vector cimport vector
from third_party.libcpp11.unordered_set cimport unordered_set
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
import marisa_trie

from wikidata_linker_utils.progressbar import get_progress_bar

from wikidata_linker_utils.bash import count_lines
from wikidata_linker_utils.anchor_filtering import clean_up_trie_source
from multiprocessing import cpu_count, Queue
from threading import Thread

from libc.stdio cimport sscanf, FILE
from libc.string cimport strchr
from libc.stdlib cimport atoi

from libc.math cimport sin, cos, atan2, sqrt

ctypedef int* int_ptr


cdef extern from "<string>" namespace "std" nogil:
    cdef cppclass string:
        string()
        string(const char *) except +
        bool operator != (const string&) const
        bool operator != (string&) const
        bool operator != (string) const
        bool operator == (const string&) const
        bool operator == (string&) const
        bool operator == (string) const

    # These are needed because Cython either thinks the string constuctor is ambigous
    # or doubts the existence of a comparison operator between strings.
    cdef bool operator != (string, const string&)
    cdef bool operator != (string, string&)
    cdef bool operator != (string, string)
    cdef bool operator == (string, const string&)
    cdef bool operator == (string, string&)
    cdef bool operator == (string, string)
    cdef bool operator != (string&, const string&)
    cdef bool operator != (string&, string&)
    cdef bool operator != (string&, string)
    cdef bool operator == (string&, const string&)
    cdef bool operator == (string&, string&)
    cdef bool operator == (string&, string)
    cdef bool operator != (const string&, const string&)
    cdef bool operator != (const string&, string&)
    cdef bool operator != (const string&, string)
    cdef bool operator == (const string&, const string&)
    cdef bool operator == (const string&, string&)
    cdef bool operator == (const string&, string)

# from libcpp.string cimport string

cdef extern from "stdio.h" nogil:
    #FILE * fopen ( const char * filename, const char * mode )
    FILE *fopen(const char *, const char *)
    #int fclose ( FILE * stream )
    int fclose(FILE *)
    #ssize_t getline(char **lineptr, size_t *n, FILE *stream);
    ssize_t getline(char **, size_t *, FILE *)


cdef class RedirectionsHolder(object):
    cdef unordered_map[string, string] _redirections

    def __init__(self, path):
        filename_byte_string = path.encode("utf-8")
        cdef char* fname = filename_byte_string
        cdef FILE* cfile
        cfile = fopen(fname, "rb")
        if cfile == NULL:
            raise FileNotFoundError(2, "No such file: '%s'" % (path,))
        cdef char *line = NULL
        cdef size_t l = 0
        cdef size_t read
        cdef char[256] source
        cdef char[256] dest
        cdef char* uppercased_in_python
        cdef char* tab_pos
        cdef char* end_pos

        with nogil:
            while True:
                read = getline(&line, &l, cfile)
                if read == -1:
                    break

                tab_pos = strchr(line, '\t')
                if (tab_pos - line) > 256 or tab_pos == NULL:
                    continue
                end_pos = strchr(tab_pos, '\n')
                if (end_pos - tab_pos) > 256:
                    continue
                return_code = sscanf(line, "%256[^\n\t]\t%256[^\n]", &source, &dest)
                if return_code != 2:
                    continue
                with gil:
                    decoded = source.decode("utf-8")
                    decoded = (decoded[0].upper() + decoded[1:]).encode("utf-8")
                    uppercased_in_python = decoded
                self._redirections[string(uppercased_in_python)] = string(dest)
        fclose(cfile)

    def __len__(self):
        return self._redirections.size()

    def __contains__(self, key):
        return self._redirections.find(key.encode("utf-8")) != self._redirections.end()

    def __getitem__(self, key):
        cdef unordered_map[string, string].iterator finder = self._redirections.find(key.encode("utf-8"))
        if finder == self._redirections.end():
            raise KeyError(key)
        return deref(finder).second.decode("utf-8")

    def get(self, key, default=None):
        cdef unordered_map[string, string].iterator finder = self._redirections.find(key.encode("utf-8"))
        if finder == self._redirections.end():
            return default
        return deref(finder).second.decode("utf-8")

    def _asdict(self):
        out = {}
        for kv in self._redirections:
            out[kv.first.decode("utf-8")] = kv.second.decode("utf-8")
        return out


def load_redirections(path):
    return RedirectionsHolder(path)


@cython.boundscheck(False)
@cython.wraparound(False)
def successor_mask(np.ndarray[int, ndim=1] values,
                   np.ndarray[int, ndim=1] offsets,
                   bad_node_pair_right,
                   np.ndarray[int, ndim=1] active_nodes):
    np_dest_array = np.zeros(len(offsets), dtype=np.bool)
    cdef bool * dest_array = bool_ptr(np_dest_array)
    cdef unordered_map[int, vector[int]] bad_node_pair_right_c
    cdef unordered_map[int, vector[bool]] bad_node_pair_right_c_backups
    for item, value in bad_node_pair_right.items():
        bad_node_pair_right_c[item] = value
        bad_node_pair_right_c_backups[item] = vector[bool](len(value), 0)
    cdef int i = 0
    cdef int j = 0
    cdef int active_nodes_max = active_nodes.shape[0]
    cdef int end = 0
    cdef int start = 0
    cdef int active_node = 0
    cdef int[:] subvalues = values
    cdef int[:] values_view = values
    cdef int * bad_node_pair_right_c_ptr
    cdef vector[bool] * bad_node_pair_right_c_backups_ptr

    with nogil:
        for i in range(active_nodes_max):
            active_node = active_nodes[i]
            end = offsets[active_node]
            if active_node == 0:
                start = 0
            else:
                start = offsets[active_node - 1]
            if bad_node_pair_right_c.find(active_node) != bad_node_pair_right_c.end():
                bad_node_pair_right_c_ptr = bad_node_pair_right_c[active_node].data()
                bad_node_pair_right_c_backups_ptr = &bad_node_pair_right_c_backups[active_node]
                for j in range(bad_node_pair_right_c[active_node].size()):
                    bad_node_pair_right_c_backups_ptr[0][j] = dest_array[bad_node_pair_right_c_ptr[j]]
                subvalues = values_view[start:end]
                for j in range(end - start):
                    dest_array[subvalues[j]] = 1
                for j in range(bad_node_pair_right_c[active_node].size()):
                    dest_array[bad_node_pair_right_c_ptr[j]] = bad_node_pair_right_c_backups_ptr[0][j]
            else:
                subvalues = values_view[start:end]
                for j in range(end - start):
                    dest_array[subvalues[j]] = 1
    return np_dest_array


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void successor_mask_internal(int* values,
                                  int* offsets,
                                  const unordered_map[int, vector[int]]& bad_node_pair_right_c,
                                  unordered_map[int, vector[bool]]& bad_node_pair_right_c_backups,
                                  vector[int]& active_nodes,
                                  const unordered_set[int]& bad_node_array_set,
                                  bool* dest_array,
                                  vector[int]& new_active_nodes) nogil:
    cdef int active_nodes_max = active_nodes.size()
    cdef int i
    cdef int j
    cdef int end = 0
    cdef int start = 0
    cdef int active_node = 0
    cdef int* subvalues
    cdef const int* bad_node_pair_right_c_ptr
    cdef vector[int] candidate_new
    for i in range(active_nodes_max):
        active_node = active_nodes[i]
        end = offsets[active_node]
        if active_node == 0:
            start = 0
        else:
            start = offsets[active_node - 1]
        if bad_node_pair_right_c.find(active_node) != bad_node_pair_right_c.end():
            bad_node_pair_right_c_ptr = bad_node_pair_right_c.at(active_node).data()
            bad_node_pair_right_c_backups_ptr = &bad_node_pair_right_c_backups[active_node]
            # make a backup of the state of the next nodes
            for j in range(bad_node_pair_right_c.at(active_node).size()):
                bad_node_pair_right_c_backups_ptr[0][j] = dest_array[bad_node_pair_right_c_ptr[j]]

            # attempt transition
            subvalues = values + start
            for j in range(end - start):
                # store changes to transition
                if not dest_array[subvalues[j]] and bad_node_array_set.find(subvalues[j]) == bad_node_array_set.end():
                    dest_array[subvalues[j]] = 1
                    candidate_new.emplace_back(subvalues[j])
            # in places where this edge should have no effect, undo changes using the history
            for j in range(bad_node_pair_right_c.at(active_node).size()):
                dest_array[bad_node_pair_right_c_ptr[j]] = bad_node_pair_right_c_backups_ptr[0][j]

            # see if changes persisted, and if so keep the proposed new nodes:
            for j in range(candidate_new.size()):
                if dest_array[candidate_new[j]]:
                    new_active_nodes.emplace_back(candidate_new[j])
            candidate_new.clear()
        else:
            subvalues = values + start
            for j in range(end - start):
                if not dest_array[subvalues[j]] and bad_node_array_set.find(subvalues[j]) == bad_node_array_set.end():
                    dest_array[subvalues[j]] = 1
                    new_active_nodes.emplace_back(subvalues[j])


@cython.boundscheck(False)
@cython.wraparound(False)
def multi_step_neighborhood_inner_work(int start,
                                       int end,
                                       relations,
                                       int max_steps,
                                       all_output,
                                       np.ndarray[int, ndim=1] dictionary):
    cdef int n_relations = len(relations)
    cdef vector[int*] relation_values
    cdef vector[int*] relation_offsets
    cdef int relation_idx
    for relation_idx in range(len(relations)):
        relation_values.emplace_back(<int*>((<long>relations[relation_idx].values.ctypes.data)))
        relation_offsets.emplace_back(<int*>((<long>relations[relation_idx].offsets.ctypes.data)))

    cdef vector[int] a
    cdef vector[int] b
    cdef int i
    cdef int origin
    cdef int step
    cdef int j
    cdef unordered_set[int] visited
    cdef unordered_map[int, int] mapping
    cdef vector[int]* origins = &a
    cdef vector[int]* successors = &b
    cdef vector[int]* temp
    cdef int total_size = 0
    cdef int output_idx = 0
    cdef vector[vector[int]] output
    cdef int dictionary_size = len(dictionary)

    with nogil:
        for i in range(dictionary_size):
            mapping[dictionary[i]] = i
        # output.resize(end - start)
        for i in range(start, end):
            # now perform multiple steps using either relation
            origins[0].clear()
            origins[0].emplace_back(i)
            successors[0].clear()
            for step in range(max_steps):
                for origin in origins[0]:
                    # do an access in an offset array:
                    for relation_idx in range(n_relations):
                        for j in range(0 if origin == 0 else relation_offsets[relation_idx][origin - 1],
                                       relation_offsets[relation_idx][origin]):
                            if visited.find(relation_values[relation_idx][j]) == visited.end():
                                successors[0].emplace_back(relation_values[relation_idx][j])
                                visited.emplace(relation_values[relation_idx][j])
                origins[0].clear()
                temp = origins
                origins = successors
                successors = temp

            output.emplace_back(vector[int]())
            output.back().reserve(visited.size())
            for el in visited:
                if mapping.find(el) != mapping.end():
                    output.back().emplace_back(mapping.at(el))
                    total_size += 1
            visited.clear()
            output_idx += 1

    cdef np.ndarray[int, ndim=1] values = np.zeros(total_size, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] offsets = np.zeros(end - start, dtype=np.int32)
    # now convert output to a single numpy array
    cdef int so_far = 0
    with nogil:
        for output_idx in range(end - start):
            offsets[output_idx] = output[output_idx].size()
            for j in range(output[output_idx].size()):
                values[so_far + j] = output[output_idx][j]
            so_far += output[output_idx].size()
    all_output.append((start, values, offsets))


@cython.boundscheck(False)
@cython.wraparound(False)
def multi_step_neighborhood(relations,
                            int max_steps,
                            np.ndarray[int, ndim=1] bad_node_array,
                            bad_node_pair_right,
                            np.ndarray[int, ndim=1] dictionary):
    # cdef unordered_map[int, vector[int]] bad_node_pair_right_c
    # cdef unordered_map[int, vector[bool]] bad_node_pair_right_c_backups
    # for item, value in bad_node_pair_right.items():
    #     bad_node_pair_right_c[item] = value
    #     bad_node_pair_right_c_backups[item] = vector[bool](len(value), 0)
    cdef int n_indices = len(relations[0].offsets)  # len(start_indices)
    all_output = []
    parallelize_work(
        n_indices,
        multi_step_neighborhood_inner_work,
        relations,
        max_steps,
        all_output,
        dictionary)
    all_output = sorted(all_output, key=lambda x: x[0])
    cdef np.ndarray[int, ndim=1] values = np.concatenate([o for _, o, _ in all_output])
    cdef np.ndarray[int, ndim=1] offsets = np.concatenate([o for _, _, o in all_output])
    cdef int i
    cdef int so_far = 0
    with nogil:
        for i in range(1, n_indices):
            offsets[i] += offsets[i - 1]
    return values, offsets


@cython.boundscheck(False)
@cython.wraparound(False)
def multi_step_successor_mask(np.ndarray[bool, ndim=1] np_dest_array,
                              relations,
                              np.ndarray[int, ndim=1] active_nodes,
                              np.ndarray[int, ndim=1] bad_node_array,
                              bad_node_pair_right,
                              int max_steps):
    cdef int step = 0
    cdef bool * dest_array = bool_ptr(np_dest_array)
    cdef unordered_map[int, vector[int]] bad_node_pair_right_c
    cdef unordered_map[int, vector[bool]] bad_node_pair_right_c_backups
    for item, value in bad_node_pair_right.items():
        bad_node_pair_right_c[item] = value
        bad_node_pair_right_c_backups[item] = vector[bool](len(value), 0)

    cdef int n_relations = len(relations)
    cdef vector[int*] relation_values
    cdef vector[int*] relation_offsets
    for relation_idx in range(len(relations)):
        relation_values.emplace_back(<int*>((<long>relations[relation_idx].values.ctypes.data)))
        relation_offsets.emplace_back(<int*>((<long>relations[relation_idx].offsets.ctypes.data)))

    cdef unordered_set[int] bad_node_array_set
    cdef vector[int] nodes_a = active_nodes
    cdef vector[int] nodes_b
    cdef vector[int]* active_nodes_ptr = &nodes_a
    cdef vector[int]* new_active_nodes_ptr = &nodes_b
    cdef int num_bad_nodes = len(bad_node_array)
    with nogil:
        for j in range(num_bad_nodes):
            bad_node_array_set.emplace(bad_node_array[j])
            dest_array[bad_node_array[j]] = 0
        while active_nodes_ptr[0].size() > 0:
            for relation_idx in range(n_relations):
                successor_mask_internal(relation_values[relation_idx],
                                        relation_offsets[relation_idx],
                                        bad_node_pair_right_c,
                                        bad_node_pair_right_c_backups,
                                        active_nodes_ptr[0],
                                        bad_node_array_set,
                                        dest_array,
                                        new_active_nodes_ptr[0])
            step += 1
            if step == max_steps:
                break
            new_active_nodes_ptr, active_nodes_ptr = active_nodes_ptr, new_active_nodes_ptr
            new_active_nodes_ptr[0].clear()


@cython.boundscheck(False)
@cython.wraparound(False)
def invert_relation(np.ndarray[int, ndim=1] values,
                    np.ndarray[int, ndim=1] offsets):

    cdef np.ndarray[int, ndim=1] new_values = np.empty_like(values)
    cdef np.ndarray[int, ndim=1] new_offsets = np.empty_like(offsets)

    cdef int max_offsets = len(offsets)
    cdef vector[vector[int]] inverted_edges = vector[vector[int]](max_offsets)
    cdef int so_far = 0
    cdef int i = 0
    cdef int j = 0
    cdef int[:] new_values_view = new_values
    cdef int[:] new_offsets_view = new_offsets
    cdef int[:] vector_view
    cdef int position = 0
    with nogil:
        for i in range(max_offsets):
            for j in range(offsets[i] - so_far):
                inverted_edges[values[so_far + j]].push_back(i)
            so_far = offsets[i]
        for i in range(max_offsets):
            for j in range(inverted_edges[i].size()):
                new_values_view[position + j] = inverted_edges[i][j]
            position += inverted_edges[i].size()
            new_offsets_view[i] = position
    return new_values, new_offsets


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void batch_is_related_inner(bool* result,
                                 int n,
                                 int_ptr indices,
                                 int_ptr offsets,
                                 int_ptr values,
                                 bool* condition_ptr,
                                 int max_relatable,
                                 bool related_or_empty) nogil:
    cdef unordered_set[int] all_neighbors
    cdef int i = 0
    cdef int j = 0
    for i in range(n):
        if indices[i] >= 0:
            for j in range(offsets[indices[i] - 1] if indices[i] > 0 else 0, offsets[indices[i]]):
                if condition_ptr == NULL or condition_ptr[values[j]]:
                    all_neighbors.emplace(values[j])
    if (max_relatable > 0 and all_neighbors.size() > max_relatable) or all_neighbors.size() == 0:
        return
    for i in range(n):
        if indices[i] >= 0 and all_neighbors.find(indices[i]) != all_neighbors.end():
            result[i] = True


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_is_related(np.ndarray[int, ndim=2] indices,
                     np.ndarray[int, ndim=1] offsets,
                     np.ndarray[int, ndim=1] values,
                     condition=None,
                     int max_relatable=-1,
                     bool related_or_empty=False):
    cdef np.ndarray[bool, ndim=2] out_is_related = np.zeros((indices.shape[0], indices.shape[1]), dtype=np.bool)
    cdef int i = 0
    cdef int n = len(indices)
    cdef int cols = indices.shape[1]
    cdef int_ptr offsets_ptr = <int*>offsets.data
    cdef int_ptr values_ptr = <int*>values.data
    cdef bool* out_is_related_ptr = bool_ptr(out_is_related)
    cdef int_ptr indices_ptr = <int*>indices.data
    cdef bool* condition_ptr = NULL
    if condition is not None:
        assert isinstance(condition, np.ndarray)
        condition_ptr = bool_ptr(condition)
    with nogil:
        for i in range(n):
            batch_is_related_inner(out_is_related_ptr + i * cols, cols, indices_ptr + i * cols, offsets_ptr, values_ptr, condition_ptr, max_relatable, related_or_empty)
    return out_is_related

@cython.boundscheck(False)
@cython.wraparound(False)
def last_non_negative_2d(np.ndarray[int, ndim=2] arr):
    cdef int i = 0
    cdef int j = 0
    cdef int n = arr.shape[0]
    cdef np.ndarray[int, ndim=1] out = np.zeros((n,), dtype=np.int32)
    cdef int m = arr.shape[1]
    cdef int_ptr arr_ptr = <int*>arr.data
    with nogil:
        for i in range(n):
            out[i] = -1
            for j in range(m - 1, -1, -1):
                if arr_ptr[i * m + j] >= 0:
                    out[i] = arr_ptr[i * m + j]
                    break
    return out


def last_non_negative(arr):
    if arr.ndim == 0:
        return arr
    if arr.size == 0:
        return -np.ones(arr.shape[:-1], dtype=np.int32)
    return last_non_negative_2d(arr.reshape(-1, arr.shape[-1]) if arr.ndim != 2 else arr).reshape(arr.shape[:-1])


# BxN candidates <-> state Bx1xK -> BxN
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void batch_is_related_pair_inner(bool* result,
                                      int n_sources,
                                      int n_destinations,
                                      int_ptr sources,
                                      int_ptr destinations,
                                      int_ptr offsets,
                                      int_ptr values,
                                      bool direct,
                                      bool indirect,
                                      bool* condition_ptr,
                                      int max_relatable,
                                      bool related_or_empty) nogil:
    cdef unordered_set[int] all_destinations
    cdef int i = 0
    cdef int j = 0
    # share a set that is the sum total of all destinations for any item in sources
    # we only compute this set once.
    if indirect:
        for i in range(n_destinations):
            if destinations[i] >= 0:
                for j in range(offsets[destinations[i] - 1] if destinations[i] > 0 else 0, offsets[destinations[i]]):
                    if condition_ptr == NULL or condition_ptr[values[j]]:
                        all_destinations.emplace(values[j])
    if direct:
        for i in range(n_destinations):
            if destinations[i] >= 0 and (condition_ptr == NULL or condition_ptr[destinations[i]]):
                all_destinations.emplace(destinations[i])
    if (max_relatable > 0 and all_destinations.size() > max_relatable):
        return
    if all_destinations.size() == 0:
        for i in range(n_sources):
            if sources[i] >= 0:
                result[i] = True
        return
    # for each item in source we traverse the graph to see if it connects to a destination
    for i in range(n_sources):
        if sources[i] >= 0:
            if related_or_empty and offsets[sources[i] - 1] == offsets[sources[i]]:
                result[i] = True
            # see if source connects to any item in destinations.
            for j in range(offsets[sources[i] - 1] if sources[i] > 0 else 0, offsets[sources[i]]):
                if all_destinations.find(values[j]) != all_destinations.end():
                    result[i] = True
                    break


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_is_related_pair(np.ndarray[int, ndim=2] sources,
                          np.ndarray[int, ndim=2] destinations,
                          np.ndarray[int, ndim=1] offsets,
                          np.ndarray[int, ndim=1] values,
                          bool direct,
                          bool indirect,
                          condition=None,
                          int max_relatable=-1,
                          bool related_or_empty=False):
    cdef np.ndarray[bool, ndim=2] out_is_related = np.zeros((sources.shape[0], sources.shape[1]), dtype=np.bool)
    assert sources.shape[0] == destinations.shape[0], "expected sources and destinations to have the same first dimension."
    cdef int i = 0
    cdef int n = len(sources)
    cdef int n_sources = sources.shape[1]
    cdef int n_destinations = destinations.shape[1]
    cdef int_ptr offsets_ptr = <int*>offsets.data
    cdef int_ptr values_ptr = <int*>values.data
    cdef bool* out_is_related_ptr = bool_ptr(out_is_related)
    cdef int_ptr sources_ptr = <int*>sources.data
    cdef int_ptr destinations_ptr = <int*>destinations.data
    cdef bool* condition_ptr = NULL
    if condition is not None:
        assert isinstance(condition, np.ndarray)
        condition_ptr = bool_ptr(condition)
    with nogil:
        for i in range(n):
            batch_is_related_pair_inner(out_is_related_ptr + i * n_sources,
                                        n_sources, n_destinations,
                                        sources_ptr + i * n_sources,
                                        destinations_ptr + i * n_destinations,
                                        offsets_ptr, values_ptr,
                                        direct,
                                        indirect,
                                        condition_ptr,
                                        max_relatable,
                                        related_or_empty)
    return out_is_related


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_is_related_pair_broadcast(np.ndarray[int, ndim=3] sources,
                                    np.ndarray[int, ndim=3] destinations,
                                    np.ndarray[int, ndim=1] offsets,
                                    np.ndarray[int, ndim=1] values,
                                    bool direct,
                                    bool indirect,
                                    condition=None,
                                    int max_relatable=-1,
                                    bool related_or_empty=False):
    assert sources.shape[0] == destinations.shape[0] or sources.shape[0] == 1 or destinations.shape[0] == 1, \
        "expected sources and destinations to have the same first dimension or be 1."
    assert sources.shape[1] == destinations.shape[1] or sources.shape[1] == 1 or destinations.shape[1] == 1, \
        "expected sources and destinations to have the same first dimension or be 1."
    cdef int i = 0
    cdef int j = 0
    cdef int n_sources_2nd_axis = sources.shape[1]
    cdef int n_sources_3rd_axis = sources.shape[2]
    cdef int n_destinations_2nd_axis = destinations.shape[1]
    cdef int n_destinations_3rd_axis = destinations.shape[2]
    cdef int broadcast_sources_1st_axis = 0 if sources.shape[0] == 1 else 1
    cdef int broadcast_sources_2nd_axis = 0 if sources.shape[1] == 1 else 1
    cdef int broadcast_destinations_1st_axis = 0 if destinations.shape[0] == 1 else 1
    cdef int broadcast_destinations_2nd_axis = 0 if destinations.shape[1] == 1 else 1

    cdef np.ndarray[bool, ndim=3] out_is_related = np.zeros((
        max(sources.shape[0], destinations.shape[0]),
        max(n_sources_2nd_axis, n_destinations_2nd_axis),
        n_sources_3rd_axis), dtype=np.bool)
    cdef int n_out_1st_axis = out_is_related.shape[0]
    cdef int n_out_2nd_axis = out_is_related.shape[1]
    cdef int n_out_3rd_axis = out_is_related.shape[2]
    
    cdef int_ptr offsets_ptr = <int*>offsets.data
    cdef int_ptr values_ptr = <int*>values.data
    cdef bool* out_is_related_ptr = bool_ptr(out_is_related)
    cdef int_ptr sources_ptr = <int*>sources.data
    cdef int_ptr destinations_ptr = <int*>destinations.data
    cdef bool* condition_ptr = NULL
    if condition is not None:
        assert isinstance(condition, np.ndarray)
        condition_ptr = bool_ptr(condition)
    with nogil:
        for i in range(n_out_1st_axis):
            for j in range(n_out_2nd_axis):
                batch_is_related_pair_inner(out_is_related_ptr + i * n_out_3rd_axis * n_out_2nd_axis + j * n_out_3rd_axis,
                                            n_sources_3rd_axis, n_destinations_3rd_axis,
                                            sources_ptr + i * n_sources_3rd_axis * n_sources_2nd_axis * broadcast_sources_1st_axis + j * n_sources_3rd_axis * broadcast_sources_2nd_axis,
                                            destinations_ptr + i * n_destinations_3rd_axis * n_destinations_2nd_axis * broadcast_destinations_1st_axis + j * n_destinations_3rd_axis * broadcast_destinations_2nd_axis,
                                            offsets_ptr, values_ptr,
                                            direct,
                                            indirect,
                                            condition_ptr,
                                            max_relatable,
                                            related_or_empty)
    return out_is_related


@cython.boundscheck(False)
@cython.wraparound(False)
def padded_gather(np.ndarray[int, ndim=1] indices,
                  np.ndarray[int, ndim=1] offsets,
                  np.ndarray[int, ndim=1] values,
                  int pad_with):
    cdef int max_rel_size = 0
    cdef int i = 0
    cdef int j = 0
    cdef int col_idx = 0
    cdef int n = len(indices)
    with nogil:
        for i in range(n):
            if indices[i] > 0:
                max_rel_size = max(max_rel_size, offsets[indices[i]] - offsets[indices[i] - 1])
            elif indices[i] == 0:
                max_rel_size = max(max_rel_size, offsets[indices[i]])
            # ignore any value of index less than 0 as that is padding.
    cdef np.ndarray[int, ndim=2] padded_out = np.empty((n, max_rel_size), dtype=values.dtype)
    cdef int[:, :] padded_out_view = padded_out
    padded_out.fill(pad_with)
    with nogil:
        for i in range(n):
            if indices[i] >= 0:
                col_idx = 0
                for j in range(offsets[indices[i] - 1] if indices[i] > 0 else 0, offsets[indices[i]]):
                    padded_out_view[i, col_idx] = values[j]
                    col_idx += 1
    return padded_out


@cython.boundscheck(False)
@cython.wraparound(False)
def convert_to_offset_array(list_list):
    cdef int num_values = len(list_list)
    cdef np.ndarray[int, ndim=1] offsets = np.zeros(num_values, dtype=np.int32)
    cdef int total_num_values = sum(len(v) for v in list_list)
    cdef np.ndarray[int, ndim=1] values = np.zeros(total_num_values, dtype=np.int32)
    cdef int[:] values_view = values
    cdef vector[int] list_list_i
    cdef int position = 0
    cdef int i = 0
    cdef int j = 0

    with nogil:
        for i in range(num_values):
            with gil:
                list_list_i = list_list[i]
            for j in range(list_list_i.size()):
                values_view[position + j] = list_list_i[j]
            position += list_list_i.size()
            offsets[i] = position
    return values, offsets


@cython.boundscheck(False)
@cython.wraparound(False)
def make_sparse(np.ndarray[int, ndim=1] dense):
    cdef np.ndarray[int, ndim=1] deltas = np.zeros_like(dense)
    deltas[1:] = dense[1:] - dense[:dense.shape[0] - 1]
    deltas[0] = dense[0]
    cdef np.ndarray[int, ndim=1] indices = np.nonzero(deltas)[0].astype(np.int32)
    cdef int original_length = len(deltas)
    cdef int num_nonzero = len(indices)
    cdef int i = 0
    cdef np.ndarray[int, ndim=1] out = np.zeros(num_nonzero * 2 + 1, dtype=np.int32)
    with nogil:
        # keep length around:
        out[0] = original_length
        for i in range(num_nonzero):
            # place index:
            out[i * 2 + 1] = indices[i]
            # place value:
            out[i * 2 + 2] = deltas[indices[i]]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def make_dense(np.ndarray[int, ndim=1] array, cumsum=False):
    cdef np.ndarray[int, ndim=1] out = np.zeros(array[0], dtype=np.int32)
    cdef int total_size = len(array)
    cdef int i = 0
    with nogil:
        for i in range(1, total_size, 2):
            out[array[i]] = array[i + 1]
    if cumsum:
        np.cumsum(out, out=out)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def offset_values_mask(np.ndarray[int, ndim=1] values,
                       np.ndarray[int, ndim=1] offsets,
                       np.ndarray[int, ndim=1] active_nodes):
    np_dest_array = np.zeros(len(values), dtype=np.bool)
    cdef bool* dest_array = bool_ptr(np_dest_array)
    cdef int i = 0
    cdef int j = 0
    cdef int active_nodes_max = len(active_nodes)
    cdef int end = 0
    cdef int start = 0
    cdef int active_node = 0
    with nogil:
        for i in range(active_nodes_max):
            active_node = active_nodes[i]
            end = offsets[active_node]
            if active_node == 0:
                start = 0
            else:
                start = offsets[active_node - 1]
            for j in range(end - start):
                dest_array[start + j] = 1
    return np_dest_array


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_offset_array_negatives(np.ndarray[int, ndim=1] values,
                                  np.ndarray[int, ndim=1] offsets):
    cdef int position = 0
    cdef np.ndarray[int, ndim=1] values_out = np.zeros_like(values)
    cdef np.ndarray[int, ndim=1] offsets_out = np.zeros_like(offsets)
    cdef int start = 0
    cdef int end = 0
    cdef int i = 0
    cdef int j = 0
    cdef int max_offsets = len(offsets)
    with nogil:
        for i in range(max_offsets):
            end = offsets[i]
            for j in range(start, end):
                if values[j] > -1:
                    values_out[position] = values[j]
                    position += 1
            offsets_out[i] = position
            start = end
    return values_out[:position], offsets_out


@cython.boundscheck(False)
@cython.wraparound(False)
def related_promote_highest(np.ndarray[int, ndim=1] values,
                            np.ndarray[int, ndim=1] offsets,
                            np.ndarray[int, ndim=1] counts,
                            condition,
                            alternative,
                            int keep_min=5):
    cdef bool* condition_ptr = bool_ptr(condition)
    cdef bool* alternative_ptr = bool_ptr(alternative)
    cdef np.ndarray[int, ndim=1] new_values = values.copy()
    cdef int i = 0
    cdef int j = 0
    cdef int start = 0
    cdef int end = 0
    cdef int [:] new_values_view = new_values
    cdef int [:] values_view = values
    cdef int [:] counts_view = counts
    cdef int [:] offsets_view = offsets
    cdef int max_offsets = len(offsets)
    cdef int alternate_count = -1
    cdef bint any_switchers
    cdef int alternate_value = -1
    with nogil:
        for i in range(max_offsets):
            end = offsets_view[i]
            any_switchers = False
            alternate_value = -1
            alternate_count = -1
            for j in range(start, end):
                if condition_ptr[j] and values_view[j] > -1:
                    any_switchers = True
                if alternative_ptr[j]:
                    if counts_view[j] > alternate_count and values_view[j] > -1:
                        alternate_count = counts_view[j]
                        alternate_value = values_view[j]
            if any_switchers and alternate_value > -1:
                if alternate_count <= keep_min:
                    alternate_value = -1
                for j in range(start, end):
                    if condition_ptr[j] and values_view[j] > -1:
                        if counts_view[j] < alternate_count:
                            new_values[j] = alternate_value
            start = end
    return new_values

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void single_usage_extend_cursor(unordered_set[int]& destination_cursor,
                                     unordered_set[int] cursor,
                                     int* offset,
                                     int* values) nogil:
    destination_cursor.clear()
    cdef int start = 0
    cdef int i = 0
    cdef int val
    for val in cursor:
        start = 0 if val == 0 else offset[val - 1]
        for i in range(start, offset[val]):
            if values[i] > -1:
                destination_cursor.insert(values[i])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void extend_cursor(unordered_set[int]& destination_cursor,
                        unordered_set[int] newcomers,
                        int* offset,
                        int* values,
                        int usage) nogil:
    destination_cursor.clear()
    cdef int start = 0
    cdef int i = 0
    cdef int val
    cdef int destination_cursor_size = destination_cursor.size()
    cdef int uses = 0
    cdef unordered_set[int] new_newcomers
    while uses < usage:
        new_newcomers.clear()
        for val in newcomers:
            start = 0 if val == 0 else offset[val - 1]
            for i in range(start, offset[val]):
                # totally new item being explored:
                if values[i] > -1:
                    if destination_cursor.find(values[i]) == destination_cursor.end():
                        new_newcomers.insert(values[i])
                        destination_cursor.insert(values[i])
        uses += 1
        if new_newcomers.size() == 0:
            break
        else:
            newcomers = new_newcomers


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint keep_high_or_highest(int original_count, int new_best_count) nogil:
    return (new_best_count > original_count) or (new_best_count > 50 and new_best_count >= 0.8 * original_count)


def binary_search(v_min, v_max, test):
    l, r, mid, solution = v_min, v_max, -1, -1
    while l <= r:
        mid = (l + r) // 2
        if test(mid):
            solution = mid
            r = mid - 1
        else:
            l = mid + 1
    return solution


def allocate_work(arr, max_work):
    last = 0
    sol = []
    work_so_far = 0

    while last < len(arr):
        next_pt = binary_search(
            last,
            len(arr) - 1,
            lambda point: arr[point] > work_so_far + max_work
        )
        if next_pt == -1:
            next_pt = len(arr)
        work_so_far = arr[next_pt - 1]
        if last == next_pt:
            return None
        sol.append((last, next_pt))
        last = next_pt
    return sol


def fair_work_allocation(offsets, num_workers):
    def check_size(work_size):
        allocated = allocate_work(offsets, work_size)
        return allocated is not None and len(allocated) <= num_workers

    best_work_size = binary_search(
        np.ceil(offsets[-1] / num_workers),
        offsets[-1],
        check_size
    )
    return allocate_work(offsets, best_work_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void extend_relations_find_largest_parent(int j, int start, int end,
                                               const unordered_map[int, bint]& is_parent,
                                               int current_count,
                                               int keep_min,
                                               int [:] counts_view,
                                               int [:] values_view,
                                               int [:] new_values_view,
                                               bool* alternative) nogil:
    cdef int max_count = -1
    cdef int max_value = -1
    cdef int oj
    if is_parent.size() > 0:
        for oj in range(start, end):
            # if the new entity is the parent according to some
            # hard rule, or the number of links to the parent
            # is greater than those to the child, consider swapping:
            if (alternative[oj] and values_view[oj] > -1 and
                is_parent.find(values_view[oj]) != is_parent.end() and
                (
                    is_parent.at(values_view[oj]) or
                    keep_high_or_highest(current_count, counts_view[oj])
                ) and counts_view[oj] > max_count) and values_view[oj] != values_view[j]:

                max_count = counts_view[oj]
                max_value = values_view[oj]
        if max_value > -1:
            if max_count > keep_min:
                new_values_view[j] = max_value
            else:
                new_values_view[j] = -1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void extend_relations_extend_path(int j,
                                       int path_idx,
                                       int [:] values_view,
                                       const vector[vector[int_ptr]]& relation_data_offsets,
                                       const vector[vector[int_ptr]]& relation_data_values,
                                       const vector[vector[int]]& relation_data_max_usage,
                                       bool promote,
                                       unordered_map[int, bint]& is_parent) nogil:
    cdef unordered_set[int] cursor
    # now starting at the current point j
    # try to walk towards the entity by extending using paths
    if values_view[j] > -1:
        cursor.insert(values_view[j])
    for step_idx in range(relation_data_offsets[path_idx].size()):
        # path can only be used once:
        if relation_data_max_usage[path_idx][step_idx] == 1:
            single_usage_extend_cursor(
                cursor,
                cursor,
                relation_data_offsets[path_idx][step_idx],
                relation_data_values[path_idx][step_idx]
            )
        else:
            # path can be used recursively:
            extend_cursor(
                cursor,
                cursor,
                relation_data_offsets[path_idx][step_idx],
                relation_data_values[path_idx][step_idx],
                relation_data_max_usage[path_idx][step_idx]
            )
        # if no entity was found in this process, stop
        if cursor.size() == 0:
            break
    # if there are entities connected via extending the cursor
    # then pick the largest one of those as the parent:
    if cursor.size() > 0:
        for val in cursor:
            is_parent_finder = is_parent.find(val)
            if is_parent_finder == is_parent.end():
                is_parent[val] = promote
            elif promote:
                is_parent[val] = True


cdef bool* bool_ptr(array):
    if array.dtype != np.bool:
        raise ValueError("Can only take boolean pointer from "
                         "array with dtype np.bool (got %r)" % (array.dtype))
    return <bool*>(<long>array.ctypes.data)

@cython.boundscheck(False)
@cython.wraparound(False)
def extend_relations_worker(int worker_idx,
                            relation_data,
                            job_queue,
                            np.ndarray[int, ndim=1] new_values,
                            np.ndarray[int, ndim=1] values,
                            np.ndarray[int, ndim=1] offsets,
                            np.ndarray[int, ndim=1] counts,
                            np_alternative,
                            np.ndarray[int, ndim=1] total_work,
                            int keep_min,
                            pbar):
    cdef int [:] offsets_view = offsets
    cdef int [:] counts_view = counts
    cdef int [:] values_view = values
    cdef int [:] new_values_view = new_values
    cdef int [:] total_work_view = total_work

    cdef vector[vector[int_ptr]] relation_data_offsets
    cdef vector[vector[int_ptr]] relation_data_values
    cdef vector[vector[int]] relation_data_max_usage
    cdef vector[bool*] relation_data_condition
    cdef vector[bool] relation_data_promote

    cdef vector[int_ptr] step_offsets_single
    cdef vector[int_ptr] step_values_single
    cdef vector[int] step_max_usage_single
    cdef bool* alternative = bool_ptr(np_alternative)

    for path, path_condition, promote in relation_data:
        for step_offsets, step_values, max_usage in path:
            step_offsets_single.push_back(<int*> (<np.ndarray[int, ndim=1]>step_offsets).data)
            step_values_single.push_back(<int*> (<np.ndarray[int, ndim=1]>step_values).data)
            step_max_usage_single.push_back(max_usage)
        relation_data_offsets.push_back(step_offsets_single)
        relation_data_values.push_back(step_values_single)
        relation_data_max_usage.push_back(step_max_usage_single)
        relation_data_condition.push_back(bool_ptr(path_condition))
        relation_data_promote.push_back(promote)
        # clear temps just being used to load vectors
        step_offsets_single.clear()
        step_values_single.clear()
        step_max_usage_single.clear()

    cdef int max_offsets = len(offsets)
    cdef int num_paths = len(relation_data)

    cdef int i
    cdef int j
    cdef int oj
    cdef int start
    cdef int end
    cdef int path_idx
    cdef int current_count
    cdef vector[bint] all_paths_false

    for path_idx in range(num_paths):
        all_paths_false.push_back(False)

    cdef vector[bint] paths_active
    cdef unordered_map[int, bint] is_parent
    cdef unordered_map[int, bint].iterator is_parent_finder
    cdef bint any_is_alternative
    cdef int start_offset
    cdef int end_offset
    cdef int work_done = 0

    while True:
        job = job_queue.get()
        if job is None:
            break
        start_offset, end_offset = job
        start = offsets_view[start_offset - 1] if start_offset != 0 else 0
        with nogil:
            for i in range(start_offset, end_offset):
                end = offsets_view[i]
                if end - start > 1:
                    # check if any of the parents can be used
                    # as valid alternatives for the current
                    # item
                    any_is_alternative = False
                    for oj in range(start, end):
                        if alternative[oj]:
                            any_is_alternative = True
                            break
                    # if there is an alternative, look for a path
                    # that connects the current entity to this
                    # new alternative
                    if any_is_alternative:
                        paths_active = all_paths_false
                        for path_idx in range(num_paths):
                            for oj in range(start, end):
                                # look if a particular alternative can be used
                                # and if the specific entity that is being
                                # refered to is not -1 (e.g. masked out)
                                # and whether the path_idx path truth table
                                # is true at this location:
                                if (alternative[oj] and
                                    values_view[oj] > -1 and
                                    relation_data_condition[path_idx][values_view[oj]]):
                                    paths_active[path_idx] = True
                                    # mark that the path has at least one
                                    # possible entity connected to it and move on
                                    break
                        for j in range(start, end):
                            is_parent.clear()
                            current_count = counts_view[j]
                            for path_idx in range(num_paths):
                                # filter by paths that are connectible to the entity
                                # (see filtering above)
                                if paths_active[path_idx]:
                                    extend_relations_extend_path(
                                        j,
                                        path_idx,
                                        values_view,
                                        relation_data_offsets,
                                        relation_data_values,
                                        relation_data_max_usage,
                                        relation_data_promote[path_idx],
                                        is_parent)
                            # select largest new parent (from cursor)
                            # to replace the current entity
                            extend_relations_find_largest_parent(
                                j,
                                start,
                                end,
                                is_parent,
                                current_count,
                                keep_min,
                                counts_view,
                                values_view,
                                new_values_view,
                                alternative)
                start = end
                work_done += 1
                if work_done % 5000 == 0:
                    total_work_view[worker_idx] = work_done
                    if worker_idx == 0:
                        with gil:
                            pbar.update(total_work.sum())
    if worker_idx == 0:
        pbar.update(total_work.sum())

@cython.boundscheck(False)
@cython.wraparound(False)
def extend_relations(relation_data,
                     np.ndarray[int, ndim=1] values,
                     np.ndarray[int, ndim=1] offsets,
                     np.ndarray[int, ndim=1] counts,
                     alternative,
                     pbar,
                     keep_min=5,
                     job_factor=6):
    cdef np.ndarray[int, ndim=1] new_values = values.copy()

    # ensure bool is used:
    relation_data = [
        (path, path_condition, promote) if path_condition.dtype == np.bool else
        (path, path_condition.astype(np.bool), promote)
        for path, path_condition, promote in relation_data
    ]

    threads = []
    num_workers = cpu_count()
    cdef np.ndarray[int, ndim=1] total_work = np.zeros(
        num_workers, dtype=np.int32)
    allocations = fair_work_allocation(offsets, num_workers * job_factor)
    job_queue = Queue()
    for job in allocations:
        job_queue.put(job)
    for worker_idx in range(num_workers):
        job_queue.put(None)
    for worker_idx in range(num_workers):
        threads.append(
            Thread(target=extend_relations_worker,
                   args=(worker_idx,
                         relation_data,
                         job_queue,
                         new_values,
                         values,
                         offsets,
                         counts,
                         alternative,
                         total_work,
                         keep_min,
                         pbar),
                   daemon=True)
        )


    pbar.start()
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    pbar.finish()
    return new_values

@cython.boundscheck(False)
@cython.wraparound(False)
def reduce_values(np.ndarray[int, ndim=1] offsets,
                  np.ndarray[int, ndim=1] values,
                  np.ndarray[int, ndim=1] counts):
    cdef int max_offsets = len(offsets)


    cdef np.ndarray[int, ndim=1] new_offsets = np.zeros_like(offsets)
    cdef np.ndarray[int, ndim=1] new_values = np.zeros_like(values)
    cdef np.ndarray[int, ndim=1] new_counts = np.zeros_like(counts)
    cdef np.ndarray[int, ndim=1] location_shift = np.zeros_like(values)

    cdef int [:] offsets_view = offsets
    cdef int [:] values_view = values
    cdef int [:] counts_view = counts

    cdef int [:] new_offsets_view = new_offsets
    cdef int [:] new_values_view = new_values
    cdef int [:] new_counts_view = new_counts

    cdef int i
    cdef int j
    cdef int start = 0
    cdef int end
    cdef unordered_map[int, int] obs
    cdef int pos = 0
    cdef int insertion_offset = 0
    cdef int index

    with nogil:
        for i in range(max_offsets):
            end = offsets_view[i]
            if end - start == 1:
                if values_view[start] > -1:
                    new_values_view[insertion_offset] = values_view[start]
                    new_counts_view[insertion_offset] = counts_view[start]
                    location_shift[start] = insertion_offset
                    new_offsets_view[i] = 1
                    insertion_offset += 1
                else:
                    new_offsets_view[i] = 0
            elif end - start == 2:
                if values_view[start] > -1 and values_view[start + 1] > -1:
                    if values_view[start] == values_view[start + 1]:
                        new_values_view[insertion_offset] = values_view[start]
                        new_counts_view[insertion_offset] = counts_view[start] + counts_view[start+1]
                        location_shift[start] = insertion_offset
                        location_shift[start + 1] = insertion_offset
                        insertion_offset += 1
                        new_offsets_view[i] = 1
                    else:
                        new_values_view[insertion_offset] = values_view[start]
                        new_counts_view[insertion_offset] = counts_view[start]
                        new_values_view[insertion_offset+1] = values_view[start+1]
                        new_counts_view[insertion_offset+1] = counts_view[start+1]
                        location_shift[start] = insertion_offset
                        location_shift[start + 1] = insertion_offset + 1
                        insertion_offset += 2
                        new_offsets_view[i] = 2
                elif values_view[start] > -1:
                    new_values_view[insertion_offset] = values_view[start]
                    new_counts_view[insertion_offset] = counts_view[start]
                    location_shift[start] = insertion_offset
                    location_shift[start + 1] = -1
                    insertion_offset += 1
                    new_offsets_view[i] = 1
                elif values_view[start + 1] > -1:
                    new_values_view[insertion_offset] = values_view[start+1]
                    new_counts_view[insertion_offset] = counts_view[start+1]
                    location_shift[start] = -1
                    location_shift[start + 1] = insertion_offset
                    insertion_offset += 1
                    new_offsets_view[i] = 1
                else:
                    new_offsets_view[i] = 0
            else:
                obs.clear()
                for j in range(start, end):
                    if values_view[j] > -1:
                        if obs.find(values_view[j]) == obs.end():
                            obs[values_view[j]] = insertion_offset
                            location_shift[j] = insertion_offset
                            new_values_view[insertion_offset] = values_view[j]
                            new_counts_view[insertion_offset] = counts_view[j]
                            insertion_offset += 1
                        else:
                            index = obs.at(values_view[j])
                            location_shift[j] = index
                            new_counts_view[index] += counts_view[j]
                    else:
                        location_shift[j] = -1
                new_offsets_view[i] = obs.size()
            start = end
    np.cumsum(new_offsets, out=new_offsets)
    new_values = new_values[:new_offsets[len(offsets) - 1]]
    new_counts = new_counts[:new_offsets[len(offsets) - 1]]
    return new_offsets, new_values, new_counts, location_shift


@cython.boundscheck(False)
@cython.wraparound(False)
def remap_offset_array(np.ndarray[int, ndim=1] mapping,
                       np.ndarray[int, ndim=1] offsets,
                       np.ndarray[int, ndim=1] values,
                       np.ndarray[int, ndim=1] counts):
    cdef int [:] mapping_view = mapping
    cdef int [:] offsets_view = offsets
    cdef int [:] values_view = values
    cdef int [:] counts_view = counts

    cdef np.ndarray[int, ndim=1] new_offsets = np.zeros_like(mapping)
    cdef np.ndarray[int, ndim=1] new_values = np.zeros_like(values)
    cdef np.ndarray[int, ndim=1] new_counts = np.zeros_like(counts)

    cdef int old_index,
    cdef int new_index
    cdef int start = 0
    cdef int j
    cdef int end
    cdef int old_start
    cdef int old_end
    cdef int max_offsets = len(mapping)

    with nogil:
        for new_index in range(max_offsets):
            old_index = mapping_view[new_index]
            if old_index > 0:
                old_start = offsets_view[old_index - 1]
            else:
                old_start = 0
            old_end = offsets_view[old_index]
            end = start + old_end - old_start
            for j in range(0, end - start):
                new_counts[start + j] = counts_view[old_start + j]
                new_values[start + j] = values_view[old_start + j]
            new_offsets[new_index] = end
            start = end
    return new_offsets, new_values, new_counts


@cython.boundscheck(False)
@cython.wraparound(False)
cdef build_trie_index2indices_array(vector[unordered_map[int, int]]& trie_index2indices):
    cdef np.ndarray[int, ndim=1] offsets = np.zeros(trie_index2indices.size(), dtype=np.int32)
    cdef int total_num_values = 0
    cdef int i
    with nogil:
        for i in range(trie_index2indices.size()):
            total_num_values += trie_index2indices[i].size()
    cdef np.ndarray[int, ndim=1] values = np.zeros(total_num_values, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] counts = np.zeros(total_num_values, dtype=np.int32)
    cdef int position = 0
    with nogil:
        for i in range(trie_index2indices.size()):
            for kv in trie_index2indices[i]:
                values[position] = kv.first
                counts[position] = kv.second
                position += 1
            offsets[i] = position
    return offsets, values, counts


def cleanup_title(dest):
    return (dest[0].upper() + dest[1:]).split('#')[0].replace('_', ' ')


def match_wikipedia_to_wikidata(dest, trie, redirections, prefix):
    prefixed_dest = prefix + "/" + dest
    dest_index = trie.get(prefixed_dest, -1)

    if dest_index == -1:
        cleaned_up_dest = cleanup_title(dest)
        prefixed_dest = prefix + "/" + cleaned_up_dest
        dest_index = trie.get(prefixed_dest, -1)

    if dest_index == -1:
        redirected_dest = redirections.get(cleaned_up_dest, None)
        if redirected_dest is not None:
            prefixed_dest = prefix + "/" + cleanup_title(redirected_dest)
            dest_index = trie.get(prefixed_dest, -1)
    if dest_index != -1:
        dest_index = dest_index[0][0]
    return dest_index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float deg2rad(float x) nogil:
    # np.pi / 180.0 * x
    return 0.017453292519943295 * x


@cython.boundscheck(False)
@cython.wraparound(False)
def distance_inner_work(int start, int end,
                        np.ndarray[int, ndim=1] latitudes,
                        np.ndarray[int, ndim=1] longitudes,
                        np.ndarray[bool, ndim=1] mask,
                        float latitude, float longitude,
                        float radius_km,
                        np.ndarray[np.float32_t, ndim=1] distances):
    # lat1, lon1 = origin
    # lat2, lon2 = destination
    # radius = 6371 # km
    cdef int i = 0
    cdef int n_vals = len(latitudes)
    cdef float dlat
    cdef float dlon
    cdef float temp
    # we want to run the operations in float32 land:
    cdef float half = 0.5
    cdef float one = 1.0
    cdef float two = 2.0
    with nogil:
        for i in range(n_vals):
            if mask[i]:
                dlat = deg2rad(latitude - latitudes[i] * 1e-7)
                dlon = deg2rad(longitude - longitudes[i] * 1e-7)
                temp = sin(dlat * half) * sin(dlat * half) + cos(deg2rad(latitudes[i] * 1e-7)) \
                       * cos(deg2rad(latitude)) * sin(dlon * half) * sin(dlon * half)
                distances[i] = radius_km * two * atan2(sqrt(temp), sqrt(one - temp))


class SliceArg(object):
    def __init__(self, obj):
        self.obj = obj

    def slice(self, start, end):
        return self.obj[start:end]


def parallelize_work(total_work, work_fn, *args):
    if total_work > 10000:
        threads = []
        num_workers = cpu_count()
        batch_size = int(np.ceil(total_work / num_workers))
        for worker_idx in range(num_workers):
            start = worker_idx * batch_size
            end = (worker_idx + 1) * batch_size
            threads.append(
                Thread(target=work_fn,
                       args=[start, end] + [arg.slice(start, end)
                                            if isinstance(arg, SliceArg) else arg for arg in args],
                       daemon=True)
            )
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
    else:
        work_fn(0, total_work, *[arg.obj if isinstance(arg, SliceArg) else arg for arg in args])



@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool c_inside_polygon(int nvert, float* polygon, float x, float y) nogil:
    cdef int i = 0
    cdef int j = nvert - 1
    cdef bool c = 0
    while i < nvert:
        if (((polygon[2 * i + 1] > y) != (polygon[2 * j + 1] > y)) and \
            (x < (polygon[2 * j] - polygon[2 * i]) * (y - polygon[2 * i + 1]) / (polygon[2 * j + 1] - polygon[2 * i + 1]) + polygon[2 * i])): # noqa
            c = not c
        # post increment i and assign to j:
        j = i
        i += 1
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
def inside_polygon_inner_work(int start, int end,
                              np.ndarray[int, ndim=1] latitudes,
                              np.ndarray[int, ndim=1] longitudes,
                              np.ndarray[bool, ndim=1] mask,
                              np.ndarray[bool, ndim=1] out,
                              np.ndarray[np.float32_t, ndim=2] polygon):
    cdef int i = 0
    cdef int n_vals = len(latitudes)
    cdef int nvert = len(polygon)
    cdef float* polygon_ptr = <float*>(<long>polygon.ctypes.data)
    with nogil:
        for i in range(n_vals):
            if mask[i]:
                out[i] = c_inside_polygon(nvert, polygon_ptr, latitudes[i] * 1e-7, longitudes[i] * 1e-7)


@cython.boundscheck(False)
@cython.wraparound(False)
def inside_polygon(np.ndarray[int, ndim=1] latitudes,
                   np.ndarray[int, ndim=1] longitudes,
                   np.ndarray[bool, ndim=1] mask,
                   np.ndarray[np.float32_t, ndim=2] polygon):
    assert len(latitudes) == len(longitudes), "expected same number of latitudes and longitudes"
    assert len(latitudes) == len(mask), "expected same number of values in mask as latitudes/longitudes."
    cdef np.ndarray[bool, ndim=1] out = np.zeros(len(latitudes), dtype=np.bool)
    parallelize_work(
        len(latitudes), inside_polygon_inner_work,
        SliceArg(latitudes), SliceArg(longitudes), SliceArg(mask), SliceArg(out), polygon)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def distance(np.ndarray[int, ndim=1] latitudes,
             np.ndarray[int, ndim=1] longitudes,
             np.ndarray[bool, ndim=1] mask,
             float latitude, float longitude,
             float radius_km=6371):
    assert len(latitudes) == len(longitudes), "expected same number of latitudes and longitudes"
    assert len(latitudes) == len(mask), "expected same number of values in mask as latitudes/longitudes."
    cdef np.ndarray[np.float32_t, ndim=1] distances = np.zeros(len(latitudes), dtype=np.float32)
    distances.fill(np.nan)
    parallelize_work(
        len(latitudes), distance_inner_work,
        SliceArg(latitudes), SliceArg(longitudes), SliceArg(mask), latitude, longitude, radius_km, SliceArg(distances))
    return distances


@cython.boundscheck(False)
@cython.wraparound(False)
def construct_mapping(anchor_trie,
                      anchor_tags,
                      wikipedia2wikidata_trie,
                      prefix,
                      redirections):
    filename_byte_string = anchor_tags.encode("utf-8")
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file: '%s'" % (anchor_tags,))

    cdef char *line = NULL
    cdef size_t l = 0
    cdef size_t read

    cdef vector[unordered_map[int, int]] trie_index2indices = vector[unordered_map[int, int]](len(anchor_trie))
    cdef vector[unordered_map[int, int]] trie_index2contexts = vector[unordered_map[int, int]](len(anchor_trie))

    cdef char[256] context
    cdef char[256] target
    cdef char[256] anchor
    cdef int anchor_int
    cdef int target_int
    cdef int context_int
    cdef int return_code
    cdef int num_lines = count_lines(anchor_tags)
    cdef int count = 0
    cdef char* tab_pos
    cdef char* end_pos
    pbar = get_progress_bar("Construct mapping", max_value=num_lines, item='lines')
    pbar.start()
    with nogil:
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            count += 1
            if count % 10000 == 0:
                with gil:
                    pbar.update(count)
            tab_pos = strchr(line, '\t')
            if (tab_pos - line) > 256 or tab_pos == NULL:
                continue
            end_pos = strchr(tab_pos, '\n')
            if (end_pos - tab_pos) > 256:
                continue
            return_code = sscanf(line, "%256[^\n\t]\t%256[^\n\t]\t%256[^\n\t]", &context, &anchor, &target)
            if return_code != 3:
                continue

            with gil:
                try:
                    target_int = match_wikipedia_to_wikidata(
                        target.decode("utf-8"),
                        wikipedia2wikidata_trie,
                        redirections,
                        prefix
                    )
                except UnicodeDecodeError:
                    continue

                if target_int != -1:
                    cleaned_up = clean_up_trie_source(anchor.decode("utf-8"), prefix=prefix)
                    if len(cleaned_up) > 0:
                        anchor_int = anchor_trie[cleaned_up]
                        context_int = match_wikipedia_to_wikidata(
                            context.decode("utf-8"),
                            wikipedia2wikidata_trie,
                            redirections,
                            prefix
                        )
                        with nogil:
                            trie_index2indices[anchor_int][target_int] += 1
                            trie_index2contexts[anchor_int][context_int] += 1
    fclose(cfile)
    pbar.finish()
    offsets, values, counts = build_trie_index2indices_array(trie_index2indices)
    context_offsets, context_values, context_counts = build_trie_index2indices_array(trie_index2contexts)
    return (
        (offsets, values, counts),
        (context_offsets, context_values, context_counts)
    )

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void pos_store_maybe_clear(vector[string]& current_pos,
                                int& current_trie_idx,
                                int& current_qid,
                                vector[unordered_map[int, int]]& multiword_trie_index2indices,
                                vector[unordered_map[int, int]]& noun_trie_index2indices,
                                vector[unordered_map[int, int]]& adj_trie_index2indices,
                                vector[unordered_map[int, int]]& other_trie_index2indices,
                                int* word_count_ptr) nogil:
    if current_pos.size() > 0:
        # flush the current pos
        if word_count_ptr[current_trie_idx] > 1:
            multiword_trie_index2indices[current_trie_idx][current_qid] += 1
        elif current_pos[0] == b"NOUN" or current_pos[0] == b"PROPN" or current_pos[0] == b"X" or current_pos[0] == b"SYM" or current_pos[0] == b"INTJ":
            noun_trie_index2indices[current_trie_idx][current_qid] += 1
        elif current_pos[0] == b"ADJ":
            adj_trie_index2indices[current_trie_idx][current_qid] += 1
        else:
            other_trie_index2indices[current_trie_idx][current_qid] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def construct_pos_mapping(anchor_trie, anchor_tags, name2index,
                          np.ndarray[np.int32_t, ndim=1] word_count,
                          np.ndarray[np.int32_t, ndim=1] values,
                          np.ndarray[np.int32_t, ndim=1] offsets):
    filename_byte_string = anchor_tags.encode("utf-8")
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file: '%s'" % (anchor_tags,))

    cdef char *line = NULL
    cdef size_t l = 0
    cdef size_t read
    # stats obtained when doing data collection on English wikipedia:
    # {'PROPN': 188572315, 'ADP': 11713004, 'X': 896671, 'ADJ': 12914601, 'NOUN': 52343737, 'CCONJ': 1927642,
    # 'PUNCT': 12469767, 'DET': 2908820, 'VERB': 3096904, 'ADV': 500508, 'PART': 1406678, 'NUM': 7303461,
    # 'AUX': 227125, 'PRON': 575149, 'SYM': 238410, 'SCONJ': 22589, 'INTJ': 18368, '_': 804}
    cdef vector[unordered_map[int, int]] adj_trie_index2indices = vector[unordered_map[int, int]](len(anchor_trie))
    cdef vector[unordered_map[int, int]] noun_trie_index2indices = vector[unordered_map[int, int]](len(anchor_trie))
    cdef vector[unordered_map[int, int]] other_trie_index2indices = vector[unordered_map[int, int]](len(anchor_trie))
    cdef vector[unordered_map[int, int]] multiword_trie_index2indices = vector[unordered_map[int, int]](len(anchor_trie))
    cdef int* word_count_ptr = <int*>((<long>word_count.ctypes.data));
    cdef char[300] tok
    cdef char[300] qid_str
    cdef char[300] context_qid
    cdef char[300] trie_idx_str
    cdef char[300] pos
    cdef int return_code
    num_lines = count_lines(anchor_tags)
    cdef long count = 0
    cdef char* tab_pos
    cdef char* end_pos
    pbar = get_progress_bar("Construct mapping", max_value=num_lines, item='lines')
    pbar.start()
    cdef vector[string] current_pos;
    cdef int current_trie_idx = -1;
    cdef int trie_idx;
    cdef int current_qid;
    cdef int qid;

    with nogil:
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            count += 1
            if count % 10000 == 0:
                with gil:
                    pbar.update(count)
            tab_pos = strchr(line, '\t')
            if (tab_pos - line) > 300 or tab_pos == NULL:
                pos_store_maybe_clear(current_pos, current_trie_idx, current_qid,
                                      multiword_trie_index2indices,
                                      noun_trie_index2indices,
                                      adj_trie_index2indices,
                                      other_trie_index2indices,
                                      word_count_ptr);
                current_pos.clear();
                current_trie_idx = -1;
                current_qid = -1;
                continue
            end_pos = strchr(tab_pos, '\n')
            if (end_pos - tab_pos) > 300:
                pos_store_maybe_clear(current_pos, current_trie_idx, current_qid,
                                      multiword_trie_index2indices,
                                      noun_trie_index2indices,
                                      adj_trie_index2indices,
                                      other_trie_index2indices,
                                      word_count_ptr);
                current_pos.clear();
                current_trie_idx = -1;
                current_qid = -1;
                continue
            return_code = sscanf(line, "%300[^\n\t]\t%300[^\n\t]\t%300[^\n\t]\t%300[^\n\t]\t%300[^\n\t]", &tok, &qid_str, &context_qid, &trie_idx_str, &pos)
            if return_code != 5:
                pos_store_maybe_clear(current_pos, current_trie_idx, current_qid,
                                      multiword_trie_index2indices,
                                      noun_trie_index2indices,
                                      adj_trie_index2indices,
                                      other_trie_index2indices,
                                      word_count_ptr);
                current_pos.clear();
                current_trie_idx = -1;
                current_qid = -1;
                continue
            trie_idx = atoi(trie_idx_str)
            with gil:
                qid = name2index[qid_str]
            if current_trie_idx == -1 or trie_idx == current_trie_idx:
                current_pos.emplace_back(pos);
                current_trie_idx = trie_idx;
                current_qid = qid;
                if word_count_ptr[current_trie_idx] == 1:
                    pos_store_maybe_clear(current_pos, current_trie_idx, current_qid,
                                          multiword_trie_index2indices,
                                          noun_trie_index2indices,
                                          adj_trie_index2indices,
                                          other_trie_index2indices,
                                          word_count_ptr);
                    current_pos.clear();
                    current_trie_idx = -1;
                    current_qid = -1;
            else:
                pos_store_maybe_clear(current_pos, current_trie_idx, current_qid,
                                      multiword_trie_index2indices,
                                      noun_trie_index2indices,
                                      adj_trie_index2indices,
                                      other_trie_index2indices,
                                      word_count_ptr);
                current_pos.clear();
                current_pos.emplace_back(pos);
                current_trie_idx = trie_idx;
                current_qid = qid;
        pos_store_maybe_clear(current_pos, current_trie_idx, current_qid,
                              multiword_trie_index2indices,
                              noun_trie_index2indices,
                              adj_trie_index2indices,
                              other_trie_index2indices,
                              word_count_ptr);
        current_pos.clear();
        current_trie_idx = -1;
        current_qid = -1;
    fclose(cfile)
    pbar.finish()

    # now place counts into a multidimensional array according to the desired values and offsets
    # given earlier
    counts = np.zeros((len(values), 4), dtype=np.int32)
    cdef int* counts_ptr = <int*>((<long>counts.ctypes.data))
    cdef int end = 0
    cdef int start = 0
    cdef int active_node = 0
    cdef int* offsets_ptr = <int*>((<long>offsets.ctypes.data))
    cdef int* values_ptr = <int*>((<long>values.ctypes.data))
    cdef int i =0
    with nogil:
        for active_node in range(adj_trie_index2indices.size()):
            end = offsets[active_node]
            if active_node == 0:
                start = 0
            else:
                start = offsets[active_node - 1]
            for i in range(start, end):
                # get the desired entity by looking at values_ptr
                # then lookup the specific counts for each pos tag:
                if adj_trie_index2indices[active_node].find(values_ptr[i]) != adj_trie_index2indices[active_node].end():
                    counts_ptr[i * 4] = adj_trie_index2indices[active_node][values_ptr[i]]
                if noun_trie_index2indices[active_node].find(values_ptr[i]) != noun_trie_index2indices[active_node].end():
                    counts_ptr[i * 4 + 1] = noun_trie_index2indices[active_node][values_ptr[i]]
                if other_trie_index2indices[active_node].find(values_ptr[i]) != other_trie_index2indices[active_node].end():
                    counts_ptr[i * 4 + 2] = other_trie_index2indices[active_node][values_ptr[i]]
                if multiword_trie_index2indices[active_node].find(values_ptr[i]) != multiword_trie_index2indices[active_node].end():
                    counts_ptr[i * 4 + 3] = multiword_trie_index2indices[active_node][values_ptr[i]]
    return counts


cdef vector[int] is_member_with_path_internal_single_value(vector[int*]& relation_values, vector[int*]& relation_offsets, unordered_set[int]& visited,
                                                           int root, int member_fields_c, double max_steps) nogil:
    cdef vector[vector[int]] candidates1
    cdef vector[vector[int]] candidates2
    cdef int n_relations = relation_values.size()

    cdef vector[vector[int]]* candidates = &candidates1
    cdef vector[vector[int]]* new_candidates = &candidates2
    cdef vector[vector[int]]* temp

    cdef vector[int] root_node;
    root_node.emplace_back(root);
    candidates[0].emplace_back(root_node)
    cdef int steps = 0
    cdef double max_steps_f = max_steps
    cdef int j
    cdef int el
    with nogil:
        while not candidates[0].empty() and steps < max_steps_f:
            new_candidates[0].clear()
            for candidate in candidates[0]:
                for relation_idx in range(n_relations):
                    for j in range(0 if candidate.back() == 0 else relation_offsets[relation_idx][candidate.back() - 1],
                                   relation_offsets[relation_idx][candidate.back()]):
                        el = relation_values[relation_idx][j]
                        if visited.find(el) == visited.end():
                            if el == member_fields_c:
                                candidate.emplace_back(relation_idx)
                                candidate.emplace_back(el)
                                return candidate
                            visited.emplace(el)
                            new_candidates[0].emplace_back(candidate)
                            new_candidates[0].back().emplace_back(relation_idx)
                            new_candidates[0].back().emplace_back(el)
            temp = new_candidates
            new_candidates = candidates
            candidates = temp
            steps += 1
    return vector[int]()



cdef vector[int] is_member_with_path_internal_multi_value(vector[int*]& relation_values, vector[int*]& relation_offsets, unordered_set[int]& visited,
                                                          int root, vector[int] member_fields, double max_steps) nogil:
    cdef unordered_set[int] member_fields_c
    for val in member_fields:
        member_fields_c.emplace(val)
    cdef vector[vector[int]] candidates1
    cdef vector[vector[int]] candidates2
    cdef int n_relations = relation_values.size()

    cdef vector[vector[int]]* candidates = &candidates1
    cdef vector[vector[int]]* new_candidates = &candidates2
    cdef vector[vector[int]]* temp

    cdef vector[int] root_node;
    root_node.emplace_back(root);
    candidates[0].emplace_back(root_node)
    cdef int steps = 0
    cdef double max_steps_f = max_steps
    cdef int j
    cdef int el
    with nogil:
        while not candidates[0].empty() and steps < max_steps_f:
            new_candidates[0].clear()
            for candidate in candidates[0]:
                for relation_idx in range(n_relations):
                    for j in range(0 if candidate.back() == 0 else relation_offsets[relation_idx][candidate.back() - 1],
                                   relation_offsets[relation_idx][candidate.back()]):
                        el = relation_values[relation_idx][j]
                        if visited.find(el) == visited.end():
                            if member_fields_c.find(el) != member_fields_c.end():
                                candidate.emplace_back(relation_idx)
                                candidate.emplace_back(el)
                                return candidate
                            visited.emplace(el)
                            new_candidates[0].emplace_back(candidate)
                            new_candidates[0].back().emplace_back(relation_idx)
                            new_candidates[0].back().emplace_back(el)
            temp = new_candidates
            new_candidates = candidates
            candidates = temp
            steps += 1
    return vector[int]()


def is_member_with_path(relations, root, fields, member_fields, bad_nodes, max_steps=None):
    if max_steps is None:
        max_steps = float("inf")
    cdef int n_relations = len(relations)
    cdef vector[int*] relation_values
    cdef vector[int*] relation_offsets
    cdef int relation_idx
    for relation_idx in range(len(relations)):
        relation_values.emplace_back(<int*>((<long>relations[relation_idx].values.ctypes.data)))
        relation_offsets.emplace_back(<int*>((<long>relations[relation_idx].offsets.ctypes.data)))

    cdef unordered_set[int] visited

    if len(member_fields) == 1:
        return is_member_with_path_internal_single_value(relation_values, relation_offsets, visited, root, member_fields[0], max_steps)
    else:
        return is_member_with_path_internal_multi_value(relation_values, relation_offsets, visited, root, member_fields, max_steps)


def iterate_anchor_lines(anchor_tags,
                         redirections,
                         wikipedia2wikidata_trie,
                         prefix):
    filename_byte_string = anchor_tags.encode("utf-8")
    cdef char* fname = filename_byte_string
    cdef FILE* cfile
    cfile = fopen(fname, "rb")
    if cfile == NULL:
        raise FileNotFoundError(2, "No such file: '%s'" % (anchor_tags,))

    cdef char *line = NULL
    cdef size_t l = 0
    cdef size_t read
    cdef char[256] context
    cdef char[256] target
    cdef char[256] anchor
    cdef string anchor_string
    cdef int anchor_int
    cdef int target_int
    cdef int context_int
    cdef int num_missing = 0
    cdef int num_broken = 0
    cdef int return_code
    cdef int num_lines = count_lines(anchor_tags)
    cdef int count = 0
    cdef char* tab_pos
    cdef char* end_pos
    cdef vector[pair[string, string]] missing
    cdef unordered_set[string] visited
    pbar = get_progress_bar("Construct mapping", max_value=num_lines, item='lines')
    pbar.start()
    with nogil:
        while True:
            read = getline(&line, &l, cfile)
            if read == -1:
                break
            count += 1
            if count % 1000 == 0:
                with gil:
                    pbar.update(count)
            tab_pos = strchr(line, '\t')
            if (tab_pos - line) > 256 or tab_pos == NULL:
                continue
            end_pos = strchr(tab_pos, '\n')
            if (end_pos - tab_pos) > 256:
                continue
            return_code = sscanf(line, "%256[^\n\t]\t%256[^\n\t]\t%256[^\n\t]", &context, &anchor, &target)
            if return_code != 3:
                num_broken += 1
                continue

            anchor_string = string(anchor)
            if visited.find(anchor_string) == visited.end():
                with gil:
                    try:
                        target_int = match_wikipedia_to_wikidata(
                            target.decode("utf-8"),
                            wikipedia2wikidata_trie,
                            redirections,
                            prefix
                        )
                    except UnicodeDecodeError:
                        num_broken += 1
                        continue

                    if target_int != -1:
                        with nogil:
                            visited.insert(anchor_string)
                        source = clean_up_trie_source(anchor.decode("utf-8"), prefix=prefix)
                        if len(source) > 0:
                            yield source
                    else:
                        num_missing += 1
                        with nogil:
                            missing.push_back(pair[string, string](anchor_string, string(target)))
    fclose(cfile)
    pbar.finish()
    print("%d/%d anchor_tags could not be found in wikidata" % (num_missing, num_lines))
    print("%d/%d anchor_tags links were malformed/too long" % (num_broken, num_lines))
    print("Missing anchor_tags sample:")
    cdef int i = 0
    for kv in missing:
        print("    " + kv.first.decode("utf-8") + " -> " + kv.second.decode("utf-8"))
        i += 1
        if i == 10:
            break

def construct_anchor_trie(anchor_tags, redirections, prefix, wikipedia2wikidata_trie):
    return marisa_trie.Trie(
        iterate_anchor_lines(
            anchor_tags=anchor_tags,
            wikipedia2wikidata_trie=wikipedia2wikidata_trie,
            redirections=redirections,
            prefix=prefix
        )
    )
