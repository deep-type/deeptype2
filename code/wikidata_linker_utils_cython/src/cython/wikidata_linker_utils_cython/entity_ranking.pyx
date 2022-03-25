cimport cython

from third_party.libcpp11.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as np
from multiprocessing import cpu_count

cdef extern from "wikidata_linker_utils_cython/rank_topk_entities_parallel.h" nogil:
    cdef cppclass OracleClassification:
        bool is_int
        void* data
        OracleClassification(bool is_int, void* data)
        int classify(int index)

    cdef cppclass ReferenceArray:
        float* data
        int rows
        int cols
        ReferenceArray(float*, int, int)

    cdef cppclass EntityCandidateOracleProbs:
        float prob
        int class_idx

    cdef cppclass EntityCandidate:
        double logit
        double prob
        double link_prob
        int option
        vector[EntityCandidateOracleProbs] oracle_probs
        EntityCandidate(double, double, double, int)

    cdef cppclass BatchRankMatch:
        int trie_index
        int start
        int stop
        vector[ReferenceArray] context

    cdef void batch_rank_topk_entities_parallel(
            const vector[OracleClassification]& oracle_classifications,
            const vector[int]& oracle_other_class,
            int k,
            double beta,
            const vector[double]& alphas,
            double min_link_prob,
            int* offsets,
            int* counts,
            int* values,
            const vector[BatchRankMatch]& matches,
            vector[vector[EntityCandidate] ]& all_candidates,
            bool return_oracle_probs,
            int num_threads);


cdef inline BatchRankMatch batch_rank_match(match, oracle_keys):
    cdef BatchRankMatch cmatch
    cmatch.trie_index = match.trie_index
    cmatch.start = match.start
    cmatch.stop = match.stop
    for key in oracle_keys:
        cmatch.context.emplace_back(
            ReferenceArray(<float*>(<np.ndarray[float, ndim=2]>match.context[key]).data,
                           match.context[key].shape[0],
                           match.context[key].shape[1]))

    return cmatch


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_rank_topk_entities(type_classifier_oracles,
                             trie_index2indices_values,
                             trie_index2indices_counts,
                             matches,
                             int k=5,
                             double beta=0.999, alpha=0.49, double min_link_prob=0.0,
                             bool return_oracle_probs=False,
                             int num_threads=0):
    # Setup arrays and pointers to do ranking:
    oracle_keys = sorted(type_classifier_oracles.keys())
    cdef vector[double] alphas
    if isinstance(alpha, float):
        for key in oracle_keys:
            alphas.push_back(alpha)
    else:
        for key in oracle_keys:
            alphas.push_back(alpha[key])

    cdef vector[OracleClassification] oracle_classifications
    cdef vector[int] oracle_other_class
    for key in oracle_keys:
        oracle_classifications.emplace_back(OracleClassification(
                type_classifier_oracles[key].classification.dtype == np.int32,
                (<np.ndarray[int, ndim=1]>type_classifier_oracles[key].classification).data))

        if type_classifier_oracles[key].classes[len(type_classifier_oracles[key].classes)-1] == "other":
            oracle_other_class.push_back(len(type_classifier_oracles[key].classes) - 1)
        else:
            oracle_other_class.push_back(-1)

    cdef np.ndarray[int, ndim=1] offsets = trie_index2indices_values.offsets
    cdef np.ndarray[int, ndim=1] counts = trie_index2indices_counts.values
    cdef np.ndarray[int, ndim=1] values = trie_index2indices_values.values

    cdef vector[BatchRankMatch] cmatches
    for match in matches:
        cmatches.push_back(batch_rank_match(match, oracle_keys))

    cdef int match_idx = 0
    cdef int i = 0
    cdef vector[vector[EntityCandidate] ] all_candidates

    if num_threads <= 0:
        num_threads = cpu_count()

    # run ranking on each entity (parallelizable operations)
    with nogil:
        batch_rank_topk_entities_parallel(
            oracle_classifications,
            oracle_other_class,
            k,
            beta,
            alphas,
            min_link_prob,
            <int*>offsets.data,
            <int*>counts.data,
            <int*>values.data,
            cmatches,
            all_candidates,
            return_oracle_probs,
            num_threads)
    # recover output in Python friendly format:
    all_topk = []
    oracles = [type_classifier_oracles[key] for key in oracle_keys]
    for candidates in all_candidates:
        topk = []
        for i in range(candidates.size()):
            candidate = {
                "link_prob": candidates[i].link_prob,
                "internal_id": candidates[i].option,
                "logit": candidates[i].logit,
                "prob": candidates[i].prob
            }
            if return_oracle_probs:
               candidate["oracle_probs"] = {
                    oracle_keys[j]: {
                        "prob": candidates[i].oracle_probs[j].prob,
                        "class": oracles[j].classes[candidates[i].oracle_probs[j].class_idx]
                    } for j in range(candidates[i].oracle_probs.size())
                }
            topk.append(candidate)
        all_topk.append(topk)
    return all_topk
