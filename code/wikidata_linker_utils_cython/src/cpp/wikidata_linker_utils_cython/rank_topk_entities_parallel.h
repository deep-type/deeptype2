#ifndef RANK_TOPK_ENTITIES_PARALLEL_H
#define RANK_TOPK_ENTITIES_PARALLEL_H

#include <vector>

struct OracleClassification {
    bool is_int;
    void* data;
    inline OracleClassification(bool is_int_, void* data_) : is_int(is_int_), data(data_) {};
    inline int classify(int index) const {
        if (is_int) {
            return ((int*)data)[index];
        } else {
            return ((bool*)data)[index];
        }
    }
};

struct ReferenceArray {
    float* data;
    int cols;
    int rows;
    inline float at(int row, int col) const {
        return data[row * cols + col];
    };
    inline ReferenceArray(float* data_, int rows_, int cols_)
        : data(data_), rows(rows_), cols(cols_) {}
};

struct BatchRankMatch {
    int trie_index;
    int start;
    int stop;
    std::vector<ReferenceArray> context;
};

struct EntityCandidateOracleProbs {
    float prob;
    int class_idx;
    EntityCandidateOracleProbs(float prob_, int class_idx_)
        : prob(prob_), class_idx(class_idx_) {};
};

struct EntityCandidate {
    double logit;
    double prob;
    double link_prob;
    int option;
    inline EntityCandidate(double logit_, double prob_, double link_prob_, int option_) :
        logit(logit_), prob(prob_), link_prob(link_prob_), option(option_) {};
    std::vector<EntityCandidateOracleProbs> oracle_probs;
};

void batch_rank_topk_entities_parallel(
    const std::vector<OracleClassification>& oracle_classifications,
    const std::vector<int>& oracle_other_class,
    int k,
    double beta,
    const std::vector<double>& alphas,
    double min_link_prob,
    int* offsets,
    int* counts,
    int* values,
    const std::vector<BatchRankMatch>& matches,
    std::vector<std::vector<EntityCandidate> >& all_candidates,
    bool return_oracle_probs,
    int num_threads);

#endif
