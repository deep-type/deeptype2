#include "rank_topk_entities_parallel.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

using std::chrono::milliseconds;
using std::function;
using std::vector;
using std::thread;
using Duration = std::chrono::duration<double>;

inline void temporal_max(std::vector<float>& output,
                         const ReferenceArray& array,
                         int start,
                         int stop) {
    for (int col_idx = 0; col_idx < output.size(); col_idx++) {
        for (int t = start; t < stop; t++) {
            output[col_idx] = std::max(output[col_idx], array.at(t, col_idx));
        }
    }
}

void rank_topk_entities(
        const std::vector<OracleClassification>& oracle_classifications,
        const std::vector<int>& oracle_other_class,
        int k,
        double beta,
        const std::vector<double>& alphas,
        double min_link_prob,
        int* offsets,
        int* counts,
        int* values,
        const BatchRankMatch& match,
        std::vector<EntityCandidate>& candidates,
        bool return_oracle_probs) {
    int choice_start = match.trie_index == 0 ? 0 : offsets[match.trie_index - 1];
    int choice_end = offsets[match.trie_index];
    bool all_other = true;

    double link_prob;
    double key_prob;
    double total_logit = 0.0;

    std::vector<std::vector<float> > context_v_maxed;
    for (int context_idx = 0; context_idx < match.context.size(); context_idx++) {
        context_v_maxed.emplace_back(match.context[context_idx].cols, 0.0);
        temporal_max(context_v_maxed[context_idx], match.context[context_idx], match.start, match.stop);
    }

    // total counts for all choices:
    double option_counts_sum = std::accumulate(counts + choice_start, counts + choice_end, 0.0);

    for (int choice_idx = choice_start; choice_idx < choice_end; choice_idx++) {
        link_prob = counts[choice_idx] / option_counts_sum;
        if (link_prob < min_link_prob) {
            continue;
        }
        all_other = true;
        for (int context_idx = 0; context_idx < oracle_other_class.size(); context_idx++) {
            if (oracle_classifications[context_idx].classify(values[choice_idx]) != oracle_other_class[context_idx]) {
                all_other = false;
                break;
            }
        }
        if (all_other) {
            continue;
        }
        candidates.emplace_back(0.0, 0.0, link_prob, values[choice_idx]);
        auto& logit = candidates.back().logit;
        logit = (1.0 - beta) + link_prob * beta;
        for (int context_idx = 0; context_idx < oracle_other_class.size(); context_idx++) {
            auto& oracle = oracle_classifications[context_idx];
            if (oracle.classify(values[choice_idx]) != oracle_other_class[context_idx]) {
                key_prob = context_v_maxed[context_idx][oracle.classify(values[choice_idx])];
            } else {
                key_prob = 1.0;
            }
            logit *= (1.0 * alphas[context_idx] + (1.0 - alphas[context_idx]) * key_prob);
            if (return_oracle_probs) {
                candidates.back().oracle_probs.emplace_back(key_prob, oracle.classify(values[choice_idx]));
            }
        }
        total_logit += logit;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const EntityCandidate& left, const EntityCandidate& right) {
        return left.logit > right.logit;
    });
    if (candidates.size() > k) {
        candidates.erase(candidates.begin() + k, candidates.end());
    }
    for (auto& candidate : candidates) {
        candidate.prob = candidate.logit / total_logit;
    }
}

class ThreadPool {
    private:
        typedef std::chrono::duration<double> Duration;
        static __thread bool in_thread_pool;
        // c++ assigns random id to each thread. This is not a thread_id
        // it's a number inside this thread pool.
        static __thread int thread_number;

        bool should_terminate;
        std::mutex queue_mutex;
        std::condition_variable is_idle;
        int active_count;

        std::deque<std::function<void()> > work;
        std::vector<std::thread> pool;
        Duration between_queue_checks;

        void thread_body(int _thread_id);
    public:
        // Creates a thread pool composed of num_threads threads.
        // threads are started immediately and exit only once ThreadPool
        // goes out of scope. Threads periodically check for new work
        // and the frequency of those checks is at minimum between_queue_checks
        // (it can be higher due to thread scheduling).
        ThreadPool(int num_threads, Duration between_queue_checks=std::chrono::milliseconds(1));

        // Run a function on a thread in pool.
        void run(std::function<void()> f);

        // Wait until queue is empty and all the threads have finished working.
        // If timeout is specified function waits at most timeout until the
        // threads are idle. If they indeed become idle returns true.
        bool wait_until_idle(Duration timeout);
        bool wait_until_idle();

        // Retruns true if all the work is done.
        bool idle() const;
        // Return number of active busy workers.
        int active_workers();
        ~ThreadPool();
};

__thread bool ThreadPool::in_thread_pool = false;
__thread int ThreadPool::thread_number = -1;


ThreadPool::ThreadPool(int num_threads, Duration between_queue_checks) :
        between_queue_checks(between_queue_checks),
        should_terminate(false),
        active_count(0) {
    // Thread pool inception is not supported at this time.
    assert(!in_thread_pool);

    ThreadPool::between_queue_checks = between_queue_checks;
    for (int thread_number = 0; thread_number < num_threads; ++thread_number) {
        pool.emplace_back(&ThreadPool::thread_body, this, thread_number);
    }
}

void ThreadPool::thread_body(int _thread_id) {
    in_thread_pool = true;
    thread_number = _thread_id;
    bool am_i_active = false;

    while (true) {
        function<void()> f;
        {
            std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
            bool was_i_active = am_i_active;
            if (should_terminate && work.empty())
                break;
            if (!work.empty()) {
                am_i_active = true;
                f = work.front();
                work.pop_front();
            } else {
                am_i_active = false;
            }

            if (am_i_active != was_i_active) {
                active_count += am_i_active ? 1 : -1;
                if (active_count == 0) {
                    // number of workers decrease so maybe all are idle
                    is_idle.notify_all();
                }
            }
        }
        // Function defines implicit conversion to bool
        // which is true only if call target was set.
        if (static_cast<bool>(f)) {
            f();
        } else {
            std::this_thread::sleep_for(between_queue_checks);
        }
        std::this_thread::yield();
    }
}

int ThreadPool::active_workers() {
    std::lock_guard<decltype(queue_mutex)> lock(queue_mutex);
    return active_count;
}

bool ThreadPool::wait_until_idle(Duration timeout) {
    std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
    is_idle.wait_for(lock, timeout, [this]{
        return active_count == 0 && work.empty();
    });
    return idle();
}

bool ThreadPool::wait_until_idle() {
    int retries = 3;
    while (retries--) {
        try {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
            is_idle.wait(lock, [this]{
                return active_count == 0 && work.empty();
            });
            return idle();
        } catch (...) {}
    }
    throw std::runtime_error(
        "exceeded retries when waiting until idle."
    );
    return false;
}

bool ThreadPool::idle() const {
    return active_count == 0 && work.empty();
}

void ThreadPool::run(function<void()> f) {
    int retries = 3;
    while (retries--) {
        try {
            std::unique_lock<decltype(queue_mutex)> lock(queue_mutex);
            work.push_back(f);
            return;
        } catch (...) {}
    }
    throw std::runtime_error(
        "exceeded retries when trying to run operation on thread pool."
    );
}

ThreadPool::~ThreadPool() {
    // Terminates thread pool making sure that all the work
    // is completed.
    should_terminate = true;
    for (auto& t : pool)
        t.join();
}


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
        int num_threads) {
    ThreadPool pool(num_threads);
    for (int match_idx = 0; match_idx < matches.size(); match_idx++) {
        all_candidates.emplace_back(std::vector<EntityCandidate>());
    }
    for (int match_idx = 0; match_idx < matches.size(); match_idx++) {
        auto match = matches[match_idx];
        auto local_match_idx = match_idx;
        pool.run([&oracle_classifications,
                  &oracle_other_class,
                  &all_candidates,
                  offsets,
                  counts,
                  match,
                  values,
                  min_link_prob,
                  beta,
                  &alphas,
                  k,
                  local_match_idx,
                  return_oracle_probs]() {
            rank_topk_entities(
                oracle_classifications,
                oracle_other_class,
                k,
                beta,
                alphas,
                min_link_prob,
                offsets,
                counts,
                values,
                match,
                all_candidates[local_match_idx],
                return_oracle_probs
            );
        });
    }
    pool.wait_until_idle();
}


