import argparse
import time
import pickle
import marisa_trie
import pandas as pd
import numpy as np
import sklearn.neural_network
from collections import Counter
from os.path import join, dirname, realpath
from wikidata_linker_utils.file import true_exists
from wikidata_linker_utils.bash import count_lines
from wikidata_linker_utils.offset_array import OffsetArray
from wikidata_linker_utils.type_collection import TypeCollection
from wikidata_linker_utils.trie import has_keys_with_prefix
from wikidata_linker_utils.progressbar import get_progress_bar
from multiprocessing import Queue, Process, Event, cpu_count
from queue import Empty

SCRIPT_DIR = dirname(realpath(__file__))
DATA_DIR = join(dirname(SCRIPT_DIR), "data")
DENSIFICATION_DIR = join(DATA_DIR, "densification")


def _prep_data_x_y(dataset):
    Y = np.array(dataset["OK?"])
    X = np.array(dataset)[:, 3:]
    X[:, 3] = np.log1p(X[:, 3].astype(np.float32))
    X = np.concatenate([[[x[0].islower()] for x in np.array(dataset)[:, 1]], X], axis=-1)
    return X, Y


def _recover_article_boundaries(input_file_iterator):
    lines = []
    article_id = None
    inside = False
    current_context = None
    blank_line_pos = -1
    for l in input_file_iterator:
        if "\t" in l:
            token, target, context, trie_context = l.split("\t")
            if current_context is None:
                current_context = context
            if context != current_context:
                # reached a new article, yield up to the last double line boundary,
                # then proceed with new article
                if blank_line_pos == -1:
                    yield lines, current_context
                    lines = []
                else:
                    yield lines[:blank_line_pos], current_context
                    lines = lines[blank_line_pos:]
                current_context = context
            # add to the current set of observed lines:
            lines.append((token, target, context, trie_context))
        else:
            lines.append((l,))
            if l == "\n":
                # blank line, possible boundary:
                blank_line_pos = len(lines)
    if len(lines) > 0:
        yield lines, current_context


def _percent_before(text, sample):
    pos = text.find(sample)
    if pos == -1:
        return 1.0
    else:
        return pos / len(text)


def _percent_after(text, sample):
    pos = text.find(sample)
    if pos == -1:
        return 1.0
    else:
        pos_from_end = pos + len(sample)
        return max(0, len(text) - pos_from_end) / len(text)

class Densifier(object):
    def __init__(self, language_path, type_collection,
                 trigger_counter, word_counter, densification_predictor):
        self.trie = marisa_trie.Trie().load(
            join(language_path, "trie.marisa")
        )
        self.trie_index2indices_values = OffsetArray.load(
            join(language_path, "trie_index2indices")
        )
        self.trie_index2indices_counts = OffsetArray(
            np.load(join(language_path, "trie_index2indices_counts.npy")),
            self.trie_index2indices_values.offsets
        )
        self.collection = type_collection
        self.article_inlinks = np.bincount(self.trie_index2indices_values.values,
                                           weights=self.trie_index2indices_counts.values).astype(np.int32)
        self.trigger_counter = trigger_counter
        self.word_counter = word_counter
        self.densification_predictor = densification_predictor
        self.densifications = 0
        self.potential_densifications = 0
        self.mean_positive_prediction = 0.0
        self.mean_negative_prediction = 0.0
        self.densification_calls = 0
        self.seen_links = 0
        self.sum_increase_ratio = 0.0
        self.aggregate_increase_ratio = 0.0
        self.prediction_time = 0.0
        self.feature_time = 0.0
        self.match_time = 0.0
        self.total_time = 0.0
        self.candidate_selection_time = 0.0
        self.candidate_filter_time = 0.0
        self.link_prep_time = 0.0
        self.skipped_match = 0


    def feature_generator(self, s, meaning, count, sum_count, existing_link):
        prob = count / sum_count
        specificity = count / self.article_inlinks[meaning]
        triggerability = self.trigger_counter.get(s, 0) / self.word_counter.get(s, 1)
        ent_name = self.collection.get_name(meaning)
        ent_name_lower = ent_name.split(" (")[0].rsplit(" - ", 1)[0].lower()
        lowercase_s = s.lower()
        return [
            s[0].islower(),
            int(lowercase_s in ent_name_lower),
            _percent_before(ent_name_lower, lowercase_s),
            _percent_after(ent_name_lower, lowercase_s),
            np.log1p(float(existing_link[meaning])),
            specificity,
            prob,
            triggerability,
        ]

    def densify(self, tokens, current_context_qid):
        start_t = time.time()
        t0 = time.time()
        self.densification_calls += 1
        # collect statistics on existing links:
        existing_link = {}
        edges = set()
        page_nlinks = 0
        for token in tokens:
            if len(token) > 1:
                page_nlinks += 1
                for idx in (self.collection.name2index[token[1]], self.collection.name2index[token[2]]):
                    if idx not in existing_link:
                        existing_link[idx] = 1
                    else:
                        existing_link[idx] += 1
                edges.add((int(token[3]), token[1]))
        self.seen_links += page_nlinks
        self.link_prep_time += time.time() - t0

        # then run trie on remainder to figure out candidate locations for modifications:
        out = []
        token_index = 0
        max_match_length = 5
        new_links = 0
        while token_index < len(tokens):
            t0 = time.time()
            token_match_end = None
            trie_index = None
            token_prefix = tokens[token_index][0].rstrip()
            if len(tokens[token_index]) == 1 and len(token_prefix) > 0 and has_keys_with_prefix(self.trie, token_prefix.lower()):
                for token_index_end in reversed(range(token_index + 1, min(len(tokens) + 1, token_index + max_match_length + 1))):
                    # make sure these tokens don't already link to something:
                    if all(len(tokens[tidx]) == 1 for tidx in range(token_index, token_index_end)):
                        as_string = " ".join([t[0].rstrip() for t in tokens[token_index:token_index_end]])
                        as_string_lowercase = as_string.lower()
                        # see if there is a trigger match for this character chain:
                        trie_index = self.trie.get(as_string_lowercase, None)
                        if trie_index is not None:
                            token_match_end = token_index_end
                            break
            self.match_time += time.time() - t0
            if token_match_end is not None and not self.trie_index2indices_values.is_empty(trie_index):
                t0 = time.time()
                counts = self.trie_index2indices_counts[trie_index]
                sum_count = counts.sum()
                possible_match = [(meaning, count) for meaning, count in zip(self.trie_index2indices_values[trie_index], counts)
                                  if meaning in existing_link and (count / sum_count > 0.01 or (trie_index, meaning) in edges)]
                self.candidate_filter_time += time.time() - t0
                if len(possible_match) > 0:
                    self.potential_densifications += 1
                    t0 = time.time()
                    features = [self.feature_generator(as_string, meaning, count, sum_count, existing_link)
                                for meaning, count in possible_match]
                    self.feature_time += time.time() - t0
                    t0 = time.time()
                    probs = self.densification_predictor.predict_proba(features)[:, 1]
                    self.prediction_time += time.time() - t0
                    t0 = time.time()
                    best_sol_idx = probs.argmax(axis=0)
                    if probs[best_sol_idx] > 0.5:
                        new_links += 1
                        self.densifications += 1
                        self.mean_positive_prediction += probs[best_sol_idx]
                        chosen_meaning, _ = possible_match[best_sol_idx]
                        new_qid = self.collection.ids[chosen_meaning]
                        # worth densifying here
                        out.extend(
                            [["{}\t{}\t{}\t{}\n".format(tokens[tidx][0].rstrip(), new_qid, current_context_qid, trie_index)]
                             for tidx in range(token_index, token_index_end)]
                        )
                    else:
                        self.mean_negative_prediction += probs[best_sol_idx]
                        # forget it:
                        out.extend(tokens[token_index:token_index_end])
                    self.candidate_selection_time += time.time() - t0
                else:
                    self.skipped_match += 1
                    out.extend(tokens[token_index:token_index_end])
                token_index = token_match_end
            else:
                out.append(tokens[token_index])
                token_index += 1

        self.sum_increase_ratio += new_links / max(page_nlinks, 1)
        self.total_time += time.time() - start_t
        return out

    def pop_stats(self):
        keys = ["potential_densifications",
                "densifications",
                "mean_positive_prediction",
                "mean_negative_prediction",
                "seen_links",
                "sum_increase_ratio",
                "densification_calls",
                "skipped_match"]
        out = {key: getattr(self, key) for key in keys}
        for key in keys:
            setattr(self, key, 0)
        return out

    def summary(self):
        print("Timing:")
        timings = {
            "prediction_time": self.prediction_time,
            "feature_time": self.feature_time,
            "match_time": self.match_time,
            "candidate_selection_time": self.candidate_selection_time,
            "candidate_filter_time": self.candidate_filter_time,
            "link_prep_time": self.link_prep_time,
        }
        remainder = self.total_time
        for k, v in timings.items():
            print("{}: {:.3f}%".format(k, 100.0 * v / self.total_time))
            remainder -= v
            setattr(self, k, 0.0)
        print("remainder: {:.3f}%".format(100.0 * remainder / self.total_time))
        self.total_time = 0.0
        print("{} potential matches, {} densifications.".format(self.potential_densifications, self.densifications,))
        if self.densifications > 0:
            print("Mean positive prediction {:.3f}".format(
                self.mean_positive_prediction / self.densifications))
        if (self.potential_densifications - self.densifications) > 0:
            print("Mean negative prediction {:.3f}".format(
                self.mean_negative_prediction / (self.potential_densifications - self.densifications)))
        if self.seen_links > 0:
            print("ratio new links to old links: {:.3f}%".format(100.0 * self.densifications / self.seen_links))
        if self.densification_calls > 0:
            print("Mean increase in links per page: {:.3f}%".format(100.0 * self.sum_increase_ratio / self.densification_calls))
        # Optionally report commonly densified links:
        # for (trie_index, chosen_meaning), freq in self.common_densifications.most_common(10):
        #     print("{} -> {} ({} times)".format(
        #         self.trie.restore_key(trie_index),
        #         self.collection.get_name(chosen_meaning),
        #         freq))


def _job_print(x):
    """
    Not actually printing because we want to silence workers,
    and only log from main thread
    """
    pass


def write_job(nworkers, path, queue, summary_queue, running):
    _job_print("Starting writer.")
    current_article_idx = 0
    received = {}
    global_stats = {"seen": 0}
    shutdown_workers = 0
    with open(path, "wt") as fout:
        while running.is_set():
            try:
                job = queue.get(block=True, timeout=0.01)
                if job is None:
                    shutdown_workers += 1
                    if shutdown_workers == nworkers:
                        # work is over
                        _job_print("All densifiers exited, safe to stop writing.")
                        break
                else:
                    new_tokens, article_idx, local_stats = job
                    _job_print("Got writing job.")
                    for key, val in local_stats.items():
                        if key not in global_stats:
                            global_stats[key] = val
                        else:
                            global_stats[key] += val
                    received[article_idx] = new_tokens
                    while current_article_idx in received:
                        for row in received[current_article_idx]:
                            fout.write("\t".join(row))
                        del received[current_article_idx]
                        current_article_idx += 1
                    global_stats["seen"] += len(new_tokens)
                    _job_print("Submitting global stats.")
                    summary_queue.put(global_stats)
                    _job_print("Done Submitting global stats.")
            except Empty:
                pass
    _job_print("Exiting writer.")


def densification_job(worker_id, job_queue, result_queue, running, args, trigger_counter, word_counter, densification_predictor, type_collection):
    _job_print("[{}] Starting worker.".format(worker_id))
    densifier = Densifier(language_path=args.language_path,
                          type_collection=type_collection,
                          trigger_counter=trigger_counter,
                          word_counter=word_counter,
                          densification_predictor=densification_predictor)
    while running.is_set():
        try:
            job = job_queue.get(block=True, timeout=0.01)
            if job is None:
                _job_print("[{}] no more jobs left, exiting".format(worker_id))
                result_queue.put(None)
                break
            tokens, article_id, article_idx = job
            _job_print("[{}] Got a job..".format(worker_id))
            new_tokens = densifier.densify(tokens, article_id)
            _job_print("[{}] Reporting result..".format(worker_id))
            result_queue.put((new_tokens, article_idx, densifier.pop_stats()))
            _job_print("[{}] Done reporting.".format(worker_id))
        except Empty:
            pass
    _job_print("[{}] Exiting worker.".format(worker_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=str,
                        help="path to a prepared tsv file containing token<TAB>entity<TAB>context<TAB>trie_index")
    parser.add_argument("output", type=str,
                        help="path to where to save the densified tsv file.")
    parser.add_argument("--source_data_date", type=str, default="2017-12")
    parser.add_argument("--source_language", type=str, default="en")
    parser.add_argument("--language_path", type=str, required=True)
    parser.add_argument("--wikidata_path", type=str, required=True)
    parser.add_argument("--num_names_to_load", type=int, default=1000000000)
    args = parser.parse_args()

    save_trigger_path = join(DENSIFICATION_DIR, args.source_data_date, args.source_language + "_trigger_counter.pkl")
    save_word_path = join(DENSIFICATION_DIR, args.source_data_date, args.source_language + "_word_counter.pkl")

    # obtain word statistics if you need them:
    if not true_exists(save_trigger_path) or not true_exists(save_word_path):
        print("Did not find precomputed statistics for word and trigger occurences {}, collecting some...".format((save_trigger_path, save_word_path)))
        
        elapsed = 0
        word_counter = Counter()
        trigger_counter = Counter()
        t0 = time.time()
        with open(args.corpus, "rt") as fin:
            for l in fin:
                if "\t" in l:
                    token, target, context, trie_context = l.split("\t")
                    trigger_counter.update([token])
                else:
                    token = l.strip()
                word_counter.update([token])
                seen += 1
                elapsed += 1
                if elapsed == 1e6:
                    print("seen: {}, words/s: {}".format(seen, seen / (time.time() - t0)))
                    elapsed = 0
                # lots of data seen already...
                if seen == 2604000000:
                    break
        os.makedirs(join(DENSIFICATION_DIR, args.source_data_date), exist_ok=True)
        with open(save_trigger_path, "wb") as fout:
            pickle.dump(trigger_counter, fout)
        with open(save_word_path, "wb") as fout:
            pickle.dump(word_counter, fout)
    else:
        print("Found precomputed statistics for word and trigger occurences {}, loading...".format((save_trigger_path, save_word_path)))
        with open(save_trigger_path, "rb") as fin:
            trigger_counter = pickle.load(fin)
        with open(save_word_path, "rb") as fin:
            word_counter = pickle.load(fin)
    densification_classifier_save_path = join(DENSIFICATION_DIR, "model.pkl")
    train = pd.read_csv(join(DENSIFICATION_DIR, "densification_data_train.csv"))
    test = pd.read_csv(join(DENSIFICATION_DIR, "densification_data_test.csv"))
    X_train, Y_train = _prep_data_x_y(train)
    X_test, Y_test = _prep_data_x_y(test)
    if not true_exists(densification_classifier_save_path):
        print("Did not find a saved densification classifier, training one:")
        densification_predictor = sklearn.neural_network.MLPClassifier(max_iter=10000, hidden_layer_sizes=[5]).fit(X_train, Y_train)
        print("Done, train accuracy: {:.2f}%".format(100.0 * np.mean(densification_predictor.predict(X_train) == Y_train)))
        print("Saving to {}...".format(densification_classifier_save_path))
        with open(densification_classifier_save_path, "wb") as fout:
            pickle.dump(densification_predictor, fout)
    else:
        print("Found a saved densification classifier, loading...")
        with open(densification_classifier_save_path, "rb") as fin:
            densification_predictor = pickle.load(fin)
    print("Test accuracy: {:.2f}%".format(100.0 * np.mean(densification_predictor.predict(X_test) == Y_test)))

    # counting lines can be slow...
    nlines = count_lines(args.corpus)
    pbar = get_progress_bar("densification", max_value=nlines)
    pbar.start()
    t0 = time.time()
    seen = 0
    nworkers = cpu_count()
    job_queue = Queue(maxsize=nworkers * 2)
    result_queue = Queue()
    summary_queue = Queue()
    running = Event()
    running.set()
    writer_process = Process(target=write_job, args=(nworkers, args.output, result_queue, summary_queue, running), daemon=True)
    writer_process.start()
    print("Starting {} densification workers".format(nworkers))
    type_collection = TypeCollection(args.wikidata_path, num_names_to_load=args.num_names_to_load)
    densifier_processes = [Process(target=densification_job,
                                   args=(job_idx, job_queue, result_queue, running,
                                         args, trigger_counter, word_counter,
                                         densification_predictor, type_collection), daemon=True)
                           for job_idx in range(nworkers)]
    for dense in densifier_processes:
        dense.start()
    del type_collection
    print("Done.")
    try:
        with open(args.corpus, "rt") as fin:
            article_idx = 0
            for tokens, article_id in _recover_article_boundaries(fin):
                job_queue.put((tokens, article_id, article_idx))
                article_idx += 1
                try:
                    summary = summary_queue.get(block=False)
                    pbar.update(summary["seen"])
                    if time.time() - t0 > 40:
                        t0 = time.time()
                        print("")
                        print("{} total matches, {} tested matches, {} densifications.".format(
                            summary["potential_densifications"] + summary["skipped_match"], summary["potential_densifications"], summary["densifications"],))
                        if summary["densifications"] > 0:
                            print("Mean positive prediction {:.3f}".format(
                                summary["mean_positive_prediction"] / summary["densifications"]))
                        if (summary["potential_densifications"] - summary["densifications"]) > 0:
                            print("Mean negative prediction {:.3f}".format(
                                summary["mean_negative_prediction"] / (summary["potential_densifications"] - summary["densifications"])))
                        if summary["seen_links"] > 0:
                            print("ratio new links to old links: {:.3f}%".format(100.0 * summary["densifications"] / summary["seen_links"]))
                        if summary["densification_calls"] > 0:
                            print("Mean increase in links per page: {:.3f}%".format(100.0 * summary["sum_increase_ratio"] / summary["densification_calls"]))
                except Empty:
                    pass
            for dense in densifier_processes:
                job_queue.put(None)
    except Exception:
        running.clear()
    finally:
        pbar.finish()
        print("Waiting on jobs to finish")
        for p in densifier_processes:
            p.join()
        while not summary_queue.empty():
            summary_queue.get()
        writer_process.join()
        running.clear()


if __name__ == "__main__":
    main()