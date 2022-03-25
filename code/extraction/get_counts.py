import argparse
import marisa_trie
import time
import multiprocessing

import numpy as np

from wikidata_linker_utils.progressbar import get_progress_bar


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trie")
    parser.add_argument("wiki")
    parser.add_argument("counts")
    parser.add_argument("--num_lines", type=int, default=100000)
    parser.add_argument("--num_threads", type=int, default=multiprocessing.cpu_count())
    return parser.parse_args(argv)


def load_corpus(path, num_lines):
    out = []
    with open(path, "rt") as fin:
        for k, l in enumerate(fin):
            if k >= num_lines:
                break
            out.append(l)
    return "".join(out)


def find_stats_worker(total_num_words, corpus, words, output_queue, progress_queue):
    counts = np.zeros(total_num_words, dtype=np.int32)
    done = 0
    for word, idx in words:
        counts[idx] = corpus.count(" " + word + " ")
        done += 1
        if done % 100 == 0:
            progress_queue.put(done)
            done = 0
    progress_queue.put(done)
    output_queue.put(counts)


def find_stats(trie, corpus, num_threads):
    t0 = time.time()
    words = trie.items()
    to_compute = [(word, idx) for word, idx in words if (word.islower() and not word[0].isdigit() and len(word.split()) < 3)]
    print("Got all words")
    total_num_words = len(words)
    total_to_compute = len(to_compute)
    chunk_size = total_to_compute // num_threads
    progress_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    workers = [
    ]
    for i in range(0, total_to_compute, chunk_size):
        worker = multiprocessing.Process(
            target=find_stats_worker,
            args=(
                total_num_words, corpus, to_compute[i:i+chunk_size], output_queue, progress_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)
    print("Launched jobs")

    done = 0
    pbar = get_progress_bar('counts ', max_value=total_to_compute, item='words')
    pbar.start()
    while done < total_to_compute:
        done = done + progress_queue.get()
        if done < total_to_compute:
            pbar.update(done)
    pbar.finish()

    final_res = None
    for worker in workers:
        res = output_queue.get()
        print("Got job result.")
        if final_res is None:
            final_res = res
        else:
            final_res = final_res + res
    for worker in workers:
        worker.join()
    t1 = time.time()
    print("%.3fs elapsed." % (t1 - t0,))
    return final_res


def main():
    args = parse_args()
    trie = marisa_trie.Trie().load(args.trie)
    print('loaded trie')
    corpus = load_corpus(args.wiki, args.num_lines)
    print('loaded corpus')
    counts = find_stats(trie, corpus, args.num_threads)
    print('got stats')
    np.save(args.counts, counts)

if __name__ == "__main__":
    main()
