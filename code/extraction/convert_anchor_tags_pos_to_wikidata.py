import argparse

from os.path import join
from os import makedirs

import marisa_trie
import numpy as np

from wikidata_linker_utils.successor_mask import construct_pos_mapping
from wikidata_linker_utils.wikidata_ids import load_wikidata_ids
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.offset_array import OffsetArray


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "anchor_tags",
        help="Location of anchor+pos tags -> wikidata mapping")
    parser.add_argument(
        "language_path", type=str,
        help="Directory where anchor_trie is stored.")
    parser.add_argument(
        "wikidata_path", type=str,
        help="Directory where wikidata is stored.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    anchor_trie = marisa_trie.Trie().load(join(args.language_path, "trie.marisa"))
    _, name2index = load_wikidata_ids(args.wikidata_path, verbose=True)
    word_count = np.zeros(len(anchor_trie), dtype=np.int32)
    pbar = get_progress_bar('detect multiword', max_value=len(anchor_trie))
    pbar.start()
    for i in (range(len(anchor_trie))):
        word_count[i] = len(anchor_trie.restore_key(i).split(" "))
        if i % 10000 == 0:
            pbar.update(i)
    pbar.finish()
    trie_index2indices = OffsetArray.load(
        join(args.language_path, "trie_index2indices")
    )
    pos_trie_index2indices_counts = construct_pos_mapping(
        anchor_trie=anchor_trie,
        anchor_tags=args.anchor_tags,
        name2index=name2index,
        word_count=word_count,
        values=trie_index2indices.values,
        offsets=trie_index2indices.offsets
    )
    print("Mapping built. Saving numpy array...")
    np.save(join(args.language_path, "pos_trie_index2indices_counts.npy"), pos_trie_index2indices_counts)
    print("Done.")


if __name__ == "__main__":
    main()
