import argparse
import marisa_trie
from os.path import join
from wikidata_linker_utils.anchor_filtering import clean_up_trie_source, anchor_is_ordinal, anchor_is_numbers_slashes
from wikidata_linker_utils.offset_array import save_record_with_offset
from wikidata_linker_utils.wikipedia import load_redirections
from wikidata_linker_utils.successor_mask import match_wikipedia_to_wikidata
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.wikidata_ids import WIKITILE_2_WIKIDATA_TRIE_NAME


def acceptable_anchor(anchor):
    return (
        len(anchor) > 0 and
        not anchor.isdigit() and
        not anchor_is_ordinal(anchor) and
        not anchor_is_numbers_slashes(anchor))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "aida_means",
        help="Location of aida_means.tsv file")
    parser.add_argument(
        "language_path", type=str,
        help="Directory where anchor_trie is stored.")
    parser.add_argument(
        "wikidata_path", type=str,
        help="Directory where wikidata exports are stored.")
    parser.add_argument(
        "redirections", type=str,
        help="Directory where wikipedia redirections are stored.")
    parser.add_argument(
        "output_path", type=str,
        help="Directory where wikidata aida_means is stored.")
    parser.add_argument("--prefix", type=str, default="enwiki")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    anchor_trie = marisa_trie.Trie().load(join(args.language_path, "trie.marisa"))
    with open(args.aida_means, "rt") as fin:
        aida_means = fin.read().splitlines()
    wiki_trie = marisa_trie.RecordTrie('i').load(
        join(args.wikidata_path, WIKITILE_2_WIKIDATA_TRIE_NAME)
    )
    redirections = load_redirections(args.redirections)
    mapping = {}
    for s in get_progress_bar("aida mention")(aida_means):
        mention, target = s.split("\t", 1)
        prev_len = len(mention)
        mention = mention.strip('"')
        assert prev_len == len(mention) + 2, "mention {}".format(mention)
        try:
            clean_key = clean_up_trie_source(mention.encode("ascii").decode('unicode-escape'), prefix="enwiki")
            if acceptable_anchor(clean_key) and clean_key in anchor_trie:
                anchor_idx = anchor_trie[clean_key]
                dest_idx = match_wikipedia_to_wikidata(
                    target.encode("ascii").decode('unicode-escape').replace("_", " "),
                    wiki_trie, redirections, args.prefix)
                if dest_idx is not None and dest_idx > -1:
                    if anchor_idx not in mapping:
                        mapping[anchor_idx] = [dest_idx]
                    else:
                        mapping[anchor_idx].append(dest_idx)
        except UnicodeEncodeError:
            continue
    save_record_with_offset(args.output_path, mapping, total_size=len(anchor_trie))



if __name__ == "__main__":
    main()
