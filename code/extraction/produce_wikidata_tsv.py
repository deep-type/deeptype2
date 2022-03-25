import argparse
import re

from os.path import join, dirname, realpath, exists

import marisa_trie
import ciseau
import numpy as np
from lxml import html

from wikidata_linker_utils.wikipedia import (
    iterate_articles, induce_wikipedia_prefix, load_redirections,
    transition_trie_index
)
from wikidata_linker_utils.json import load_config
from wikidata_linker_utils.offset_array import OffsetArray
from wikidata_linker_utils.type_collection import TypeCollection
from wikidata_linker_utils.anchor_filtering import acceptable_anchor, clean_up_trie_source
from wikidata_linker_utils.successor_mask import match_wikipedia_to_wikidata
from wikidata_linker_utils.wikidata_ids import WIKITILE_2_WIKIDATA_TRIE_NAME

SCRIPT_DIR = dirname(realpath(__file__))


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("out")
    parser.add_argument("--relative_to", type=str, default=None)
    return parser.parse_args(args=args)


link_pattern = re.compile(r"\[\[([^\]\[:]*)\]\]")
ref_pattern = re.compile(r"<ref[^<>]*>[^<]+</ref>")
double_bracket_pattern = re.compile(r"{{[^{}]+}}")
title_pattern = re.compile(r"==+([^=]+)==+")
bullet_point_pattern = re.compile(r"^([*#])", re.MULTILINE)


def merge_tags(words, tags, start_sent):
    out = [(w, [], []) for w in words]
    for tag_start, tag_end, tag, anchor_idx in tags:
        so_far = start_sent
        for k, word in enumerate(words):
            begins = tag_start <= so_far or (tag_start > so_far and tag_start < so_far + len(word))
            ends = (so_far + len(word) <= tag_end) or (tag_end < so_far + len(word) and tag_end > so_far)
            if begins and ends:
                out[k][1].append(tag)
                out[k][2].append(anchor_idx)
            so_far += len(word)
            if so_far >= tag_end:
                break
    return out


def pick_relevant_tags(tagged_sequence, char_offset, char_offset_end):
    relevant_tags = []
    for word, tags in tagged_sequence:
        if tags is not None:
            start, end, dest_index, anchor_idx = tags
            if start >= char_offset and start < char_offset_end:
                relevant_tags.append((start, end, dest_index, anchor_idx))
            if start >= char_offset_end:
                break
    return relevant_tags


def convert_document_to_labeled_tags(annotated, sentences):
    paragraphs = []
    paragraph = []
    char_offset = 0
    for sentence in sentences:
        sentence_length = sum(len(w) for w in sentence)
        sentence_tags = pick_relevant_tags(
            annotated,
            char_offset,
            char_offset + sentence_length
        )
        sentence_with_tags = merge_tags(
            sentence,
            sentence_tags,
            char_offset
        )
        sentence_with_tags = [
            (
                w,
                tags[0] if len(tags) > 0 else None,
                anchor_tags[0] if len(tags) > 0 else None,
            ) for w, tags, anchor_tags in sentence_with_tags
        ]
        if "\n" in sentence[-1]:
            paragraph.extend(sentence_with_tags)
            paragraphs.append(paragraph)
            paragraph = []
        else:
            paragraph.extend(sentence_with_tags)
        char_offset += sentence_length
    if len(paragraph) > 0:
        paragraphs.append(paragraph)
    return paragraphs


def annotate_document(doc,
                      collection,
                      wiki_trie,
                      anchor_trie,
                      trie_index2indices,
                      trie_index2indices_counts,
                      trie_index2indices_transitions,
                      redirections,
                      prefix,
                      fix_destination):
    out = []
    current_position = 0
    current_position_no_brackets = 0
    for match in re.finditer(link_pattern, doc):
        start = match.start()
        end = match.end()

        if current_position != start:
            out.append(
                (doc[current_position:start], None)
            )
            current_position_no_brackets += start - current_position
        current_position = end

        match_string = match.group(1).strip()
        if "|" in match_string:
            link, anchor = match_string.rsplit("|", 1)
            link = link.strip().split("#")[0]
            anchor = anchor
            anchor_stripped = anchor.strip()
        else:
            anchor = match_string
            anchor_stripped = match_string.strip()
            link = anchor_stripped

        if len(anchor) > 0 and len(link) > 0:
            anchor = clean_up_trie_source(anchor, lowercase=False)
            lowercase_anchor = anchor.lower()
            if acceptable_anchor(lowercase_anchor, anchor_trie):
                anchor_idx = anchor_trie[lowercase_anchor]
                dest_index = match_wikipedia_to_wikidata(link, wiki_trie, redirections, prefix)
                if dest_index is not None and dest_index != -1:
                    all_options = trie_index2indices[anchor_idx]
                    if len(all_options) > 0:
                        if trie_index2indices_transitions is not None:
                            dest_index = transition_trie_index(
                                anchor_idx, dest_index,
                                trie_index2indices_transitions,
                                all_options
                            )
                        try:
                            if fix_destination is not None:
                                all_counts = trie_index2indices_counts[anchor_idx]
                                new_dest_index, all_options, all_counts, keep = fix_destination(
                                    anchor,
                                    dest_index,
                                    all_options,
                                    all_counts,
                                    collection
                                )
                            else:
                                new_dest_index = dest_index
                                keep = True

                            if keep and new_dest_index != -1:
                                out.append(
                                    (
                                        anchor,
                                        (
                                            current_position_no_brackets,
                                            current_position_no_brackets + len(anchor),
                                            collection.ids[new_dest_index],
                                            anchor_idx
                                        )
                                    )
                                )
                                current_position_no_brackets += len(anchor)
                                continue
                        except IndexError:
                            # missing element
                            pass
        current_position_no_brackets += len(anchor)
        out.append(
            (anchor, None)
        )

    if current_position != len(doc):
        out.append(
            (doc[current_position:len(doc)], None)
        )
    return out


def remove_pattern(doc, begin_pattern, end_pattern):
    while True:
        beginning_sortable = doc.find(begin_pattern)
        if beginning_sortable == -1:
            break
        end_sortable = doc.find(end_pattern, beginning_sortable)
        if end_sortable == -1:
            break
        doc = doc[:beginning_sortable] + doc[end_sortable + len(end_pattern):]
    return doc


def remove_nested_pattern(doc, begin_pattern, end_pattern, min_nesting):
    last_char = 0
    while True:
        begin_char = doc.find(begin_pattern, last_char)
        if begin_char == -1:
            break
        stack_depth = 1
        max_depth = 1
        last_char = begin_char + len(begin_pattern)
        while stack_depth > 0:
            possible_ending = doc.find(end_pattern, last_char)
            possible_nesting = doc.find(begin_pattern, last_char)
            if possible_ending == -1 or possible_nesting == -1:
                break
            if possible_nesting < possible_ending:
                stack_depth += 1
                max_depth = max(stack_depth, max_depth)
                last_char = possible_nesting + len(begin_pattern)
            else:
                stack_depth -= 1
                last_char = possible_ending + len(end_pattern)
        if max_depth > min_nesting:
            doc = doc[:begin_char] + doc[last_char:]
            last_char = begin_char
    return doc


def convert(article_name,
            doc,
            collection,
            wiki_trie,
            anchor_trie,
            trie_index2indices,
            trie_index2indices_counts,
            trie_index2indices_transitions,
            redirections,
            prefix,
            fix_destination):
    doc = doc.replace("\t", " ")
    doc = remove_nested_pattern(doc, "{{", "}}", min_nesting=0)
    doc = remove_pattern(doc, "{|", "|}")
    doc = remove_nested_pattern(doc, "[[", "]]", min_nesting=1)
    # remove ref tags:
    # doc = re.sub(ref_pattern, " ", doc)
    doc = re.sub(title_pattern, r"\n\n\1\. ", doc)
    doc = re.sub(bullet_point_pattern, r"\1 ", doc)

    article_index = match_wikipedia_to_wikidata(
        article_name, wiki_trie, redirections, prefix
    )
    # find location of tagged items in wikipedia:
    annotated = annotate_document(
        doc,
        collection,
        wiki_trie,
        anchor_trie,
        trie_index2indices,
        trie_index2indices_counts,
        trie_index2indices_transitions,
        redirections,
        prefix,
        fix_destination
    )
    text_without_brackets = "".join(text for text, _ in annotated)
    sentences = ciseau.sent_tokenize(
        text_without_brackets,
        normalize_ascii=False,
        keep_whitespace=True
    )
    return (
        convert_document_to_labeled_tags(
            annotated, sentences
        ),
        collection.ids[article_index] if article_index is not None else "other"
    )


def main():
    args = parse_args()
    config = load_config(
        args.config,
        ["wiki", "language_path", "wikidata", "redirections"],
        defaults={
            "num_names_to_load": 0,
            "prefix": None,
            "sample_size": 100,
            "fix_links": False
        },
        relative_to=args.relative_to
    )
    prefix = config.prefix or induce_wikipedia_prefix(config.wiki)

    collection = TypeCollection(
        config.wikidata,
        num_names_to_load=0
    )
    collection.load_blacklist(join(SCRIPT_DIR, "blacklist.json"))

    trie_index2indices = OffsetArray.load(
        join(config.language_path, "trie_index2indices"),
        compress=True
    )
    trie_index2indices_counts = OffsetArray(
        np.load(join(config.language_path, "trie_index2indices_counts.npy")),
        trie_index2indices.offsets
    )
    if exists(join(config.language_path, "trie_index2indices_transition_values.npy")):
        trie_index2indices_transitions = OffsetArray(
            np.load(join(config.language_path, "trie_index2indices_transition_values.npy")),
            np.load(join(config.language_path, "trie_index2indices_transition_offsets.npy")),
        )
    else:
        trie_index2indices_transitions = None

    anchor_trie = marisa_trie.Trie().load(join(config.language_path, "trie.marisa"))
    wiki_trie = marisa_trie.RecordTrie('i').load(
        join(config.wikidata, WIKITILE_2_WIKIDATA_TRIE_NAME)
    )
    redirections = load_redirections(config.redirections)

    fix_destination = None
    if config.fix_links:
        import link_fixer
        link_fixer.init(collection)
        fix_destination = link_fixer.fix

    seen = 0
    with open(args.out, "wt") as fout:
        try:
            for article_name, article in iterate_articles(config.wiki, skip_templated_lines=True):
                doc = html.fromstring("<text>" + article + "</text>")
                # remove ref-tags..
                for el in doc.cssselect("ref"):
                    el.text = ""
                article = doc.text_content()
                fixed_article, article_qid = convert(
                    article_name,
                    article,
                    collection=collection,
                    anchor_trie=anchor_trie,
                    wiki_trie=wiki_trie,
                    trie_index2indices=trie_index2indices,
                    trie_index2indices_counts=trie_index2indices_counts,
                    trie_index2indices_transitions=trie_index2indices_transitions,
                    redirections=redirections,
                    prefix=prefix,
                    fix_destination=fix_destination
                )
                for paragraph in fixed_article:
                    for word, qid, anchor_idx in paragraph:
                        if qid is not None:
                            fout.write(word.rstrip() + "\t" + "\t".join([qid, article_qid, str(anchor_idx)]) + "\n")
                        else:
                            fout.write(word.rstrip() + "\n")
                    fout.write("\n")
                seen += 1
                if seen >= config.sample_size:
                    break
        finally:
            fout.flush()
            fout.close()


if __name__ == "__main__":
    main()
