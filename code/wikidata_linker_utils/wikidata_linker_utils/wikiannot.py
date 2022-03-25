from .wikipedia import load_wikipedia_pageid_to_wikidata
from collections import Counter

unicode_table = [
    ("à", "a"),
    ("é", "e"),
    ("ê", "e"),
    ("ë", "e"),
    ("û", "u"),
    ("ü", "u"),
    ("Ä", "A"),
    ("À", "A"),
    ("Â", "A"),
    ("Ã", "A"),
    ("Å", "A"),
    ("É", "E"),
    ("È", "E"),
    ("Ê", "E"),
    ("Ô", "O"),
    ("ô", "o"),
    ("\xa0", " "),
    ("ð", "o"),
    ("ó", "o"),
    ("á", "a"),
    ("í", "i"),
    ("i̇", "i"),
    ("£", "L"),
    ("¥", "Y"),
    ("€", "E"),
    ("–", "-")
]


def replace_letters(text):
    for source, dest in unicode_table:
        text = text.replace(source, dest)
    return text


def asciify(text):
    return replace_letters(text)


class WikiAnnotDoc(object):
    def __init__(self, line, tags):
        self.line = line
        self.tags = tags

    def matches_filter(self, text):
        return text in self.line

    def links(self, wiki_trie, redirections, prefix):
        lower_line = asciify(self.line.lower())

        tag_seq = []
        for anchor, tag in self.tags:
            anchor = anchor.lower()
            pos = lower_line.find(" " + asciify(anchor) + " ")
            if pos == -1:
                pos = lower_line.find(" " + asciify(anchor))
                if pos == -1:
                    pos = lower_line.find(asciify(anchor))
                    if pos == -1:
                        if "-" in anchor:
                            pos = lower_line.find(asciify(anchor.replace("-", "")))
                            if pos != -1:
                                anchor = anchor.replace("-", "")
                        if pos == -1:
                            print("could not find %r in %r." % (anchor, lower_line,))
                            continue
                else:
                    pos = pos + 1
            else:
                pos = pos + 1
            tag_seq.append((pos, pos + len(anchor), tag))
        tag_seq.sort(key=lambda x: x[0])
        current = 0
        for start, end, tag in tag_seq:
            if start < current:
                if end - current > 0:
                    yield self.line[current:end], None
                    current = end
                continue
            if start > current:
                yield self.line[current:start], None
            yield self.line[start:end], tag
            current = end
        if current != len(self.line):
            yield self.line[current:], None


def load_wikiannot_docs(path, start, size, data_dir, name2index, article2id=None, redirections=None,
                        wikipedia_sql_props=None):
    with open(path, "rt") as fin:
        dataset = fin.read().splitlines()
    groups = []
    if wikipedia_sql_props is None:
        wikipedia_sql_props = load_wikipedia_pageid_to_wikidata(data_dir)
    common_missing = Counter()
    line_idx = 1
    ex_seen = 0
    while len(groups) < size and line_idx < len(dataset):
        # most of the dataset is (example <newline> tags) but there is a newline that gets misdetected by splitlines()
        # above and throws this off, so we instead use a while loop to figure out where the tags and examples are stored.
        if "\t" in dataset[line_idx]:
            ex_seen += 1
            if ex_seen > start:
                tokens = dataset[line_idx].split("\t")
                newtokens = []
                for j in range(1, len(tokens), 2):
                    idx = wikipedia_sql_props.get(tokens[j], None)
                    if idx is not None:
                        dest_index = name2index.get(idx.upper(), None)
                        if dest_index is not None:
                            newtokens.append((tokens[j - 1], dest_index))
                        else:
                            common_missing.update([(tokens[j], tokens[j - 1])])
                    else:
                        common_missing.update([(tokens[j], tokens[j - 1])])
                if len(newtokens) > 0:
                    groups.append(WikiAnnotDoc(" ".join(dataset[line_idx - 1].strip().split()), newtokens))
            line_idx += 1
        else:
            line_idx += 1
    return groups
