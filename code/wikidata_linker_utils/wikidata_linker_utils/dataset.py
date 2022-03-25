import numpy as np
import subprocess
import h5py
import hashlib
import base64
import time
import pickle
import ciseau
import multiprocessing
from collections import Counter

from os.path import exists, splitext, join, abspath
from .wikidata_ids import load_wikidata_ids
from .offset_array import OffsetArray, SparseAttribute
from .progressbar import get_progress_bar
from .make_callable import make_callable
from .wikipedia import load_redirections
from . import wikidata_properties as wprop
from .type_collection import TypeCollection
from .logic import logical_ors, logical_negate
from .trie import has_keys_with_prefix
from .hashing import nested_hash_vals
from .type_interaction_constants import (
    CONSTRAINTS_REQUIRED_PADDING, CONSTRAINTS_CREATED_PADDING, METADATA_DIMS,
    METADATA_LIST_DIM, METADATA_SCORE_DIM, METADATA_INSIDE_LIST
)


STOP_WORDS = {
    # english
    "a", "an", "in", "the", "of", "it", "from", "with", "this", "that", "they",
    "he", "she", "some", "where", "what", "since", "his", "her", "their",
    "later", "into", "at", "for", "when", "why", "how", "with", "whether", "if",
    "thus", "then", "and", "but", "on", "during", "while", "as", "within", "was",
    "is", "under", "above", "do", "does", "until",
    # french
    "le" , "la", "les", "il", "elle", "ce", "ça", "ci", "ceux", "ceci", "cela",
    "celle", "se", "cet", "cette", "dans", "avec", "con", "sans", "pendant",
    "durant", "avant", "après", "puis", "que", "qui", "quoi", "dont", "ou",
    "où",  "depuis", "à", "de", "du", "un", "une",  "l'", "et", "est", "au",
    "fait", "va", "vont", "sur", "sous", "entre", "en", "pour", "ensuite",
    "sinon", "encore", "devant", "derrière", "dedans", "depuis", "d'", "qu'",
    # spanish
    "el", "lo", "la", "ese", "esto", "si", "este", "esta", "cual", "eso", "ella",
    "y", "a", "su", "los", "las", "una", "uno", "para", "asi", "dentro", "después",
    "desde", "al", "por", "del", "cuando", "cuan"
}

PUNCTUATION = {" ", ".", ",", ";", "!", "(", ")", "-", ":"}
STANDARD_TERMS = {"inc", "corp", "ltd", "sr", "phd", "jr"}


def only_punc_or_upper(remainder_tokens):
    for tok in remainder_tokens:
        if tok.strip().lower() in STANDARD_TERMS:
            continue
        for c in tok:
            if not (c.isupper() or c in PUNCTUATION or c.isdigit()):
                return False
    return True


class MergedTrigger(object):
    __slots__ = ["old", "new"]
    def __init__(self, old, new):
        self.old = old
        self.new = new


def realign_string_labels_with_trie(links, anchor_trie, article2id=None, trie_index2indices_values=None,
                                    only_tag_ending=False, merge_triggers=False):
    full_string_doc = "".join([text for text, _ in links])
    tokens = ciseau.tokenize(full_string_doc,
                             normalize_ascii=False)
    tags_len = []
    so_far = 0
    text2trie_idx = {}
    for link, idx in links:
        if idx is not None:
            if link.startswith((" ", "\"", "'", ".")):
                curlen = len(link)
                link = link.lstrip(" \"'.")
                so_far += curlen - len(link)
                link = link.rstrip(" \"'.")
            simplified_link = link.rstrip().lower()
            trie_idx = None
            if isinstance(idx, ScenarioExample):
                trie_idx = idx.anchor_idx
                idx = idx.label
            if trie_idx is None and anchor_trie.get(simplified_link) is None:
                link_tokens = ciseau.tokenize(link, normalize_ascii=False)
                remainder = "".join(link_tokens[1:]).rstrip()
                simplified_shortened_link = remainder.lower()
                # this rule might fail when looking at non-english text...
                if len(link_tokens) > 0 and link_tokens[0].rstrip().lower() in STOP_WORDS and simplified_shortened_link in anchor_trie:
                    curlen = len(link)
                    link = remainder
                    so_far += curlen - len(link)
                    trie_idx = anchor_trie[simplified_shortened_link]
                else:
                    trie_idx = None
                    # try to simplify further...
                    new_ending = None
                    as_string = ""
                    token_index_end = 0
                    if len(link_tokens) > 0 and has_keys_with_prefix(anchor_trie, link_tokens[0].rstrip().lower()):
                        for token_index_end in reversed(range(1, len(link_tokens) + 1)):
                            as_string = "".join(link_tokens[:token_index_end]).rstrip()
                            as_string_lowercase = as_string.lower()
                            trie_idx = anchor_trie.get(as_string_lowercase, None)
                            if trie_idx is not None:
                                new_ending = token_index_end
                                break
                    if trie_idx is not None and (len(as_string) / len(link)) >= 0.8 or (len(link) - len(as_string)) < 5 or only_punc_or_upper(link_tokens[token_index_end:]):
                        link = as_string
                    elif trie_idx is not None and trie_index2indices_values is not None and idx in trie_index2indices_values[trie_idx]:
                        link = as_string
                    else:
                        # try to look it up directly
                        # if this is the sole link and it points to the right entity, then this is an unambiguous token.
                        has_articles_with_id = has_keys_with_prefix(article2id, "enwiki/" + link.rstrip()) if article2id is not None else False
                        if has_articles_with_id:
                            values = []
                            counts = []
                            valid = False
                            for art, dest in article2id.iteritems("enwiki/" + link.rstrip()):
                                if dest[0] == idx:
                                    valid = True
                                values.append(dest[0])
                                counts.append(1)
                            if valid:
                                trie_idx = UnambiguousAnchor(values=values, counts=counts)
                                link = link.rstrip()
                            else:
                                trie_idx = None
                        else:
                            trie_idx = None
                # give up on recovering these matches
                trie_idx = None
                idx = None
            elif trie_idx is None:
                trie_idx = anchor_trie[simplified_link]
            tags_len.append((so_far, so_far + len(link), idx, trie_idx, link.rstrip()))
            if trie_idx is not None:
                text2trie_idx[link.rstrip()] = trie_idx
        so_far += len(link)
    y_seq = [None for _ in range(len(tokens))]
    trie_seq = [None for _ in range(len(tokens))]
    x_cursor = 0
    so_far = 0

    if merge_triggers:
        new_tags_len = []
        known_replacements = {}
        for start, end, idx, trie_idx, trie_text in tags_len:
            if trie_idx is None:
                new_tags_len.append((start, end, idx, trie_idx, trie_text))
            else:
                if trie_text in known_replacements:
                    longest_match = known_replacements[trie_text]
                else:
                    longest_match = trie_text
                    for other_text in text2trie_idx.keys():
                        split_other = other_text.split()
                        if (len(other_text) > len(longest_match) and
                            trie_text.lower() == split_other[-1].lower() and
                            "in" not in split_other and
                            "of" not in split_other):
                            longest_match = other_text
                    known_replacements[trie_text] = longest_match
                if len(longest_match) > len(trie_text):
                    # print("Merging trigger {} with trigger {}".format(trie_text, longest_match))
                    new_tags_len.append((start, end, idx, MergedTrigger(trie_idx, text2trie_idx[longest_match]), longest_match))
                else:
                    new_tags_len.append((start, end, idx, trie_idx, trie_text))
        tags_len = new_tags_len

    for start, end, idx, trie_idx, trie_text in tags_len:
        while so_far + len(tokens[x_cursor]) <= start:
            so_far += len(tokens[x_cursor])
            x_cursor += 1
        while so_far < end:
            so_far += len(tokens[x_cursor])
            if not only_tag_ending:
                y_seq[x_cursor] = idx
                trie_seq[x_cursor] = trie_idx
            elif so_far >= end:
                y_seq[x_cursor] = idx
                trie_seq[x_cursor] = trie_idx
            x_cursor += 1

    return tokens, y_seq, trie_seq


def realign_string_labels(links):
    tokens, y_seq, _ = realign_string_labels_with_trie(links, None)
    return tokens, y_seq


def count_examples(lines, comment, ignore_value, column_indices):
    example_length = 0
    has_labels = False
    found = 0
    for line in lines:
        if len(line) == 0 or (comment is not None and line.startswith(comment)):
            if example_length > 0 and has_labels:
                found += 1
            example_length = 0
            has_labels = False
        else:
            example_length += 1
            if not has_labels:
                cols = line.split("\t")
                if len(cols) > 1:
                    if ignore_value is not None:
                        for col_index in column_indices:
                            if cols[col_index] != ignore_value:
                                has_labels = True
                                break

                    else:
                        has_labels = True
    if example_length > 0 and has_labels:
        found += 1
    return found


def _update_newline_separated_counts(batch):
    counts = Counter()
    for el in batch:
        counts.update(el.split("\n"))
    return counts


def retokenize_example(x, y):
    tokens = ciseau.tokenize(" ".join(w for w in x), normalize_ascii=False)
    out_y = []
    regular_cursor = 0
    tokens_length_total = 0
    regular_length_total = len(x[regular_cursor]) + 1 if len(x) > 0 else 0
    if regular_cursor + 1 == len(x):
        regular_length_total -= 1
    for i in range(len(tokens)):
        tokens_length_total = tokens_length_total + len(tokens[i])
        while regular_length_total < tokens_length_total:
            regular_cursor += 1
            regular_length_total = regular_length_total + len(x[regular_cursor]) + 1
            if regular_cursor + 1 == len(x):
                regular_length_total -= 1
        out_y.append(y[regular_cursor])
    assert(regular_cursor + 1 == len(x)), "error with %r" % (x,)
    return ([tok.rstrip() for tok in tokens], out_y)


def convert_lines_to_examples(lines, comment, ignore_value,
                              column_indices, x_column, empty_column,
                              retokenize=False):
    examples = []
    x = []
    y = []
    for line in lines:
        if len(line) == 0 or (comment is not None and line.startswith(comment)):
            if len(x) > 0:
                if not all(row == empty_column for row in y):
                    examples.append((x, y))
                x = []
                y = []
        else:
            cols = line.split("\t")
            x.append(cols[x_column])
            if len(cols) == 1:
                y.append(empty_column)
            else:
                if ignore_value is not None:
                    y.append(
                        tuple(
                            cols[col_index] if col_index is not None and cols[col_index] != ignore_value else None
                            for col_index in column_indices
                        )
                    )
                else:
                    y.append(
                        tuple(
                            cols[col_index] if col_index is not None else None
                            for col_index in column_indices
                        )
                    )
    if len(x) > 0 and not all(row == empty_column for row in y):
        examples.append((x, y))
    if retokenize:
        examples = [retokenize_example(x, y) for x, y in examples]
    return examples


def load_tsv(path, x_column, y_columns, objective_names, comment, ignore_value,
             retokenize):
    """"
    Deprecated method for loading a tsv file as a training/test set for a model.

    Arguments:
    ----------
        path: str, location of tsv file
        x_column: int
        y_columns: list<dict>, objectives in this file along with their column.
            (e.g. `y_columns=[{"objective": "POS", "column": 2}, ...])`)
        objective_names: name of all desired columns
        comment: line beginning indicating it's okay to skip
        ignore_value: label value that should be treated as missing
        retokenize: run tokenizer again.
    Returns
    -------
        list<tuple> : examples loaded into memory

    Note: can use a lot of memory since entire file is loaded.
    """
    objective2column = {col["objective"]: col["column"] for col in y_columns}
    column_indices = [objective2column.get(name, None) for name in objective_names]

    if all(col_index is None for col_index in column_indices):
        return []

    empty_column = tuple(None for _ in objective_names)
    with open(path, "rt") as fin:
        lines = fin.read().splitlines()

    return convert_lines_to_examples(lines,
                                     ignore_value=ignore_value,
                                     empty_column=empty_column,
                                     x_column=x_column,
                                     column_indices=column_indices,
                                     comment=comment,
                                     retokenize=retokenize)


class RandomizableDataset(object):
    ignore_y = False
    def set_rng(self, rng):
        self.rng = rng

    def ignore_hash(self):
        return self.kwargs.get("ignore_hash", False)

    def set_randomize(self, randomize):
        self.randomize = randomize

    def set_ignore_y(self, ignore):
        self.ignore_y = ignore
        self.reset_cache()

    def get_word_vocab(self):
        raise NotImplementedError()

    def reset_cache(self):
        raise NotImplementedError()

    def hash(self, hasher):
        raise NotImplementedError()

    def create_column2col_indices(self, ignore_value):
        objective2column = {
            col["objective"]: (
                str(col["column"]),
                self._classifications.get_classifier(col["classification"])
            ) for col in self.y_columns
        }
        self.ignore_value = ignore_value
        if self.ignore_value is not None:
            for _, classifier in objective2column.values():
                if self.ignore_value in classifier.classes:
                    classifier.classes[classifier.classes.index(self.ignore_value)] = None

        self.column2col_indices = {}
        for col_idx, name in enumerate(self.objective_names):
            if name not in objective2column:
                continue
            column, classifier = objective2column[name]
            if column not in self.column2col_indices:
                self.column2col_indices[column] = [(classifier, col_idx)]
            else:
                self.column2col_indices[column].append((classifier, col_idx))

    def scenario_init(self, path, x_column, y_columns, objective_names,
                      tries, classifications, ignore_value, kwargs, randomize=False, rng=None):
        self.x_column = str(x_column)
        if len(y_columns) != 1:
            raise ValueError("expected a single y_column for this dataset but got %r." % (
                len(y_columns),))
        self.y_columns = y_columns
        self.y_column = y_columns[0]
        self.ignore_value = ignore_value
        self.objective_names = objective_names
        self.randomize = randomize
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng
        self._classifications = classifications
        self.path = path
        (self.trie_index2indices_values,
         self.trie_index2indices_counts,
         self.pos_trie_index2indices_counts) = tries.get_trie(self.y_column["language_path"])
        self.objective_column = self.objective_names.index(self.y_column["objective"])
        counts = np.bincount(
            self.trie_index2indices_values.values,
            weights=self.trie_index2indices_counts.values).astype(np.int32)
        self.index2incoming_count = np.ones(len(counts) + 1, dtype=np.int32)
        self.index2incoming_count[:len(counts)] = counts

        if self.pos_trie_index2indices_counts is not None:
            counts = np.bincount(
                self.trie_index2indices_values.values,
                weights=np.sum(self.pos_trie_index2indices_counts.values, axis=-1)).astype(np.int32)
            self.pos_index2incoming_count = np.ones(len(counts) + 1, dtype=np.int32)
            self.pos_index2incoming_count[:len(counts)] = counts

        self.uniform_sample = kwargs.get("uniform_sample", True)
        self.index2incoming_cumprob = None if self.uniform_sample else np.cumsum(self.index2incoming_count / self.index2incoming_count.sum())
        self.kwargs = kwargs
        self.filter_examples = make_callable(kwargs["filter_examples"]) if kwargs.get("filter_examples") is not None else None


class TSVDataset(RandomizableDataset):
    _fhandle = None
    _fhandle_position = 0
    _examples = None
    _example_indices = None
    _example_index = 0
    _eof = False
    ignore_y = False

    def __init__(self, path, x_column, y_columns, objective_names, comment, ignore_value, kwargs,
                 retokenize=False, chunksize=50000000, randomize=False, rng=None):
        """"
        Arguments:
        ----------
            path: str, location of tsv file
            kwargs: dict, configuration options for the dataset
            x_column: int
            y_columns: list<dict>, objectives in this file along with their column.
                (e.g. `y_columns=[{"objective": "POS", "column": 2}, ...])`)
            objective_names: name of all desired columns
            comment: line beginning indicating it's okay to skip
            ignore_value: label value that should be treated as missing
            chunksize: how many bytes to read from the file at a time.
            rng: numpy RandomState
            retokenize: run tokenizer on x again.
        """
        self.path = path
        self.kwargs = kwargs
        self.randomize = randomize
        self.x_column = x_column
        self.y_columns = y_columns
        self.objective_names = objective_names
        self.comment = comment
        self.ignore_value = ignore_value
        self.retokenize = retokenize
        self.chunksize = chunksize
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng
        # column picking setup:
        objective2column = {col["objective"]: col["column"] for col in y_columns}
        self.column_indices = [objective2column.get(name, None) for name in objective_names]
        self.empty_column = tuple(None for _ in objective_names)
        if all(col_index is None for col_index in self.column_indices):
            self.length = 0
        else:
            self._compute_length()

    def hash(self, hasher):
        hasher.update(type(self).__name__.encode("utf-8") + self.path.encode("utf-8"))

    def _signature(self):
        try:
            file_sha1sum = subprocess.check_output(
                ["sha1sum", self.path], universal_newlines=True
            ).split(" ")[0]
        except FileNotFoundError:
            file_sha1sum = subprocess.check_output(
                ["shasum", self.path], universal_newlines=True
            ).split(" ")[0]
        sorted_cols = list(
            map(
                str,
                sorted(
                    [col for col in self.column_indices if col is not None]
                )
            )
        )
        return "-".join([file_sha1sum] + sorted_cols)

    def _compute_length(self):
        length_file = (
            splitext(self.path)[0] +
            "-length-" +
            self._signature() + ".txt"
        )
        if exists(length_file):
            with open(length_file, "rt") as fin:
                total = int(fin.read())
        else:
            total = 0
            while True:
                total += self._count_examples()
                if self._eof:
                    break
            with open(length_file, "wt") as fout:
                fout.write(str(total) + "\n")
        self.length = total

    def __len__(self):
        return self.length

    def close(self):
        if self._fhandle is not None:
            self._fhandle.close()
            self._fhandle = None
        self._fhandle_position = 0
        self._eof = False
        self._examples = None
        self._example_indices = None

    def __del__(self):
        self.close()

    def _read_file_until_newline(self):
        if self._fhandle is None:
            self._fhandle = open(self.path, "rb")
        if self._eof:
            self._fhandle_position = 0
            self._fhandle.seek(0)
            self._eof = False

        read_chunk = None
        while True:
            new_read_chunk = self._fhandle.read(self.chunksize)
            if read_chunk is None:
                read_chunk = new_read_chunk
            else:
                read_chunk += new_read_chunk
            if len(new_read_chunk) < self.chunksize:
                del new_read_chunk
                self._fhandle_position += len(read_chunk)
                self._eof = True
                break
            else:
                del new_read_chunk
                newline_pos = read_chunk.rfind(b"\n\n")
                if newline_pos != -1:
                    # move to last line end position (so that we don't get
                    # half an example.)
                    self._fhandle.seek(self._fhandle_position + newline_pos + 2)
                    self._fhandle_position += newline_pos + 2
                    read_chunk = read_chunk[:newline_pos]
                    break
        return read_chunk

    def _count_examples(self):
        read_chunk = self._read_file_until_newline()
        return count_examples(
            read_chunk.decode("utf-8").splitlines(),
            ignore_value=self.ignore_value,
            column_indices=self.column_indices,
            comment=self.comment
        )

    def _load_examples(self):
        read_chunk = self._read_file_until_newline()
        if self._examples is not None:
            del self._examples
        self._examples = convert_lines_to_examples(
            read_chunk.decode("utf-8").splitlines(),
            ignore_value=self.ignore_value,
            empty_column=self.empty_column,
            x_column=self.x_column,
            column_indices=self.column_indices,
            comment=self.comment,
            retokenize=self.retokenize
        )
        self._example_indices = np.arange(len(self._examples))
        if self.randomize:
            # access loaded data randomly:
            self.rng.shuffle(self._example_indices)
        self._example_index = 0

    def get_word_vocab(self):
        counts = Counter()
        while True:
            read_chunk = self._read_file_until_newline()
            lines = read_chunk.decode("utf-8").splitlines()
            for line in lines:
                if len(line) == 0 or (self.comment is not None and line.startswith(self.comment)):
                    pass
                else:
                    cols = line.split("\t")
                    counts.update([cols[self.x_column]])
            if self._eof:
                break
        return counts

    def __getitem__(self, index):
        """Retrieve the next example (index is ignored)"""
        if index >= self.length:
            raise StopIteration()
        if self._example_indices is None or self._example_index == len(self._example_indices):
            self._load_examples()
        while len(self._examples) == 0:
            self._load_examples()
            if len(self._examples) > 0:
                break
            if self._eof:
                raise StopIteration()
        ex = self._examples[self._example_indices[self._example_index]]
        self._example_index += 1
        return ex

    def set_randomize(self, randomize):
        if randomize != self.randomize:
            self.randomize = randomize

    def reset_cache(self):
        # has no effect on tsvs
        pass


class OracleClassification(object):
    def __init__(self, classes, classification, path):
        self.classes = classes
        self.classification = classification
        self.path = path
        self.contains_other = self.classes[-1] == "other" if self.classes is not None else False

    def num_entities(self):
        return len(self.classification)

    def classify(self, index):
        return self.classification[index]

    def padded_gather(self, *args, **kwargs):
        return self.classification.padded_gather(*args, **kwargs)

    def batch_is_related(self, *args, **kwargs):
        return self.classification.batch_is_related(*args, **kwargs)


class IdentityOracleClassification(object):
    def __init__(self, classes, variable_length):
        self.variable_length = variable_length
        self.classes = classes

    def num_entities(self):
        return len(self.classes)

    def classify(self, index):
        return index


def load_oracle_classification(path, variable_length=False):
    if exists(join(path, "classes.txt")):
        with open(join(path, "classes.txt"), "rt", encoding="UTF-8") as fin:
            classes = fin.read().splitlines()
    else:
        classes = None
    classification = (
        OffsetArray.load(join(path, "classification"))
        if variable_length else np.load(join(path, "classification.npy"))
    )
    return OracleClassification(classes, classification, path)


class ClassificationHandler(object):
    def __init__(self, wikidata_path, classification_path, num_names_to_load, verbose=False):
        self.classification_path = classification_path
        self.wikidata_path = wikidata_path
        self.verbose = verbose
        self.type_collection = TypeCollection(wikidata_path, verbose=verbose, num_names_to_load=num_names_to_load, cache=False)
        self.name2index = self.type_collection.name2index
        self.classifiers = {}

    @property
    def article2id(self):
        return self.type_collection.article2id

    def num_entities(self):
        return len(self.type_collection.ids)

    def get_classifier(self, name, variable_length=False):
        if name not in self.classifiers:
            if name is None:
                self.classifiers[name] = IdentityOracleClassification(self.type_collection.ids, variable_length=variable_length)
            else:
                self.classifiers[name] = load_oracle_classification(
                    join(self.classification_path, name), variable_length=variable_length
                )
        return self.classifiers[name]


class ExtensibleTrie(object):
    def __init__(self, anchor_trie):
        self.anchor_trie = anchor_trie
        self._extras = {}
        self._idx2key = {}

    def __contains__(self, value):
        return value in self.anchor_trie or value in self._extras

    def __getitem__(self, value):
        if value in self.anchor_trie:
            return self.anchor_trie.__getitem__(value)
        else:
            return self._extras.__getitem__(value)

    def get(self, value, fallback=None):
        return self.anchor_trie.get(value, fallback) if value in self.anchor_trie else self._extras.get(value, fallback)

    def insert(self, value):
        if value in self.anchor_trie:
            return self.anchor_trie[value]
        elif value in self._extras:
            return self._extras[value]
        else:
            initial_idx = len(self.anchor_trie)
            extra_idx = len(self._extras)
            new_idx = initial_idx + extra_idx
            self._extras[value] = new_idx
            self._idx2key[new_idx] = value
            return new_idx

    def iterkeys(self, prefix=None):
        for key in self._extras.keys():
            if prefix is None or key.startswith(prefix):
                yield key
        for key in self.anchor_trie.iterkeys(prefix=prefix):
            yield key

    def restore_key(self, index):
        if index in self._idx2key:
            return self._idx2key[index]
        return self.anchor_trie.restore_key(index)


class TrieHandler(object):
    def __init__(self, trie_path):
        self.trie_path = trie_path
        self.tries = {}
        self.pos_tries = {}
        self.anchor_tries = {}
        self.redirections = {}
        self._to_insert = {}

    def insert_entity(self, anchor, entity_id, count=None):
        if anchor not in self._to_insert:
            self._to_insert[anchor] = []
        self._to_insert[anchor].append((entity_id, count))

    def get_trie(self, name):
        if name not in self.tries:
            trie_index2indices = OffsetArray.load(
                join(self.trie_path, name, "trie_index2indices")
            )
            trie_index2indices_counts = OffsetArray(
                np.load(join(self.trie_path, name, "trie_index2indices_counts.npy")),
                trie_index2indices.offsets
            )
            if exists(join(self.trie_path, name, "pos_trie_index2indices_counts.npy")):
                pos_trie_index2indices_counts = OffsetArray(
                    np.load(join(self.trie_path, name, "pos_trie_index2indices_counts.npy")),
                    trie_index2indices.offsets
                )
            else:
                pos_trie_index2indices_counts = None
            if len(self._to_insert) > 0:
                # dynamically creating new entity options.
                anchor_trie = self.get_anchor_trie(name)
                for key, value in self._to_insert.items():
                    if key in anchor_trie:
                        anchor_idx = anchor_trie.get(key)
                        median_count = np.median(trie_index2indices_counts[anchor_idx])
                    else:
                        anchor_idx = anchor_trie.insert(key)
                        median_count = 1
                    for entity_idx, entity_count in value:
                        if entity_count is None:
                            entity_count = median_count
                        # TODO add insert multiple.
                        trie_index2indices.insert(anchor_idx, [entity_idx], update_offsets=False)
                        if pos_trie_index2indices_counts is not None:
                            pos_trie_index2indices_counts.insert(anchor_idx, [entity_count], update_offsets=False)
                        trie_index2indices_counts.insert(anchor_idx, [entity_count])
            self.tries[name] = (trie_index2indices, trie_index2indices_counts, pos_trie_index2indices_counts)
        return self.tries[name]

    def get_redirections(self, name):
        if name not in self.redirections:
            self.redirections[name] = load_redirections(join(self.trie_path, name + "_redirections.tsv"))
        return self.redirections[name]

    def get_anchor_trie(self, name):
        if name not in self.anchor_tries:
            import marisa_trie
            self.anchor_tries[name] = ExtensibleTrie(marisa_trie.Trie().load(
                join(self.trie_path, name, "trie.marisa")
            ))
        return self.anchor_tries[name]


class ClassificationRandomizableDataset(RandomizableDataset):
    def __init__(self, path, x_column, y_columns, objective_names,
                 classifications, ignore_value, kwargs, randomize=False, rng=None):
        self.path = path
        self.kwargs = kwargs
        self.x_column = str(x_column)
        self.y_columns = y_columns
        self.objective_names = objective_names
        # Note: randomize not implemented yet here...
        self.randomize = randomize
        if rng is None:
            rng = np.random.RandomState(0)
        self.rng = rng
        self._classifications = classifications
        self.create_column2col_indices(ignore_value)


class H5Dataset(ClassificationRandomizableDataset):
    handle_open = False
    ignore_y = False
    _max_generated_example = 0
    _min_generated_example = 0

    def setup_h5(self):
        self.handle = h5py.File(self.path, "r")
        handle_keys = list(self.handle.keys())
        if self.x_column not in handle_keys:
            raise ValueError("missing column %r in dataset %r (found %r)." % (
                self.x_column, self.path, list(self.handle.keys())))
        for col in y_columns:
            if str(col["column"]) not in handle_keys:
                raise ValueError("missing column %r in dataset %r (found %r)." % (
                    str(col["column"]), self.path, list(self.handle.keys())))
        
        self.handle_open = True
        self.length = len(self.handle[self.x_column])
        self.chunksize = self.handle[self.x_column].chunks[0]
        self._example_indices = None
        self.handle_column = str(self.y_column["column"])
        self.anchor_column = str(self.y_column["anchor_column"])


    def __init__(self, path, x_column, y_columns, objective_names,
                 classifications, ignore_value, kwargs, randomize=False, rng=None):
        super().__init__(path=path, kwargs=kwargs, randomize=randomize, rng=rng,
                         objective_names=objective_names, x_column=x_column,
                         y_columns=y_columns, classifications=classifications,
                         ignore_value=ignore_value)
        self.setup_h5()

    def close(self):
        if self.handle_open:
            self.handle.close()
            self.handle_open = False

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def get_word_vocab(self):
        assert self.handle_open, "cannot get get_word_vocab on closed H5 dataset."
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        full_counter = Counter()
        try:
            jobs = []
            t0 = time.time()
            for i in get_progress_bar("read data from H5 ", item="batches")(range(0, self.length, 500000)):
                batch = self.handle["0"][i:i + 500000]
                jobs.append(pool.apply_async(_update_newline_separated_counts, (batch,)))
            for job in get_progress_bar("accumulate counts ", item="counters")(jobs):
                full_counter += job.get()
            elapsed = time.time() - t0
        finally:
            pool.close()
        print("Done with counts for %r in %.3fs" % (self.path, elapsed), flush=True)
        return full_counter

    def _build_examples(self, index):
        x = [x_chunk.split("\n") for x_chunk in self.handle[self.x_column][index:index + self.chunksize]]
        y = [[[None for k in range(len(self.objective_names))] for j in range(len(x[i]))] for i in range(len(x))]
        if not self.ignore_y:
            for handle_column, col_content in self.column2col_indices.items():
                col_ids = [[self._classifications.name2index[name] if name != "" else None
                            for name in y_chunk.split("\n")]
                           for y_chunk in self.handle[handle_column][index:index + self.chunksize]]
                for i in range(len(col_ids)):
                    for j, idx in enumerate(col_ids[i]):
                        if idx is not None:
                            for classifier, k in col_content:
                                y[i][j][k] = classifier.classify(idx)

        return x, y

    def set_randomize(self, randomize):
        if self.randomize != randomize:
            self.randomize = randomize
            if self._max_generated_example != self._min_generated_example:
                self.xorder = np.arange(self._min_generated_example, self._max_generated_example)
                self.rng.shuffle(self.xorder)

    def reset_cache(self):
        """Reset the cache of H5 wipes the current loaded buffer of data."""
        self._max_generated_example = 0
        self._min_generated_example = 0
        self.x = None
        self.y = None

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()
        if self.randomize:
            if self._example_indices is None or index == 0:
                self._example_indices = np.arange(0, len(self), self.chunksize)
                self.rng.shuffle(self._example_indices)
            # transformed index:
            index = (self._example_indices[index // self.chunksize] + (index % self.chunksize)) % len(self)

        if index < self._min_generated_example or index >= self._max_generated_example:
            self.x, self.y = self._build_examples(index)
            # store bounds of generated data:
            self._min_generated_example = index
            self._max_generated_example = index + len(self.x)

            if self.randomize:
                self.xorder = np.arange(self._min_generated_example, self._max_generated_example)
                self.rng.shuffle(self.xorder)
        if self.randomize:
            index = self.xorder[index - self._min_generated_example]
        return self.x[index - self._min_generated_example], self.y[index - self._min_generated_example]

    def hash(self, hasher):
        hasher.update(type(self).__name__.encode("utf-8") + self.path.encode("utf-8"))


class StandardClassificationDataset(ClassificationRandomizableDataset):
    def reset_cache(self):
        # has no effect on standard datasets
        pass

    def hash(self, hasher):
        hasher.update(type(self).__name__.encode("utf-8") + self.path.encode("utf-8"))

    def __init__(self, path, x_column, y_columns, objective_names,
                 classifications, ignore_value, kwargs, corpus_loader, randomize=False, rng=None):
        super().__init__(path=path, kwargs=kwargs, randomize=randomize, rng=rng,
                         objective_names=objective_names, x_column=x_column,
                         y_columns=y_columns, classifications=classifications,
                         ignore_value=ignore_value)
        corpus = corpus_loader(path=self.path,
                               data_dir=self._classifications.wikidata_path,
                               name2index=self._classifications.name2index,
                               article2id=self._classifications.article2id)
        self.x = []
        self.y = []
        redirections = classifications.redirections if kwargs.get("redirections_required") else None
        for doc in corpus:
            doc_links = doc.links(wiki_trie=classifications.article2id,
                                  redirections=redirections,
                                  prefix=self.kwargs.get("prefix", "enwiki"))
            tokens, labels = realign_string_labels(doc_links)
            example_x = [token.rstrip() for token in tokens]
            example_y = [[None for k in range(len(self.objective_names))] for _ in labels]
            for col_content in self.column2col_indices.values():
                for tstep, idx in enumerate(labels):
                    if idx is not None:
                        for classifier, k in col_content:
                            example_y[tstep][k] = classifier.classify(idx)
            x.append(example_x)
            y.append(example_y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def _place_index_at_start(array, index):
    return np.concatenate([array[index:index + 1], array[:index], array[index + 1:]])


def empty_metadata(n):
    return np.zeros((n, METADATA_DIMS), dtype=np.int32)


def _extend_candidate_metadata(existing, n):
    extended = empty_metadata(len(existing) + n)
    extended[:len(existing)] = existing
    # copy "listedness" over
    extended[len(existing):, METADATA_LIST_DIM] = existing[0, METADATA_LIST_DIM]
    return extended


SCENARIO_EXAMPLE_INDEX_FIELDS = ["indices", "counts", "entity_counts", "pos_counts", "pos_entity_counts",
                                 "created_constraints", "constraint_required", "candidate_metadata"]
SCENARIO_EXAMPLE_PADDED_FIELDS = {"constraint_required": CONSTRAINTS_REQUIRED_PADDING, "created_constraints": CONSTRAINTS_CREATED_PADDING}
SCENARIO_EXAMPLE_EXTENSIBLE_FIELDS = {"candidate_metadata": _extend_candidate_metadata}


class ScenarioExample(object):
    __slots__ = SCENARIO_EXAMPLE_INDEX_FIELDS + ["label", "original_label", "uniform_sample", "index2incoming_cumprob", "index2incoming_count", "anchor_idx"]

    def __init__(self, label, indices, counts, entity_counts, original_label,
                 uniform_sample, index2incoming_cumprob, index2incoming_count, anchor_idx, pos_counts, pos_entity_counts,
                 created_constraints, constraint_required, candidate_metadata):
        self.label = label
        self.indices = indices
        self.counts = counts
        self.entity_counts = entity_counts
        self.original_label = original_label
        self.uniform_sample = uniform_sample
        self.index2incoming_cumprob = index2incoming_cumprob
        self.index2incoming_count = index2incoming_count
        self.pos_counts = pos_counts
        self.pos_entity_counts = pos_entity_counts
        self.anchor_idx = anchor_idx
        self.created_constraints = created_constraints
        self.constraint_required = constraint_required
        self.candidate_metadata = candidate_metadata

    def __str__(self):
        return "ScenarioExample(label={}, indices={}, counts={}, entity_counts={}, original_label={}, anchor_idx={}, pos_counts={}, pos_entity_counts={}, )".format(
            self.label, self.indices, self.counts, self.entity_counts, self.original_label,
            self.anchor_idx, self.pos_counts, self.pos_entity_counts)

    def __repr__(self):
        return str(self)

    def init_metadata(self):
        if self.candidate_metadata is None:
            self.candidate_metadata = empty_metadata(len(self.indices))

    def iterate_not_none_fields(self):
        for fieldname in SCENARIO_EXAMPLE_INDEX_FIELDS:
            field = getattr(other, fieldname)
            if field is not None:
                yield 
    
    def copy_index_fields(self, other):
        for fieldname in SCENARIO_EXAMPLE_INDEX_FIELDS:
            field = getattr(other, fieldname)
            if field is not None:
                setattr(self, fieldname, field.copy())

    def reorder_indices(self, indices):
        for fieldname in SCENARIO_EXAMPLE_INDEX_FIELDS:
            field = getattr(self, fieldname)
            if field is not None:
                setattr(self, fieldname, field[indices])

    def extend_indices(self, **fields):
        for fieldname in SCENARIO_EXAMPLE_INDEX_FIELDS:
            field = getattr(self, fieldname)
            if field is not None:
                if fieldname in SCENARIO_EXAMPLE_PADDED_FIELDS:
                    if fields[fieldname] is None:
                        # just insert some padding.
                        padding = np.ones(len(fields["new_indices"]), dtype=np.int32)
                        padding.fill(SCENARIO_EXAMPLE_PADDED_FIELDS[fieldname])
                        setattr(self, fieldname, np.concatenate([field, padding]))
                    else:
                        insertion = fields[fieldname]
                        if insertion.shape[1] != field.shape[1]:
                            # create a larger storage medium for both
                            padded = np.empty((len(field) + len(insertion), max(field.shape[1], insertion.shape[1])), dtype=np.int32)
                            padded.fill(SCENARIO_EXAMPLE_PADDED_FIELDS[fieldname])
                            padded[:len(field), :field.shape[1]] = field
                            padded[len(field):, :insertion.shape[1]] = insertion
                            setattr(self, fieldname, padded)
                        else:
                            # just glue the pieces together
                            setattr(self, fieldname, np.concatenate([field, fields[fieldname]]))
                else:
                    # no padding required, just glue this piece in.
                    if fields[fieldname] is None:
                        assert fieldname in SCENARIO_EXAMPLE_EXTENSIBLE_FIELDS
                        setattr(self, fieldname, SCENARIO_EXAMPLE_EXTENSIBLE_FIELDS[fieldname](field, len(fields["indices"])))
                    else:
                        setattr(self, fieldname, np.concatenate([field, fields[fieldname]]))

    def place_index_at_start(self, index):
        for fieldname in SCENARIO_EXAMPLE_INDEX_FIELDS:
            field = getattr(self, fieldname)
            if field is not None:
                setattr(self, fieldname, _place_index_at_start(field, index))


class UnambiguousAnchor(object):
    __slots__ = ["values", "counts"]
    def __init__(self, values, counts):
        self.values = values
        self.counts = counts


class H5ScenarioDataset(H5Dataset):
    def setup_h5(self):
        self.handle = h5py.File(self.path, "r")
        self.handle_column = str(self.y_column["column"])
        self.anchor_column = str(self.y_column["anchor_column"])
        handle_keys = list(self.handle.keys())
        if self.x_column not in handle_keys:
            raise ValueError("missing column %r in dataset %r (found %r)." % (
                self.x_column, self.path, handle_keys))
        if self.handle_column not in handle_keys:
            raise ValueError("missing column %r in dataset %r (found %r)." % (
                self.handle_column, self.path, handle_keys))
        if self.anchor_column not in handle_keys:
            raise ValueError("missing column %r in dataset %r (found %r)." % (
                self.anchor_column, self.path, handle_keys))
        self.handle_open = True
        self.length = len(self.handle[self.x_column])
        self.chunksize = self.handle[self.x_column].chunks[0]
        self._example_indices = None

    def __init__(self, path, x_column, y_columns, objective_names,
                 tries, classifications, ignore_value, kwargs, randomize=False, rng=None):
        self.scenario_init(path=path, x_column=x_column, y_columns=y_columns, objective_names=objective_names,
                           tries=tries, classifications=classifications,
                           ignore_value=ignore_value, kwargs=kwargs, randomize=randomize, rng=rng)
        self.setup_h5()
    
    def _build_examples(self, index):
        x = [x_chunk.split("\n") for x_chunk in self.handle[self.x_column][index:index + self.chunksize]]
        y = [[[None for k in range(len(self.objective_names))] for j in range(len(x[i]))] for i in range(len(x))]
        if not self.ignore_y and self.objective_column != -1:
            qids = [y_chunk.split("\n") for y_chunk in self.handle[self.handle_column][index:index + self.chunksize]]
            col_ids = [[self._classifications.name2index[name] if name != "" else None for name in qid_row] for qid_row in qids]
            anchor_ids = [[int(name) if name != "" else None
                           for name in y_chunk.split("\n")]
                          for y_chunk in self.handle[self.anchor_column][index:index + self.chunksize]]
            for i in range(len(col_ids)):
                for j, label in enumerate(col_ids[i]):
                    if label is not None:
                        anchor_idx = anchor_ids[i][j]
                        label_count = 1
                        for sub_i in range(len(self.trie_index2indices_values[anchor_idx])):
                            if self.trie_index2indices_values == label:
                                label_count = self.trie_index2indices_counts[anchor_idx][sub_i]
                                break
                        ex = ScenarioExample(
                            label,
                            self.trie_index2indices_values[anchor_idx],
                            self.trie_index2indices_counts[anchor_idx],
                            entity_counts=self.index2incoming_count[np.minimum(self.trie_index2indices_values[anchor_idx], len(self.index2incoming_count) - 1)],
                            original_label=qids[i][j],
                            uniform_sample=self.uniform_sample,
                            index2incoming_cumprob=self.index2incoming_cumprob,
                            index2incoming_count=self.index2incoming_count,
                            anchor_idx=anchor_idx,
                            pos_counts=self.pos_trie_index2indices_counts[anchor_idx] if self.pos_trie_index2indices_counts is not None else None,
                            pos_entity_counts=self.pos_index2incoming_count[np.minimum(self.trie_index2indices_values[anchor_idx], len(self.index2incoming_count) - 1)] if self.pos_trie_index2indices_counts is not None else None,
                            created_constraints=None,
                            constraint_required=None,
                            candidate_metadata=None,
                        )
                        if self.filter_examples is None or self.filter_examples(ex):
                            y[i][j][self.objective_column] = ex
        return x, y



def _has_keys_with_prefix_period(trie, token):
    return has_keys_with_prefix(trie, token) or has_keys_with_prefix(trie, token.replace(".", ""))


def _trie_get_period(trie, token):
    res = trie.get(token, None)
    if res is None:
        return trie.get(token.replace(".", ""))
    return res


def densifier(text, anchor_trie, trie_index2indices_values, trie_index2indices_counts, is_valid, max_match_length=5):
    import ciseau
    matches = []
    tokens = ciseau.tokenize(text, normalize_ascii=False)
    STOP_WORDS = {"i", "not", "if", "the", "else", ".", ","}
    token_index = 0
    while token_index < len(tokens):
        token_match_end = None
        trie_index = None
        token_prefix = tokens[token_index].rstrip()
        if len(token_prefix) > 0 and token_prefix[0].isupper() and token_prefix.lower() not in STOP_WORDS and _has_keys_with_prefix_period(anchor_trie, token_prefix.lower()):
            for token_index_end in reversed(range(token_index + 1, min(len(tokens) + 1, token_index + max_match_length + 1))):
                full_string = "".join(tokens[token_index:token_index_end])
                as_string = full_string.rstrip()
                as_string_lowercase = as_string.lower()
                if len(as_string_lowercase) == 1:
                    continue
                trie_index = _trie_get_period(anchor_trie, as_string_lowercase)
                if trie_index is not None and "," not in as_string_lowercase:
                    # Replaces silly trigger matches of the form 'california.' by 'california'
                    if tokens[token_index_end-1].strip() == ".":
                        alt_full_string = "".join(tokens[token_index:token_index_end - 1])
                        alt_as_string = alt_full_string.rstrip()
                        alt_as_string_lowercase = alt_as_string.lower()
                        alt_trie_index = _trie_get_period(anchor_trie, alt_as_string_lowercase)
                        if alt_trie_index is not None and alt_trie_index != trie_index:
                            alt_anchor_count = trie_index2indices_counts[alt_trie_index].sum()
                            anchor_count = trie_index2indices_counts[trie_index].sum()
                            if alt_anchor_count > anchor_count:
                                token_match_end = token_index_end - 1
                                trie_index = alt_trie_index
                                full_string = alt_full_string
                                as_string = alt_as_string
                                as_string_lowercase = alt_as_string_lowercase
                                break

                    token_match_end = token_index_end
                    break

        if token_match_end is not None and np.any(is_valid[trie_index2indices_values[trie_index]]):
            matches.append((as_string, ScenarioExample(
                            -1,
                            None,
                            None,
                            entity_counts=None,
                            original_label=None,
                            uniform_sample=None,
                            index2incoming_cumprob=None,
                            index2incoming_count=None,
                            anchor_idx=trie_index,
                            pos_counts=None,
                            pos_entity_counts=None,
                            created_constraints=None,
                            constraint_required=None,
                            candidate_metadata=None,
                        )))
            if len(as_string) < len(full_string):
                matches.append((full_string[len(as_string):], None))
            token_index = token_match_end
        else:
            if len(matches) > 0 and matches[-1][1] is None:
                matches[-1] = (matches[-1][0] + tokens[token_index], None)
            else:
                matches.append((tokens[token_index], None))
            token_index += 1
    return matches


def _reduce_according_to_index(anchor_values, anchor_counts, entity_counts, pos_counts, pos_entity_counts, candidate_metadata):
    unique_anchor_values = np.unique(anchor_values)
    if len(unique_anchor_values) == len(anchor_values):
        return anchor_values, anchor_counts, entity_counts, pos_counts, pos_entity_counts, candidate_metadata
    anchor_value2position = {anchor_value: position for position, anchor_value in enumerate(unique_anchor_values)}
    unique_anchor_counts = np.zeros_like(unique_anchor_values)
    unique_entity_counts = np.zeros_like(unique_anchor_values)
    unique_pos_counts = np.zeros_like(unique_anchor_values) if pos_counts is not None else None
    unique_pos_entity_counts = np.zeros_like(unique_anchor_values) if pos_entity_counts is not None else None
    unique_candidate_metadata = empty_metadata(len(unique_anchor_values)) if candidate_metadata is not None else None
    for idx, anchor_value in enumerate(anchor_values):
        unique_anchor_counts[anchor_value2position[anchor_value]] += anchor_counts[idx]
        unique_entity_counts[anchor_value2position[anchor_value]] += entity_counts[idx]
        if candidate_metadata is not None:
            unique_candidate_metadata[anchor_value2position[anchor_value]] = candidate_metadata[idx]
        if pos_counts is not None:
            unique_pos_counts[anchor_value2position[anchor_value]] += pos_counts[idx]
        if pos_entity_counts is not None:
            unique_pos_entity_counts[anchor_value2position[anchor_value]] += pos_entity_counts[idx]
    return unique_anchor_values, unique_anchor_counts, unique_entity_counts, unique_pos_counts, unique_pos_entity_counts


def pad_arrays(arrays, fill_value, dtype=np.int32):
    shape_1 = max(map(len, arrays))
    padded = np.zeros((len(arrays), shape_1), dtype=dtype)
    padded.fill(fill_value)
    for i, row in enumerate(arrays):
        padded[i, :len(row)] = row
    return padded


class Namespace(object):
    def __init__(self, **kwargs):
        self._default_value = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_default(self, value):
        self._default_value = value

    def __getattr__(self, name):
        assert self._default_value is not None
        return self._default_value


class StandardScenarioDataset(RandomizableDataset):
    def reset_cache(self):
        # has no effect on standard datasets
        pass

    def hash(self, hasher):
        hasher.update(type(self).__name__.encode("utf-8") + self.path.encode("utf-8"))

    def get_word_vocab(self):
        counts = Counter()
        for ex in self.x:
            counts.update(ex)
        return counts

    def _apply_filter_to_label(self, label, token, selection_mask, name, soft_filter=False):
        l = label
        if soft_filter:
            l.init_metadata()
            # how much to downweigh a beam that carries this candidate.
            l.candidate_metadata[np.logical_not(selection_mask), METADATA_SCORE_DIM] = -1000
            if found_gt and l.candidate_metadata[l.label == l.indices, METADATA_SCORE_DIM][0] < 0:
                print(f"Penalizing ground truth [{name}] {self._classifications.type_collection.get_name(l.label, web=True)} {token}")
        else:
            l.reorder_indices(selection_mask)

    def _propagate_filter(self, label, example_y, labels, start, increment=1):
        l = label
        start_anchor_idx = l.anchor_idx
        while start >= 0 and start < len(labels) and example_y[start][self.objective_column] is not None and example_y[start][self.objective_column].anchor_idx == start_anchor_idx:
            example_y[start][self.objective_column].copy_index_fields(l)
            start += increment

    def _apply_filter_forwards(self, example_y, tokens, labels, tstep, condition, name, soft_filter=False, increment=1):
        l = example_y[tstep][self.objective_column]
        selection_mask = condition[l.indices]
        current_pick_could_match_condition = np.any(selection_mask)
        if current_pick_could_match_condition:
            self._apply_filter_to_label(l, tokens[tstep], selection_mask, name, soft_filter=soft_filter)
            self._propagate_filter(l, example_y, labels, tstep + increment, increment=increment)

    def _place_comma_place(self, labels, tokens, example_y, booleans):
        t0 = time.time()
        detected_places = {}
        inside_parenthesis = False
        for tstep, label in enumerate(labels):
            stripped_token = tokens[tstep].strip()
            if stripped_token == "(":
                inside_parenthesis = True
            elif stripped_token == ")":
                inside_parenthesis = False
            if (label is None and
                stripped_token == "," and
                tstep > 0 and
                tstep + 1 < len(labels) and
                example_y[tstep - 1][self.objective_column] is not None and
                example_y[tstep + 1][self.objective_column] is not None and
                len(example_y[tstep - 1][self.objective_column].indices) > 0 and
                len(example_y[tstep + 1][self.objective_column].indices) > 0):
                # check if previous label could be a place
                prev_label = example_y[tstep - 1][self.objective_column]
                next_label = example_y[tstep + 1][self.objective_column]
                likely_country_country = booleans.is_country[prev_label.indices[prev_label.counts.argmax()]] and booleans.is_country[next_label.indices[next_label.counts.argmax()]]
                if likely_country_country:
                    continue

                top_prev_pick_could_be_us_state = np.any(booleans.is_us_state[prev_label.indices])
                top_next_pick_could_be_usa = next_label.indices[next_label.counts.argmax()] == self._classifications.type_collection.name2index["Q30"]
                # Skip cases of the form us-state, USA
                if top_prev_pick_could_be_us_state and top_next_pick_could_be_usa:
                    continue

                next_condition_mask = booleans.is_region
                prev_condition_mask = booleans.is_town_or_county
                alt_next_condition_mask = booleans.is_town
                alt_prev_condition_mask = booleans.is_institution
                top_prev_pick_could_be_place = np.any(prev_condition_mask[prev_label.indices])
                top_next_pick_could_be_place = np.any(next_condition_mask[next_label.indices])
                no_place_place_variations = False
                if not top_prev_pick_could_be_place:
                    next_condition_mask = alt_next_condition_mask
                    prev_condition_mask = alt_prev_condition_mask
                    top_prev_pick_could_be_place = np.any(prev_condition_mask[prev_label.indices])
                    top_next_pick_could_be_place = np.any(next_condition_mask[next_label.indices])
                    no_place_place_variations = True
                if top_prev_pick_could_be_place and top_next_pick_could_be_place:
                    t0_0 = time.time()
                    # coherent pairs
                    while True:
                        c = self._classifications.type_collection
                        curr_selection_mask = prev_condition_mask[prev_label.indices]
                        next_selection_mask = next_condition_mask[next_label.indices]
                        next_indices = next_label.indices[next_selection_mask]
                        # sometimes you need a parent region
                        next_indices_extend = next_label.indices[booleans.is_region_extensible[next_label.indices]]
                        extended_next_indices = set(next_indices) | set(np.where(c.satisfy(
                            [wprop.COUNTRY + ".inv", wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY + ".inv"],
                            next_indices_extend))[0])
                        supported_regions = set()
                        prev_2_region = {}
                        for k, idx in enumerate(prev_label.indices):
                            if curr_selection_mask[k]:
                                matches = c.satisfy([wprop.COUNTRY + ".inv", wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY + ".inv"],
                                                    [idx])
                                overlaps = [idx for idx in extended_next_indices if matches[idx]]
                                if len(overlaps) == 0:
                                    curr_selection_mask[k] = False
                                else:
                                    prev_2_region[idx] = overlaps
                                for reg in overlaps:
                                    supported_regions.add(reg)
                        # stop iterating if we found a matching place, place pattern, or we've run out of variations.
                        if len(supported_regions) > 0 or no_place_place_variations:
                            break
                        else:
                            # try a variant where we use (institution, town)
                            no_place_place_variations = True
                            next_condition_mask = alt_next_condition_mask
                            prev_condition_mask = alt_prev_condition_mask

                    if len(supported_regions) > 0:
                        next_selection_mask = []
                        next_2_region = {}
                        possible_next_indices_count = 0
                        for idx in next_label.indices:
                            if not next_condition_mask[idx]:
                                next_selection_mask.append(False)
                            else:
                                found_support = False
                                if idx in supported_regions:
                                    found_support = True
                                    next_2_region[idx] = [idx]
                                inv_state = c.satisfy(
                                    [wprop.COUNTRY + ".inv", wprop.LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY + ".inv"],
                                    [idx])
                                for reg in supported_regions:
                                    if inv_state[reg]:
                                        if idx not in next_2_region:
                                            next_2_region[idx] = [reg]
                                        else:
                                            next_2_region[idx].append(reg)
                                        found_support = True
                                        possible_next_indices_count += 1
                                possible_next_indices_count += found_support
                                next_selection_mask.append(found_support)
                        next_selection_mask = np.array(next_selection_mask)
                        expose_constraints = len(supported_regions) > 1 and possible_next_indices_count > 1
                        # allow cases of the form PLACE/TEAM, PLACE if you are inside a parenthesis.
                        if inside_parenthesis:
                            curr_selection_mask = np.logical_or(booleans.is_national_team[prev_label.indices], curr_selection_mask)
                        # switch prev and next choices to only be places
                        if prev_label.anchor_idx not in detected_places:
                            detected_places[prev_label.anchor_idx] = tstep - 1
                        supported_regions_list = list(supported_regions)
                        for l, start_loc, increment, selection_mask, is_prev in ((prev_label, tstep - 2, -1, curr_selection_mask, True),
                                                                                 (next_label, tstep + 2, 1, next_selection_mask, False)):
                            self._apply_filter_to_label(l, tokens[tstep - 1:tstep + 2], selection_mask, "place,place")
                            if is_prev and expose_constraints:
                                l.created_constraints = pad_arrays([prev_2_region[idx] if idx in prev_2_region else supported_regions_list for idx in l.indices], fill_value=CONSTRAINTS_CREATED_PADDING)
                            elif not is_prev and expose_constraints:
                                l.constraint_required = pad_arrays([next_2_region[idx] for idx in l.indices], fill_value=CONSTRAINTS_REQUIRED_PADDING)
                            self._propagate_filter(l, example_y, labels, start_loc, increment=increment)
                    t0_1 = time.time()
                else:
                    pass

    def first_occurence_of_anchor(self, anchor_idx, tstep, example_y):
        return tstep == 0 or example_y[tstep - 1][self.objective_column] is None or example_y[tstep - 1][self.objective_column].anchor_idx != anchor_idx

    def linked_doc_to_example(self, doc, aida_means_trie, booleans, inception_date):
        doc_links = list(doc.links(wiki_trie=self._classifications.article2id,
                                   redirections=self._redirections,
                                   prefix=self.kwargs.get("prefix", "enwiki")))

        densify = self.kwargs.get("densify", False)
        if densify:
            t0 = time.time()
            new_doc_links = []
            full_densification = densifier("".join([link for link, _ in doc_links]), self.anchor_trie, self.trie_index2indices_values, self.trie_index2indices_counts, booleans.is_valid_dense)
            so_far = 0
            so_far_to_label = {}
            for link, label in full_densification:
                if label is not None:
                    so_far_to_label[so_far] = label
                so_far += len(link)

            so_far = 0
            for link, label in doc_links:
                if label is None:
                    # try to densify using the anchor_trie
                    new_doc_links.extend(densifier(link, self.anchor_trie, self.trie_index2indices_values, self.trie_index2indices_counts, booleans.is_valid_dense))
                else:
                    if so_far in so_far_to_label:
                        so_far_to_label[so_far].label = label
                        new_doc_links.append((link, so_far_to_label[so_far]))
                    else:
                        new_doc_links.append((link, label))
                so_far += len(link)
            doc_links = new_doc_links
            self.densifier_t += time.time() - t0
        t0 = time.time()
        tokens, labels, anchor_ids = realign_string_labels_with_trie(doc_links, self.anchor_trie, self._classifications.article2id,
                                                                     trie_index2indices_values=self.trie_index2indices_values,
                                                                     only_tag_ending=self.kwargs.get("only_tag_ending", False),
                                                                     merge_triggers=self.kwargs.get("merge_triggers", False))
        example_x = [token.rstrip() for token in tokens]
        example_y = [[None for k in range(len(self.objective_names))] for _ in labels]
        doc_dates = set([tok for tok in example_x if tok.isdigit() and len(tok) == 4])
        for tstep, label in enumerate(labels):
            anchor_idx = anchor_ids[tstep]
            if label is not None:
                if anchor_idx is None or booleans.is_list_disambiguation[label]:
                    continue
                if isinstance(anchor_idx, UnambiguousAnchor):
                    anchor_values = anchor_idx.values
                    anchor_values = anchor_idx.counts
                    entity_counts = np.ones_like(anchor_values)
                else:
                    candidate_metadata = None
                    if isinstance(anchor_idx, MergedTrigger):
                        anchor_idx = anchor_idx.new
                    first_occurence_of_anchor = self.first_occurence_of_anchor(anchor_idx, tstep, example_y)
                    anchor_values = self.trie_index2indices_values[anchor_idx]
                    anchor_counts = self.trie_index2indices_counts[anchor_idx]
                    entity_counts = self.index2incoming_count[np.minimum(anchor_values, len(self.index2incoming_count) - 1)]
                    pos_counts = self.pos_trie_index2indices_counts[anchor_idx] if self.pos_trie_index2indices_counts is not None else None
                    pos_entity_counts = self.pos_index2incoming_count[np.minimum(self.trie_index2indices_values[anchor_idx], len(self.index2incoming_count) - 1)] if self.pos_trie_index2indices_counts is not None else None

                    if example_x[tstep] in self._classifications.type_collection.state_abbreviations():
                        full_anchor = self._classifications.type_collection.state_abbreviations()[example_x[tstep]]
                        new_anchor_values = self.trie_index2indices_values[self.anchor_trie[full_anchor.lower()]]
                        new_anchor_counts = self.trie_index2indices_counts[self.anchor_trie[full_anchor.lower()]]
                        keep_new = booleans.is_us_state[new_anchor_values]
                        new_anchor_values = new_anchor_values[keep_new]
                        new_anchor_counts = new_anchor_counts[keep_new]
                        if len(new_anchor_values) > 0:
                            top_abbrev_state = new_anchor_counts.argmax()
                            new_anchor_values = new_anchor_values[top_abbrev_state:top_abbrev_state+1]
                            new_anchor_counts = new_anchor_counts[top_abbrev_state:top_abbrev_state+1]
                            # remove previous states
                            prev_select_mask = np.logical_not(booleans.is_us_state[anchor_values])
                            anchor_values = np.concatenate([anchor_values[prev_select_mask], new_anchor_values])
                            anchor_counts = np.concatenate([anchor_counts[prev_select_mask], new_anchor_counts])
                            entity_counts = self.index2incoming_count[np.minimum(anchor_values, len(self.index2incoming_count) - 1)]
                            pos_counts = np.concatenate([pos_counts[prev_select_mask], new_anchor_counts]) if self.pos_trie_index2indices_counts is not None else None
                            pos_entity_counts = np.concatenate([pos_entity_counts[prev_select_mask], self.pos_index2incoming_count[np.minimum(new_anchor_values, len(self.index2incoming_count) - 1)] ]) if self.pos_trie_index2indices_counts is not None else None
                            candidate_metadata = np.concatenate([candidate_metadata, empty_metadata(len(new_anchor_counts))]) if candidate_metadata is not None else None
                            anchor_values, anchor_counts, entity_counts, pos_counts, pos_entity_counts, candidate_metadata = _reduce_according_to_index(
                                anchor_values, anchor_counts, entity_counts, pos_counts, pos_entity_counts, candidate_metadata
                            )

                    if aida_means_trie is not None:
                        aida_means_values = set(aida_means_trie[anchor_idx])
                        if len(aida_means_values) > 0:
                            anchor_idxes = [k for k, idx in enumerate(anchor_values) if idx in aida_means_values]
                            # if label not in aida_means_values and label in anchor_values:
                            #     print("Reduced candidates from {} to {}".format(len(anchor_values), len(anchor_idxes)))
                            #     print("Label is gone ", anchor_idx, example_x[tstep])
                            anchor_values = anchor_values[anchor_idxes]
                            anchor_counts = anchor_counts[anchor_idxes]
                            entity_counts = entity_counts[anchor_idxes]
                            if candidate_metadata is not None:
                                candidate_metadata = candidate_metadata[anchor_idxes]
                            if pos_counts is not None:
                                pos_counts = pos_counts[anchor_idxes]
                            if pos_entity_counts is not None:
                                pos_entity_counts = pos_entity_counts[anchor_idxes]
                        else:
                            pass

                    selection_mask = np.logical_not(booleans.is_list_disambiguation[anchor_values])
                    remainder_size = selection_mask.sum()
                    original_anchor_values = anchor_values
                    anchor_values = anchor_values[selection_mask]
                    anchor_counts = anchor_counts[selection_mask]
                    entity_counts = entity_counts[selection_mask]
                    if candidate_metadata is not None:
                        candidate_metadata = candidate_metadata[selection_mask]
                    if pos_counts is not None:
                        pos_counts = pos_counts[selection_mask]
                    if pos_entity_counts is not None:
                        pos_entity_counts = pos_entity_counts[selection_mask]

                    
                    for condition, disallowed_group in [(booleans.is_town, booleans.is_venue)]:
                        if np.any(condition[anchor_values]):
                            anchor_values_are_disallowed = disallowed_group[anchor_values]
                            selection_mask = np.logical_not(anchor_values_are_disallowed)
                            selection_mask[anchor_counts.argmax()] = True
                            anchor_values = anchor_values[selection_mask]
                            anchor_counts = anchor_counts[selection_mask]
                            entity_counts = entity_counts[selection_mask]
                            if candidate_metadata is not None:
                                candidate_metadata = candidate_metadata[selection_mask]
                            if pos_counts is not None:
                                pos_counts = pos_counts[selection_mask]
                            if pos_entity_counts is not None:
                                pos_entity_counts = pos_entity_counts[selection_mask]
                if label == -1:
                    label = None
                
                ex = ScenarioExample(
                    label,
                    anchor_values,
                    anchor_counts,
                    entity_counts=entity_counts,
                    original_label=None,
                    uniform_sample=self.uniform_sample,
                    index2incoming_cumprob=self.index2incoming_cumprob,
                    index2incoming_count=self.index2incoming_count,
                    anchor_idx=anchor_idx,
                    pos_counts=pos_counts,
                    pos_entity_counts=pos_entity_counts,
                    created_constraints=None,
                    constraint_required=None,
                    candidate_metadata=candidate_metadata
                )
                if self.filter_examples is None or self.filter_examples(ex):
                    example_y[tstep][self.objective_column] = ex
        
        self.other_t += time.time() - t0
        # Finds cases of RANK, RANKED ENTITY (e.g. 'Captain Kirk') and tries to ensure both are consistent
        # filters results to only contain consistent pairs.
        t0 = time.time()
        c = self._classifications.type_collection
        for tstep, label in enumerate(labels):
            if (label is not None and
                tstep > 0 and
                example_y[tstep - 1][self.objective_column] is not None and
                example_y[tstep][self.objective_column] is not None and
                len(example_y[tstep - 1][self.objective_column].indices) > 0 and
                len(example_y[tstep][self.objective_column].indices) > 0 and
                example_y[tstep][self.objective_column].anchor_idx != example_y[tstep - 1][self.objective_column].anchor_idx):
                # check if previous label could be a rank
                prev_label = example_y[tstep - 1][self.objective_column]
                prev_selection_mask = booleans.is_rank[prev_label.indices]
                prev_pick_could_be_rank = np.any(prev_selection_mask)
                if not prev_pick_could_be_rank:
                    continue
                
                # check if next label could be a person with that rank
                next_label = example_y[tstep][self.objective_column]
                curr_selection_mask = booleans.is_ranked[next_label.indices]
                current_pick_could_be_ranked = np.any(curr_selection_mask)
                if not current_pick_could_be_ranked:
                    continue
                prev_indices = set(prev_label.indices[prev_selection_mask])
                supported_ranks = set()
                prev_2_rank = {}
                for k, idx in enumerate(next_label.indices):
                    if curr_selection_mask[k]:
                        matches = c.relation(wprop.MILITARY_RANK)[idx]
                        overlaps = [match for match in matches if match in prev_indices]
                        if len(overlaps) == 0:
                            curr_selection_mask[k] = False
                        for reg in overlaps:
                            supported_ranks.add(reg)
                if len(supported_ranks) > 0:
                    for idx, val in enumerate(prev_label.indices):
                        if prev_selection_mask[idx] and val not in supported_ranks:
                            prev_selection_mask[idx] = False
                    self._apply_filter_to_label(next_label, tokens[tstep], curr_selection_mask, "rank")
                    self._propagate_filter(next_label, example_y, labels, tstep + 1, increment=1)
                    self._apply_filter_to_label(prev_label, tokens[tstep - 1], prev_selection_mask, "rank")
                    self._propagate_filter(prev_label, example_y, labels, tstep - 2, increment=-1)
        self.rank_person_t += time.time() - t0

        t0 = time.time()
        possible_list_links = []
        possible_list_endings = []
        for tstep, label in enumerate(labels):
            if (tstep > 0 and
                tstep + 1 < len(labels) and
                label is None and
                tokens[tstep].strip() == "," and
                example_y[tstep - 1][self.objective_column] is not None and
                example_y[tstep + 1][self.objective_column] is not None):
                possible_list_links.append((tstep - 1, tstep + 1))
            elif (tstep > 0 and
                    tstep + 1 < len(labels) and
                    label is None and
                    tokens[tstep].strip() == "and" and
                    example_y[tstep - 1][self.objective_column] is not None and
                    example_y[tstep + 1][self.objective_column] is not None):
                    possible_list_endings.append((tstep - 1, tstep + 1))
            elif (tstep > 0 and
                    tstep + 2 < len(labels) and
                    label is None and
                    tokens[tstep].strip() == "," and
                    tokens[tstep + 1].strip() == "and" and
                    example_y[tstep - 1][self.objective_column] is not None and
                    example_y[tstep + 2][self.objective_column] is not None):
                    possible_list_endings.append((tstep - 1, tstep + 2))
        
        lists = {}
        link2list = {}
        for start, end in possible_list_endings:
            lists[end] = []
            end_anchor_idx = example_y[end][self.objective_column].anchor_idx
            end_end = end
            while end_end < len(example_y) and example_y[end_end][self.objective_column] is not None and example_y[end_end][self.objective_column].anchor_idx == end_anchor_idx:
                lists[end].append(end_end)
                end_end += 1
            lists[end] = lists[end][::-1]
            link2list[end] = end
            start_anchor_idx = example_y[start][self.objective_column].anchor_idx
            while start >= 0 and example_y[start][self.objective_column] is not None and example_y[start][self.objective_column].anchor_idx == start_anchor_idx:
                link2list[start] = end
                lists[end].append(start)
                start -= 1
        for start, end in reversed(possible_list_links):
            if end in link2list:
                start_anchor_idx = example_y[start][self.objective_column].anchor_idx
                while start >= 0 and example_y[start][self.objective_column] is not None and example_y[start][self.objective_column].anchor_idx == start_anchor_idx:
                    lists[link2list[end]].append(start)
                    link2list[start] = link2list[end]
                    start -= 1
        
        conditionals = [booleans.is_town_not_airport, booleans.is_region, booleans.is_county, booleans.is_institution, booleans.is_national_team, booleans.is_airport]
        for list_ending, values in lists.items():
            is_consistent_list = [all(np.any(condition[example_y[v][self.objective_column].indices[example_y[v][self.objective_column].counts.argsort()[::-1][:2]]]) for v in values)
                                    for condition in conditionals]
            all_could_be = [all(np.any(condition[example_y[v][self.objective_column].indices]) for v in values)
                            for condition in conditionals]
            if any(all_could_be):
                list_size = len(set([example_y[v][self.objective_column].anchor_idx for v in values]))
                if detect_lists_inconsistent_team_airport and not any(is_consistent_list) and list_size < 3:
                    pass
                else:
                    selection_mask = logical_ors([condition for condition, allowed in zip(conditionals, all_could_be) if allowed])
                    for step, l in zip(values, [example_y[v][self.objective_column] for v in values]):
                        self._apply_filter_to_label(l, tokens[step], selection_mask[l.indices], "list")
                    
                for l in [example_y[v][self.objective_column] for v in values[:-1]]:
                    l.init_metadata()
                    l.candidate_metadata[:, METADATA_LIST_DIM] = METADATA_INSIDE_LIST
        self.list_t += time.time() - t0
        return example_x, example_y

    def __init__(self, path, x_column, y_columns, objective_names,
                 tries, classifications, ignore_value, kwargs, corpus_loader, randomize=False, rng=None):
        self.scenario_init(path=path, x_column=x_column, y_columns=y_columns, objective_names=objective_names,
                           tries=tries, classifications=classifications,
                           ignore_value=ignore_value, kwargs=kwargs, randomize=randomize, rng=rng)
        self.anchor_trie = tries.get_anchor_trie(self.y_column["language_path"])
        self._redirections = tries.get_redirections(self.kwargs.get("prefix", "enwiki").replace("wiki", ""))
        self._corpus_loader = corpus_loader
        if self.kwargs.get("cache_corpus_loader_args", False):
            m = hashlib.sha1()
            m.update(abspath(self._classifications.wikidata_path).encode("utf-8"))
            nested_hash_vals(m, self.kwargs.get("standard_dataset_loader"))
            m = m.hexdigest()[:12]
        else:
            m = hashlib.sha1(abspath(self._classifications.wikidata_path).encode("utf-8")).hexdigest()[:12]
        cached_path = path + m + "prelink.pkl"
        if exists(cached_path):
            print("Loading cached version of corpus from \"{}\"".format(cached_path))
            with open(cached_path, "rb") as fin:
                corpus = pickle.load(fin)
        else:
            corpus = corpus_loader(path=path,
                                   data_dir=self._classifications.wikidata_path,
                                   name2index=self._classifications.name2index,
                                   article2id=self._classifications.article2id,
                                   redirections=self._redirections)
            with open(cached_path, "wb") as fout:
                pickle.dump(corpus, fout)
        self.x = []
        self.y = []
        self.xorder = None
        c = classifications.type_collection
        if "Q4167410" in c.name2index and "Q13406463" in c.name2index:
            disambiguation_page = c.name2index["Q4167410"]
            list_page = c.name2index["Q13406463"]
            history_page = c.name2index["Q17524420"]
            history_page2 = c.name2index["Q309"]
            stateless = c.name2index["Q1151616"]
            ghost_town = c.name2index["Q74047"]
            all_list_page_disambig_pages = np.where(c.satisfy([wprop.SUBCLASS_OF], [disambiguation_page, list_page, history_page, history_page2, stateless, ghost_town]))[0]
            booleans = Namespace()
            booleans.is_list_disambiguation = c.satisfy([wprop.INSTANCE_OF], all_list_page_disambig_pages)
            # child of sports venue.
            booleans.is_venue = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [c.name2index["Q1076486"]])
            booleans.is_town = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [
                # city
                c.name2index["Q515"],
                # town
                c.name2index["Q3957"],
                # consistuency
                c.name2index["Q192611"],
                # Arad thing
                c.name2index["Q34843301"],
                # commune
                c.name2index["Q484170"],
                # Municipality of Austria
                c.name2index["Q667509"],
                # airport
                c.name2index["Q1248784"],
                # censsu
                c.name2index["Q498162"],
                # human settlement
                c.name2index["Q486972"]
                ])
            booleans.is_town_not_airport = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [
                # city
                c.name2index["Q515"],
                # town
                c.name2index["Q3957"],
                # consistuency
                c.name2index["Q192611"],
                # Arad thing
                c.name2index["Q34843301"],
                # commune
                c.name2index["Q484170"],
                # Municipality of Austria
                c.name2index["Q667509"],
                # censsu
                c.name2index["Q498162"],
                # human settlement
                c.name2index["Q486972"]
                ])
            booleans.is_town = np.logical_and(
                booleans.is_town,
                np.logical_not(
                    c.satisfy(
                        [wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                        [
                           # must not be a country
                           c.name2index["Q6256"],
                           # must not be a continent
                           c.name2index["Q5107"]
                        ])))
            booleans.is_town_not_airport = np.logical_and(
                booleans.is_town_not_airport,
                np.logical_not(
                    c.satisfy(
                        [wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                        [
                           # must not be a country
                           c.name2index["Q6256"],
                           # must not be a continent
                           c.name2index["Q5107"]
                        ])))
            booleans.is_institution = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                                                # institution
                                                [c.name2index["Q178706"]])
            booleans.is_county = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                                           # county
                                           [c.name2index["Q28575"]])
            booleans.is_town_or_county = np.logical_or(booleans.is_county, booleans.is_town)
            booleans.is_country = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                                            # country or nation
                                            [c.name2index["Q6256"], c.name2index["Q6266"], c.name2index["Q22890"]])
            booleans.is_national_team = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                                         # sports team
                                         [c.name2index["Q12973014"], c.name2index["Q476028"]])
            booleans.is_airport = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                                   # sports team
                                   [c.name2index["Q1248784"]])
            booleans.is_region = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [
                # sovereign state
                c.name2index["Q3624078"],
                # country
                c.name2index["Q6256"],
                # consistuency
                c.name2index["Q192611"],
                # US State
                c.name2index["Q35657"],
                # canadian province
                c.name2index["Q11828004"],
                # island
                c.name2index["Q23442"]
                ])
            booleans.is_valid_dense = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                [
                    # country
                    c.name2index["Q6256"],
                    # consistuency
                    c.name2index["Q192611"],
                    # US State
                    c.name2index["Q35657"],
                    # canadian province
                    c.name2index["Q11828004"],
                    # island
                    c.name2index["Q23442"],
                    # city
                    c.name2index["Q515"],
                    # town
                    c.name2index["Q3957"],
                    # consistuency
                    c.name2index["Q192611"],
                    # Arad thing
                    c.name2index["Q34843301"],
                    # commune
                    c.name2index["Q484170"],
                    # Municipality of Austria
                    c.name2index["Q667509"],
                    # airport
                    c.name2index["Q1248784"],
                    # public building
                    c.name2index["Q294422"],
                    # sport
                    c.name2index["Q31629"], c.name2index["Q349"],
                    # human
                    c.name2index["Q5"],
                    # taxon
                    c.name2index["Q16521"],
                    # populated place
                    c.name2index["Q486972"],
                    c.name2index["Q43229"],
                    c.name2index["Q431289"],
                    c.name2index["Q271669"],
                    c.name2index["Q16510064"],
                    c.name2index["Q4989906"],
                    c.name2index["Q12737077"],
                    c.name2index["Q211503"],
                    c.name2index["Q12973014"],
                    c.name2index["Q215380"],
                    c.name2index["Q43229"],
                ])
            booleans.is_region_extensible = logical_negate(c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [
                # island
                c.name2index["Q23442"]
                ]), [c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [
                               # inland island
                               c.name2index["Q202199"]])])
            booleans.is_rank = c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [c.name2index["Q56019"]])
            booleans.is_ranked = c.relation(wprop.MILITARY_RANK).edges() > 0
            # US States or territories (e.g. Washington DC)
            booleans.is_us_state = c.satisfy([wprop.INSTANCE_OF], [c.name2index["Q35657"]])
            if self.kwargs.get("at_in_team_market_language", True):
                booleans.is_at_in = logical_ors([booleans.is_town, booleans.is_region, booleans.is_institution, booleans.is_county, booleans.is_venue, booleans.is_national_team,
                    # geographical region, market, business, or language
                    c.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF],
                              [c.name2index["Q82794"], c.name2index["Q37654"], c.name2index["Q34770"], c.name2index["Q4830453"]]),
                ])
            else:
                booleans.is_at_in = logical_ors([booleans.is_town, booleans.is_region, booleans.is_institution, booleans.is_county, booleans.is_venue])
            inception_date = c.attribute(wprop.INCEPTION)
        else:
            booleans = Namespace(is_list_disambiguation=np.zeros(len(c.ids), dtype=np.bool))
            booleans.set_default(booleans.is_list_disambiguation)
            inception_date = SparseAttribute(booleans.is_list_disambiguation, booleans.is_list_disambiguation)

        aida_means_trie = OffsetArray.load(self.kwargs["aida_means"]) if self.kwargs.get("aida_means", None) is not None else None
        example_filter = self.kwargs.get("example_filter")
        total_labels = 0
        for doc in get_progress_bar("linked_doc_to_example", item="examples")(corpus):
            if example_filter is None or doc.matches_filter(example_filter):
                example_x, example_y = self.linked_doc_to_example(
                    doc=doc,
                    aida_means_trie=aida_means_trie,
                    booleans=booleans,
                    inception_date=inception_date)
                self.x.append(example_x)
                self.y.append(example_y)
                for w in example_y:
                    if len(w) > 0 and w[0] is not None:
                        total_labels += 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.randomize and (index == 0 or self.xorder is None):
            self.xorder = np.arange(len(self.x))
            self.rng.shuffle(self.xorder)
        if self.randomize:
            return self.x[self.xorder[index]], self.y[self.xorder[index]]
        else:
            return self.x[index], self.y[index]


def must_be_ambiguous(example):
    return len(example.indices) > 1


class CombinedDataset(object):
    _which_dataset = None
    _dataset_counters = None

    def set_rng(self, rng):
        self.rng = rng
        for dataset in self.datasets:
            dataset.rng = rng

    def set_randomize(self, randomize):
        self.randomize = randomize
        for dataset in self.datasets:
            dataset.set_randomize(randomize)

    def set_ignore_y(self, ignore):
        for dataset in self.datasets:
            if hasattr(dataset, "set_ignore_y"):
                dataset.set_ignore_y(ignore)

    def close(self):
        for dataset in self.datasets:
            dataset.close()

    def get_word_vocab(self):
        counts = None
        for dataset in self.datasets:
            if dataset.ignore_hash():
                pass
            else:
                new_counts = dataset.get_word_vocab()
                if counts is None:
                    counts = new_counts
                else:
                    counts += new_counts
        return counts

    def hash(self, hasher):
        for dataset in self.datasets:
            if dataset.ignore_hash():
                pass
            else:
                dataset.hash(hasher)

    def _build_which_dataset(self):
        self._which_dataset = np.empty(self.length, dtype=np.int16)
        self._dataset_counters = np.zeros(len(self.datasets), dtype=np.int64)
        offset = 0
        for index, dataset in enumerate(self.datasets):
            # ensure each dataset is seen as much as its content
            # says:
            self._which_dataset[offset:offset + len(dataset)] = index
            offset += len(dataset)

    def __getitem__(self, index):
        if index == 0:
            if self.randomize:
                # visit datasets in random orders:
                self.rng.shuffle(self._which_dataset)
            self._dataset_counters[:] = 0
        which = self._which_dataset[index]
        idx = self._dataset_counters[which]
        self._dataset_counters[which] += 1
        return self.datasets[which][idx]

    def __init__(self, datasets, rng=None, randomize=False):
        self.datasets = datasets
        if rng is None:
            rng = np.random.RandomState(0)
        self.set_rng(rng)
        self.set_randomize(randomize)
        self.length = sum(len(dataset) for dataset in datasets)
        self._build_which_dataset()

    def __len__(self):
        return self.length


class Oversample(object):
    def __init__(self, dataset, oversample, rng=None, randomize=False):
        self.oversample = oversample
        self._dataset = dataset
        self.internal_mapping = np.arange(len(self._dataset))
        if rng is None:
            rng = np.random.RandomState(0)
        self.set_rng(rng)
        self.set_randomize(randomize)
        assert isinstance(oversample, int) and oversample >= 1

    def close(self):
        self._dataset.close()

    def __getitem__(self, index):
        renorm_index = index % len(self._dataset)
        if renorm_index == 0:
            if self.randomize:
                # visit datasets in random orders:
                self.rng.shuffle(self.internal_mapping)
        idx = self.internal_mapping[renorm_index]
        return self._dataset[idx]

    def __len__(self):
        return self.oversample * len(self._dataset)

    def set_rng(self, rng):
        self._dataset.set_rng(rng)

    def ignore_hash(self):
        return self._dataset.ignore_hash()

    def set_randomize(self, randomize):
        self.randomize = True
        return self._dataset.set_randomize(randomize)

    def set_ignore_y(self, ignore):
        return self._dataset.set_ignore_y(ignore)

    def get_word_vocab(self):
        return self._dataset.get_word_vocab()

    def reset_cache(self):
        return self._dataset.reset_cache()

    def hash(self, hasher):
        return self._dataset.hash(hasher)
