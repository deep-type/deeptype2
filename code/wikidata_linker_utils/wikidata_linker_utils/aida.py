from .wikipedia import load_wikipedia_pageid_to_wikidata, match_wikipedia_to_wikidata

class AidaDoc(object):
    __slots__ = ["_links"]

    def __init__(self, links):
        self._links = links
    
    def matches_filter(self, text):
        for span, _ in self._links:
            if text.lower() in span.lower():
                return True
        return False

    def links(self, wiki_trie, redirections, prefix):
        for link in self._links:
            yield link

def load_aida_docs(path, data_dir, name2index, article2id=None, redirections=None,
                   wikipedia_sql_props=None, ignore=None):
    with open(path, "rt") as fin:
        lines = fin.read().splitlines()
    if wikipedia_sql_props is None:
        wikipedia_sql_props = load_wikipedia_pageid_to_wikidata(data_dir)
    docs = []
    doc = []
    doc_linestart = None
    text = ""
    inside = False
    for line in lines:
        if line.startswith("-DOCSTART-"):
            if len(text) > 0:
                doc.append((text, None))
                text = ""
            if len(doc) > 0:
                if len(doc) > 1 or (len(doc) == 1 and doc[0][1] is not None):
                    assert doc_linestart is not None
                    skip = False
                    if ignore is not None:
                        for key in ignore:
                            if key in doc_linestart:
                                skip = True
                    if not skip:
                        docs.append(AidaDoc(doc))
                doc = []
            doc_linestart = line
            inside = False
        elif len(line.strip()) == 0:
            # whitespace in the document, e.g. between sentences...
            # text += "\n "
            inside = False
        else:
            cols = line.split("\t")
            token = cols[0]
            if len(cols) > 5:
                if not inside or cols[1] == "B":
                    mention = cols[2]
                    # if article2id is None:
                    wiki_id = cols[5]
                    idx = wikipedia_sql_props.get(wiki_id, None)
                    dest_index = name2index.get(idx.upper(), None) if idx is not None else None
                    # code for doing lookup using wikipedia page name (breaks when names change):
                    # wikipedia_title = cols[4].replace("_", " ").replace("http://en.wikipedia.org/wiki/", "")
                    # dest_index = match_wikipedia_to_wikidata(
                    #     wikipedia_title,
                    #     article2id,
                    #     redirections,
                    #     "enwiki"
                    # )
                    if dest_index is not None:
                        if len(text) > 0:
                            doc.append((text, None))
                            text = ""
                        doc.append((mention + " ", dest_index))
                        inside = True
                    else:
                        text += mention + " "
            else:
                text += token + " "
                inside = False
    if len(text) > 0:
        doc.append((text, None))
        text = ""
    if len(doc) > 1 or (len(doc) == 1 and doc[0][1] is not None):
        assert doc_linestart is not None
        skip = False
        if ignore is not None:
            for key in ignore:
                if key in doc_linestart:
                    skip = True
        if not skip:
            docs.append(AidaDoc(doc))
    return docs


def load_aida_qid_docs(path, data_dir, name2index, article2id=None, redirections=None,
                       wikipedia_sql_props=None, ignore=None):
    with open(path, "rt") as fin:
        lines = fin.read().splitlines()
    if wikipedia_sql_props is None:
        wikipedia_sql_props = load_wikipedia_pageid_to_wikidata(data_dir)
    docs = []
    doc = []
    doc_linestart = None
    text = ""
    inside = False
    for line in lines:
        if line.startswith("-DOCSTART-"):
            if len(text) > 0:
                doc.append((text, None))
                text = ""
            if len(doc) > 0:
                if len(doc) > 1 or (len(doc) == 1 and doc[0][1] is not None):
                    assert doc_linestart is not None
                    skip = False
                    if ignore is not None:
                        for key in ignore:
                            if key in doc_linestart:
                                skip = True
                    if not skip:
                        docs.append(AidaDoc(doc))
                doc = []
            doc_linestart = line
            inside = False
        elif len(line.strip()) == 0:
            # whitespace in the document, e.g. between sentences...
            # text += "\n "
            inside = False
        else:
            cols = line.split("\t")
            token = cols[0]
            if len(cols) > 3:
                if not inside or cols[1] == "B":
                    mention = cols[2]
                    # if article2id is None:
                    idx = cols[3]
                    dest_index = name2index.get(idx.upper(), None) if idx is not None else None
                    # code for doing lookup using wikipedia page name (breaks when names change):
                    # wikipedia_title = cols[4].replace("_", " ").replace("http://en.wikipedia.org/wiki/", "")
                    # dest_index = match_wikipedia_to_wikidata(
                    #     wikipedia_title,
                    #     article2id,
                    #     redirections,
                    #     "enwiki"
                    # )
                    if dest_index is not None:
                        if len(text) > 0:
                            doc.append((text, None))
                            text = ""
                        doc.append((mention + " ", dest_index))
                        inside = True
                    else:
                        text += mention + " "
            else:
                text += token + " "
                inside = False
    if len(text) > 0:
        doc.append((text, None))
        text = ""
    if len(doc) > 1 or (len(doc) == 1 and doc[0][1] is not None):
        assert doc_linestart is not None
        skip = False
        if ignore is not None:
            for key in ignore:
                if key in doc_linestart:
                    skip = True
        if not skip:
            docs.append(AidaDoc(doc))
    return docs
