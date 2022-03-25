import json
from .wikipedia import match_wikipedia_to_wikidata


class TacDoc(object):
    __slots__ = ["_doc", "_queries"]

    def __init__(self, doc, queries):
        self._doc = doc
        self._queries = queries

    def matches_filter(self, text):
        return text in self._doc

    def links(self, wiki_trie, redirections, prefix):
        queries = [{"text": q["text"].rstrip(), "entity": q["entity"]} for q in self._queries]
        positions = [(query, self._doc.find(query["text"]))
                     for query in queries]
        positions.sort(key=lambda x: x[1])
        so_far = 0
        for query, pos in positions:
            if pos >= 0 and pos >= so_far:
                # print("[", query["text"], "]", pos)
                yield (self._doc[so_far:pos], None)
                dest_index = match_wikipedia_to_wikidata(
                    query["entity"],
                    wiki_trie,
                    redirections,
                    prefix
                )
                if dest_index is None:
                    print(query["text"], query["entity"])
                yield (query["text"], dest_index)
                so_far = pos + len(query["text"])
        if so_far != len(self._doc):
            yield (self._doc[so_far:], None)


def load_tac_docs(path, data_dir, name2index, article2id=None, redirections=None):
    with open(path, "rt") as fin:
        docs = json.load(fin)
    return [TacDoc(doc["doc"], doc["queries"]) for doc in docs]
