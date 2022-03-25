import re

STOP_WORDS = {'a', 'an', 'in', 'the', 'of', 'it', 'from', 'with', 'this', 'that', 'they', 'he',
              'she', 'some', 'where', 'what', 'since', 'his', 'her', 'their', 'le', 'la', 'les', 'il',
              'elle', 'ce', 'ça', 'ci', 'ceux', 'ceci', 'cela', 'celle', 'se', 'cet', 'cette',
              'dans', 'avec', 'con', 'sans', 'pendant', 'durant', 'avant', 'après', 'puis', 'el', 'lo', 'la',
              'ese', 'esto', 'que', 'qui', 'quoi', 'dont', 'ou', 'où', 'si', 'este', 'esta', 'cual',
              'eso', 'ella', 'depuis', 'y', 'a', 'à', 'su', 'de', "des", 'du', 'los', 'las', 'un', 'une', 'una',
              'uno', 'para', 'asi', 'later', 'into', 'dentro', 'dedans', 'depuis', 'después', 'desde',
              'al', 'et', 'por', 'at', 'for', 'when', 'why', 'how', 'with', 'whether', 'if',
              'thus', 'then', 'and', 'but', 'on', 'during', 'while', 'as', 'within', 'was', 'is',
              'est', 'au', 'fait', 'font', 'va', 'vont', 'sur', 'en', 'pour', 'del', 'cuando',
              'cuan', 'do', 'does', 'until', 'sinon', 'encore', 'to', 'by', 'be', 'which',
              'have', 'not', 'were', 'has', 'also', 'its', 'isbn', 'pp.', "&amp;", "p.", 'ces', 'o'}


PREFIXED_STOP_WORDS = {
   "enwiki": {
      'a', 'an', 'in', 'the', 'of', 'it', 'from', 'with', 'this', 'that', 'they', 'he',
      'she', 'some', 'where', 'what', 'since', 'his', 'her', 'their', 'later'
      'thus', 'then', 'and', 'but', 'on', 'during', 'while', 'as', 'within', 'was', 'is',
      'do', 'does', 'until', 'to', 'by', 'be', 'which', 'have', 'not', 'were', 'has', 'also', 'its', 
      'pp.', "&amp;", "p.", 
   },
   "frwiki": {
      'elle', 'ce', 'ça', 'ci', 'ceux', 'ceci', 'cela', 'celle', 'se', 'cet', 'cette',
      'est', 'au', 'fait', 'font', 'va', 'vont', 'sur', 'en', 'pour', 'del', 'cuando',
      'dans', 'avec', 'con', 'sans', 'pendant', 'durant', 'avant', 'après', 'puis', 'el', 'lo', 'la',
      'depuis', 'à', 'de', 'du', 'un', 'une', 'dont', 'que', 'qui', 'quoi', 'ou', 'où', 
      'sinon', 'encore', 'ces',
      'pp.', "&amp;", "p.", 
   },
   "eswiki": {
      'ese', 'esto','si', 'este', 'esta', 'cual',
      'eso', 'ella', 'y', 'a', 'su', 'de', "des", 'los', 'las', 'un', 'une', 'una',
      'cuan', 'o',
      'pp.', "&amp;", "p.", 
   }
}


def starts_with_apostrophe_letter(word):
    return (
        word.startswith("l'") or
        word.startswith("L'") or
        word.startswith("d'") or
        word.startswith("D'") or
        word.startswith("j'") or
        word.startswith("J'") or
        word.startswith("t'") or
        word.startswith("T'")
    )

PUNCTUATION = {"'", ",", "-", "!", ".", "?", ":", "’"}
BAD_END_PUNCTUATION = {"'", ",", "-", "!", "?", ":", "’"}


def clean_up_trie_source(source, lowercase=True, prefix=None):
    source = source.rstrip().strip('()[]')
    if len(source) > 0 and (source[-1] in BAD_END_PUNCTUATION or source[0] in PUNCTUATION):
        return ""
    # remove l'
    if starts_with_apostrophe_letter(source):
        source = source[2:]
    if source.endswith("'s"):
        source = source[:-2]
    tokens = source.split()
    stop_words = STOP_WORDS if prefix is None else PREFIXED_STOP_WORDS[prefix]
    while len(tokens) > 0 and tokens[0].lower() in stop_words:
        tokens = tokens[1:]
    while len(tokens) > 0 and tokens[-1].lower() in stop_words:
        tokens = tokens[:-1]
    joined_tokens = " ".join(tokens)
    # trim to max 256 characters
    joined_tokens = joined_tokens[:256]
    if lowercase:
        return joined_tokens.lower()
    return joined_tokens



ORDINAL_ANCHOR = re.compile("^\d+(st|th|nd|rd|er|eme|ème|ère)$")
NUMBER_PUNCTUATION = re.compile("^\d+([\/\-,\.:;%]\d*)+$")


def anchor_is_ordinal(anchor):
    return ORDINAL_ANCHOR.match(anchor) is not None


def anchor_is_numbers_slashes(anchor):
    return NUMBER_PUNCTUATION.match(anchor) is not None


def acceptable_anchor(anchor, anchor_trie, blacklist=None):
    return (
        len(anchor) > 0 and
        not anchor.isdigit() and
        not anchor_is_ordinal(anchor) and
        not anchor_is_numbers_slashes(anchor) and
        anchor in anchor_trie and
        (blacklist is None or anchor not in blacklist)
    )
