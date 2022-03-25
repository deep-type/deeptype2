import numpy as np
from . import wikidata_properties as wprop

def redo_using_parenthesis(ex, collection, is_country, verbose=True):
    inside_paren = False
    inside_colon = False
    num_paren = " ".join([el["text"] for el in ex]).count("(")
    paren_els = []
    current_paren = []
    for t, el in enumerate(ex):
        for c in el["text"]:
            if c == ":":
                inside_colon = True
            elif c == "(":
                inside_paren = True
            elif c == ")":
                if len(current_paren) > 0:
                    paren_els.append(current_paren)
                    current_paren = []
                inside_paren = False
        
        if inside_paren and "correct" in el and inside_colon:
            current_paren.append(el)

    for current_paren in paren_els:
        if len(current_paren) > 1:
            # only consider cases where parentheses only concern one item.
            continue
        for el in current_paren:
            # take highest picked model option under country contraint:
            wiki_default = [el["candidates"][i]["prob"] if is_country[collection.name2index[el["candidates"][i]["id"]]] else -1 for i in range(len(el["candidates"]))]
            if len(wiki_default) == 0:
                continue
            default_pick = np.argmax(wiki_default)
            if is_country[collection.name2index[el["candidates"][default_pick]["id"]]] and num_paren > 1:
                if not el["candidates"][default_pick]["label"]:
                    el["correct"] = False
                else:
                    el["correct"] = True
                for cand in el["candidates"]:
                    cand["predicted"] = False
                el["candidates"][default_pick]["predicted"] = True
    return ex


def redo_looking_for_dates(ex, collection, is_event, verbose=True):
    tokens = set([w for w in " ".join([el["text"] for el in ex]).split() if w.isdigit() and len(w) == 4])
    for t, el in enumerate(ex):
        if "correct" in el:
            predicted_cand_idx = None
            for idx, cand in enumerate(el["candidates"]):
                if cand["predicted"]:
                    predicted_cand_idx = idx
                    break
            if predicted_cand_idx is not None and is_event[collection.name2index[el["candidates"][predicted_cand_idx]["id"]]]:
                # find date support
                if "prev_correct_date" not in el:
                    el["prev_correct_date"] = el["correct"]
                
                filtered = []
                for cand in el["candidates"]:
                    dates = [w for w in cand["name"].split() if len(w) == 4 and w.isdigit()]
                    filtered.append(len(dates) == 0 or dates[0] in tokens)
                    
                if verbose:
                    print("---")
                    for cand, filt in zip(el["candidates"], filtered):
                        if filt:
                            print(cand["name"])
                    print("---")
                # wiki_default = [el["candidates"][i]["float_inputs"][9][0] for i in range(len(el["candidates"]))]
                # take highest picked model option under country contraint:
                wiki_default = [el["candidates"][i]["prob"] if filtered[i] else -1 for i in range(len(el["candidates"]))]
                if len(wiki_default) == 0:
                    continue
                default_pick = np.argmax(wiki_default)
                
                
                if not el["candidates"][default_pick]["label"]:
                    el["correct"] = False
                else:
                    for cand_idx, cand in enumerate(el["candidates"]):
                        if cand["predicted"]:
                            if verbose and cand_idx != default_pick:
                                print("Now correct")
                    el["correct"] = True
            else:
                if "prev_correct_date" in el:
                    el["correct"] = el["prev_correct_date"]
        elif "prev_correct_date" in el:
            el["correct"] = el["prev_correct_date"]
    return ex


NATIONALITY_WORDS = set(("english", "american", "french", "spanish", "german", "portuguese", "greek",
                         "italian", "brazilian", "israeli", "lebanese", "swedish", "turkish", "kurdish",
                         "japanese", "chinese", "british", "scottish", "welsh", "estonian",
                         "australian", "kurdish", "korean", "koreans", "canadian", "czech", "czechs", "britons"))


def redo_nationalities(ex, collection, is_country, wiki_prob_feature_index, verbose=True):
    for t, el in enumerate(ex):
        if "correct" in el and el["text"].rstrip().lower() in NATIONALITY_WORDS and len(el["candidates"]) > 0:
            # take default definition for this word
            wiki_default = [el["candidates"][i]["float_inputs"][wiki_prob_feature_index][0] if is_country[collection.name2index[el["candidates"][i]["id"]]] else -1 for i in range(len(el["candidates"]))]
            default_pick = np.argmax(wiki_default)
            if "prev_correct_nationality" not in el:
                el["prev_correct_nationality"] = el["correct"]
            if not el["candidates"][default_pick]["label"]:
                el["correct"] = False
            else:
                for cand_idx, cand in enumerate(el["candidates"]):
                    if cand["predicted"]:
                        if verbose and cand_idx != default_pick:
                            print("Now correct")
                el["correct"] = True
        else:
            if "prev_correct_nationality" in el:
                el["correct"] = el["prev_correct_nationality"]
    return ex


class PostProcesser(object):
    def __init__(self, collection, wiki_prob_feature_index):
        self.collection = collection
        self.wiki_prob_feature_index = wiki_prob_feature_index
        self.is_country = collection.satisfy([wprop.INSTANCE_OF], [
            collection.name2index["Q6256"],
            collection.name2index["Q22890"],
            collection.name2index["Q6266"],
            collection.name2index["Q1024900"],
            collection.name2index["Q3336843"],
            collection.name2index["Q21"]])
        self.is_event = collection.satisfy([wprop.INSTANCE_OF, wprop.SUBCLASS_OF], [
            collection.name2index["Q476300"],
            collection.name2index["Q20852192"]])

    def postprocess(self, ex, verbose=False):
        ex = redo_using_parenthesis(ex, collection=self.collection, is_country=self.is_country, verbose=verbose)
        ex = redo_looking_for_dates(ex, collection=self.collection, is_event=self.is_event, verbose=verbose)
        if self.wiki_prob_feature_index is not None:
            ex = redo_nationalities(ex, collection=self.collection, is_country=self.is_country, verbose=verbose, wiki_prob_feature_index=self.wiki_prob_feature_index)
        return ex


def setup_postprocess(collection, wiki_prob_feature_index):
    return PostProcesser(collection, wiki_prob_feature_index)
