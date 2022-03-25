"""
Tool for producing human-readable error report, e.g. for doing paper's error-analysis.
"""
import argparse
import pickle, json, numpy as np
from wikidata_linker_utils.report import display_example, ex2html, find_batch_idx
from wikidata_linker_utils.training_config import Config
from wikidata_linker_utils.type_collection import TypeCollection
from os.path import join, exists, basename, dirname
from os import makedirs
from collections import Counter
from tqdm import tqdm


def _confusion2html(source, dest, count):
    join_dest = '<br/>'.join(dest)
    join_source = '<br/>'.join(source)
    out = (f"<div style='display: inline-block;padding-right: 10px'>{count}</div>"
           f"<div style='display: inline-block; padding: 5px;border-radius: 10px; background: rgba(108, 186, 243, 0.2)'>{join_source}</div>"
           f"<div style='display: inline-block; padding: 5px;border-radius: 10px; background: rgba(237, 44, 44, 0.2)'>{join_dest}</div>")
    return out


def _update_mistakes(el, ex, ex_id, types2mistakes, examples, example_ids):
    if "prediction_instance" in el:
        key = ((tuple(el["instance"]), tuple(el["prediction_instance"])),)
        types2mistakes.update(key)
        if key not in examples:
            examples[key] = [ex]
            example_ids[key] = {ex_id}
        elif ex_id not in example_ids[key]:
            examples[key].append(ex)
            example_ids[key].add(ex_id)


def matches_filter(report, i, vocab, model_name, example_filter):
    batch, batch_idx = find_batch_idx(report, i, model_name)
    if batch is None:
        return False
    seq_lens = batch['{}/Inputs/sequence_lengths:0'.format(model_name)]
    seq_len = seq_lens[batch_idx]
    words = vocab[batch['{}/Inputs/input_placeholders_0:0'.format(model_name)][:seq_len, batch_idx]]
    return example_filter in " ".join(words).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("model_dir")
    parser.add_argument("report")
    parser.add_argument("output")
    parser.add_argument("--num_examples", type=int, default=-1)
    parser.add_argument("--num_names_to_load", type=int, default=0)
    parser.add_argument("--cwd", type=str, default=None)
    parser.add_argument("--error_analysis", action="store_true", default=False)
    parser.add_argument("--compare_report", type=str, default=None)
    parser.add_argument("--example_filter", required=False, type=str, default=None)
    parser.add_argument("--only_results", action="store_true", default=False)
    parser.add_argument("--group_by_filter", type=str, default=None)
    args = parser.parse_args()
    conf = Config.load(args.config, cwd=args.cwd)
    with open(join(args.model_dir, "model.json"), "rt") as fin:
        model_props = json.load(fin)
    # usually this is the feature for words but could be wrong
    vocab = np.array(model_props["feature_index2words"][0])
    with open(args.report, "rb") as fin:
        report = pickle.load(fin)
    if args.compare_report is not None:
        with open(args.compare_report, "rb") as fin:
            alternative_report = pickle.load(fin)
    else:
        alternative_report = None
    # assuming there's only one objective of type scenario
    projections = [featurization["name"] for featurization in conf.objectives[0]["classifications"]]
    for interaction in conf.objectives[0].get("feature_interactions", []):
        projections.append(interaction.get("name", "_".join(interaction["features"])))

    projection_vocab = {}
    for name in projections:
        proj_path = join(conf.classification_path, name, "classes.txt")
        if exists(proj_path):
            with open(proj_path, "rt") as fin:
                projection_vocab[name] = np.array([l.strip() for l in fin.readlines()])
        else:
            projection_vocab[name] = None
    collection = TypeCollection(conf.wikidata_path, num_names_to_load=args.num_names_to_load)
    collection._web_get_name = False
    total_num_examples = sum([len(batch["unary_scores"]) for batch in report])
    if alternative_report is not None:
        total_num_alternative_examples = sum([len(batch["unary_scores"]) for batch in alternative_report])
        assert total_num_alternative_examples == total_num_examples, "unequal number of examples in alternative and main report."

    uppercase_input_index = -1
    for feat_idx, feat in enumerate(model_props["features"]):
        if feat["type"] == "uppercase":
            uppercase_input_index = feat_idx
            break

    wiki_prob_feature_index = None
    for idx, proj in enumerate(conf.objectives[0]["classifications"]):
        if proj.get("type") == "wikipedia_probs":
            wiki_prob_feature_index = idx
            break

    page = f"""<!DOCTYPE html>
    <html lang="en">
    <head><meta charset="utf-8">
    <title>Report for {basename(args.report)}</title>
    </head>
    <body>
    """
    types2mistakes, types2mistakes_alt = Counter(), Counter()
    example_ids, examples, examples_alt, example_ids_alt = {}, {}, {}, {}
    max_count = args.num_examples if args.num_examples > 0 else total_num_examples
    supervised = 0
    correct = 0
    alternative_fixes = 0
    alternative_breaks = 0
    display_kwargs = dict(vocab=vocab,
                          model_name=model_props["name"],
                          uppercase_input_index=uppercase_input_index,
                          projections=projections,
                          collection=collection,
                          projection_vocab=projection_vocab,
                          only_results=args.only_results,
                          group_by_filter=args.group_by_filter)
    example_filter = args.example_filter.lower() if args.example_filter is not None else None
    recall_at_k = {k: 0 for k in [2, 3, 4, 5]}
    for i in tqdm(range(max_count)):
        if example_filter is None or matches_filter(report, i, vocab=vocab, example_filter=example_filter, model_name=model_props["name"]):
            ex = display_example(report, example_idx=i, **display_kwargs)
            # ex = display_example(report, example_idx=i, **display_kwargs)
            if alternative_report is not None:
                alt_ex = display_example(alternative_report, example_idx=i, **display_kwargs)
                for el, alt_el in zip(ex, alt_ex):
                    # cases where main is incorrect, and alternative is correct.
                    if "correct" in el and el["supervised"] and not el.get("correct", True) and alt_el.get("correct", True):
                        _update_mistakes(el, ex, i, types2mistakes, examples, example_ids)
                        alternative_fixes += 1
                    # cases where main is correct, and alternative is incorrect.
                    elif "correct" in el and el["supervised"] and el.get("correct", True) and not alt_el.get("correct", True):
                        _update_mistakes(alt_el, alt_ex, i, types2mistakes_alt, examples_alt, example_ids_alt)
                        alternative_breaks += 1
            elif args.error_analysis:
                for el in ex:
                    if "correct" in el and el["supervised"] and not el.get("correct", True):
                        _update_mistakes(el, ex, i, types2mistakes, examples, example_ids)
                        supervised += 1
                        correct_answer_pos = -1
                        for idx, cand in enumerate(el["candidates"]):
                            if cand["label"]:
                                correct_answer_pos = idx
                                break
                        assert correct_answer_pos != -1
                        # 0-index switch to 1-index, e.g. if answer is in 5th position, we'll have noted
                        # correct_answer_pos=4, we now switch back to 5.
                        correct_answer_pos += 1
                        for k in recall_at_k.keys():
                            # take note of whether the right answer came up at or before k:
                            recall_at_k[k] += correct_answer_pos <= k
                    elif "correct" in el and el["supervised"] and el.get("correct", True):
                        correct += 1
                        supervised += 1
                        for k in recall_at_k.keys():
                            recall_at_k[k] += 1
            else:
                page += ex2html(ex).data
    max_num_examples = 10
    if alternative_report is not None:
        for title, t2m, exes in [("Cases where alternative fixes", types2mistakes, examples),
                                 ("Cases where alternative breaks", types2mistakes_alt, examples_alt)]:
            page += f"<h2>{title}</h2>\n"
            for (source, dest), count in t2m.most_common(max_count):
                page += _confusion2html([collection.get_name(collection.name2index[idx]).split(" - ")[0].split("(")[0].strip()
                                        if idx in collection.name2index else idx for idx in source],
                                        [collection.get_name(collection.name2index[idx]).split(" - ")[0].split("(")[0].strip()
                                        if idx in collection.name2index else idx for idx in dest], count)
                key = ((source, dest),)
                for ex in exes[key][:max_num_examples]:
                    page += ex2html(ex, highlight=key).data
        print("alternative_fixes", alternative_fixes)
        print("alternative_breaks", alternative_breaks)
    elif args.error_analysis:
        print("correct", correct)
        print("supervised", supervised)
        print("acc", (correct / supervised) if supervised > 0 else 0.0)
        for k, correct_at_k in sorted(recall_at_k.items()):
            print(f"r@{k}", (correct_at_k / supervised) if supervised > 0 else 0.0)

        for (source, dest), count in types2mistakes.most_common(max_count):
            page += _confusion2html([collection.get_name(collection.name2index[idx]).split(" - ")[0].split("(")[0].strip()
                                     if idx in collection.name2index else idx for idx in source],
                                    [collection.get_name(collection.name2index[idx]).split(" - ")[0].split("(")[0].strip()
                                     if idx in collection.name2index else idx for idx in dest], count)
            key = ((source, dest),)
            for ex in examples[key][:max_num_examples]:
                page += ex2html(ex, highlight=key).data
    page += "</html>"
    makedirs(dirname(args.output), exist_ok=True)
    with open(args.output, "wt") as fout:
        fout.write(page)


if __name__ == "__main__":
    main()