import numpy as np
import argparse
import time
import pickle
import ciseau
from os.path import join
from os import makedirs
from wikidata_linker_utils.batchifier import iter_batches_single_threaded
from wikidata_linker_utils.entrypoint import load_model, boolean_argument
from wikidata_linker_utils.progressbar import get_progress_bar
from wikidata_linker_utils.scoped_timer import scoped_timer_summarize
import wikidata_linker_utils.wikidata_properties as wprop


def _convert_to_fp16(value):
    newsize = 0.0
    oldsize = 0.0
    if hasattr(value, "dtype"):
        oldsize = value.nbytes
        if value.dtype in (np.float32, np.float16, np.float64):
            return value.astype(np.float16)
    elif isinstance(value, (list, tuple)):
        return type(value)([_convert_to_fp16(el) for el in value])
    elif isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = _convert_to_fp16(value[key])
    return value


def create_example(collection, dataset, text, word_meaning):
    from wikidata_linker_utils.dataset import ScenarioExample
    x = [w.rstrip() for w in ciseau.tokenize(text)]
    x_lower = [w.lower() for w in x]
    y = [[None] for _ in x]
    for word, meaning in word_meaning:
        idx = x_lower.index(word.lower().strip())
        label = collection.name2index[meaning]
        anchor_idx = dataset.anchor_trie[word.lower()]
        anchor_values = dataset.trie_index2indices_values[anchor_idx]
        anchor_counts = dataset.trie_index2indices_counts[anchor_idx]
        entity_counts = dataset.index2incoming_count[np.minimum(anchor_values, len(dataset.index2incoming_count) - 1)]
        y[idx][0] = ScenarioExample(
            label,
            anchor_values,
            anchor_counts,
            entity_counts=entity_counts,
            original_label=meaning,
            uniform_sample=dataset.uniform_sample,
            index2incoming_cumprob=dataset.index2incoming_cumprob,
            index2incoming_count=dataset.index2incoming_count,
            anchor_idx=anchor_idx,
            pos_counts=None,
            pos_entity_counts=None
        )
    dataset.x = [x]
    dataset.y = [y]


def produce_report(session, model, dataset, blank_triggers, beam_width, batch_size, max_length, save_attention, verbose=True):
    pbar = get_progress_bar("validation", item="batches")
    batches = iter_batches_single_threaded(model=model,
                                           dataset=dataset,
                                           batch_size=batch_size,
                                           train=False,
                                           max_length=max_length,
                                           blank_triggers=blank_triggers,
                                           pbar=pbar)
    unary_scores = model.unary_scores[0]
    scenario_feature_scores = model.scenario_feature_scores
    token_correct = model.token_correct[0]
    token_correct_total = model.token_correct_total[0]
    nll = model.nll
    nll_total = model.nll_total
    is_training = model.is_training
    greedy_decoding = model.greedy_decoding
    beam_width_ph = model.beam_width
    decoded = model.decoded[0]
    attention_weights = model.attention_weights if save_attention else []
    # decoding_styles = [(True, True), (False, True)] if greedy_decoding is not None else [(None, True)]
    decoding_styles = [(True, True)] if greedy_decoding is not None else [(None, True)]
    output = []
    total_token_correct = {decoding_style: 0 for decoding_style, _ in decoding_styles}
    total_token_correct_total = {decoding_style: 0 for decoding_style, _ in decoding_styles}
    total_nll = {decoding_style: 0 for decoding_style, _ in decoding_styles}
    total_nll_total = {decoding_style: 0 for decoding_style, _ in decoding_styles}
    elapsed = 0.0
    for batch_idx, feed_dict in enumerate(batches):
        feed_dict[is_training] = False
        for decoding_style, save_result in decoding_styles:
            if decoding_style is not None:
                feed_dict[greedy_decoding] = decoding_style
            if beam_width_ph is not None:
                feed_dict[beam_width_ph] = beam_width
            t0 = time.time()
            batch_unary_scores, batch_scenario_feature_scores, batch_token_correct, batch_token_correct_total, batch_nll, batch_nll_total, batch_decoded, batch_attention_weights = session.run(
                (unary_scores, scenario_feature_scores, token_correct, token_correct_total, nll, nll_total, decoded, attention_weights), feed_dict)
            elapsed += time.time() - t0
            if save_result:
                saved_feed_dict = {key.name: value for key, value in feed_dict.items()}
                saved_feed_dict["unary_scores"] = batch_unary_scores
                saved_feed_dict["scenario_feature_scores"] = batch_scenario_feature_scores
                saved_feed_dict["decoded"] = batch_decoded
                if save_attention:
                    saved_feed_dict["attention_weights"] = batch_attention_weights
                output.append(_convert_to_fp16(saved_feed_dict))
            total_token_correct[decoding_style] += batch_token_correct
            total_token_correct_total[decoding_style] += batch_token_correct_total
            total_nll[decoding_style] += batch_nll
            total_nll_total[decoding_style] += batch_nll_total
            if verbose:
                print(decoding_style,
                      total_token_correct[decoding_style] / max(1, total_token_correct_total[decoding_style]),
                      batch_nll / max(1, batch_nll_total),
                      total_nll[decoding_style] / max(1, total_nll_total[decoding_style]))
    if verbose:
        print(total_token_correct, total_token_correct_total)
        print("elapsed", elapsed)
        print(scoped_timer_summarize())
    return output


def filter_dataset(dataset, keyword):
    if not hasattr(dataset, "orig_x"):
        dataset.orig_x = dataset.x
        dataset.orig_y = dataset.y

    new_x = []
    new_y = []
    for x, y in zip(dataset.x, dataset.y):
        if keyword in " ".join(x):
            new_x.append(x)
            new_y.append(y)
    dataset.x = new_x
    dataset.y = new_y
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", required=True, type=str)
    parser.add_argument("--dataset_filter", required=False, type=str, default=None)
    parser.add_argument("--example_filter", required=False, type=str, default=None)
    parser.add_argument("--max_length", type=int, default=200)
    boolean_argument(parser, "blank_triggers", default=False)
    boolean_argument(parser, "save_attention", default=False)
    with load_model(parser=parser) as (session, model, config, args):
        print("loading dataset...")
        t0 = time.time()
        wiki_prob_feature_index = None
        for idx, proj in enumerate(config.objectives[0]["classifications"]):
            if proj.get("type") == "wikipedia_probs":
                wiki_prob_feature_index = idx
                break

        config.setup_classification_handler()
        config.setup_trie_handler()
        collection = config.classifications.type_collection
        collection._web_get_name = False
        do_fix = True
        import wikidata_linker_utils.sequence_model
        with wikidata_linker_utils.sequence_model.as_default_type_collection(collection):
            print("set default type collection")
            pass
        with wikidata_linker_utils.sequence_model.as_default_classification_handler(config.classifications):
            print("set default classification handler")
            pass
        validation_batch_size = args.batch_size if args.validation_batch_size is None else args.validation_batch_size
        if args.dataset_filter is not None:
            config.datasets = [v for v in config.datasets if args.dataset_filter in v.get("name", v["path"].split('/')[-1].split(".")[0])]
        if args.example_filter is not None:
            for d in config.datasets:
                d["example_filter"] = args.example_filter

        validation_set = config.load_dataset("dev", merge=False)
        collection = config.classifications.type_collection
        if not isinstance(validation_set, dict):
            validation_set = {"": validation_set}
        for title, dataset in validation_set.items():
            output = produce_report(session=session,
                                    dataset=dataset,
                                    model=model,
                                    beam_width=args.beam_width,
                                    max_length=args.max_length,
                                    blank_triggers=args.blank_triggers,
                                    batch_size=validation_batch_size,
                                    save_attention=args.save_attention)

            makedirs(args.report_dir, exist_ok=True)
            save_path = join(args.report_dir, title + "_report.pkl")
            print("Saving report to {}".format(save_path))
            with open(save_path, "wb") as fout:
                pickle.dump(output, fout)


if __name__ == "__main__":
    main()
