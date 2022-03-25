import numpy as np
import uuid
import json
from IPython.display import HTML, display_html


def find_batch_idx(report, example_idx, model_name):
    so_far = 0
    for batch in report:
        seq_lens = batch['{}/Inputs/sequence_lengths:0'.format(model_name)]
        batch_size = len(seq_lens)
        try:
            if so_far + batch_size <= example_idx:
                so_far += batch_size
                continue
            batch_idx = example_idx - so_far
            seq_len = seq_lens[batch_idx]
        except:
            print("so_far", so_far)
            print("batch_idx", batch_idx)
            print("example_idx", example_idx)
            print("len(seq_lens)", len(seq_lens))
            raise
        return batch, batch_idx
    return None, None


def display_example(report, vocab, model_name, example_idx, projections, projection_vocab, uppercase_input_index,
                    collection, only_results=False, group_by_filter=None):
    batch, batch_idx = find_batch_idx(report, example_idx, model_name)
    if batch is None:
        return None
    seq_lens = batch['{}/Inputs/sequence_lengths:0'.format(model_name)]
    seq_len = seq_lens[batch_idx]
    words = vocab[batch['{}/Inputs/input_placeholders_0:0'.format(model_name)][:seq_len, batch_idx]]
    uppercase = batch['{}/Inputs/input_placeholders_{}:0'.format(model_name, uppercase_input_index)][:seq_len, batch_idx] if uppercase_input_index != -1 else None
    labels_mask = batch['{}/Inputs/type/labels_mask:0'.format(model_name)][:seq_len, batch_idx, 0]
    candidate_ids = batch['{}/type_1/candidates_ids:0'.format(model_name)][batch_idx, :seq_len]
    supervised = batch['{}/type_1/supervised_time_major:0'.format(model_name)][:seq_len, batch_idx] if '{}/type_1/supervised_time_major:0'.format(model_name) in batch else labels_mask
    if "decoded" in batch:
        predictions = batch["decoded"][batch_idx, :seq_len]
    else:
        predictions = np.argmax(batch["unary_scores"][batch_idx, :seq_len], axis=-1)
    correct = predictions == 0
    out = []
    out_str = ""
    segment_ids = batch.get('{}/type_1/MultiwordPool/segment_ids:0'.format(model_name))
    segment_locations = batch.get('{}/type_1/MultiwordPool/segment_locations:0'.format(model_name))
    if segment_ids is not None:
        segment_ids = segment_ids[:seq_len, batch_idx]
        segment_locations = segment_locations[:seq_len, batch_idx]
    ids_and_cumsum = {}
    float_inputs = []
    for name in projections:
        seglenth = '{}/type_1/VariableAssignments/Embed/{}/segment_lengths:0'.format(model_name, name)
        ids_key = '{}/type_1/VariableAssignments/Embed/{}/unique_assignments:0'.format(model_name, name)
        float_input_key = '{}/type_1/VariableAssignments/Embed/{}/float_input:0'.format(model_name, name)
        wikipedia_probs_input_key = '{}/type_1/VariableAssignments/Embed/{}/wikipedia_probs_input:0'.format(model_name, name)
        if ids_key in batch:
            ids = batch[ids_key]
            if seglenth in batch:
                # is variable size
                cumsum = np.cumsum(batch[seglenth])
            else:
                cumsum = np.cumsum(np.ones(len(ids), dtype=np.int32))
            ids_and_cumsum[name] = (cumsum, ids)
        else:
            pass
        if float_input_key in batch:
            float_inputs.append(batch[float_input_key])
        elif wikipedia_probs_input_key in batch:
            float_inputs.append(batch[wikipedia_probs_input_key])
        else:
            float_inputs.append(None)

    starting_segment_id = None
    if segment_ids is not None:
        for idx in segment_ids:
            if idx > 0:
                if starting_segment_id is None:
                    starting_segment_id = idx
                else:
                    starting_segment_id = min(idx, starting_segment_id)
    
    segment_id2prediction_tstep = {idx: k for k, idx in enumerate(segment_locations)}
    for tstep, word in enumerate(words):
        if uppercase is not None and uppercase[tstep] and len(word) > 0:
            word = word[0].upper() + word[1:]

        if segment_ids is not None:
            # pooling is occuring so prediction tstep is not the text timestep
            step_has_label = segment_ids[tstep] != 0
            prediction_tstep = prediction_tstep = segment_id2prediction_tstep[segment_ids[tstep]] if step_has_label else None
        else:
            prediction_tstep = tstep
            step_has_label = labels_mask[tstep]
        if only_results:
            # only care about supervised datapoints
            step_has_label = step_has_label and supervised[prediction_tstep]
        if step_has_label:
            if tstep > 0 and ((segment_ids is not None and segment_ids[tstep - 1] == segment_ids[tstep]) or (segment_ids is None and candidate_ids[tstep - 1, 0] == candidate_ids[tstep, 0])):
                # continue
                out[-1]["text"] += " " + word
            else:
                # restart
                if len(out_str) > 0:
                    out.append({"text": out_str})
                    out_str = ""
                
                batch_labels = batch['{}/Inputs/type/labels:0'.format(model_name)][prediction_tstep, batch_idx]
                batch_labels_mask = batch['{}/Inputs/type/labels_mask:0'.format(model_name)][prediction_tstep, batch_idx]
                label_internal_idx = batch_labels[0]
                instance = []
                for name, (cumsum, ids) in ids_and_cumsum.items():
                    # print(cumsum[label_internal_idx - 1])
                    # print(cumsum[label_internal_idx])
                    # print(ids)
                    # print(ids[cumsum[label_internal_idx - 1]])
                    # print(ids[cumsum[label_internal_idx - 1] if label_internal_idx > 0 else 0: cumsum[label_internal_idx]])
                    if group_by_filter is None or group_by_filter in name:
                        instance += list(projection_vocab[name][ids[cumsum[label_internal_idx - 1] if label_internal_idx > 0 else 0: cumsum[label_internal_idx]]])
                label_wikidata = collection.ids[candidate_ids[prediction_tstep, 0]]
                res = {"text": word,
                        "label": label_wikidata,
                        "correct": correct[prediction_tstep],
                        "supervised": supervised[prediction_tstep],
                        "projections": projections,
                        "candidates": sorted([
                            {
                                "id": collection.ids[candidate_id],
                                "name": "" if only_results else collection.get_name(candidate_id).rsplit(" - ", 1)[0].rsplit(" (", 1)[0],
                                "prob": prob,
                                "label": idx == 0,
                                "predicted": predictions[prediction_tstep] == idx,
                                "feature_scores": [fscore[batch_idx, prediction_tstep, idx] for fscore in batch["scenario_feature_scores"]],
                                "float_inputs": [finput[batch_labels[idx]] if finput is not None else None for finput in float_inputs]
                            }
                            for idx, (candidate_id, present, prob)
                            in enumerate(zip(candidate_ids[prediction_tstep], batch_labels_mask, batch["unary_scores"][batch_idx, prediction_tstep]))
                            if present
                        ], key=lambda x: x["prob"], reverse=True),
                        "instance": instance,
                        "prob": batch["unary_scores"][batch_idx, prediction_tstep, 0]}
                if not correct[prediction_tstep]:
                    label_internal_idx = batch_labels[predictions[prediction_tstep]]
                    instance = []
                    for name, (cumsum, ids) in ids_and_cumsum.items():
                        if group_by_filter is None or group_by_filter in name:
                            instance += list(projection_vocab[name][ids[cumsum[label_internal_idx - 1] if label_internal_idx > 0 else 0: cumsum[label_internal_idx]]])
                    res.update({"prediction": collection.ids[candidate_ids[prediction_tstep, predictions[prediction_tstep]]],
                                "prediction_prob": batch["unary_scores"][batch_idx, prediction_tstep, predictions[prediction_tstep]],
                                "prediction_instance": instance})
                out.append(res)
        else:
            if len(out_str) > 0:
                out_str += " " + word
            else:
                out_str = word
    if len(out_str) > 0:
        out.append({"text": out_str})
        out_str = ""
    return out


def confusion2html( source, dest, count):
    join_dest = '<br/>'.join(dest)
    join_source = '<br/>'.join(source)
    out = (f"<div style='display: inline-block;padding-right: 10px'>{count}</div>"
           f"<div style='display: inline-block; padding: 5px;border-radius: 10px; background: rgba(108, 186, 243, 0.2)'>{join_source}</div>"
           f"<div style='display: inline-block; padding: 5px;border-radius: 10px; background: rgba(237, 44, 44, 0.2)'>{join_dest}</div>")
    return HTML(out)


def ex2html(example, highlight=None):
    out = "<div class='row'><div class='column' style='padding-bottom: 40px; line-height: 325%; position: relative; width: 300px;'>"
    projections = []
    for tstep, el in enumerate(example):
        if "projections" in el:
            projections = [subel.replace("_", " ") for subel in el["projections"]]
    hover_id = str(uuid.uuid1())
    registered_ids = []
    for tstep, el in enumerate(example):
        if tstep > 0:
            out += f"<span> </span>"
        text = el["text"].replace("<", "&lt;").replace(">", "&gt;")
        if "label" in el:
            color = "blue" if el["correct"] else "red"
            color = "lightblue" if not el["supervised"] else color
            subcaption = ""
            
            if not el["correct"] and "prediction" in el and "prediction_instance" in el:
                highlighted = ((tuple(el["instance"]), tuple(el["prediction_instance"])),) == highlight if highlight is not None else False
                prediction_link = f"https://www.wikidata.org/wiki/{el['prediction']}"
                subcaption = f"<div  style=\"position:absolute; top: 30px; font-size: 10px; line-height: 100%\"><div style=\"position:relative;white-space: nowrap\"><a target=\"_blank\" href=\"https://www.wikidata.org/wiki/{el['label']}\">{el['label']}({el['prob']:.2f})</a> -></div><div style=\"position:relative;white-space: nowrap\"><a target=\"_blank\" href=\"{prediction_link}\">{el['prediction']}({el['prediction_prob']:.2f})</a></div></div>"
                if highlighted:
                    color = "orange"
            else:
                prediction_link = f"https://www.wikidata.org/wiki/{el['label']}"
            registered_id = str(uuid.uuid1())
            registered_ids.append(
                (registered_id, el["candidates"])
            )
            out += f"<div style=\"display: inline-block; position:relative\"><a style=\"color:{color}\" target=\"_blank\" href=\"{prediction_link}\" id=\"{registered_id}\">{text}</a>{subcaption}</div>"
        else:
            out += f"<span>{text}</span>"
    out += "</div>"
    out += f"<div class='column' style='line-height: 110%; font-size:15px; width: 500px;' id=\"{hover_id}\"></div>"
    out += """<script>
        (function () {
        var selected_item = null;
        function show_hover_box(e, candidates) {
            var projections = """ + json.dumps(projections) + """;
            if (selected_item != null) {
                selected_item.style.background = "";
            }
            selected_item = e.target;
            selected_item.style.background = "yellow";
            var el = document.getElementById('""" + hover_id + """');
            var html = "<table><tr><th style='text-align: left;'>Name</th><th style='text-align: left;'>Prob</th>";
            for (var i = 0; i < projections.length; i++) {
                html += "<th style='text-align: left;'>" + projections[i] + "</th>";
            }
            html += "</tr>";
            for (var i = 0; i < candidates.length; i++) {
                html += "<tr" + (candidates[i].label ? ' style=\"background:#F3FDC0\"' : "") + "><td style='text-align: left;'>" + candidates[i].name + "</td><td>" + (100.0 * candidates[i].prob).toFixed(1) + "%" + "</td>";
                for (var j = 0; j < candidates[i].feature_scores.length; j++) {
                    html += "<td>" + candidates[i].feature_scores[j].toFixed(2);
                    if (candidates[i].float_inputs[j] != null) {
                        var minimized_finput = [];
                        for (var x = 0; x < candidates[i].float_inputs[j].length; x++) {
                            minimized_finput.push(candidates[i].float_inputs[j][x].toFixed(2));
                        }
                        html += "<br/><span style='font-size: 8px; color: #444'>" + minimized_finput.join(", ") + "</span>";
                    }
                    html += "</td>";
                }
                html += "</tr>";
            }
            html += "</table>";
            el.innerHTML = html;
        };
    """
    for registered_id, candidates in registered_ids:
        candidate_feature_scores_arr = np.array([candidate["feature_scores"] for candidate in candidates])
        mean_score = np.median(candidate_feature_scores_arr) if np.prod(candidate_feature_scores_arr.shape) > 0 else 0.0
        for candidate in candidates:
            candidate["feature_scores"] = candidate["feature_scores"] - mean_score
        out += "document.getElementById('" + registered_id + """').addEventListener('mouseover', function (e) {
            show_hover_box(e, """ + json.dumps([{"id": candidate["id"], "name": candidate["name"], "prob": float(candidate["prob"]),
                                                 "feature_scores": list(map(float, candidate["feature_scores"])),
                                                 "label": candidate["label"],
                                                 "float_inputs": [list(map(float, finput)) if finput is not None else None
                                                                  for finput in candidate["float_inputs"]]}
                                                 for candidate in candidates]) + """)});\n"""
    out += "})();</script><style>.column{float: left;height: 375px; overflow-y:scroll}.row:after{content: \"\";display: table;clear: both;}</style></div>"
    return HTML(out)


def print_with_explanation(mistakes, examples):
    for (source, dest), count in mistakes:
        display_html(confusion2html([collection.get_name(collection.name2index[idx]).split(" - ")[0].split("(")[0].strip() for idx in source],
                                    [collection.get_name(collection.name2index[idx]).split(" - ")[0].split("(")[0].strip() for idx in dest], count))
        key = ((source, dest),)
        for ex in examples[key][:2]:
            display_html(ex2html(ex, highlight=key))
