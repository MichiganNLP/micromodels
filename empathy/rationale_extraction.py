"""
Experiment on Empathetic Rationale Extraction on Epitome data.
"""

import os

import numpy as np
from nltk.tokenize import word_tokenize

from empathy.constants import EMP_TASKS as TASKS
from empathy.utils import (
    setup_emp_config,
    init_featurizer,
    load_emp_data,
    save_json,
    load_json,
)
from empathy.eval_utils import evaluate

MM_HOME = os.environ.get("MM_HOME")
MM_MODEL_DIR = os.path.join(MM_HOME, "models")
EMP_HOME = os.environ.get("EMP_HOME")
FEATURE_DIR = os.path.join(EMP_HOME, "featurized")
EMP_DATA_DIR = os.path.join(EMP_HOME, "data")

MMS = ["empathy_" + task for task in TASKS]
METRICS = ["t_f1", "iou_f1"]
ZERO_METRICS = ["always_zero_t_f1", "always_zero_iou_f1"]


THRESHOLD = 0.75


def reformat_data(data):
    """Reformat data to list of queries"""
    query_groups = [x["response_tokenized"] for x in data]
    reformatted = []
    groupings = {}
    query_idx = 0
    for idx, queries in enumerate(query_groups):
        for _ in queries:
            groupings[query_idx] = idx
            query_idx += 1
        reformatted.extend(queries)
    return reformatted, groupings


def run_bert(featurizer, data):
    """Featurize"""
    queries, query_idxs = reformat_data(data)
    bert_results = featurizer.run_bert(queries)
    for utt_idx, _ in bert_results.items():
        query_idx = query_idxs[utt_idx]
        bert_results[utt_idx]["labels"] = {
            task: {
                "level": data[query_idx][task]["level"],
                "rationales": data[query_idx][task]["rationales"],
            }
            for task in TASKS
        }
    return bert_results


def extract_rationales(orig_data, bert_data, task, threshold, seed):
    """
    Extraction rationales using micromodels.
    """
    _, query_idxs = reformat_data(orig_data)
    results_by_query = {}
    for utt_idx, data_obj in bert_data.items():
        query_idx = query_idxs[int(utt_idx)]

        if query_idx not in results_by_query:
            results_by_query[query_idx] = {
                "results": [],
                "query": orig_data[query_idx]["response_tokenized"],
                "id": "%s_%s"
                % (
                    orig_data[query_idx]["sp_id"],
                    orig_data[query_idx]["rp_id"],
                ),
            }
        results_by_query[query_idx]["results"].append(data_obj["results"])

    rationales = []
    groundtruths = []
    for query_idx, result_obj in results_by_query.items():
        _results = result_obj["results"]
        _query_utts = result_obj["query"]
        _rationales = []

        _mm = "empathy_%s" % task
        _scores = [_result[_mm]["max_score"] for _result in _results]
        binary = [1 if x > threshold else 0 for x in _scores]
        for _result in _results:
            top_k = _result[_mm]["top_k_scores"]
            if top_k[0][1] > threshold:
                _seed = top_k[0][0]

        for hit_idx, hit in enumerate(binary):
            segment = _query_utts[hit_idx]
            segment_toks = word_tokenize(segment)

            if hit == 0:
                _rationales.extend([0] * len(segment_toks))
            else:
                _rationales.extend([1] * len(segment_toks))
        rationales.append(_rationales)

        _groundtruth = orig_data[query_idx][task]["iob_format"]
        groundtruths.append(_groundtruth)
        assert len(_groundtruth) == len(_rationales)
    assert len(groundtruths) == len(rationales)
    return rationales, groundtruths


def run(train_data, test_data, seed, featurize=True):
    """Run end to end experiment"""

    emp_configs = setup_emp_config(train_data)
    featurizer = init_featurizer(emp_configs)

    output_path_test = os.path.join(FEATURE_DIR, "emp_test_%d.json" % seed)

    if featurize:
        print("Featurizing data")
        bert_results_val = run_bert(featurizer, test_data)
        save_json(bert_results_val, output_path_test)

    else:
        bert_results_val = load_json(output_path_test)

    threshold = THRESHOLD
    results = {task: {} for task in TASKS}
    for task in TASKS:
        rationales, groundtruth_rationales = extract_rationales(
            test_data, bert_results_val, task, threshold, seed
        )
        results[task] = evaluate(rationales, groundtruth_rationales)

    return results


def aggregate_results(full_results, run_results):
    """
    Aggregate results
    """
    for task in TASKS:
        for metric in METRICS + ZERO_METRICS:
            full_results[task]["f1s"][metric].append(run_results[task][metric])


def print_results(results):
    """
    Print results
    """
    for task in TASKS:
        print("--------------------------")
        print(" Task: %s" % task)
        t_f1 = np.mean(results[task]["f1s"]["t_f1"])
        iou_f1 = np.mean(results[task]["f1s"]["iou_f1"])
        print(" T_F1:", t_f1)
        print(" IOU F1:", iou_f1)

        t_f1_std = np.std(results[task]["f1s"]["t_f1"])
        iou_f1_std = np.std(results[task]["f1s"]["iou_f1"])
        print(" T_F1 STD DEV:", t_f1_std)
        print(" IOU F1 STD DEV:", iou_f1_std)


def main():
    """Driver"""
    results = {
        task: {
            "f1s": {
                "t_f1": [],
                "iou_f1": [],
                "always_zero_t_f1": [],
                "always_zero_iou_f1": [],
            },
        }
        for task in TASKS
    }
    results["data_stats"] = {
        "train_size": [],
        "val_size": [],
        "test_size": [],
    }

    for seed in range(42, 42 + 10):
        print("Running Seed %d" % seed)

        emp_data_path = os.path.join(EMP_DATA_DIR, "combined_iob.json")
        train_data, val_data, test_data, _ = load_emp_data(
            emp_data_path, seed, train_ratio=0.75, val_ratio=0.05
        )

        _results = run(train_data, test_data, seed, featurize=True)
        aggregate_results(results, _results)

    print_results(results)


if __name__ == "__main__":
    main()
