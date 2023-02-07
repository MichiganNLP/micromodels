"""
Utility functions for evaluating rationale extraction.
"""
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score


def evaluate(predictions, gold, iou_threshold=0.5):
    """
    T-F1, IOU-F1
    """
    t_f1_binary, t_f1_micro, t_f1_macro = t_f1(predictions, gold)
    iou_f1_score = iou_f1(predictions, gold, threshold=iou_threshold)
    return {
        "t_f1_binary": t_f1_binary,
        "t_f1_micro": t_f1_micro,
        "t_f1_macro": t_f1_macro,
        "t_f1": t_f1_binary,
        "iou_f1": iou_f1_score,
    }


def get_spans(array):
    """
    Get spans
    """
    spans = []
    span_start_idx = -1
    span_end_idx = -1
    for inner_idx, inner_elem in enumerate(array):
        if inner_elem == 1:
            if span_start_idx == -1:
                span_start_idx = inner_idx
            else:
                continue
        if inner_elem == 0:
            if span_start_idx == -1:
                continue
            span_end_idx = inner_idx
            spans.append((span_start_idx, span_end_idx))

            span_start_idx = -1
            span_end_idx = -1
    if span_start_idx != -1:
        span_end_idx = inner_idx
        spans.append((span_start_idx, span_end_idx))
    return spans


def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)


def iou_f1(predictions, gold, threshold=0.5):
    """
    IOU-F1
    """
    assert len(predictions) == len(gold)
    all_f1_vals = []
    for idx, pred in enumerate(predictions):
        gold_instance = gold[idx]
        assert len(pred) == len(gold_instance)

        pred_spans = get_spans(pred)
        gold_spans = get_spans(gold_instance)

        ious = defaultdict(dict)
        for pred_span in pred_spans:
            best_iou = 0.0
            for gold_span in gold_spans:
                num = len(
                    set(range(pred_span[0], pred_span[1]))
                    & set(range(gold_span[0], gold_span[1]))
                )
                denom = len(
                    set(range(pred_span[0], pred_span[1]))
                    | set(range(gold_span[0], gold_span[1]))
                )
                iou = 0 if denom == 0 else num / denom

                if iou > best_iou:
                    best_iou = iou
            ious[pred_span] = best_iou

        threshold_tps = sum(int(x >= threshold) for x in ious.values())

        micro_r = threshold_tps / len(gold_spans) if len(gold_spans) > 0 else 0
        micro_p = threshold_tps / len(pred_spans) if len(pred_spans) > 0 else 0
        micro_f1 = _f1(micro_r, micro_p)
        if len(pred_spans) == 0 and len(gold_spans) == 0:
            all_f1_vals.append(1)
        else:
            all_f1_vals.append(micro_f1)
    return np.mean(all_f1_vals)


def t_f1(predictions, gold):
    """
    T-F1
    """
    assert len(predictions) == len(gold)
    binaries = []
    micros = []
    macros = []
    for idx, pred in enumerate(predictions):
        gold_instance = gold[idx]
        assert len(pred) == len(gold_instance)

        binary_f1 = f1_score(
            gold_instance, pred, average="binary", zero_division=1
        )
        micro_f1 = f1_score(gold_instance, pred, average="micro")
        macro_f1 = f1_score(gold_instance, pred, average="macro")

        binaries.append(binary_f1)
        micros.append(micro_f1)
        macros.append(macro_f1)
    return np.mean(binaries), np.mean(micros), np.mean(macros)
