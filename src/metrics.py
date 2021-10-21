"""
Metrics for eval
"""

from sklearn.metrics import f1_score


def _stats(predictions, ground_truths, pos_val=1, neg_val=0):
    """
    Return true positives, false positives, true negatives, false negatives.
    """
    assert len(predictions) == len(ground_truths)
    results = {"true_pos": 0, "false_pos": 0, "true_neg": 0, "false_neg": 0}
    for idx, val in enumerate(ground_truths):
        if val == pos_val:
            if predictions[idx] == pos_val:
                results["true_pos"] += 1
            elif predictions[idx] == neg_val:
                results["false_neg"] += 1
            else:
                raise ValueError("Invalid Value")

        elif val == neg_val:
            if predictions[idx] == pos_val:
                results["false_pos"] += 1
            elif predictions[idx] == neg_val:
                results["true_neg"] += 1
            else:
                raise ValueError("Invalid Value")

        else:
            raise ValueError("Invalid Value")
    return results


def recall(predictions, ground_truths, pos_val=1, neg_val=0):
    """
    Recall
    """
    assert len(predictions) == len(ground_truths)
    stats = _stats(predictions, ground_truths, pos_val, neg_val)
    top = stats["true_pos"]
    if top == 0:
        return 0
    bottom = stats["true_pos"] + stats["false_neg"]
    return top / bottom


def precision(predictions, ground_truths, pos_val=1, neg_val=0):
    """
    Precision
    """
    assert len(predictions) == len(ground_truths)
    stats = _stats(predictions, ground_truths, pos_val, neg_val)
    top = stats["true_pos"]
    if top == 0:
        return 0
    bottom = stats["true_pos"] + stats["false_pos"]
    return top / bottom


def f1(predictions, ground_truths):
    """
    F1 Score
    """
    pos_f1 = f1_score(predictions, ground_truths, average="weighted")
    micro_f1 = f1_score(predictions, ground_truths, average="micro")
    macro_f1 = f1_score(predictions, ground_truths, average="macro")

    return (
        pos_f1,
        micro_f1,
        macro_f1,
    )
