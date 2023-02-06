"""
Empathy Prediction experiment.
"""

import os
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
)
from interpret.glassbox import ExplainableBoostingClassifier
from empathy.constants import EMP_TASKS as TASKS
from empathy.utils import setup_emp_config, init_featurizer, load_emp_data

EMP_HOME = os.environ.get("EMP_HOME")
EMP_DATA_DIR = os.path.join(EMP_HOME, "data")


def featurize(data, featurizer=None):
    """
    Featurize for EBM
    """
    if featurizer is None:
        mm_configs = setup_emp_config(data)
        featurizer = init_featurizer(mm_configs)

    queries = [data_obj["response_post"] for data_obj in data]
    results = featurizer.run_bert(queries)

    featurized = []
    er_labels = []
    int_labels = []
    exp_labels = []
    for idx, data_obj in enumerate(data):

        _results = results[idx]

        scores = [
            _results["results"]["empathy_emotional_reactions"]["max_score"],
            _results["results"]["empathy_interpretations"]["max_score"],
            _results["results"]["empathy_explorations"]["max_score"],
        ]

        featurized.append(scores)
        er_labels.append(data_obj["emotional_reactions"]["level"])
        int_labels.append(data_obj["interpretations"]["level"])
        exp_labels.append(data_obj["explorations"]["level"])

    feature_names = ["emotional_reactions", "interpretations", "explorations"]
    return feature_names, featurized, [er_labels, int_labels, exp_labels]


def main():
    """Driver"""
    em_accs = []
    em_f1s = []
    int_accs = []
    int_f1s = []
    exp_accs = []
    exp_f1s = []

    for seed in range(42, 42 + 10):
        print("Running Seed %d" % seed)

        emp_data_path = os.path.join(EMP_DATA_DIR, "combined_iob.json")
        train_data, val_data, test_data, _ = load_emp_data(
            emp_data_path, seed, train_ratio=0.75, val_ratio=0.05
        )

        feature_names, train_features, train_labels = featurize(train_data)

        em_ebm = ExplainableBoostingClassifier(feature_names=feature_names)
        int_ebm = ExplainableBoostingClassifier(feature_names=feature_names)
        exp_ebm = ExplainableBoostingClassifier(feature_names=feature_names)

        em_ebm.fit(train_features, train_labels[0])
        int_ebm.fit(train_features, train_labels[1])
        exp_ebm.fit(train_features, train_labels[2])

        _, test_features, test_labels = featurize(test_data)

        em_preds = em_ebm.predict(test_features)
        int_preds = int_ebm.predict(test_features)
        exp_preds = exp_ebm.predict(test_features)

        em_accuracy = accuracy_score(test_labels[0], em_preds)
        em_f1 = f1_score(test_labels[0], em_preds, average="macro")

        int_accuracy = accuracy_score(test_labels[1], int_preds)
        int_f1 = f1_score(test_labels[1], int_preds, average="macro")

        exp_accuracy = accuracy_score(test_labels[2], exp_preds)
        exp_f1 = f1_score(test_labels[2], exp_preds, average="macro")

        em_accs.append(em_accuracy)
        em_f1s.append(em_f1)
        int_accs.append(int_accuracy)
        int_f1s.append(int_f1)
        exp_accs.append(exp_accuracy)
        exp_f1s.append(exp_f1)

    print("EM")
    print("Acc", np.mean(em_accs))
    print("F1", np.mean(em_f1s))
    print("INT")
    print("Acc", np.mean(int_accs))
    print("F1", np.mean(int_f1s))
    print("EXP")
    print("Acc", np.mean(exp_accs))
    print("F1", np.mean(exp_f1s))


if __name__ == "__main__":
    main()
