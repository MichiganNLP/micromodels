"""
Snorkel based classification
"""
from typing import List, Mapping, Any

import os
import json
from nltk import tokenize
from FeatureClassifier import FeatureClassifier


BASE_DIR = "/home/andrew/cbt_dialogue_system/featurizer/"
MODEL_DIR = os.path.join(BASE_DIR, "models")


class BatchSnorkelClassifier(FeatureClassifier):
    """
    Snorkel based classification
    """

    def __init__(self, name):
        self.snorkel_query = None
        super(BatchSnorkelClassifier, self).__init__(name)

    def setup(self, config):
        """
        setup stage
        """
        self.snorkel_query = config["snorkel_query"](config["name"])

    def _train(
        self, train_data: List[str], train_args: Mapping[Any, Any] = None
    ) -> None:
        """
        Train Snorkel, which is equivalent to evaluating any
        snorkel packets.
        """
        train_args = train_args or {}
        tokenized = []
        for _op in train_data:
            tokenized.extend(tokenize.sent_tokenize(_op))
        self.snorkel_query.train(tokenized)
        save_path = train_args.get("save_path")
        if save_path:
            data = self.snorkel_query.build_train_data(tokenized)
            with open(save_path, "w") as file_p:
                json.dump(data, file_p, indent=2)

    def save_model(self, model_path):
        """
        Save model
        """
        self.snorkel_query.save_seed(model_path)

    def load_model(self, model_path):
        """
        Load model
        """
        self.snorkel_query.load_seed(model_path)

    def _infer(self, query):
        """
        infer
        """
        return self.snorkel_query.infer(query)

    def is_loaded(self):
        """
        Check if snorket_query is set.
        """
        return self.snorkel_query is not None

    def batch_infer(self, queries):
        """
        batch inference
        """
        return self.snorkel_query.batch_infer(queries)


if __name__ == "__main__":
    from Orchestrator import Orchestrator
    from snorkel_queries.constants import MID_DATA, NANO_DATA
    from snorkel_queries import (
        self_description,
        extreme_description,
        self_blame,
        hateful,
        future_tense,
        should,
        dismiss_positive,
        sadness,
        phq_apathy,
        phq_tired,
        phq_sleep,
        phq_focus,
        phq_food,
        phq_suicidal,
        diagnosed,
        gad_scared,
        PHQSuicidal,
    )

    configs = [
        {
            "model_type": "snorkel",
            "feature_name": "self_description",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": self_description,
                "save_path": "/tmp/self_description.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "extreme_description",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": extreme_description,
                "save_path": "/tmp/extreme_description.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "future_tense",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": future_tense,
                "save_path": "/tmp/future_tense.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "hateful",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": hateful,
                "save_path": "/tmp/hateful.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "sadness",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": sadness,
                "save_path": "/tmp/sadness.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "should",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": should,
                "save_path": "/tmp/should.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "self_blame",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": self_blame,
                "save_path": "/tmp/self_blame.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_apathy",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": phq_apathy,
                "save_path": "/tmp/phq_apathy.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_focus",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": phq_focus,
                "save_path": "/tmp/phq_focus.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_food",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": phq_food,
                "save_path": "/tmp/phq_food.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_sleep",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": phq_sleep,
                "save_path": "/tmp/phq_sleep.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_tired",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": phq_tired,
                "save_path": "/tmp/phq_tired.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_suicidal",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": phq_suicidal,
                "save_path": "/tmp/phq_suicidal.json",
            },
        },
        {
            "model_type": "snorkel",
            "feature_name": "diagnosed",
            "data": MID_DATA,
            "args": {
                "snorkel_packet": diagnosed,
                "save_path": "/tmp/diagnosed.json",
            },
        },
    ]
    configs = [
        {
            "model_type": "snorkel",
            "feature_name": "phq_suicidal",
            "data": MID_DATA,
            "setup_args": {
                "snorkel_query": PHQSuicidal
            }
        }
    ]

    orchestrator = Orchestrator(configs)
    orchestrator.train_all()
    print(orchestrator.infer_config("I am an idiot", configs[0]))
    features = orchestrator.infer("I am an idiot")
    print(json.dumps(features, indent=2))
    features = orchestrator.infer("I want to kill myself")
    print(json.dumps(features, indent=2))
