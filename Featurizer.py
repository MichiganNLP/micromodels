"""
Featurize text data
"""

import os
import json
from Orchestrator import Orchestrator
from snorkel_queries.constants import NANO_DATA


class Featurizer:
    """
    Featurizer that embeds text data to a linguistic feature vector
    """
    def __init__(self, configs):
        """
        Initialize orchestrator
        """
        self.configs = configs
        self.orchestrator = Orchestrator(configs)

    def featurize(self, queries, save_dir):
        """
        Featurize list of strings
        """
        if os.path.isdir(save_dir):
            raise RuntimeError("Directory %s already exists!" % save_dir)
        os.mkdir(save_dir)

        self.orchestrator.train_all()

        for config in self.configs:
            name = self.orchestrator._get_model_name(config)
            save_file = os.path.join(save_dir, name)

            features = self.orchestrator.batch_infer_config(config, queries)
            feature_vector = features[name]

            result = {
                "config": config,
                "features": feature_vector,
                "queries": queries
            }

            with open(save_file + ".json", "w") as file_p:
                json.dump(
                    result,
                    file_p,
                    indent=2,
                    default=lambda o: f"<<non-serializable: {type(o).__repr__}>>",
                )



if __name__ == "__main__":
    from snorkel_queries import (
        SelfDescription,
        FutureTense,
        Hateful,
        PHQSuicidal,
        SubstanceAbuse,
        Victimhood,
    )
    self_description = SelfDescription
    configs = [
        {
            "model_type": "snorkel",
            "feature_name": "self_description",
            "data": NANO_DATA,
            "setup_args": {
                "snorkel_query": SelfDescription,
            }
        },
        {
            "model_type": "snorkel",
            "feature_name": "future_tense",
            "data": NANO_DATA,
            "setup_args": {
                "snorkel_query": FutureTense
            }
        },
        {
            "model_type": "snorkel",
            "feature_name": "hateful",
            "data": NANO_DATA,
            "setup_args": {
                "snorkel_query": Hateful
            }
        },
        {
            "model_type": "snorkel",
            "feature_name": "phq_suicidal",
            "data": NANO_DATA,
            "setup_args": {
                "snorkel_query": PHQSuicidal
            }
        },
        {
            "model_type": "snorkel",
            "feature_name": "substance_abuse",
            "data": NANO_DATA,
            "setup_args": {
                "snorkel_query": SubstanceAbuse
            }
        },
        {
            "model_type": "snorkel",
            "feature_name": "victimhood",
            "data": NANO_DATA,
            "setup_args": {
                "snorkel_query": Victimhood
            }
        },
    ]
    featurizer = Featurizer(configs)
    test_queries = [
        "This is a test",
        "I'm an idiot",
        "I am so sad",
        "I hate this world",
        "I want to kill myself",
        "I am an alcoholic"
    ]
    featurizer.featurize(test_queries, "testing_snorkel_featurized")
