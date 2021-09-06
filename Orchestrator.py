"""
Orchestrate featurizers
"""

from typing import List, Mapping

import os
import json
import sys
import time
from pathos.multiprocessing import ProcessingPool as Pool
from factory import FEATURE_CLASSIFIER_FACTORY
from utils import preprocess


class Orchestrator:
    """
    An Orchestrator is responsible for training, maintaining, loading,
    and infering from multiple featurizer models.
    """
    def __init__(self, configs=None):
        self.configs = configs
        self.pipeline = []
        self.cache = {}
        self.model_basepath = os.path.join(
            os.environ.get("CBT_DS_PATH"), "featurizer/models"
        )

    def set_configs_from_file(self, config_file):
        """
        Load config file
        """
        with open(config_file, "r") as file_p:
            configs = json.load(file_p)
        self.set_configs(configs)

    def set_configs(self, configs):
        """
        Set config
        """
        self._verify_configs(configs)
        self.configs = configs

    def set_model_basepath(self, path):
        """
        set model_basepath
        """
        self.model_basepath = path

    def flush_cache(self):
        """
        Flush cache
        """
        self.cache = {}

    def _verify_configs(self, configs, train=False):
        """
        Verify list of configs
        """
        for config in configs:
            self._verify_config(config, train=train)
        feature_names = [self._get_model_name(config) for config in configs]
        if len(feature_names) != len(configs):
            raise ValueError("Duplicate feature names?")

    @staticmethod
    def _verify_config(config, train=False):
        """
        Verify single config
        """
        required_fields = ["feature_name", "model_type"]
        for field in required_fields:
            if field not in config:
                raise ValueError("%s missing" % field)

    @staticmethod
    def _get_model_name(config):
        """
        Get model name from config.
        """
        return "%s_%s" % (config["feature_name"], config["model_type"])

    def train_config(self, config, force_train=False):
        """
        Train based on a single config
        """
        self._verify_config(config, train=True)
        model_type = config.get("model_type")
        train_data = config.get("data")
        args = config.get("args", {})
        model_name = self._get_model_name(config)

        try:
            model = FEATURE_CLASSIFIER_FACTORY[model_type](model_name)
        except KeyError:
            raise KeyError("Invalid model type")

        if force_train or model_name not in self.cache:
            setup_config = config.get("setup_args", {})
            if "name" not in setup_config:
                setup_config["name"] = self._get_model_name(config)
            model.setup(setup_config)
            model.train(train_data, args)
            model_path = config.get(
                "model_path", os.path.join(self.model_basepath, model_name)
            )
            model.save_model(model_path)
            self.cache[model_name] = model
        return self.cache[model_name]

    def train_all(self):
        """
        Train based on self.config
        """
        for config in self.configs:
            print("Training %s" % self._get_model_name(config))
            self.train_config(config)

    def load_model(self, config, force_reload=False):
        """
        Load model based on config
        """
        self._verify_config(config)
        model_name = self._get_model_name(config)
        if force_reload or model_name not in self.cache:
            model_path = config.get(
                "model_path", os.path.join(self.model_basepath, model_name)
            )
            model = FEATURE_CLASSIFIER_FACTORY[config["model_type"]](
                model_name
            )
            setup_config = config.get("setup_args", {})
            if "name" not in setup_config:
                setup_config["name"] = model_name
            model.setup(setup_config)
            model.load_model(model_path)
            self.cache[model_name] = model
        return self.cache[model_name]

    def infer_config(self, query, config):
        """
        Infer based on a config
        """
        self._verify_config(config)
        model_name = self._get_model_name(config)
        model = self.cache.get(model_name)
        if not model:
            model = self.load_model(config)
        return {model_name: model.infer(query, do_preprocess=True)}

    def infer_all(self, query):
        """
        Infer from all the models in self.configs
        """
        features = {}
        for config in self.configs:
            features.update(self.infer_config(query, config))
        return features

    def infer(self, query):
        """
        wrapper around infer_all
        """
        return self.infer_all(query)

    def batch_infer_config(self, config, queries):
        """
        Batch inference if applicable
        """
        self._verify_config(config)
        model_name = self._get_model_name(config)
        model = self.cache.get(model_name)
        if not model:
            model = self.load_model(config)
        if not hasattr(model, "batch_infer"):
            raise RuntimeError(
                "model %s does not have batch_infer" % model_name
            )
        return {model_name: model.batch_infer(queries)}

    def infer_aggregate(self, queries: List[str]) -> Mapping[str, int]:
        """
        Given a list of queries, run inference on each one and
        aggregate the results.
        """
        aggregate = {}
        for query in queries:
            predictions = self.infer(query)
            for feature_name, result in predictions.items():
                if feature_name not in aggregate:
                    aggregate[feature_name] = 0
                if result == "true":
                    aggregate[feature_name] += 1
        return aggregate

    def infer_aggregate_parallel(
        self, queries: List[str], num_proc: int
    ) -> Mapping[str, int]:
        """
        Given a list of queries, run inference on each one and
        aggregate the results.
        Run in parallel.
        """
        if num_proc < 2:
            return self.infer_aggregate(queries)

        chunk_size = len(queries) // num_proc
        chunks = [
            queries[i : i + chunk_size]
            for i in range(0, len(queries), chunk_size)
        ]
        pool = Pool(num_proc)
        try:
            results = pool.amap(self.infer_aggregate, chunks)
            while not results.ready():
                time.sleep(1)
            infer_results = results.get()
        except (KeyboardInterrupt, SystemExit):
            print("Inference Canceled")
            pool.terminate()
            sys.exit(1)

        aggregate = {}
        for infer_result in infer_results:
            for feature_name, feature_count in infer_result.items():
                if feature_name not in aggregate:
                    aggregate[feature_name] = 0
                aggregate[feature_name] += feature_count
        return aggregate


def main():
    """ Driver """
    configs = [
        {
            "model_type": "svm",
            "data": "data/testing.json",
            "feature_name": "gender",
        },
        {
            "model_type": "cql",
            "feature_name": "gender",
            "data": "data/testing.json",
            "args": {
                "config": [
                    {
                        "type": "liwc",
                        "category": "pronouns",
                        "dominant": "I",
                        "threshold": 0.5,
                    }
                ]
            },
        },
    ]
    orchestrator = Orchestrator()
    orchestrator.set_configs(configs)
    orchestrator.train_all()
    features = orchestrator.infer("I am a boy")
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
