"""
Orchestrate micromodels.
An Orchestrator is responsible for training, loading,
and running inference of multiple micromodels.
"""

from typing import List, Mapping, Any

import os
import json
import sys
import time
from pathos.multiprocessing import ProcessingPool as Pool
from src.factory import FEATURE_CLASSIFIER_FACTORY
from src.micromodels.AbstractMicromodel import AbstractMicromodel


def get_model_name(config: Mapping[str, Any]) -> str:
    """
    Get name of a micromodel based on its config.

    :param config: a micromodel configuration
    :return: name of micromodel
    :raise ValueError: raised when configuration is invalid
    """
    if "feature_name" not in config:
        raise ValueError("Missing %s in config." % "feature_name")
    if "model_type" not in config:
        raise ValueError("Missing %s in config." % "model_type")
    return "%s_%s" % (config["feature_name"], config["model_type"])


class Orchestrator:
    """
    An Orchestrator is responsible for training, loading, and predicting
    from multiple micromodels.
    """

    def __init__(self, base_path: str, configs: List[Mapping[str, Any]] = None):
        """
        :param base_path: filesystem path to where micromodels will be stored.
        :param configs: list of micromodel configurations.
        """
        self.pipeline = []
        self.cache = {}
        self.model_basepath = os.path.join(base_path, "models")
        if configs is not None:
            self.set_configs(configs)

    def set_configs_from_file(self, config_file: str) -> None:
        """
        Load and set Orchestrator's configuration based on input file.

        :param config_file: filepath to configuration file
        :returns: None
        """
        with open(config_file, "r") as file_p:
            configs = json.load(file_p)
        self.set_configs(configs)

    def set_configs(self, configs: List[Mapping[str, Any]]) -> None:
        """
        Set Orchestrator's configuration.

        :param configs: List of micromodel configurations.
        :return: None
        """
        self._verify_configs(configs)
        self.configs = configs

    def set_model_basepath(self, path: str) -> None:
        """
        Set model_basepath attribute.

        :param path: filepath to directory storing all micromodels.
        :return: None
        """
        self.model_basepath = path

    def flush_cache(self):
        """
        Flush cache
        """
        self.cache = {}

    def _verify_configs(self, configs: List[Mapping[str, Any]]) -> None:
        """
        Verify list of micromodel configurations.

        :param configs: list of micromodel configurations
        :return: None
        :raises ValueError: raised when a configuration is invalid or
            there are duplicate feature names.
        """
        for config in configs:
            self._verify_config(config)
        feature_names = [get_model_name(config) for config in configs]
        if len(list(set(feature_names))) != len(configs):
            raise ValueError(
                "Number of feature names does not match the number of config \
                entries. Are there duplicate feature names?"
            )

    @staticmethod
    def _verify_config(config: Mapping[str, Any]) -> None:
        """
        Verify a single config.

        :param config: a micromodel configuration
        :return: None
        :raises ValueError: raised when configuration is invalid
        """
        required_fields = ["feature_name", "model_type"]
        for field in required_fields:
            if field not in config:
                raise ValueError("Invalid config: %s missing" % field)

    def train_config(
        self, config: Mapping[str, Any], force_train=False
    ) -> AbstractMicromodel:
        """
        Fetch micromodel as specified in the input config.
        If the micromodel has not been trained yet, this method will
        train the model, add it to the cache, and return the model.
        :force_train: allows users to always retrain a micromodel.

        :param config: a micromodel configuration
        :param force_train: flag for forcing a retrain
        :return: AbstractMicromodel
        :raise KeyError: raised when an invalid model type is specified.
        """
        self._verify_config(config)
        model_type = config.get("model_type")
        train_data = config.get("data")
        args = config.get("args", {})
        model_name = get_model_name(config)

        try:
            model = FEATURE_CLASSIFIER_FACTORY[model_type](model_name)
        except KeyError as ex:
            raise KeyError("Invalid model type %s" % model_type) from ex

        if not force_train and model_name in self.cache:
            return self.cache[model_name]

        setup_config = config.get("setup_args", {})
        if "name" not in setup_config:
            setup_config["name"] = model_name
        model.setup(setup_config)
        print("Training %s" % model_name)
        model.train(train_data, args)
        model_path = config.get(
            "model_path", os.path.join(self.model_basepath, model_name)
        )
        model.save_model(model_path)
        print("%s saved to %s" % (model_name, model_path))
        self.cache[model_name] = model
        return self.cache[model_name]

    def train_all(self) -> None:
        """
        Train all micromodels specified in self.configs.
        """
        for config in self.configs:
            if config.get("train", True):
                self.train_config(config)

    def load_models(self) -> None:
        """
        Load all models into cache.
        """
        for config in self.configs:
            self.load_model(config)

    def load_model(self, config: Mapping[str, Any], force_reload: bool = False):
        """
        Load model based on config
        """
        self._verify_config(config)
        model_name = get_model_name(config)

        if not force_reload and model_name in self.cache:
            return self.cache[model_name]

        model_path = config.get(
            "model_path", os.path.join(self.model_basepath, model_name)
        )
        model = FEATURE_CLASSIFIER_FACTORY[config["model_type"]](model_name)
        setup_config = config.get("setup_args", {})
        if "name" not in setup_config:
            setup_config["name"] = model_name
        model.setup(setup_config)
        model.load_model(model_path)
        self.cache[model_name] = model
        return self.cache[model_name]

    def infer_config(self, query, config):
        """
        Run inference from the micromodel specified in a config.
        """
        self._verify_config(config)
        model_name = get_model_name(config)
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
        model_name = get_model_name(config)
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

    @property
    def num_micromodels(self):
        """
        Return the number of micromodels
        """
        if self.configs is None:
            return 0
        return len(self.configs)
