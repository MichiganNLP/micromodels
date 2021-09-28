"""
Orchestrate micromodels.
An Orchestrator is responsible for training, loading,
and running inference of multiple micromodels.
"""

from typing import List, Mapping, Any, Optional

import os
from src.factory import FEATURE_CLASSIFIER_FACTORY
from src.micromodels.AbstractMicromodel import AbstractMicromodel


def get_model_name(config: Mapping[str, Any]) -> str:
    """
    Get name of a micromodel based on its config.

    :param config: a micromodel configuration.
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

    def __init__(
        self, base_path: str, configs: Optional[List[Mapping[str, Any]]] = None
    ):
        """
        :param base_path: filesystem path to where micromodels will be stored.
        :param configs: list of micromodel configurations.
        """
        self.pipeline = []
        self.cache = {}
        self.model_basepath = base_path
        if configs is not None:
            self.set_configs(configs)

    def set_configs(self, configs: List[Mapping[str, Any]]) -> None:
        """
        Set Orchestrator's configuration.

        :param configs: List of micromodel configurations.
        :return: None
        """
        self._verify_configs(configs)
        self.configs = configs

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

        :param config: a micromodel configuration.
        :param force_train: flag for forcing a retrain.
        :return: Instance of a micromodel.
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
            self._load_model(config)

    def _load_model(
        self, config: Mapping[str, Any], force_reload: bool = False
    ):
        """
        Load model based on config

        :param config: Micromodel configuration.
        :param force_reload: flag for forcing a reload.
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

    def infer_config(
        self, query: str, config: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """
        Run inference from the micromodel specified in a config.

        :param query: string utterance to run inference on.
        :param config: Micromodel configuration.
        :return: Mapping of micromodel name to the inference result.
        """
        self._verify_config(config)
        model_name = get_model_name(config)
        model = self.cache.get(model_name)
        if not model:
            model = self._load_model(config)
        return {model_name: model.infer(query, do_preprocess=True)}

    def infer(self, query: str) -> Mapping[str, Any]:
        """
        Infer from all the models in self.configs

        :param query: string utterance to run interence on.
        :return: Mapping of micromodel names to inference results.
        """
        features = {}
        for config in self.configs:
            features.update(self.infer_config(query, config))
        return features

    def flush_cache(self) -> None:
        """
        Flush cache of micromodels.
        """
        self.cache = {}

    @property
    def num_micromodels(self):
        """
        Return the number of micromodels
        """
        if self.configs is None:
            return 0
        return len(self.configs)
