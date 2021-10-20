"""
Orchestrate micromodels.
An Orchestrator is responsible for building, loading,
and running inference of multiple micromodels.
"""

from typing import List, Mapping, Any, Optional

import os
from src.factory import MICROMODEL_FACTORY
from src.micromodels.AbstractMicromodel import AbstractMicromodel


def get_model_name(config: Mapping[str, Any]) -> str:
    """
    Get name of a micromodel based on its config.

    :param config: a micromodel configuration.
    :return: name of micromodel
    :raise ValueError: raised when configuration is invalid
    """
    if "name" not in config:
        raise ValueError("Missing %s in config." % "name")
    if "model_type" not in config:
        raise ValueError("Missing %s in config." % "model_type")
    return "%s_%s" % (config["name"], config["model_type"])


class Orchestrator:
    """
    An Orchestrator is responsible for building, loading, and predicting
    from multiple micromodels.
    """

    def __init__(
        self, base_path: str, configs: List[Mapping[str, Any]] = None
    ):
        """
        :param base_path: filesystem path to where micromodels will be stored.
        :param configs: list of micromodel configurations.
        """
        self.cache = {}
        self.model_basepath = base_path
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
        names = [get_model_name(config) for config in configs]
        if len(list(set(names))) != len(configs):
            raise ValueError(
                "Number of names does not match the number of config \
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
        required_fields = ["name", "model_type"]
        for field in required_fields:
            if field not in config:
                raise ValueError("Invalid config: %s missing" % field)

    def build_micromodel_from_config(
        self, config: Mapping[str, Any], force_rebuild=False
    ) -> AbstractMicromodel:
        """
        Fetch micromodel as specified in the input config.
        If the micromodel has not been built yet, this method will
        build the model, add it to the cache, and return the model.

        :param config: a micromodel configuration.
        :param force_rebuild: flag for forcing a re-build.
        :return: Instance of a micromodel.
        :raise KeyError: raised when an invalid model type is specified.
        """
        self._verify_config(config)
        model_type = config.get("model_type")
        model_name = get_model_name(config)

        try:
            model = MICROMODEL_FACTORY[model_type](
                model_name, **config.get("setup_args", {})
            )
        except KeyError as ex:
            raise KeyError("Invalid model type %s" % model_type) from ex

        if not force_rebuild and model_name in self.cache:
            return self.cache[model_name]

        print("Training %s" % model_name)
        model.build()
        model_path = config.get(
            "model_path", os.path.join(self.model_basepath, model_name)
        )
        model.save_model(model_path)
        print("%s saved to %s" % (model_name, model_path))
        self.cache[model_name] = model
        return self.cache[model_name]

    def build_all_micromodels(self) -> None:
        """
        Build all micromodels specified in self.configs.
        """
        for config in self.configs:
            if config.get("build", True):
                self.build_micromodel_from_config(config)

    def load_models(self, force_reload: bool = False) -> None:
        """
        Load all models into cache.
        """
        for config in self.configs:
            self._load_model(config, force_reload=force_reload)

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
        model = MICROMODEL_FACTORY[config["model_type"]](
            model_name, **config.get("setup_args", {})
        )
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

    def batch_infer_config(
        self, queries: List[str], config: Mapping[str, Any]
    ) -> Mapping[str, List[Any]]:
        """
        Run batch inference from the micromodel specified in config.
        :param queries: List of string utterance to run inference on.
        :param config: Micromodel configuration.
        :return: Mapping of micromodel name to the inference result.
        """
        # TODO: Preprocessing
        self._verify_config(config)
        model_name = get_model_name(config)
        model = self.cache.get(model_name)
        if not model:
            model = self._load_model(config)
        return {model_name: model.batch_infer(queries)}

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
