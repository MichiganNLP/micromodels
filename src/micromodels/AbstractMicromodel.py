"""
Feature Classifier Abstract Class
"""
from typing import Mapping, Any

from src.utils import preprocess


class AbstractMicromodel:
    """
    Abstract classifier for deriving linguistic features.
    """

    def __init__(self, name: str):
        self.name = name
        self.trainable = None

    def setup(self, config: Mapping[str, Any]):
        """
        Set up any necessary configurations per micromodel.
        """
        return self._setup(config)

    def _setup(self, config: Mapping[str, Any]):
        """
        Inner setup method.
        """
        raise NotImplementedError("setup() not implemented.")

    def train(
        self, training_data_path: str, train_args: Mapping[str, Any] = None
    ) -> None:
        """
        Train a micromodel.
        """
        raise NotImplementedError("train() not implemented.")

    def save_model(self, model_path: str):
        """
        Save model to file system
        """
        raise NotImplementedError("save_model() not implemented.")

    def load_model(self, model_path: str):
        """
        Load micromodel.
        """
        raise NotImplementedError("load_model() not implemented.")

    def infer(self, query: str, do_preprocess: bool = True):
        """
        Infer classifier
        """
        if do_preprocess:
            query = preprocess(query)
            # query = standard_preprocess(query)
        return self._infer(query)

    def _infer(self, query: str):
        """
        Inner infer function
        """
        raise NotImplementedError("_infer() not implemented.")

    def is_loaded(self) -> bool:
        """
        Indicate whether model is loaded.
        """
        raise NotImplementedError("is_loaded() not implemented.")
