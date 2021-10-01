"""
Micromodel Abstract Class
"""
from typing import Mapping, Any, Optional

from src.utils import preprocess


class AbstractMicromodel:
    """
    Abstract classifier for deriving linguistic features.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def setup(self, config: Mapping[str, Any]) -> None:
        """
        Set up any necessary configurations per micromodel.
        The following properties are required for every micromodel:

        - name: Given name of the micromodel.
        - model_type: Type of algorithm used for the micromodel. (Ex: svm)
        - model_path: Filepath for saving or loading micromodel.
        - (Optional) data: Filepath to training data, if any. Note that
          not all micromodels need training data (Ex: logic micromodels).
        - (Optional) setup_args: Dictionary of micromodel specific parameters.
          For micromodel specific parameters, refer to their respective
          documentation.

        :param config: micromodel configuration.
        """
        raise NotImplementedError("setup() not implemented.")

    def _setup(self, config: Mapping[str, Any]) -> None:
        """
        Inner setup method.

        :param config: micromodel configuration.
        """

    def train(self, training_data_path: str) -> None:
        """
        Train a micromodel.

        :param training_data_path: filepath to training data.
            Note that the format of the training data depends on how
            the inner _train() method for each micromodel.
        """
        raise NotImplementedError("train() not implemented.")

    def infer(
        self, query: str, do_preprocess: Optional[bool] = True
    ) -> Mapping[str, Any]:
        """
        Run inference.

        :param query: string utterance to query.
        :param do_preprocess: flag to determine whether to preprocess query.
        :return: Mapping of micromodel name to inference results.
        """
        if do_preprocess:
            query = preprocess(query)
        return self._infer(query)

    def _infer(self, query: str) -> Mapping[str, Any]:
        """
        Inner infer function.

        :param query: string utterance to query.
        :return: Mapping of micromodel name to inference results.
        """
        raise NotImplementedError("_infer() not implemented.")

    def save_model(self, model_path: str) -> None:
        """
        Save model to file system.

        :param model_path: filepath to save model.
        """
        raise NotImplementedError("save_model() not implemented.")

    def load_model(self, model_path: str) -> None:
        """
        Load micromodel.

        :param model_path: filepath to load model from.
        """
        raise NotImplementedError("load_model() not implemented.")

    def is_loaded(self) -> bool:
        """
        Indicate whether model is loaded.
        """
        raise NotImplementedError("is_loaded() not implemented.")
