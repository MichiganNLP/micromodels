"""
Micromodel Abstract Class
"""
from typing import Mapping, Any, Optional, List, Tuple

from src.utils import preprocess


class AbstractMicromodel:
    """
    Abstract classifier for deriving linguistic features.
    """
    def __init__(self, name: str, **kwargs) -> None:
        """
        See documentation for each micromodel for details on kwargs.
        """
        self.name = name

    def build(self) -> None:
        """
        Train a micromodel.
        """
        raise NotImplementedError("build() not implemented.")

    def infer(self, query: str, do_preprocess: Optional[bool] = True) -> Any:
        """
        Run inference.

        :param query: string utterance to query.
        :param do_preprocess: flag to determine whether to preprocess query.
        :return: Mapping of micromodel name to inference results.
        """
        if do_preprocess:
            query = preprocess(query)
        return self._infer(query)

    def _infer(self, query: str) -> Any:
        """
        Inner infer function.

        :param query: string utterance to query.
        :return: Inference results.
        """
        raise NotImplementedError("_infer() not implemented.")

    def batch_infer(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Batch inference.

        :param queries: List of query objects (indices and list of strings).
        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        # TODO: preprocessing
        return self._batch_infer(query_groups)

    def _batch_infer(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Inner batch inference.

        :param queries: List of query objects (indices and list of strings).
        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        raise NotImplementedError("_batch_infer() not implemented.")

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
