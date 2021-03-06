"""
Logic Micromodel.
"""
from typing import Mapping, Any, List, Callable

import dill
from src.micromodels.AbstractMicromodel import AbstractMicromodel
from src.micromodels.mm_utils import run_parallel


class LogicClassifier(AbstractMicromodel):
    """
    Logic-based classifier
    """

    def __init__(self, name: str, **kwargs) -> None:
        """
        kwargs:
        :param logic_func: (Callable), required.
            Inner logic function for micromodel.
        :param pool_size: (int), Optional, defaults to 4.
            Pool size for multiprocessing batch_run.
        """
        logic_func = kwargs.get("logic_func")
        if logic_func is None:
            raise ValueError(
                "Must pass in a callable function for 'logic_func'."
            )

        self.logic = logic_func
        self.pool_size = kwargs.get("pool_size", 4)
        super().__init__(name)

    def build(self) -> None:
        """
        No need to build logical classifiers. No-op method.
        """
        return

    def save_model(self, model_path: str) -> None:
        """
        Save model to file system. Uses dill to dump inner Callable function.

        :param model_path: Filepath to store model.
        """
        with open(model_path, "wb") as file_p:
            dill.dump(self.logic, file_p)

    def load_model(self, model_path: str) -> None:
        """
        Load model. Sets self.logic with a Callable instance.

        :param model_path: Filepath to load logic.
        """
        with open(model_path, "rb") as file_p:
            self.logic = dill.load(file_p)

    def _run(self, query: str) -> bool:
        """
        Inner run method. Calls self.logic method.

        :param query: String utterance to query.
        :return: Boolean result.
        """
        return self.logic(query)

    def _batch_run(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Batch run method.

        :param queries: List of query utterances.
        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        return run_parallel(self.logic, query_groups, self.pool_size)

    def is_loaded(self) -> bool:
        """
        Check if self.logic is set.
        :return: True if self.logic is set.
        """
        return self.logic is not None
