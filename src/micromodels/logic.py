"""
Logic Micromodel.
"""
from typing import Mapping, Any, List, Tuple

import sys
import time
import dill
from pathos.multiprocessing import ProcessingPool as Pool
from src.micromodels.AbstractMicromodel import AbstractMicromodel
from src.micromodels.mm_utils import run_parallel


class LogicClassifier(AbstractMicromodel):
    """
    Logic-based classifier
    """

    def __init__(self, name: str, **kwargs) -> None:
        self.logic = None
        self.parallelize = False
        self.pool_size = kwargs.get("pool_size", 4)
        super().__init__(name)

    def setup(self, config: Mapping[str, Any]) -> None:
        """
        The following properties are required for a logic micromodel:

        - logic_func: Callable function with the actual logic implementation.
        - (Optional) parallelize: Boolean value for applying the current
          micromodel on the input data in parallel.
        """
        self.logic = config["logic_func"]
        self.parallelize = config.get("parallelize", False)

    def train(self, training_data_path: str) -> None:
        """
        No need to train logical classifiers. No-op method.

        :param training_data_path: Filepath to training data.
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

    def _infer(self, query: str) -> bool:
        """
        Inner infer method. Calls self.logic method.

        :param query: String utterance to query.
        :return: Boolean result.
        """
        return self.logic(query)

    def _batch_infer(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Batch inference method.

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
