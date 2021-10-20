"""
SVM Micromodel.
"""
from typing import Mapping, Any, List

import pickle
import random
import json
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from src.micromodels.AbstractMicromodel import AbstractMicromodel
from src.micromodels.mm_utils import run_parallel
from src.utils import preprocess_list

random.seed(a=42)


class SVM(AbstractMicromodel):
    """
    SVM implementation of a micromodel.
    """

    def __init__(self, name: str, **kwargs) -> None:
        """
        kwargs:
        :param training_data_path: (str), required.
            File path to training data file (json).
        :param pool_size: (int), Optional, defaults to 4.
            Pool size for multiprocessing batch_infer.
        """
        super().__init__(name)
        self.svm_model = None
        if "training_data_path" not in kwargs:
            raise ValueError("SVM requires 'training_data_path' argument.")

        self.training_data_path = kwargs["training_data_path"]
        self.pool_size = kwargs.get("pool_size", 4)

    def build(self) -> None:
        """
        Build a SVM micromodel.

        :param training_data_path: Filepath to training data.
        """
        if self.training_data_path is None:
            raise RuntimeError("self.training_data_path is not set!")
        with open(self.training_data_path, "r") as file_p:
            train_data = json.load(file_p)
        self._build(train_data)

    def _build(self, train_data: Mapping[str, List[str]]) -> None:
        """
        Inner train method.

        :param train_data: dict of labels [utterances]
        :return: Instance of a SklearnClassifier.
        """
        if "true" not in train_data or "false" not in train_data:
            raise ValueError(
                "Invalid training data: Must have keys 'true' and 'false'."
            )
        positives = train_data["true"]
        negatives = train_data["false"]
        if len(positives) <= 0:
            raise ValueError(
                "Invalid training data, could not find positive examples."
            )
        if len(negatives) <= 0:
            raise ValueError(
                "Invalid training data, could not find negative examples."
            )

        if len(negatives) > len(positives) * 2:
            train_data["false"] = random.sample(negatives, len(positives) * 2)

        preprocessed = {label: [] for label in train_data.keys()}
        for label, utterances in train_data.items():
            preprocessed[label] = preprocess_list(utterances)

        formatted = []
        for label, utterances in preprocessed.items():
            for utterance in utterances:
                words = {token: True for token in utterance.split()}
                formatted.append((words, label))

        self.svm_model = SklearnClassifier(SVC(kernel="linear"))
        self.svm_model.train(formatted)

    def save_model(self, model_path: str) -> None:
        """
        Dumps SVM model to file, using pickle.

        :param model_path: Filepath to save SVM model.
        """
        if self.svm_model is None:
            raise ValueError("self.svm_model is None")

        with open(model_path, "wb") as file_p:
            pickle.dump(self.svm_model, file_p)

    def load_model(self, model_path: str) -> None:
        """
        Loads SVM model, and sets self.svm_model.
        Expects a pickled file.

        :param model_path: Filepath for loading SVM model.
        """
        with open(model_path, "rb") as file_p:
            self.svm_model = pickle.load(file_p)

    def _infer(self, query: str) -> bool:
        """
        Innfer infer method. Calls self.svm_model.classify().

        :param query: Utterance to classify.
        :return: Inference result, either True or False.
        """
        if self.svm_model is None:
            raise ValueError("self.svm_model is None")

        tokens = {token: True for token in query.strip().split()}
        pred = self.svm_model.classify(tokens)
        return {"true": True, "false": False}[pred]

    def _batch_infer(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Batch inference method.

        :param queries: List of query utterances.
        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        return run_parallel(self._infer, query_groups, self.pool_size)

    def is_loaded(self) -> bool:
        """
        Check if self.svm_model is set.
        :return: True if self.svm_model is set.
        """
        return self.svm_model is not None
