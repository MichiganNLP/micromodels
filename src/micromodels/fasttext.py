"""
FastText Micromodel.
"""
from typing import Mapping, Any, List

import pickle
import random
import json
import tempfile
import fasttext
from src.micromodels.AbstractMicromodel import AbstractMicromodel
from src.micromodels.mm_utils import run_parallel
from src.utils import preprocess_list, preprocess 

random.seed(a=42)


class FastText(AbstractMicromodel):
    """
    Fasttext implementation of a micromodel.
    """

    def __init__(self, name: str, **kwargs) -> None:
        """
        kwargs:
        :param training_data_path: (str), required.
            File path to training data file (json).
        :param pool_size: (int), Optional, defaults to 4.
            Pool size for multiprocessing batch_run.
        """
        super().__init__(name)
        self.fasttext_model = None
        if (
            "training_data_path" not in kwargs
            and "training_data" not in kwargs
        ):
            raise ValueError(
                "FastText Micromodel requires either 'training_data' or 'training_data_path' arguments."
            )

        self.training_data_path = kwargs.get("training_data_path")
        self.training_data = kwargs.get("training_data")
        if (
            self.training_data_path is not None
            and self.training_data is not None
        ):
            raise ValueError(
                "FastText Micromodel can't have both 'training_data_path' and 'training_data' set."
            )
        self.pool_size = kwargs.get("pool_size", 4)

    def build(self) -> None:
        """
        Build a FastText micromodel.

        :param training_data_path: Filepath to training data.
        """
        if self.training_data_path is None and self.training_data is None:
            raise RuntimeError(
                "self.training_data_path and self.training_data is not set!"
            )

        if self.training_data_path is not None:
            with open(self.training_data_path, "r") as file_p:
                train_data = json.load(file_p)
        elif self.training_data is not None:
            train_data = self.training_data
        else:
            raise RuntimeError("Unexpected branch.")

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

        fp = tempfile.NamedTemporaryFile()
        with open(fp.name, "w") as file_p:
            for label, utterances in preprocessed.items():
                for utterance in utterances:
                    file_p.write("__label__%s %s\n" % (label, utterance))

        self.fasttext_model = fasttext.train_supervised(fp.name, epoch=20)
        fp.close()


    def save_model(self, model_path: str) -> None:
        """
        Dumps Fasttext model to file, using pickle.

        :param model_path: Filepath to save Fasttext model.
        """
        if self.fasttext_model is None:
            raise ValueError("self.fasttext_model is None")
        self.fasttext_model.save_model(model_path)

    def load_model(self, model_path: str) -> None:
        """
        Loads Fasttext model, and sets self.fasttext_model.
        Expects a pickled file.

        :param model_path: Filepath for loading Fasttext model.
        """
        self.fasttext_model = fasttext.load_model(model_path)

    def _run(self, query: str) -> bool:
        """
        Innfer run method. Calls self.fasttext_model.classify().

        :param query: Utterance to classify.
        :return: Inference result, either True or False.
        """
        if self.fasttext_model is None:
            raise ValueError("self.fasttext_model is None")

        query = preprocess(query)
        prediction = self.fasttext_model.predict(query)[0]
        if not prediction:
            breakpoint()
            return None
        prediction = prediction[0][len("__label__"):]
        return {"true": True, "false": False}[prediction]

    def _batch_run(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Batch run method.

        :param queries: List of query utterances.
        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        return run_parallel(self._run, query_groups, self.pool_size)

    def is_loaded(self) -> bool:
        """
        Check if self.fasttext_model is set.
        :return: True if self.fasttext_model is set.
        """
        return self.fasttext_model is not None
