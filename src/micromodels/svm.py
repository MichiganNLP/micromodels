"""
SVM Micromodel.
"""

import pickle
import random
import json
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from src.micromodels.AbstractMicromodel import AbstractMicromodel
from src.utils import preprocess_list

random.seed(a=42)


class SVM(AbstractMicromodel):
    """
    SVM implementation of a micromodel.
    """

    def __init__(self, name):
        super().__init__(name)
        self.svm_model = None

    def _setup(self, config):
        """
        Setup stage - nothing to do for svm.
        """

    def train(self, training_data_path, **kwargs) -> None:
        """
        Train SVM.
        """
        with open(training_data_path, "r") as file_p:
            train_data = json.load(file_p)

        return self._train(train_data, kwargs)

    def _train(self, train_data, _train_args=None):
        """
        Train model
        :param train_data: dict of labels [utterances]
        """
        positives = train_data["true"]
        negatives = train_data["false"]
        if len(positives) <= 0 or len(negatives) <= 0:
            raise ValueError("Invalid training data")

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
        return self.svm_model

    def save_model(self, model_path):
        """
        Save model
        """
        if self.svm_model is None:
            raise ValueError("self.svm_model is None")

        with open(model_path, "wb") as file_p:
            pickle.dump(self.svm_model, file_p)

    def load_model(self, model_path):
        """
        Load model
        """
        with open(model_path, "rb") as file_p:
            self.svm_model = pickle.load(file_p)

    def _infer(self, query):
        """
        Infer model
        """
        if self.svm_model is None:
            raise ValueError("self.svm_model is None")

        tokens = {token: True for token in query.strip().split()}
        pred = self.svm_model.classify(tokens)
        return {"true": True, "false": False}[pred]

    def is_loaded(self):
        """
        Check if model is loaded.
        """
        return self.svm_model is not None


if __name__ == "__main__":
    test = SVM("testing")
    from datetime import datetime

    _model_path = "/home/andrew/micromodels/models/diagnosed_svm"
    start = datetime.now()
    test.load_model(_model_path)
    end = datetime.now()
    print("Loading took", end - start)
    start = datetime.now()
    print(test.infer("I was diagnosed with depression"))
    end = datetime.now()
    print("Inference took", end - start)
    print(test.infer("I have been diagnosed with depression."))
