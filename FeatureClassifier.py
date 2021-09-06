"""
Feature Classifier Abstract Class
"""
from typing import List, Mapping, Any

import json
from snorkel_engine.preprocessors import standard_preprocess
from utils import preprocess


class FeatureClassifier:
    """
    Abstract classifier for deriving features of Linguistic Properties
    """

    def __init__(self, name):
        """
        Init
        """
        self.name = name
        self.trainable = None

    def setup(self, config):
        """
        Set up any necessary configurations
        """
        pass

    def train(self, training_data_path, train_args=None):
        """
        Train a FeatureClassifier
        """
        with open(training_data_path, "r") as file_p:
            train_data = json.load(file_p)

        train_args = train_args or {}
        return self._train(train_data, train_args)

    def train_tuples(self, training_data, train_args=None):
        """
        Train a classifier where training_data is given
        in a format of [(utterance, label), ...].
        """
        labels = set(instance[1] for instance in training_data)
        preprocessed = {label: [] for label in labels}
        for (utterance, label) in training_data:
            utterance = preprocess(utterance)
            preprocessed[label].append(utterance)
        return self._train(preprocessed, train_args)

    def _train(
        self, training_data: List[str], train_args: Mapping[Any, Any]
    ) -> None:
        """
        Inner train method
        """
        raise NotImplementedError("_train() not implemented.")

    def save_model(self, model_path):
        """
        Save model to file system
        """
        raise NotImplementedError("save_model() not implemented.")

    def load_model(self, model_path):
        """
        Load FeatureClassifier model
        """
        raise NotImplementedError("load_model() not implemented.")

    def test(self, testing_dataset):
        """
        Test classifier
        """
        raise NotImplementedError("test() not implemented.")

    def test_tuples(self, test_dataset):
        """
        Test classifier, where data is in the format of
        [(utterance, label), ...]
        """
        total_count = len(test_dataset)
        num_correct = 0
        for (utterance, label) in test_dataset:
            pred = self.infer(utterance)
            if pred == label:
                num_correct += 1
            else:
                print("--")
                print(utterance)
                print("groundtruth:", label)
                print("prediction :", pred)
                print("--")
        print("num_correct", num_correct)
        print("total_count", total_count)
        return num_correct / total_count

    def infer(self, query, do_preprocess=True):
        """
        Infer classifier
        """
        if do_preprocess:
            #query = preprocess(query)
            query = standard_preprocess(query)
        return self._infer(query)

    def _infer(self, query):
        raise NotImplementedError("_infer() not implemented.")

    def is_loaded(self):
        """
        Return boolean on whether model is loaded.
        """
