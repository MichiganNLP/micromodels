"""
Logic based classification
"""
from typing import Mapping, Any
import dill
from src.micromodels.AbstractMicromodel import AbstractMicromodel
from src.utils import run_parallel_cpu


class LogicClassifier(AbstractMicromodel):
    """
    Logic based classification
    """

    def __init__(self, name):
        self.logic = None
        self.parallelize = False
        super().__init__(name)

    def _setup(self, config):
        """
        setup stage
        """
        self.logic = config["logic_func"]
        self.parallelize = config.get("parallelize", False)

    def train(
        self, training_data_path: str, train_args: Mapping[str, Any] = None
    ) -> None:
        """
        No need to train logical classifiers.
        """

    def save_model(self, model_path):
        """
        Save model
        """
        with open(model_path, "wb") as file_p:
            dill.dump(self.logic, file_p)

    def load_model(self, model_path):
        """
        Load model
        """
        with open(model_path, "rb") as file_p:
            self.logic = dill.load(file_p)

    def _infer(self, query):
        """
        infer
        """
        return self.logic(query)

    def is_loaded(self):
        """
        Check if snorket_query is set.
        """
        return self.logic is not None

    def batch_infer(self, queries):
        """
        batch inference
        """
        if self.parallelize:
            preds = run_parallel_cpu(self.logic, queries)
        else:
            preds = [self.logic(query) for query in queries]
        return preds


if __name__ == "__main__":
    clf = LogicClassifier("testing")

    def _foo(utt):
        """ Dummy logic func """
        return "depress" in utt

    clf.setup({"logic_func": _foo, "parallelize": True})
    clf.save_model("testing_logic_clf")

    clf = LogicClassifier("Testing2")
    clf.load_model("testing_logic_clf")

    print(clf.infer("I am depressed"))
    print(clf.infer("I am happy"))

    print(clf.batch_infer(["I am depressed", "I am happy"]))
