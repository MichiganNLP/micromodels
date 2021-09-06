"""
Logic based classification
"""

from typing import List, Mapping, Any
import dill
from FeatureClassifier import FeatureClassifier
from snorkel_queries.helpers import (
    parallelize,
)


class LogicClassifier(FeatureClassifier):
    """
    Logic based classification
    """

    def __init__(self, name):
        self.logic = None
        self.parallelize = False
        super(LogicClassifier, self).__init__(name)

    def setup(self, config):
        """
        setup stage
        """
        self.logic = config["logic_func"]
        self.parallelize = config.get("parallelize", False)

    def _train(
        self, train_data: List[str], train_args: Mapping[Any, Any] = None
    ) -> None:
        """
        No need to train logical classifiers.
        """
        pass

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
            preds = parallelize(self.logic, queries)
        else:
            preds = self.logic(queries)
        idxs = [
            idx for idx, pred in enumerate(preds)
            if pred
        ]
        return idxs

if __name__ == "__main__":
    clf = LogicClassifier("testing")
    def foo(utt):
        return "depress" in utt
    clf.setup({"logic_func": foo})
    clf.save_model("testing_logic_clf")

    clf = LogicClassifier("Testing2")
    clf.load_model("testing_logic_clf")


    print(clf.infer("I am depressed"))
    print(clf.infer("I am happy"))


