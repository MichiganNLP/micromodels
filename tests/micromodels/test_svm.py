"""
Unittest for SVM micromodel
"""


import os
import unittest
from nltk.classify import SklearnClassifier
from src.micromodels.svm import SVM
from tests.micromodels import MicromodelUnittest


MM_BASE_PATH = os.environ.get("MM_HOME")


class TestSVM(MicromodelUnittest):
    """
    Testcases for Orchestrator
    """

    def __init__(self, methodName=""):
        super().__init__(methodName)

    def _set_micromodel(self, config):
        """
        Set micromodel.
        """
        self.micromodel = SVM(self.config["name"], **config["setup_args"])

    def tearDown(self):
        """
        Remove any models trained from training
        """
        model_path = self.config["model_path"]
        if os.path.isfile(model_path):
            print("Deleting %s" % model_path)
            os.remove(model_path)

    def initialize_config(self):
        """
        Return configuration for micromodel.
        """
        return {
            "model_type": "svm",
            "name": "test_svm",
            "model_path": os.path.join(self.model_path, "test_svm"),
            "setup_args": {
                "training_data_path": os.path.join(
                    self.data_path, "dog_vs_cat.json"
                ),
            },
        }

    def test_methods(self):
        """
        Tests the following methods in the following order:

        build(), save_model(), load_model(), run(), batch_run().
        """
        mm = self.micromodel

        mm.build()
        self.assertIsInstance(mm.svm_model, SklearnClassifier)

        model_path = self.config["model_path"]
        mm.save_model(model_path)
        self.assertTrue(os.path.isfile(model_path))

        mm.svm_model = None
        mm.load_model(model_path)
        self.assertIsInstance(mm.svm_model, SklearnClassifier)

        prediction = mm.run("Cute cat !")
        self.assertEqual(prediction, True)

        predictions = mm.batch_run(
            [
                ["This is a cat", "This is a dog"],
                ["testing dog", "testing cat"],
                ["testing cat", "cat one two", "dog one two"],
            ]
        )
        self.assertEqual(predictions[0], [0])
        self.assertEqual(predictions[1], [1])
        self.assertEqual(predictions[2], [0, 1])


if __name__ == "__main__":
    unittest.main()
