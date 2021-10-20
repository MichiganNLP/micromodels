"""
Unittest for logic micromodel
"""
import os
import unittest
from src.micromodels.logic import LogicClassifier
from tests.micromodels import MicromodelUnittest


MM_BASE_PATH = os.environ.get("MM_HOME")


class TestLogicClassifier(MicromodelUnittest):
    """
    Testcases for Orchestrator
    """

    def __init__(self, methodName=""):
        super().__init__(methodName)

    def _set_micromodel(self, config):
        """
        Set micromodel.
        """
        self.micromodel = LogicClassifier(
            self.config["name"], **config["setup_args"]
        )

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

        def _logic(utterance: str) -> bool:
            """
            Logic to be used by the logic-micromodel.
            """
            return "cat" in utterance.lower()

        return {
            "model_type": "logic",
            "name": "test_logic",
            "model_path": os.path.join(self.model_path, "test_logic"),
            "setup_args": {"logic_func": _logic},
        }

    def test_methods(self):
        """
        Tests the following methods in the following order:

        save_model(), load_model(), run(), batch_run().
        """
        mm = self.micromodel

        self.assertTrue(callable(mm.logic))

        model_path = self.config["model_path"]
        mm.save_model(model_path)
        self.assertTrue(os.path.isfile(model_path))

        mm.logic = None
        mm.load_model(model_path)
        self.assertTrue(callable(mm.logic))

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
