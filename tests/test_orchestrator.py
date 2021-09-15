"""
Unittests for orchestrator
"""

import os
import unittest
from src.Orchestrator import Orchestrator


MM_BASE_PATH = os.environ.get("MM_HOME")


class TestOrchestrator(unittest.TestCase):
    """
    Testcases for Orchestrator
    """

    def setUp(self):
        self.base_path = os.path.join(MM_BASE_PATH, "tests")
        self.data_path = os.path.join(self.base_path, "test_data")
        self.model_path = os.path.join(self.base_path, "test_models")

        self.configs = self.initialize_config()
        self.orchestrator = Orchestrator(self.model_path)
        self.orchestrator.set_configs(self.configs)

    def tearDown(self):
        """
        Remove any models trained from training
        """
        micromodel_files = [config["model_path"] for config in self.configs]
        for filepath in micromodel_files:
            if os.path.isfile(filepath):
                print("Deleting %s" % filepath)
                os.remove(filepath)

    def initialize_config(self):
        """
        Initialize configuration for Orchestrator.
        The configurations should include every type of micromodel.
        TODO: Add test for snorkel.
        """

        def _logic(utterance: str) -> bool:
            """
            Logic to be used by the logic-micromodel.
            """
            return "depress" in utterance.lower()

        configs = [
            {
                "model_type": "svm",
                "data": os.path.join(self.data_path, "gender.json"),
                "feature_name": "gender",
                "model_path": os.path.join(self.model_path, "gender_svm"),
            },
            {
                "model_type": "logic",
                "feature_name": "depression",
                "model_path": os.path.join(self.model_path, "depression_logic"),
                "setup_args": {"logic_func": _logic},
            },
        ]
        return configs

    def test_train_all(self):
        """
        Test training all configurations
        """
        self.orchestrator.train_all()
        micromodel_names = [
            "%s_%s" % (config["feature_name"], config["model_type"])
            for config in self.configs
        ]

        for micromodel_name in micromodel_names:
            self.assertIn(micromodel_name, self.orchestrator.cache)


if __name__ == "__main__":
    unittest.main()
