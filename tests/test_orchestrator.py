"""
Unittests for orchestrator
"""
import os
import shutil
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
        self.orchestrator = Orchestrator(self.model_path, self.configs)

    def tearDown(self):
        """
        Remove any models trained from training
        """
        micromodel_files = [config["model_path"] for config in self.configs]
        for _path in micromodel_files:
            if os.path.isfile(_path):
                print("Deleting %s" % _path)
                os.remove(_path)
            if os.path.isdir(_path):
                print("Deleting %s" % _path)
                shutil.rmtree(_path)

    def initialize_config(self):
        """
        Initialize configuration for Orchestrator.
        The configurations should include every type of micromodel.
        """

        def _logic(utterance: str) -> bool:
            """
            Logic to be used by the logic-micromodel.
            """
            return "test" in utterance.lower()

        configs = [
            {
                "model_type": "svm",
                "name": "test_svm",
                "model_path": os.path.join(self.model_path, "test_svm"),
                "setup_args": {
                    "training_data_path": os.path.join(
                        self.data_path, "dog_vs_cat.json"
                    ),
                },
            },
            {
                "model_type": "logic",
                "name": "test_logic",
                "model_path": os.path.join(self.model_path, "test_logic"),
                "setup_args": {"logic_func": _logic},
            },
            {
                "model_type": "bert_query",
                "name": "test_bert_query",
                "model_path": os.path.join(self.model_path, "test_bert_query"),
                "setup_args": {
                    "threshold": 0.8,
                    "seed": [
                        "This is a test",
                        "Arya is a hungry cat.",
                    ],
                    "infer_config": {
                        "k": 2,
                        "segment_config": {"window_size": 5, "step_size": 3},
                    },
                },
            },
        ]
        return configs

    @property
    def micromodel_names(self):
        """
        Get names of micromodels according to self.configs
        """
        return [config["name"] for config in self.configs]

    def test_build_micromodels(self):
        """
        Test training all configurations
        """
        self.orchestrator.build_micromodels()
        for micromodel_name in self.micromodel_names:
            self.assertIn(micromodel_name, self.orchestrator.cache)

    def test_load_micromodels(self):
        """
        Test loading of models.
        """
        self.orchestrator.build_micromodels()
        self.orchestrator.flush_cache()
        self.orchestrator.load_micromodels()
        for micromodel_name in self.micromodel_names:
            self.assertIn(micromodel_name, self.orchestrator.cache)

    def test_run_micromodels(self):
        """
        Test infering all micromodels.
        """
        self.orchestrator.build_micromodels()
        inference = self.orchestrator.run_micromodels("cat says meow .")
        for micromodel in self.micromodel_names:
            self.assertIsInstance(inference[micromodel], bool)
            if micromodel.endswith("svm"):
                self.assertEqual(inference[micromodel], True)

        inference = self.orchestrator.run_micromodels("This is a test.")
        for micromodel in self.micromodel_names:
            self.assertIsInstance(inference[micromodel], bool)
            if micromodel.endswith("logic") or micromodel.endswith(
                "bert_query"
            ):
                self.assertEqual(inference[micromodel], True)


if __name__ == "__main__":
    unittest.main()
