"""
Parent class for micromodel unit tests.
"""
import os
import unittest


MM_BASE_PATH = os.environ.get("MM_HOME")

class MicromodelUnittest(unittest.TestCase):
    """
    Parent class for all micromodel unittests.
    """
    def setUp(self):
        self.base_path = os.path.join(MM_BASE_PATH, "tests")
        self.data_path = os.path.join(self.base_path, "test_data")
        self.model_path = os.path.join(self.base_path, "test_models")

        self.config = self.initialize_config()
        self._set_micromodel(self.config)

    def _set_micromodel(self, config):
        """
        Set micromodel.
        """
        raise NotImplementedError("_set_micromodel() is not implemented.")

    def initialize_config(self):
        """
        Return configuration for micromodel.
        """
        raise NotImplementedError("initialize_config() is not implemented.")
