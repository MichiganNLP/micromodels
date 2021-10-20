"""
Unittest for Bert-Query micromodel
"""
import os
import shutil
import unittest
import numpy as np
from sentence_transformers import SentenceTransformer
from src.micromodels.bert_query import BertQuery
from tests.micromodels import MicromodelUnittest


class TestBertQuery(MicromodelUnittest):
    """
    Testcases for Orchestrator
    """

    def __init__(self, methodName=""):
        super().__init__(methodName)

    def _set_micromodel(self, config):
        """
        Set micromodel.
        """
        self.micromodel = BertQuery(
            self.config["name"], **config["setup_args"]
        )

    def tearDown(self):
        """
        Remove any models trained from training
        """
        model_path = self.config["model_path"]
        if os.path.isdir(model_path):
            print("Deleting %s" % model_path)
            shutil.rmtree(model_path)

    def initialize_config(self):
        """
        Return configuration for micromodel.
        """
        return {
            "model_type": "bert_query",
            "name": "test_bert_query",
            "model_path": os.path.join(self.model_path, "test_bert_query"),
            "setup_args": {
                "threshold": 0.8,
                "seed": [
                    "This is a cat",
                    "Arya is a hungry cat.",
                ],
                "infer_config": {
                    "k": 2,
                    "segment_config": {"window_size": 5, "step_size": 3},
                },
            },
        }

    def test_methods(self):
        """
        Tests the following methods in the following order:

        save_model(), load_model(), run(), batch_run().
        """
        mm = self.micromodel

        mm.build()
        self.assertIsInstance(mm.bert, SentenceTransformer)

        model_path = self.config["model_path"]
        mm.save_model(model_path)
        self.assertTrue(os.path.isdir(model_path))

        mm.seed = None
        mm.seed_encoding = None
        mm.load_model(model_path)
        self.assertIsInstance(mm.seed, list)
        self.assertIsInstance(mm.seed_encoding, np.ndarray)

        prediction = mm.run("That is a cat")
        self.assertEqual(prediction, True)

        predictions = mm.batch_run(
            [
                ["This is a cat", "This is a dog"],
                ["testing dog", "that is a cat"],
                ["is this a cat", "that is probably a cat", "dog one two"],
            ]
        )
        self.assertEqual(predictions[0], [0])
        self.assertEqual(predictions[1], [1])
        self.assertEqual(predictions[2], [0, 1])


if __name__ == "__main__":
    unittest.main()
