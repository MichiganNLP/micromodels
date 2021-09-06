"""
Keyword lookup
"""

from nltk import tokenize
from FeatureClassifier import FeatureClassifier


class KeywordLookup(FeatureClassifier):
    """
    Keyword lookup Feature Classifier
    """

    def __init__(self):
        super(KeywordLookup, self).__init__()
        self.key_words = ["worthless", "totally", "total", "complete"]

    def _train(self, training_data):
        """
        No training needed
        """

    def save_model(self, model_path):
        """
        No saving required
        """

    def load_model(self, model_path):
        """
        No saving required
        """

    def _infer(self, query):
        """
        Get sentiment of query
        """
        query = tokenize.word_tokenize(query)
        return any(token in query for token in self.key_words)

    def test(self, testing_dataset):
        """
        Test sentiment analyzer
        """
        return None

    def is_loaded(self):
        """
        check if model is loaded.
        """
        return True
