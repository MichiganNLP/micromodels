"""
Feature Classifier Factory
"""

from src.micromodels.svm import SVM
from src.micromodels.logic import LogicClassifier
from src.micromodels.bert_query import BertQuery

FEATURE_CLASSIFIER_FACTORY = {
    "svm": SVM,
    "logic": LogicClassifier,
    "bert_query": BertQuery,
}
