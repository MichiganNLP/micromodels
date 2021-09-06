"""
Feature Classifier Factory
"""

from feature_classifiers.batch_snorkel_classifier import BatchSnorkelClassifier
from feature_classifiers.svm import SVM
from feature_classifiers.logic import LogicClassifier

FEATURE_CLASSIFIER_FACTORY = {
    "svm": SVM,
    "snorkel": BatchSnorkelClassifier,
    "logic": LogicClassifier,
}
