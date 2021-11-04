"""
Micromodels Factory
"""

from src.micromodels.svm import SVM
from src.micromodels.logic import LogicClassifier
from src.micromodels.bert_query import BertQuery
from src.micromodels.FastText import FastText

MICROMODEL_FACTORY = {
    "svm": SVM,
    "logic": LogicClassifier,
    "bert_query": BertQuery,
    "fasttext": FastText,
}
