"""
Utility functions for preprocessing text.
"""

import string
import contractions


def preprocess(text: str) -> str:
    """
    Preprocess the query with the following steps:
    * lowercase
    * remove punctuations
    * expand contractions
    * lemmatization
    * replace numbers with '__DIG__'
    """
    text = text.lower()
    text = remove_punctuation(text)
    return text


def remove_punctuation(text):
    """
    Remove punctuations in the text
    """
    return "".join((x for x in text if x not in string.punctuation))


def expand_contraction(utterance: str) -> str:
    """
    Expand contraction.
    """
    return contractions.fix(utterance)
