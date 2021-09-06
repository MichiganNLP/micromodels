"""
Utility functions
"""

import os
import csv
import string
import nltk
from pathos.multiprocessing import ProcessPool as Pool

_POS_TAGGER = nltk.tag.perceptron.PerceptronTagger()
_WORDNET_LEMMA = nltk.stem.WordNetLemmatizer()


def run_parallel_cpu(func, utterances):
    """
    Run condition_func via multiprocessing.
    """
    pool = Pool()
    pool.restart()
    results = pool.imap(func, utterances)
    pool.close()
    pool.join()
    pool.terminate()

    return results


def _remove_punctuation(text):
    """
    Remove punctuation in the text
    """
    return "".join((x for x in text if x not in string.punctuation))


def _pos_tagging(text):
    """
    tag POS
    """
    return _POS_TAGGER.tag(text)


def lemmatize(token, pos):
    """
    lemmatize tokens
    """
    return _WORDNET_LEMMA.lemmatize(token, pos=pos)


def preprocess_list(utterances):
    """
    Preprocess a list of utterances
    """
    return list(run_parallel_cpu(preprocess, utterances))


def preprocess(query):
    """
    Preprocess the query with following steps:
     - to all lowercase
     - expand contraction
     - remove punctuation
     - part-of-speech tagging
     - lemmatize all words
    """
    query = query.lower()
    query = _remove_punctuation(query)

    pos_tag = _pos_tagging(query.split())
    preprocessed = []

    for word, tag in pos_tag:
        if tag.startswith("NN"):
            word = lemmatize(word, pos="n")
        elif tag.startswith("VB"):
            word = lemmatize(word, pos="v")
        elif tag.startswith("JJ"):
            word = lemmatize(word, pos="a")
        elif tag.startswith("RB"):
            word = lemmatize(word, pos="r")

        if word.isdigit():
            word = "__DIG__" * len(word)
        preprocessed.append(word)

    return " ".join(preprocessed)

def expand_contraction(utterance):
    """
    Expand contraction
    """
    return _CONTRACTION_EXPANDER.expand(utterance)


class ContractionExpander(object):
    """
    Contraction Expander
    """

    def __init__(self):
        """
        Reads the expansion information from the csv and caches the data.
        """
        self.expansions = {}
        csvpath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "expansions.csv"
        )
        with open(csvpath) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.expansions[row["contraction"]] = row["expansion"]

        # Make another verson of the dictionay in the case the contractions
        # are missing apostrophes.
        self.no_apos_expansions = {}
        for key in self.expansions:
            if key in ["we'll", "i'll"]:
                continue
            self.no_apos_expansions[remove_apostrophe(key)] = self.expansions[
                key
            ]

    def expand(self, words):
        """
        Expands all contractions in a string of words to their english
        equivalent expansions.
        Also takes into account capitalizatin and lack of apostrophes
        in the contraction.
        """

        # Remove all punctuation other_than...
        #import pdb
        #pdb.set_trace()
        other_than = ["'", "$", "."]
        punctuation = "".join(
            (x for x in string.punctuation if x not in other_than)
        )
        words = "".join((x for x in words if x not in punctuation))

        # split on all whitespace.
        word_list = words.split()
        new_word_list = []
        for word in word_list:
            # Remember if the word is capitalized or not.
            is_capitalized = word[0].isupper()
            word = word.lower()
            if word in self.expansions:
                word = self.expansions[word]
            elif word in self.no_apos_expansions:
                word = self.no_apos_expansions[word]

            if is_capitalized:
                word = word.capitalize()

            new_word_list.append(word)

        # Remake the original string with the expanded words.
        return_str = ""
        for word in new_word_list:
            return_str += word + " "

        # Remove the trailing whitespace.
        return return_str[:-1]


def remove_apostrophe(word):
    """
    Removes all instances of "'" in a word and returns the resulting word.
    """

    while "'" in word:
        index = word.find("'")
        word = word[:index] + word[index + 1 :]

    return word


_CONTRACTION_EXPANDER = ContractionExpander()
