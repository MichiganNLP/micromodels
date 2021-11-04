"""
Utility functions
"""

import sys
import string
from time import time
import numpy as np
import nltk
from pathos.multiprocessing import ProcessPool as Pool

_POS_TAGGER = nltk.tag.perceptron.PerceptronTagger()
_WORDNET_LEMMA = nltk.stem.WordNetLemmatizer()


def run_parallel_cpu(func, utterances):
    """
    Run func via multiprocessing.
    """
    pool = Pool()
    pool.restart()
    results = pool.imap(func, utterances)
    pool.close()
    pool.join()
    pool.terminate()

    return results


def parallelize(func, utterances, num_procs=4):
    """
    Parallelize lexical searches.
    :func: must return a list of idxs
    """

    def wrapper(_utterances):
        """
        _utterances is a list of tuples (idx, utterance)
        """
        _utts = [_utt[1] for _utt in _utterances]
        idxs = func(_utts)
        orig_idxs = np.array(_utterances)[idxs]
        orig_idxs = [int(_utt[0]) for _utt in orig_idxs]
        return orig_idxs

    if num_procs < 2:
        return func(utterances)
    if len(utterances) < 100:
        return func(utterances)

    chunk_size = len(utterances) // num_procs
    if chunk_size < 1:
        return func(utterances)

    utterances = list(enumerate(utterances))
    chunks = [
        utterances[i : i + chunk_size]
        for i in range(0, len(utterances), chunk_size)
    ]
    pool = Pool(num_procs)
    try:
        results = pool.amap(wrapper, chunks)
        while not results.ready():
            time.sleep(1)
        infer_results = results.get()
    except (KeyboardInterrupt, SystemExit):
        pool.terminate()
        sys.exit(1)

    idxs = []
    for idx_group in infer_results:
        idxs.extend(idx_group)
    idxs = sorted(idxs)
    return idxs


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


def remove_punctuation(text):
    """
    Remove punctuations in the text
    """
    return "".join((x for x in text if x not in string.punctuation))


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
    query = remove_punctuation(query)

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
