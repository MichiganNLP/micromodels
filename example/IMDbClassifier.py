"""
Example Task-Specific Classifier.
Classify IMDb movie review sentiments.

For details on the task, see https://ai.stanford.edu/~amaas/data/sentiment/
"""
from typing import List, Mapping, Tuple, Any

import json
import random
from nltk import tokenize
from src.TaskClassifier import TaskClassifier


class IMDbClassifier(TaskClassifier):
    """
    Classifier for IMDb movie reviews.
    """

    positive_value = "positive"
    negative_value = "negative"

    def __init__(
        self, mm_basepath: str, configs: List[Mapping[str, Any]] = None
    ) -> None:
        """
        Initialize IMDB sentiment classifier
        """
        super().__init__(mm_basepath, configs)

    def load_data(
        self, data_path: str = None, **kwargs
    ) -> Tuple[List[Tuple[List[str], str]], List[Tuple[List[str], str]]]:
        """
        Load IMDB data, including sentence-tokenizing the input text and
        splitting the data into train and test splits.

        :param data_path: Filepath to imdb data in json form.
        :param train_ratio: ratio of data to use for training.
        :return: A pair of lists, one for training and one for test.
            Each list contains a tuple pair, which represents a single
            instance of data point. The first element is a list of sentences
            and the second element is the label.
        """
        train_ratio = kwargs.get("train_ratio", 0.7)
        with open(data_path, "r") as file_p:
            data = json.load(file_p)

        tokenized = {"positive": [], "negative": []}
        for label, sentences in data.items():
            for sentence in sentences:
                sentence = sentence.lower()
                tokenized[label].append(tokenize.sent_tokenize(sentence))
        positives = tokenized["positive"]
        negatives = tokenized["negative"]
        random.shuffle(positives)
        random.shuffle(negatives)
        train_size = int(len(positives) * train_ratio)
        train = [
            (utterances, "positive") for utterances in positives[:train_size]
        ] + [(utterances, "negative") for utterances in negatives[:train_size]]

        test = [
            (utterances, "positive") for utterances in positives[train_size:]
        ] + [(utterances, "negative") for utterances in negatives[train_size:]]
        return train, test
