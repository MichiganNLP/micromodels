"""
Example of a task-specific classifier.
This example experiment applies a list of micromodels
on the IMDB sentiment analysis task.

For details on the task, see https://ai.stanford.edu/~amaas/data/sentiment/
"""
from typing import List, Mapping, Tuple


import os
import json
import random
from nltk import tokenize
from src.TaskClassifier import TaskClassifier


def load_data(
    data_path: str = None, **kwargs
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


def get_configs():
    """
    Setup micromodel configurations.
    """
    positive_keywords = ["good", "great", "wonderful", "beautiful"]

    def _positive_keyword_lookup(utterance):
        """ inner logic """
        return any(keyword in utterance for keyword in positive_keywords)

    mm_base_path = os.environ.get("MM_HOME")
    models_basepath = os.path.join(mm_base_path, "models")

    imdb_data_dir = os.path.join(mm_base_path, "example/data")
    imdb_data_path = os.path.join(imdb_data_dir, "imdb_dataset.json")
    svm_data_path = os.path.join(imdb_data_dir, "svm_train_data.json")

    configs = [
        {
            "model_type": "logic",
            "name": "good_logic",
            "model_path": os.path.join(models_basepath, "good_keyword_logic"),
            "setup_args": {"logic_func": _positive_keyword_lookup},
        },
        {
            "model_type": "svm",
            "name": "film_sentiment_svm",
            "model_path": os.path.join(models_basepath, "film_sentiment_svm"),
            "setup_args": {"training_data_path": svm_data_path},
        },
        {
            "model_type": "bert_query",
            "name": "positive_sentiment_bert_query",
            "model_path": os.path.join(
                models_basepath, "positive_sentiment_bert_query"
            ),
            "setup_args": {
                "threshold": 0.8,
                "seed": [
                    "This is a great movie",
                    "I really liked this movie",
                    "That was a fantastic movie.",
                    "Great plot",
                    "Very enjoyable movie",
                    "Great cast and great actors"
                ],
                "infer_config": {
                    "k": 2,
                    "segment_config": {"window_size": 5, "step_size": 3},
                },
            },
        },
    ]
    return configs


def example_experiment():
    """ Driver """
    mm_base_path = os.environ.get("MM_HOME")
    models_basepath = os.path.join(mm_base_path, "models")
    imdb_data_dir = os.path.join(mm_base_path, "example/data")
    imdb_data_path = os.path.join(imdb_data_dir, "imdb_dataset.json")

    configs = get_configs()
    clf = TaskClassifier(models_basepath, configs)

    train_data, test_data = load_data(imdb_data_path, train_ratio=0.7)
    featurized = clf.featurize_data(train_data)
    feature_vector = featurized["feature_vector"]
    labels = featurized["labels"]
    clf.fit(feature_vector, labels)
    print(
        clf.infer(
            ["I liked this movie.", "It was a good movie", "It was wonderful"]
        )
    )
    print(
        clf.infer(
            ["I hated this movie.", "It was a bad movie", "It was awful"]
        )
    )
    print(json.dumps(clf.test(test_data), indent=2))


if __name__ == "__main__":
    example_experiment()
