"""
Example of a task-specific classifier.
This example experiment applies a list of micromodels
on the IMDB sentiment analysis task.

For details on the task, see https://ai.stanford.edu/~amaas/data/sentiment/
"""


import os
import json
from example.IMDbClassifier import IMDbClassifier


def example_experiment():
    """ Driver """
    positive_keywords = ["good", "wonderful"]

    def _positive_keyword_lookup(utterance):
        """ inner logic """
        return any(keyword in utterance for keyword in positive_keywords)

    mm_base_path = os.environ.get("MM_HOME")
    imdb_data_path = os.path.join(mm_base_path, "example/imdb_dataset.json")
    clf = IMDbClassifier(mm_base_path)

    configs = [
        {
            "model_type": "logic",
            "name": "good",
            "setup_args": {"logic_func": _positive_keyword_lookup},
            "model_path": "/home/andrew/micromodels/models/good_logic",
        },
        {
            "model_type": "logic",
            "name": "good2",
            "setup_args": {"logic_func": _positive_keyword_lookup},
            "model_path": "/home/andrew/micromodels/models/good_logic",
        },
    ]
    clf.set_configs(configs)
    clf.load_micromodels()

    train_data, test_data = clf.load_data(imdb_data_path, train_ratio=0.7)
    clf.train(train_data)
    clf.train(train_data)
    print(
        clf.infer(
            ["I liked this movie.", "It was a good movie", "It was wonderful"]
        )
    )
    print(
        clf.infer(["I hated this movie.", "It was a bad movie", "It was awful"])
    )
    print(json.dumps(clf.test(test_data[:10]), indent=2))


if __name__ == "__main__":
    example_experiment()
