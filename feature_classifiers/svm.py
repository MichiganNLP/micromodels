"""
SVM Feature classifier
"""

import pickle
import random
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from FeatureClassifier import FeatureClassifier
from utils import preprocess_list

random.seed(a=42)


class SVM(FeatureClassifier):
    """
    SVM implementation of Feature Classifier
    """

    def __init__(self, name):
        super(SVM, self).__init__(name)
        self.trainable = True
        self.svm_model = None

    def _train(self, train_data, train_args):
        """
        Train model
        :param train_data: dict of labels: [utterances]
        """
        positives = train_data["true"]
        negatives = train_data["false"]
        if len(positives) <= 0 or len(negatives) <= 0:
            raise ValueError("Invalid training data")

        if len(negatives) > len(positives) * 2:
            train_data["false"] = random.sample(
                negatives, len(positives) * 2
            )

        preprocessed = {label: [] for label in train_data.keys()}
        for label, utterances in train_data.items():
            preprocessed[label] = preprocess_list(utterances)

        formatted = []
        for label, utterances in preprocessed.items():
            for utterance in utterances:
                words = {token: True for token in utterance.split()}
                formatted.append((words, label))

        self.svm_model = SklearnClassifier(SVC(kernel="linear"))
        print("Start training")
        self.svm_model.train(formatted)
        return self.svm_model

    def save_model(self, model_path):
        """
        Save model
        """
        if self.svm_model is None:
            raise ValueError("self.svm_model is None")

        with open(model_path, "wb") as file_p:
            pickle.dump(self.svm_model, file_p)

    def load_model(self, model_path):
        """
        Load model
        """
        with open(model_path, "rb") as file_p:
            self.svm_model = pickle.load(file_p)

    def _infer(self, query):
        """
        Infer model
        """
        if self.svm_model is None:
            raise ValueError("self.svm_model is None")

        tokens = {token: True for token in query.strip().split()}
        pred = self.svm_model.classify(tokens)
        #return {"true": True, "false": False}[pred]
        return pred

    def test(self, testing_dataset):
        """
        Test classifier
        """
        return None

    def is_loaded(self):
        """
        Check if model is loaded.
        """
        return self.svm_model is not None


if __name__ == "__main__":
    test = SVM("testing")
    #data = "/home/andrew/cbt_dialogue_system/featurizer/data/diagnosed.json"
    #test.train(data)
    #print(test.infer("I was diagnosed with depression"))
    #print(test.infer("I have been diagnosed with depression."))
    #print(test.infer("@ writing has always helped me deal with emotional stuff. i used to free write a lot when i was diagnosed with depression."))
    #test.save_model("testing")
    from datetime import datetime
    model_path = "/home/andrew/cbt_dialogue_system/featurizer/models/diagnosed_svm"
    start = datetime.now()
    test.load_model(model_path)
    end = datetime.now()
    print("Loading took", end - start)
    start = datetime.now()
    print(test.infer("I was diagnosed with depression"))
    end = datetime.now()
    print("Inference took", end - start)
    print(test.infer("I have been diagnosed with depression."))
    print(test.infer("@ writing has always helped me deal with emotional stuff. i used to free write a lot when i was diagnosed with depression."))
