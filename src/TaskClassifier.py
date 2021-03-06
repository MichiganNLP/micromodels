"""
Task-specific classifier.
"""
from typing import List, Mapping, Tuple, Any

import json
import numpy as np
from interpret import show
from interpret.glassbox import ExplainableBoostingClassifier
from src.Orchestrator import Orchestrator, get_model_name
from src.aggregators.SimpleRatioAggregator import SimpleRatioAggregator
from src.metrics import f1


def to_binary_vectors(
    mm_output: Mapping[int, List[int]], utterance_lengths: Mapping[int, int]
) -> List[np.ndarray]:
    """
    Format the output of micromodels into one-hot encodings.

    :param mm_output: output vector from micromodels. mm_output is a
        nested dictionary that maps the name of the micromodels to
        an inner dictionary. The inner dictionary maps an index of
        utterance groups to a list of indices that correspond to "hits"
        from each micromodel.
    :param utterance_lengths: mapping of utterance group indices to the
        number of sentences in the group.
    :return: A list of ndarrays, where the length of each ndarray is the
        number of sentences in the corresponding utterance group.
        i.e., i'th ndarray.shape = (utterance_lengths[i], )
    """
    binary_vectors = []
    for utt_idx, length in utterance_lengths.items():
        vec = np.zeros(length, dtype=int)
        vec[mm_output[utt_idx]] = 1
        binary_vectors.append(vec)
    return binary_vectors


class TaskClassifier:
    """
    Classifier for specific tasks.
    """

    positive_value = 1
    negative_value = 0

    def __init__(
        self, mm_basepath: str, configs: List[Mapping[str, Any]] = None
    ) -> None:
        """
        Initialize the classifier (EBM), orchestrator, and aggregators.

        :param mm_basepath: fle path to where micromodels are stored.
        :param configs: list of configurations for each micromodel.
        """
        self.model = None
        self.orchestrator = Orchestrator(mm_basepath, configs)
        # TODO: Load this from config file.
        # TODO: Make this a list of aggregators
        self.aggregator = SimpleRatioAggregator()
        self._init_micromodels()

    def _init_micromodels(self) -> None:
        """
        Initialize all micromodels, by either loading them from file
        or training them if needed.
        """
        print("Initializing micromodels...")
        self.orchestrator.build_micromodels()
        self.orchestrator.load_micromodels(force_reload=True)

    def featurize_data(
        self, data: List[Tuple[List[str], Any]]
    ) -> Mapping[str, Any]:
        """
        Featurize data, where the input is a list of tuples.
        The first element of the tuple is a list of utterances and the
        second element is their label.

        :param data: list of data instances. See load_data() for details
            on the data format.

        :return: A dictionary with the following format:
        ```
            {
                "original_data": data,
                "binary_vectors": {
                    "micromodel_name: {
                        utterance_group_idx: List[int]
                    }, ...
                },
                "feature_vector": ndarray of shape (len(data), # micromodels),
                "labels": List of labels
            }
        ```
        """
        utterances = [instance[0] for instance in data]
        labels = [instance[1] for instance in data]

        micromodel_output = self.run_micromodels(utterances)
        utterance_lengths = {
            idx: len(sentences) for idx, sentences in enumerate(utterances)
        }
        featurized = None

        for mm_output in micromodel_output.values():
            # mm_output: {utt_idx: list[int]}:
            # map utterance ids to list of matched indices

            # mm_output is a list of binary vectors, represented as ndarrays
            # Convert to actual binary vectors.
            mm_outputs = to_binary_vectors(mm_output, utterance_lengths)

            feature_values = self.aggregator.aggregate(mm_outputs)
            if featurized is None:
                featurized = feature_values
            else:
                featurized = np.vstack([featurized, feature_values])
        featurized = np.transpose(featurized)
        return {
            "original_data": data,
            "binary_vectors": micromodel_output,
            "feature_vector": featurized,
            "labels": labels,
        }

    def dump_features(self, data: Mapping[str, Any], output_path: str) -> None:
        """
        Dump binary vectors, features, and labels to file, along with
        the original input text data.
        See featurize_data() for details on the format of features.

        :param data: Object with binary vectors, feature vector, and labels.
        :param output_path: Filepath for data.
        """
        data["feature_vector"] = data["feature_vector"].tolist()
        with open(output_path, "w") as file_p:
            json.dump(data, file_p, indent=2)

    def load_features(self, input_path: str) -> Mapping[str, Any]:
        """
        Load binary vectors, features, and labels from file.

        :param input_path: Filepath for features.
        """
        with open(input_path, "r") as file_p:
            data = json.load(file_p)
        data["feature_vector"] = np.array(data["feature_vector"])
        return data

    def run_micromodel(
        self,
        config: Mapping[str, Any],
        utterances: List[List[str]],
    ) -> Mapping[int, List[int]]:
        """
        Run a single micromodel that's specified in config.
        Processes the input utterances in parallel.

        :param config: micromodel specs.
        :param utterances: list of utterance groups.

        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        model_type = config.get("model_type")
        if not model_type:
            raise RuntimeError("model_type not found in config.")
        model_name = get_model_name(config)
        if not model_name:
            raise RuntimeError("name not found in config.")

        binary_vectors = self.orchestrator.run_micromodel_batch(
            model_name, utterances
        )
        return binary_vectors

    def run_micromodels(
        self, utterances: List[List[str]]
    ) -> Mapping[str, Mapping[int, List[int]]]:
        """
        Run micromodels on group of utterances.

        :param utterances: list of utterance groups. This is
            a list of lists, where each inner list corresponds
            to a single utterance group, a.k.a. a list of sentences.
        :return: micromodel results. These are represented in the
            following way:
            {

                micromodel_name: {

                    utterance_group_idx: [

                        hit_1_idx, hit_2_idx, ..., hit_n_idx

                    ],

                ...

            }
            where utterance_group_idx is the index of each utterance group,
            and hit_n_idx are the indices within each utterance_group that
            correspond to hits.
        """
        micromodel_outputs = {}
        for idx, config in enumerate(self.orchestrator.configs):
            self.orchestrator.flush_cache()
            micromodel_name = get_model_name(config)
            print(
                "Running micromodel %s (%d/%d)."
                % (micromodel_name, idx + 1, len(self.orchestrator.configs))
            )
            micromodel_outputs[micromodel_name] = self.run_micromodel(
                config, utterances
            )
        return micromodel_outputs

    def fit(self, featurized_data: np.ndarray, labels: List[str]) -> None:
        """
        Train task-specific classifier using already featurized data.

        :param featurized_data: ndarray of shape
            (len(input_data), number of micromodels).
        :param labels: list of labels.
        """
        input_shape = featurized_data.shape
        if len(input_shape) != 2:
            raise RuntimeError(
                "Invalid shape for featurized_data: %s" % (input_shape,)
            )
        if input_shape[1] != self.orchestrator.num_micromodels:
            raise RuntimeError(
                "Mismatch between input shape (%s) and number of micromodels (%d)"
                % (input_shape, self.orchestrator.num_micromodels)
            )
        if featurized_data.shape[1] < 2:
            raise RuntimeError(
                "EBM requires more than 1 micromodel to be trained."
            )

        self.model = ExplainableBoostingClassifier()
        self.model.fit(featurized_data, labels)

    def predict(self, data: np.ndarray) -> List[Any]:
        """
        Run inference on input data.

        :param data: ndarray of shape (n_samples, n_features).
        :return: List of inference results
        """
        predictions = []
        for row in data:
            prediction, _ = self._infer_featurized([row])
            predictions.append(prediction)
        return predictions

    def infer(self, utterances: List[str]) -> Tuple[Any, float]:
        """
        Infer on a single instance (list of utterances).

        :param utterances: A single utterance group.
        :return: tuple of prediction and probability.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        formatted = [(utterances, None)]
        feature_vector = self.featurize_data(formatted)["feature_vector"]
        return self._infer_featurized(feature_vector)

    def _infer_featurized(
        self, feature_vector: np.ndarray
    ) -> Tuple[Any, float]:
        """
        Infer on a single featurized vector.

        :param feature_vector: ndarray of size
            (number of utterances, number of micromodels).
        :return: tuple of preidction and probability.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        prediction = self.model.predict(feature_vector)[0]
        probability = self.model.predict_proba(feature_vector)[0][1]
        return prediction, probability

    def test(
        self, test_data: List[Tuple[List[str], Any]]
    ) -> Mapping[str, Any]:
        """
        Run tests where test_data is in the form of
        [([sentence_1, sentence_2, ...], label), ...]
        See load_data() for more details.

        This method evaluates the following metrics:
        * F1 score
        * Accuracy

        :param test_data: data to test.
        :return: test restuls.
        """
        featurized = self.featurize_data(test_data)
        x_test = featurized["feature_vector"]
        groundtruth = featurized["labels"]
        predictions = self.predict(x_test)

        num_correct = 0
        for idx, pred in enumerate(predictions):
            if pred == groundtruth[idx]:
                num_correct += 1

        accuracy = num_correct / len(groundtruth)
        return {
            "accuracy": accuracy,
            "f1": f1(
                predictions,
                groundtruth,
            ),
        }

    def _test_featurized(
        self, featurized_data: np.ndarray, labels: List[str]
    ) -> Mapping[str, Any]:
        """
        Test task-specific classifier using already featurized data.

        :param featurized_data: ndarray of shape
            (len(input_data), number of micromodels).
        :param labels: list of labels.
        :return: test results.
        """
        predictions = self.predict(featurized_data)

        num_correct = 0
        for idx, pred in enumerate(predictions):
            if pred == labels[idx]:
                num_correct += 1

        accuracy = num_correct / len(labels)
        return {
            "accuracy": accuracy,
            "f1": f1(
                predictions,
                labels,
            ),
        }

    def explain_global(self) -> None:
        """
        Explain model's global feature importance scores.
        """
        if self.model is None:
            raise RuntimeError("EBM model is not set!")
        ebm_global = self.model.explain_global()
        show(ebm_global)

    def explain_local(self, query: str) -> None:
        """
        Explain model's decision on input query.

        :param query: Input utterance.
        """

    def inspect_provenance(
        self, features: Mapping[str, Any], micromodel_name: str
    ) -> List[str]:
        """
        Show the input text that corresponds to "hits" based on binary vectors.

        :param features: Dictionary with feature information.
            Requires 'original_data' and 'binary_vectors' fields.
        :param micromodel_name: Name of the micromodel to inspect.
        :return: List of string utterances that correspond to hits.
        """
        for req in ["original_data", "binary_vectors"]:
            if req not in features:
                raise ValueError("%s missing in features argument!" % req)
        text_data = features["original_data"]
        binary_vectors = features["binary_vectors"]
        if micromodel_name not in binary_vectors:
            raise RuntimeError(
                "Could not find binary vectors for %s" % micromodel_name
            )
        binary_vectors = binary_vectors[micromodel_name]
        assert len(text_data) == len(binary_vectors)

        all_hits = []
        for idx, input_data in enumerate(text_data):
            # Input data is in the format of
            # [ ([sentence 1, sentence 2, ...], label), ...]
            sentences = input_data[0]
            hit_idxs = binary_vectors[idx]
            hits = np.array(sentences)[hit_idxs]
            all_hits.extend(hits)
        return all_hits
