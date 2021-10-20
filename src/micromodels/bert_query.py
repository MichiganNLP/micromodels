"""
Bert Query Micromodel.
"""
from typing import Mapping, Any, List

import os
import json
from collections import defaultdict
import numpy as np
import torch
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from src.micromodels.AbstractMicromodel import AbstractMicromodel


def _get_segments(
    utterance: str, window_size: int, step_size: int
) -> List[str]:
    """
    Break down utterance into segments based on input parameters.

    :param utterance: input query.
    :window_size: size of each segment.
    :step_size: step size for the sliding window.
    """
    tokens = tokenize.word_tokenize(utterance)
    left_idx = 0
    right_idx = window_size

    segments = []
    while left_idx < len(tokens):
        segments.append(
            " ".join(tokens[left_idx : min(right_idx, len(tokens))])
        )
        left_idx += step_size
        right_idx += step_size
    return segments


def _batch_similarity_search(
    bert: SentenceTransformer,
    seed_utterances: List[str],
    seed_encoding: np.ndarray,
    queries: List[str],
    config: Mapping[str, Any] = None,
) -> Mapping[int, Mapping[str, Any]]:
    """
    Do batched similarity search.

    :param bert: Bert model for doing similarity search.
    :param seed_utterances: List of seed utterances.
    :param seed_encoding: Encoding of seed utterances, with
        shape of (len(seed_utterances), embedding_size)
    :param queries: Input queries
    :param config: config specific to doing batch similarity search.
        Properties:

        - k: number of similar sentences to return per query. Defaults to 3.
        - segment_config: if "segment_config" is specified in the config,
          this function will split the input queries into smaller segments and
          do a similarity search on the segments.

        segment_config requires 2 properties:

        - window_size (int), defaults to 7
        - step_size (int), defaults to 4

    :return: Mapping of indices of input queries to result objects in the
        following format:
        {
            "query": str,
            "max_score": float - maximum similarity score
            "top_k_scores": List[Tuple[str, float]] of length k, where the
                tuples consist of the top k most similar seed utterances and
                their similarity scores.
        }
    """
    config = config or {}
    orig_queries = queries

    segment_config = config.get("segment_config")

    # In case there are segments,
    # keep track of which segments map to which
    # utterance in the original list of queries.
    query_idx_map = []
    if segment_config is not None:
        window_size = segment_config.get("window_size", 7)
        step_size = segment_config.get("step_size", 4)
        all_segments = [
            _get_segments(query, window_size, step_size) for query in queries
        ]
        for idx, segments in enumerate(all_segments):
            for segment in segments:
                query_idx_map.append((idx, segment))

    else:
        query_idx_map = list(enumerate(queries))

    query_idx_map = [
        (idx, utt.encode("utf-8", "replace").decode())
        for (idx, utt) in query_idx_map
    ]

    query_embeddings = bert.encode(
        [query[1] for query in query_idx_map],
        batch_size=16,
        convert_to_tensor=True,
    )

    if isinstance(seed_encoding, (list, np.ndarray)):
        seed_encoding = torch.from_numpy(seed_encoding)
    seed_encoding = seed_encoding.to(query_embeddings.device)

    # cos_scores.shape: (len(queries) x len(seed_utterances))
    cos_scores = st_util.pytorch_cos_sim(query_embeddings, seed_encoding)

    # top_results[0].shape: (len(queries) x k), top k similarity scores
    # top_results[1].shape: (len(queries) x k), top k similar indexes
    # The idxs are idxs to seed_utterances
    k = config.get("k", 3)
    top_results = torch.topk(cos_scores, k=min(k, len(seed_utterances)))
    scores, idxs = top_results

    results = defaultdict(list)
    for idx, query_obj in enumerate(query_idx_map):
        orig_idx = query_obj[0]

        top_scores = [
            float(score) for score in scores[idx]
        ]  # len(scores[idx]) = k

        similar_idxs = idxs[idx]
        similar_utts = [seed_utterances[idx] for idx in similar_idxs]
        top_k_scores = list(zip(similar_utts, top_scores))

        results[orig_idx].append(
            {"max_score": top_scores[0], "top_k_scores": top_k_scores}
        )

    # In case there were segments, map segment results into original utterance.
    for idx, result_obj in results.items():
        top_idx = 0
        if len(result_obj) > 1:
            top_idx = np.argmax([entry["max_score"] for entry in result_obj])
        top_entry = result_obj[top_idx]
        top_entry["query"] = orig_queries[idx]
        results[idx] = top_entry

    return results


class BertQuery(AbstractMicromodel):
    """
    Bert Query implementation of a micromodel.
    """

    def __init__(self, name: str, **kwargs) -> None:
        """
        kwargs:
        :param threshold: (float) Threshold value for bert similarity score.
        :param seed: (List[str]) Seed for Bert-Query.
        :param device: (str) device to use for Transformer.
            Optional, defaults to 'cuda:1'.

        :param infer_config: (Mapping[str, Any]) Configurations specific to
            run_bert(). See run_bert() method for details.
        Bert Query micromodels also support the following options:

        :param bert_model: (SentenceTransformer), Optional, defaults to None.
            If not specified, will use the pre-trained model
            "paraphrase-xlm-r-multilingual-v1".
        """
        required_params = ["threshold", "seed"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(
                    "Bert-Query is missing the parameter '%s'" % param
                )

        self.threshold = kwargs["threshold"]
        self.seed = kwargs["seed"]
        self.seed_encoding = None
        self.infer_config = kwargs.get("infer_config")
        device = kwargs.get("device", "cuda:1")
        if "bert_model" in kwargs:
            self.bert = kwargs["bert"]

        else:
            self.bert = SentenceTransformer(
                "paraphrase-xlm-r-multilingual-v1", device=device
            )

        super().__init__(name)

    def _set_seed(self, seed: List[str]) -> None:
        """
        Set self.seed.
        """
        self.seed = seed

    def _set_seed_encoding(self, seed_encoding: np.ndarray) -> None:
        """
        Set self.seed_encoding.
        """
        self.seed_encoding = seed_encoding

    def train(self) -> None:
        """
        Build seed encodings during training.
        """
        if self.seed is None:
            raise RuntimeError("Seed is not set!")
        self.seed_encoding = self.bert.encode(self.seed)

    def _infer(self, query: str) -> Any:
        """
        Inner infer method. Calls self.run_bert().

        :param query: String utterance to query.
        :return: Boolean result.
        """
        similarity_results = self.run_bert([query], config=self.infer_config)
        return similarity_results[0]["max_score"] >= self.threshold

    def _batch_infer(self, query_groups: List[List[str]]) -> List[List[int]]:
        """
        Batch inference.

        :param queries: List of utterance groups.
        :return: List of binary vectors, which is represented as a list of
            indices that correspond to a hit.
        """
        binary_vectors = []
        for queries in query_groups:
            similarity_results = self.run_bert(
                queries, config=self.infer_config
            )
            binary_vector = [
                query_idx
                for query_idx, result in similarity_results.items()
                if result["max_score"] >= self.threshold
            ]
            binary_vectors.append(binary_vector)
        return binary_vectors

    def save_model(self, model_path: str) -> None:
        """
        Dumps seed and seed encodings to files. :model_path: should
        point to a directory.
        self.seed will be saved in 'seed_data.json', while
        self.seed_encodings will be saved as 'seed_encodings.npy'.

        :param model_path: directory to save seed and seed encodings.
        """
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        seed_path = os.path.join(model_path, "seed_data.json")
        with open(seed_path, "w") as file_p:
            json.dump({"seed": self.seed}, file_p, indent=2)

        encoding_path = os.path.join(model_path, "seed_encodings.npy")
        np.save(encoding_path, self.seed_encoding)

    def load_model(self, model_path: str) -> None:
        """
        Load seed and seed encodings.
        """
        if not os.path.isdir(model_path):
            raise ValueError("Could not find directory %s" % model_path)

        seed_path = os.path.join(model_path, "seed_data.json")
        encoding_path = os.path.join(model_path, "seed_encodings.npy")
        with open(seed_path, "r") as file_p:
            seed = json.load(file_p)["seed"]
        seed_encoding = np.load(encoding_path, allow_pickle=True)

        self._set_seed(seed)
        self._set_seed_encoding(seed_encoding)

    def is_loaded(self) -> bool:
        """
        Check if self.seed and self.seed_encoding is set.
        """
        return self.seed is not None and self.seed_encoding is not None

    def run_bert(
        self, queries: List[str], config: Mapping[str, Any] = None
    ) -> Mapping[int, Mapping[str, Any]]:
        """
        Run bert similarity checks on input queries against seed encodings.

        :param queries: List of utterances to query.
        :param config: config specific to doing batch similarity search.
            Properties:

            - k: number of similar sentences to return per query. Defaults to 3.
            - segment_config: if "segment_config" is specified in the config,
              this function will split the input queries into smaller segments
              and do a similarity search on the segments.

            segment_config requires 2 properties:

            - window_size (int), defaults to 7
            - step_size (int), defaults to 4

        :return: Mapping of indices of input queries to result objects in the
            following format:
            {
                "query": str,
                "max_score": float - maximum similarity score
                "top_k_scores": List[Tuple[str, float]] of length k, where the
                    tuples consist of the top k most similar seed utterances and
                    their similarity scores.
            }
        """
        if self.seed is None:
            raise RuntimeError("Seed is not set!")
        if self.seed_encoding is None:
            raise RuntimeError("Seed encodings are not set!")
        return _batch_similarity_search(
            self.bert, self.seed, self.seed_encoding, queries, config=config
        )
