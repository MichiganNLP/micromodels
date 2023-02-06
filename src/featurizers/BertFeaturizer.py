"""
Bert Featurizer
"""
from typing import List
from collections import defaultdict

import torch
import numpy as np
from nltk import tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

from src.Orchestrator import Orchestrator, get_model_name
from src.micromodels.bert_query import batch_similarity_search


BATCH_SIZE = 32


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
    discontinue = False
    while left_idx < len(tokens):
        if discontinue:
            break
        _segment_tokens = tokens[left_idx : min(right_idx, len(tokens))]
        if len(_segment_tokens) < window_size:
            discontinue = True
        segments.append(" ".join(_segment_tokens))
        left_idx += step_size
        right_idx += step_size
    return segments


def _get_segments_sent(utterance) -> List[str]:
    return tokenize.sent_tokenize(utterance)


def _build_query_idx_map(queries, segment_config):
    all_segments = [_get_segments_sent(query) for query in queries]
    query_idx_map = []
    for idx, segments in enumerate(all_segments):
        for segment in segments:
            segment = segment.encode("utf-8", "replace").decode()
            query_idx_map.append((idx, segment))

    return query_idx_map


def _get_mm_results(mm_obj, cos_scores, query_idx_map):
    """
    Retrieve results for each mm_obj from cos_scores, where
    mm_obj has the following format:
    {
        "mm_name": str,
        "start_idx": int,
        "end_idx": int,
        "seed_utts: List[str]
    }
    and cos_scores is a tensor of the shape
    (
        total # of segments (from all queries) x \
        total # of seed utterances \
        (from all micromodels in the same segment_config group)
    )
    """
    start_idx = mm_obj["start_idx"]
    end_idx = mm_obj["end_idx"]
    seed_utts = mm_obj["seed"]

    _cos_scores = cos_scores[:, start_idx : end_idx + 1]
    # top_results = torch.topk(_cos_scores, k=min(10, len(seed_utts)))
    top_results = torch.topk(_cos_scores, k=len(seed_utts))
    scores, score_idxs = top_results

    mm_results = defaultdict(list)
    for segment_idx, query_obj in enumerate(query_idx_map):
        orig_idx = query_obj[0]
        segment_utt = query_obj[1]
        top_scores = [float(score) for score in scores[segment_idx]]
        similar_idxs = score_idxs[segment_idx]
        similar_utts = [seed_utts[_idx] for _idx in similar_idxs]
        top_k_scores = list(zip(similar_utts, top_scores))

        mm_results[orig_idx].append(
            {
                "max_score": top_scores[0],
                "top_k_scores": top_k_scores,
                "segment": segment_utt,
            }
        )
    return mm_results


def _update_results(results, mm_results, mm_name):
    """
    Update results based on mm_results
    """
    for utt_idx, result_obj in mm_results.items():
        top_idx = 0
        if len(result_obj) > 1:
            top_idx = np.argmax([entry["max_score"] for entry in result_obj])
        top_entry = result_obj[top_idx]
        results[utt_idx]["results"][mm_name] = top_entry


class BertFeaturizer(Orchestrator):
    """
    Bert specific featurizer
    """

    def __init__(self, mm_basepath, configs, **kwargs):
        super().__init__(mm_basepath, configs)
        if not all(config["model_type"] == "bert_query" for config in configs):
            raise ValueError("Invalid micromodel type found in configs.")
        device = kwargs.get("device", "cuda:0")
        self.bert = SentenceTransformer(
            "paraphrase-xlm-r-multilingual-v1", device=device
        )
        self.setup()

    def setup(self):
        """
        Set up bert for batched similarity searching
        """
        self.seeds = {
            config["name"]: config["setup_args"]["seed"]
            for config in self.configs
        }
        self.config_idxs = {}
        self.config_idxs_inv = {}
        self.config_mapping = {}
        _idx = 0
        for config in self.configs:
            mm_name = config["name"]
            segment_config = tuple(
                sorted(
                    config["setup_args"]["infer_config"][
                        "segment_config"
                    ].items()
                )
            )

            if segment_config not in self.config_idxs:
                self.config_idxs[segment_config] = _idx
                self.config_idxs_inv[_idx] = dict(segment_config)
                _idx += 1
            self.config_mapping[mm_name] = self.config_idxs[segment_config]

        self.all_seeds = []
        utt_idx = 0
        curr_idxs = {
            _seg_config_idx: 0
            for _seg_config_idx, _ in self.config_idxs_inv.items()
        }
        utt_idx2mm = {}
        self.seed_group = defaultdict(list)
        self.seed_utts_group = defaultdict(list)
        for mm_name, seed in self.seeds.items():
            self.all_seeds.extend(seed)
            config_id = self.config_mapping[mm_name]
            self.seed_group[config_id].append(
                {
                    "mm_name": mm_name,
                    "start_idx": curr_idxs[config_id],
                    "end_idx": curr_idxs[config_id] + len(seed) - 1,
                    "seed": seed,
                }
            )
            self.seed_utts_group[config_id].extend(seed)
            curr_idxs[config_id] += len(seed)

            _utt_idxs = range(utt_idx, utt_idx + len(seed))
            utt_idx2mm.update(zip(_utt_idxs, [mm_name] * len(_utt_idxs)))

        self.seed_group_encoding = {
            config_id: self.encode(utts)
            for config_id, utts in self.seed_utts_group.items()
        }

    def encode(self, utterances):
        """
        Encode utterances using Bert
        """
        return self.bert.encode(
            utterances, batch_size=BATCH_SIZE, convert_to_tensor=True
        )

    def run_bert(self, queries: List[str]):
        """
        Run bert and get scores.
        infer_config: {
            "segment_config": {
                "window_size": ...,
                "step_size": ...
            }
        }
        """
        orig_queries = queries
        results = {
            utt_idx: {"query": query, "results": {}}
            for utt_idx, query in enumerate(orig_queries)
            if query != ""
        }

        for config_id, mm_objs in self.seed_group.items():
            segment_config = self.config_idxs_inv[config_id]
            query_idx_map = _build_query_idx_map(queries, segment_config)

            query_embeddings = self.encode(
                [query[1] for query in query_idx_map]
            )
            seed_encoding = self.seed_group_encoding[config_id]

            # cos_scores.shape: (len(query_segments) x len(seed_utterances))
            cos_scores = st_util.pytorch_cos_sim(
                query_embeddings, seed_encoding
            )

            for mm_obj in mm_objs:
                mm_name = mm_obj["mm_name"]
                mm_results = _get_mm_results(mm_obj, cos_scores, query_idx_map)
                _update_results(results, mm_results, mm_name)
        return results

    def apply_threshold(self, bert_results, threshold):
        """
        Output binary vectors from results based on threshold.
        """
        binary_vectors = defaultdict(list)
        for emotion, results in bert_results.items():
            for result_obj in results.values():
                hit = 0
                if result_obj["max_score"] >= threshold:
                    hit = 1
                binary_vectors[emotion].append(hit)
        return binary_vectors
