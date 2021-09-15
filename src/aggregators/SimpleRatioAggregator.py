"""
Simple-Ratio Aggregator.
Simply compute the ratio of "hits" per micromodel.
"""

import numpy as np
from src.aggregators.AbstractAggregator import AbstractAggregator


class SimpleRatioAggregator(AbstractAggregator):
    """
    Simple Ratio Aggregator:
    Compute the ratio of "hits" per micromodel.
    """

    def aggregate_single(self, mm_vector):
        """
        aggregate
        """
        matched_count = np.count_nonzero(mm_vector)
        return matched_count / len(mm_vector)

    def aggregate(self, mm_vectors):
        """
        Aggregate
        """

        def normalize(feature_value, mean, std):
            """
            Normalize feature value.
            """
            return (feature_value - mean) / std

        ratios = np.array(
            [self.aggregate_single(vector) for vector in mm_vectors]
        )
        # TODO:
        # mean = np.mean(ratios)
        # std = np.std(ratios)

        # normalized_ratios = [
        #    normalize(ratio, mean, std) for ratio in ratios
        # ]
        # return normalized_ratios
        return ratios
