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
        Aggregate a single micromodel vector.
        """
        matched_count = np.count_nonzero(mm_vector)
        return matched_count / len(mm_vector)

    def aggregate(self, mm_vectors):
        """
        Aggregate all micromodel vectors.
        """
        ratios = np.array(
            [self.aggregate_single(vector) for vector in mm_vectors]
        )
        return ratios
