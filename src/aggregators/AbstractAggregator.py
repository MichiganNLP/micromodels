"""
Abstract Aggregator class.
"""


class AbstractAggregator:
    """
    Abstract class for aggregating micromodel outputs.
    """

    def __init__(self):
        pass

    def aggregate_single(self, mm_vector):
        """
        Aggregate a single micromodel output.
        """
        raise NotImplementedError("aggregate_single() not implemented.")

    def aggregate(self, mm_vectors):
        """
        Aggregate micromodel outputs.
        """
        raise NotImplementedError("aggregate() not implemented.")
