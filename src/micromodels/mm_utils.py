"""
Utility functions for micromodels
"""
from typing import List, Callable, Tuple

import sys
import time
from pathos.multiprocessing import ProcessingPool as Pool


def run_parallel(
    func: Callable, query_groups: List[List[str]], pool_size: int
) -> List[int]:
    """
    Run func via multiprocessing.
    """
    def inner_func(
        _query_groups: List[Tuple[int, List[str]]]
    ) -> List[List[int]]:
        """
        Inference for parallelizing.
        """
        _binary_vectors = []
        for group_idx, sentences in _query_groups:
            hit_idxs = [
                idx
                for idx, sentence in enumerate(sentences)
                if func(sentence)
            ]
            _binary_vectors.append((group_idx, hit_idxs))
        return _binary_vectors

    query_groups = list(enumerate(query_groups))
    chunk_size = len(query_groups) // pool_size
    if chunk_size < 1:
        binary_vectors = inner_func(query_groups)
        return [
            binary_vector[1] for binary_vector in binary_vectors
        ]

    chunks = [
        query_groups[i : i + chunk_size]
        for i in range(0, len(query_groups), chunk_size)
    ]
    pool = Pool(pool_size)
    pool.restart()
    try:
        _results = pool.amap(inner_func, chunks)
        while not _results.ready():
            time.sleep(1)
        infer_results = _results.get()
    except (KeyboardInterrupt, SystemExit):
        pool.terminate()
        sys.exit(1)

    binary_vectors = []
    for idx_group in infer_results:
        binary_vectors.extend(idx_group)

    pool.close()
    pool.join()
    pool.terminate()

    return [
        binary_vector[1]
        for binary_vector in sorted(binary_vectors, key=lambda x: x[0])
    ]
