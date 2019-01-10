"""Functions to read a SVO file into appropriate data structures
"""


import itertools
import numpy as np
from collections import defaultdict


def build_index(svo_stream):
    """Reads a SVO file into two indexes:
    the first maps (S, O) entities to the [(V, N, r)] they occur with
    (where r is True if the sentence is SVO and False if OVS),
    the second maps V contexts to the [((S, O), N)] they occur with.

    :param svo_stream: tab-separated SVO{N} data, such as an open file
    :return: tuple with both indexes
    """
    pair_to_contexts = defaultdict(lambda: [])
    context_to_pairs = defaultdict(lambda: [])

    for line in svo_stream:
        s, v, o, n = line.split('\t')
        n = int(n)
        pair = tuple(sorted([s, o]))
        rev = pair == tuple([s, o])
        pair_to_contexts[pair].append((v, n, rev))
        context_to_pairs[v].append((pair, n))

    return pair_to_contexts, context_to_pairs


def normalize(matrix):
    """Normalization rule to make rows sum up to 1
    """
    row_means = matrix.sum(axis=1).reshape(-1, 1)
    return matrix / row_means


def cooccurrence_dense_matrix(pair_to_contexts, unique_contexts):
    """Builds a co-ocurrence matrix from the values of SVO

    :param pair_to_contexts: SVO index of (S, O) -> [(V, N, r)]
    :param unique_contexts: list of unique contexts (V's) in the SVO
    :return: numpy.ndarray with shape (|unique_contexts|, |unique_contexts|)
    """
    # matrix as a dict (valid representation for sparse matrices)
    matrix = defaultdict(lambda: 0)

    for pair, contexts in pair_to_contexts.items():
        # ignore value of n
        contexts = [ctx for (ctx, _, _) in contexts]

        for v1, v2 in itertools.combinations_with_replacement(contexts, 2):
            # the contexts v1 and v2 co-occur within the same (S, O) pair
            matrix[(v1, v2)] += 1
            matrix[(v2, v1)] += 1

    # converting dict-matrix to numpy array
    n = len(unique_contexts)
    matrix_array = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            v1 = unique_contexts[i]
            v2 = unique_contexts[j]
            matrix_array[i, j] = matrix[(v1, v2)]

    return matrix_array


def cooccurrence_sparse_matrix(pair_to_contexts, unique_contexts):
    """Builds a co-ocurrence matrix from the values of SVO

    :param pair_to_contexts: SVO index of (S, O) -> [(V, N, r)]
    :param unique_contexts: list of unique contexts (V's) in the SVO
    :return: ? with shape (|unique_contexts|, |unique_contexts|)
    """
    # TODO
    pass
