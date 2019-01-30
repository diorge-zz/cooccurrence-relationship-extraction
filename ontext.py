import itertools
import numpy as np
from collections import defaultdict


class BuildCooccurrenceMatrix:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Build_cooccurrence_matrix'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return ['pair_to_contexts', 'unique_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['comatrix']

    def apply(self, pair_to_contexts, unique_contexts, **kwargs):
        matrix = defaultdict(lambda: 0)

        for pair, contexts in pair_to_contexts.items():
            # ignore value of n
            contexts = [ctx for (ctx, _, _) in contexts]

            for v1, v2 in itertools.combinations_with_replacement(contexts, 2):
                # the contexts v1 and v2 co-occur within the same (S, O) pair
                matrix[(v1, v2)] += 1
                matrix[(v2, v1)] += 1

        n = len(unique_contexts)
        matrix_array = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                v1 = unique_contexts[i]
                v2 = unique_contexts[j]
                matrix_array[i, j] = matrix[(v1, v2)]

        return {'comatrix': matrix_array}


class NormalizeMatrix:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Normalize_matrix'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['comatrix']

    def creates(self):
        return []

    def returns(self):
        return ['comatrix']

    def apply(self, comatrix, **kwargs):
        row_means = comatrix.sum(axis=1).reshape(-1, 1)
        normalized = comatrix / row_means
        return {'comatrix': normalized}
