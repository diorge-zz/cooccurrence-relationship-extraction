import itertools
import logging
from collections import defaultdict

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


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


class OntextKmeans:
    def __init__(self, k=5, cache=False):
        self.k = k
        self.cache = cache

    def __repr__(self):
        return f'Ontext_kmeans_{self.k}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['comatrix', 'unique_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['cluster_data', 'groups', 'centroids', 'medoids',
                'relation_names', 'relation_count']

    def apply(self, comatrix, unique_contexts, **kwargs):
        if comatrix.size == 0:
            logging.info('comatrix is shaped (0, 0)')

            return {'cluster_data': None,
                    'groups': [],
                    'centroids': [],
                    'medoids': [],
                    'relation_names': [],
                    'relation_count': 0}

        clusterer = KMeans(n_clusters=self.k, init='k-means++')

        clusterer.fit(comatrix)

        groups = clusterer.predict(comatrix)
        centroids = clusterer.cluster_centers_
        medoids, _ = pairwise_distances_argmin_min(centroids, comatrix)
        relation_names = unique_contexts[medoids]

        return {'cluster_data': clusterer,
                'groups': groups,
                'centroids': centroids,
                'medoids': medoids,
                'relation_names': relation_names,
                'relation_count': len(relation_names)}


class InstanceRanker:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Instance_ranker'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['contexts_to_pairs', 'groups', 'comatrix', 'relation_count',
                'unique_contexts', 'cluster_data']

    def creates(self):
        return []

    def returns(self):
        return ['instances_scores']

    def apply(self, contexts_to_pairs, groups, comatrix,
              relation_count, unique_contexts, cluster_data, **kwargs):
        scores = []

        for group_id in range(relation_count):
            scores.append(defaultdict(lambda: 0))
            cluster_contexts = unique_contexts[groups == group_id]
            centroid = cluster_data.cluster_centers_[group_id]

            for context, occurrences in contexts_to_pairs.items():
                if context in cluster_contexts:
                    c_value = comatrix[np.where(unique_contexts == context)][0]
                    sd = np.std(c_value - centroid)
                    for pair, n in occurrences:
                        scores[-1][pair] += n / (1 + sd)

        return {'instances_scores': [dict(score) for score in scores]}


class EvidenceForPromotion:
    def __init__(self, promoted_instances, cache=False):
        self.promoted_instances = promoted_instances
        self.cache = cache

    def __repr__(self):
        return f'Evidence_for_promotion_{self.promoted_instances}'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['instances_scores', 'pair_to_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['group_pairs', 'promoted_pairs', 'evidence_sentences']

    def apply(self, instances_scores, pair_to_contexts, **kwargs):
        all_group_pairs = []
        all_promoted_pairs = []
        all_evidence_sentences = []

        for group_id, scores in enumerate(instances_scores):
            sorted_scores = sorted(scores.items(), reverse=True)
            group_pairs = [pair for (pair, score) in sorted_scores]
            top_pairs = group_pairs[:self.promoted_instances]

            evidence_sentences_gen = \
                self.sentences_with_pairs(pair_to_contexts, top_pairs)
            evidence_sentences = np.array(list(evidence_sentences_gen))

            all_group_pairs.append(group_pairs)
            all_promoted_pairs.append(top_pairs)
            all_evidence_sentences.append(evidence_sentences)

        return {'group_pairs': all_group_pairs,
                'promoted_pairs': all_promoted_pairs,
                'evidence_sentences': all_evidence_sentences}

    def sentences_with_pairs(self, pair_to_contexts, pairs):
        """The sentences in pair_to_contexts in which the pair is
        in the pairs collection
        """
        for pair in pairs:
            s, o = pair
            for context, occurrence_count, ordering in pair_to_contexts[pair]:
                if ordering:
                    yield ' '.join([s, context, o])
                else:
                    yield ' '.join([o, context, s])
