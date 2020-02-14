"""The New Clustering Method."""


import itertools
from collections import defaultdict, namedtuple
from operator import itemgetter
from typing import Any, DefaultDict, Dict, List, Tuple

import hcsw

import networkx as nx

import numpy as np


class BuildCooccurrenceGraph:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'Build_cooccurrence_graph'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return ['svo']

    def required_data(self):
        return ['pair_to_contexts', 'unique_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['cograph']

    def apply(self, pair_to_contexts, unique_contexts, **kwargs):
        cograph = nx.Graph()
        cograph.add_nodes_from(unique_contexts)

        for pair, contexts in pair_to_contexts.items():
            # ignore value of n
            contexts = [ctx for (ctx, _, _) in contexts]

            for v1, v2 in itertools.combinations_with_replacement(contexts, 2):
                # the contexts v1 and v2 co-occur within the same (S, O) pair
                if cograph.has_edge(v1, v2):
                    existing_edge = cograph[v1][v2]
                    existing_weight = existing_edge['weight']
                    new_edge_attr = {(v1, v2): {'weight': existing_weight + 1}}

                    nx.set_edge_attributes(cograph, new_edge_attr)
                else:
                    cograph.add_edge(v1, v2, weight=1)

        return {'cograph': cograph}


class NcmHcsw:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'NcmHcsw'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['cograph', 'unique_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['groups']

    def apply(self, cograph: nx.Graph, unique_contexts, **kwargs):
        weights = [weight
                   for (from_node, to_node, weight)
                   in cograph.edges(data='weight')]
        mean_weight = np.mean(weights)

        result = hcsw.hcsw_disconnected(cograph, mean_weight * 2)

        groups = hcsw.label(result, unique_contexts)

        return {'groups': groups}


class Medoids:
    def __init__(self, cache=False):
        self.cache = cache

    def __repr__(self):
        return 'NcmMedoids'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['cograph', 'groups', 'unique_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['relation_names']

    def apply(self, cograph, groups, unique_contexts, **kwargs):
        node_centrality = nx.degree_centrality(cograph)

        centrality_array = np.zeros_like(unique_contexts, dtype=np.float)

        medoids = []

        for i, context in enumerate(unique_contexts):
            centrality_array[i] = node_centrality[context]

        for group_code in range(groups.max() + 1):
            indexes = np.where(groups == group_code)[0]
            group_centralities = centrality_array[indexes]

            central_index = indexes[group_centralities.argmax()]
            medoids.append(unique_contexts[central_index])

        return {'relation_names': medoids}


class PromotePairs:
    def __init__(self,
                 only_commonest: bool = True,
                 pairs_to_promote: int = 50,
                 cache: bool = False):
        """only_commonest removes scores below 1
        """
        self.only_commonest = only_commonest
        self.pairs_to_promote = pairs_to_promote
        self.cache = cache

    def __repr__(self):
        return 'NcmPromotePairs'

    def __str__(self):
        return repr(self)

    def required_files(self):
        return []

    def required_data(self):
        return ['unique_contexts', 'groups', 'pair_to_contexts']

    def creates(self):
        return []

    def returns(self):
        return ['promoted_pairs', 'group_pairs', 'groups_to_prune']

    def apply(self,
              unique_contexts: 'np.ndarray[str]',
              groups: 'np.ndarray[int]',
              pair_to_contexts: Dict[Tuple[str, str],
                                     List[Tuple[str, int, bool]]],
              **kwargs
              ) -> Dict[str, Any]:

        total_pairs = len(pair_to_contexts)
        total_groups = groups.max() + 1

        occurrence_count = np.zeros(shape=(total_pairs, total_groups))
        pair_to_index: Dict[Tuple[str, str], int] = {}

        for (name1, name2), context_list in pair_to_contexts.items():
            index = len(pair_to_index)
            pair_to_index[(name1, name2)] = index

            for context, occurrences, is_reversed in context_list:
                context_index = np.where(unique_contexts == context)[0]
                group_index = groups[context_index]

                occurrence_count[index, group_index] += occurrences

        Score = namedtuple('Score', 'group score')

        scores: Dict[Tuple[str, str], Score] = {}

        for (name1, name2), index in pair_to_index.items():
            vector = occurrence_count[index, :]
            maximum_index = vector.argmax()
            maximum = vector[maximum_index]

            score = maximum / (vector.sum() - maximum + 1)

            scores[name1, name2] = Score(maximum_index, score)

        if self.only_commonest:
            scores = {k: v for k, v in scores.items() if v.score >= 1}

        group_pairs_dict: DefaultDict[int, List[Tuple[str, str, int]]] = \
            defaultdict(list)

        for (name1, name2), score in scores.items():
            group_pairs_dict[score.group].append((name1, name2, score.score))

        group_pairs = []
        promoted_pairs = []
        groups_to_prune = []

        for i in range(total_groups):
            candidates = group_pairs_dict[i]

            # sort descending by score
            candidates = sorted(candidates, key=itemgetter(2), reverse=True)

            # throw away the score
            candidates = [(c[0], c[1]) for c in candidates]

            group_pairs.append(candidates)
            promoted_pairs.append(candidates[:self.pairs_to_promote])

            if not candidates:
                groups_to_prune.append(i)

        return {'group_pairs': group_pairs,
                'promoted_pairs': promoted_pairs,
                'groups_to_prune': groups_to_prune}
