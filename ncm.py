"""The New Clustering Method."""


import itertools

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
