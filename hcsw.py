"""HCSw: Highly Connected Subgraphs (weighted).

Based on "A clustering algorithm based on graph connectivity",
         Hartuv, E. and Shamir, R (2000)
"""


from typing import Any, List

import networkx as nx

import numpy as np


def highly_connected(graph: nx.Graph,
                     sum_of_removed_weights: float,
                     multiplier_threshold: float = 2
                     ) -> bool:
    """Does removing edges of such weights make the graph weight half?

    The "half" part can be adjusted by the multiplier threshold
    (higher values makes it easier to be considered highly connected)
    """
    threshold = multiplier_threshold * sum_of_removed_weights
    return threshold > graph.number_of_nodes()


def hcsw(graph: nx.Graph,
         multiplier_threshold: float = 2
         ) -> nx.Graph:
    """Clusters a connected undirected weighted graph.

    Returns a graph with the same nodes but not necessarily connected
    """

    # singular graphs are already clustered
    if graph.number_of_nodes() < 2:
        return graph

    cut_weight, partitions = nx.algorithms.connectivity.stoer_wagner(graph)

    if not highly_connected(graph, cut_weight, multiplier_threshold):
        sub_graphs = [graph.subgraph(v).copy() for v in partitions]

        component_1 = hcsw(sub_graphs[0])
        component_2 = hcsw(sub_graphs[1])

        graph = nx.compose(component_1, component_2)

    return graph


def hcsw_disconnected(graph: nx.Graph,
                      multiplier_threshold: float = 2
                      ) -> nx.Graph:
    components = nx.connected_components(graph)

    clustered = [hcsw(graph.subgraph(subgraph).copy(), multiplier_threshold)
                 for subgraph in components]

    result = nx.Graph()

    for component in clustered:
        result = nx.compose(result, component)

    return result


def label(partitioned_graph: nx.Graph,
          node_order: List[Any]
          ) -> np.ndarray:
    order_map = {node: code for code, node in enumerate(node_order)}

    labels = np.zeros(len(node_order), dtype=np.int) - 1

    gen = enumerate(nx.connected_components(partitioned_graph))
    for cluster_code, cluster_nodes in gen:
        for node in cluster_nodes:
            index = order_map[node]
            labels[index] = cluster_code

    return labels
