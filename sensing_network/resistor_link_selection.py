'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import numpy as np
import networkx as nx
from sensing_network.convert_utils import to_nxgraph


def select_resistor_links(nodes,
                          links,
                          link_weights=None,
                          G=None,
                          exact='auto',
                          n_permutations=100):
    if G is None:
        G = to_nxgraph(nodes, links, link_weights)
    if exact == 'auto':
        exact = True if len(G.nodes()) < 6 else False

    if len(nx.get_edge_attributes(G, 'weight')) > 0:
        node_order, _ = _floyd_warshall_node_visits(
            G, exact=exact, n_permutations=n_permutations)
        resistor_links = _node_order_to_links_in_shortestpath(G, node_order)
    else:
        # complexity: O(V) * O(V + E)
        min_n_paths = None
        resistor_links = None
        for node in nodes:
            tmp_resistor_links = list(nx.dfs_edges(G, source=node))
            n_paths = len(to_paths(tmp_resistor_links))
            if (min_n_paths is None) or (n_paths < min_n_paths):
                min_n_paths = n_paths
                resistor_links = tmp_resistor_links

        node_order = []
        for s, t in resistor_links:
            if (len(node_order) == 0):
                node_order.append(s)
            node_order.append(t)

    resistor_links = np.array(resistor_links, dtype=int).tolist()
    node_order = np.array(node_order, dtype=int).tolist()

    return resistor_links, node_order


def to_paths(links):
    paths = []
    for s, t in links:
        if len(paths) == 0:
            paths.append([s, t])
        else:
            if s == paths[-1][-1]:
                paths[-1].append(t)
            else:
                paths.append([s, t])
    return paths


def _floyd_warshall_node_visits(G, exact=False, n_permutations=100):
    D = nx.floyd_warshall_numpy(G)

    n_nodes = D.shape[0]
    best_dist = np.Inf
    best_order = None

    def node_order_to_dist(node_order):
        dist = 0
        for i in range(len(node_order) - 1):
            s = node_order[i]
            t = node_order[i + 1]
            dist += D[s, t]
        return dist

    if exact:
        # check all possible permutations/node orders
        import itertools
        for path in itertools.permutations(range(n_nodes)):
            dist = node_order_to_dist(path)
            if dist < best_dist:
                best_dist = dist
                best_order = path
    else:
        # generate n_pertmutations orders and evaluate them
        for i in range(n_permutations):
            path = np.random.permutation(n_nodes)
            dist = node_order_to_dist(path)
            if dist < best_dist:
                best_dist = dist
                best_order = path

    return best_order, best_dist


def _node_order_to_links_in_shortestpath(G, node_order):
    paths = []
    for i in range(len(node_order) - 1):
        s = node_order[i]
        t = node_order[i + 1]
        paths.append(nx.dijkstra_path(G, s, t))

    # subnetwork of G by taking only related links to the paths
    G_sub = nx.Graph()
    G_sub.add_nodes_from(G.nodes())
    for path in paths:
        for i in range(len(path) - 1):
            s = path[i]
            t = path[i + 1]
            G_sub.add_edge(s, t)

    return list(G_sub.edges())


if __name__ == '__main__':
    # four node graphlet example
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    resistor_links, _ = select_resistor_links(nodes, links)
    print(resistor_links)
