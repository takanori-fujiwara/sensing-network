'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import networkx as nx
import numpy as np
from sensing_network.convert_utils import to_nxgraph


def layout(nodes,
           links,
           G=None,
           mean_link_length=None,
           method=nx.spring_layout):
    if G is None:
        G = to_nxgraph(nodes, links)
    node_positions = method(G, dim=3)
    node_positions = np.array(list(node_positions.values()))

    # scale positions based on mean_link_length
    if mean_link_length:
        link_lengths = np.array([
            np.linalg.norm(node_positions[s] - node_positions[t])
            for s, t in links
        ])
        node_positions *= mean_link_length / link_lengths.mean()

    return node_positions


def plot_network(node_positions, links):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    link_positions = np.array([[node_positions[s], node_positions[t]]
                               for s, t in links])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*node_positions.T, s=100, ec='w')
    ax.add_collection(Line3DCollection(link_positions))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # four node graphlet example
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    node_positions = layout(nodes, links)
    plot_network(node_positions, links)
