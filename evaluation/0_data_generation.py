'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

# This code requires graph-tool
# Follow installation instructions (only supporting macOS and Linux): https://graph-tool.skewed.de/
# Note: brew installed graph_tool only supports Python3.11

import json
import os

import graph_tool.all as gt
import networkx as nx

from sensing_network.resistor_link_selection import select_resistor_links, to_paths


def get_n_paths(nodes, links):
    resistor_links, node_order = select_resistor_links(nodes, links)
    return len(to_paths(resistor_links))


def process_g(g):
    # g = gt.extract_largest_component(g, prune=True) # gt has bug here
    nodes = g.get_vertices().tolist()
    links = g.get_edges().tolist()
    # g.set_directed(False)
    # gt.remove_parallel_edges(g) # gt has bug here
    # gt.remove_self_loops(g)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(links)
    G.remove_edges_from(nx.selfloop_edges(G))

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    G = nx.relabel.convert_node_labels_to_integers(G,
                                                   first_label=0,
                                                   ordering='default')

    nodes = list(G.nodes())
    links = list(G.edges())
    n_paths = get_n_paths(nodes, links)

    return nodes, links, n_paths


network_list_csv = 'networks.csv'
with open(network_list_csv, 'w') as f:
    f.write('name,n_nodes,n_links,n_paths,data_source\n')

saved_network_names = {}

network_funcs = [
    'bull', 'chvatal', 'cubical', 'desargues', 'diamond', 'dodecahedral',
    'heawood', 'hoffman_singleton', 'house', 'icosahedral', 'krackhardt_kite',
    'moebius_kantor', 'octahedral', 'pappus', 'petersen', 'sedgewick_maze',
    'tetrahedral', 'truncated_cube', 'truncated_tetrahedron', 'tutte'
]
# removed 'frucht' due to graph-tool's implementation error

for name in network_funcs:
    g = eval(f'gt.collection.{name}_graph()')
    n_nodes = len(g.get_vertices())
    if (n_nodes < 2000) and (n_nodes > 1) and (saved_network_names.get(name)
                                               is None):
        print(name, n_nodes)
        nodes, links, n_paths = process_g(g)

        saved_network_names[name] = True
        if not os.path.exists('./data/'):
            os.makedirs('./data')
        with open(f'./data/{name}.json', 'w') as f:
            json.dump({'nodes': nodes, 'links': links}, f)
        with open(network_list_csv, 'a') as f:
            f.write(
                f'{name},{len(nodes)},{len(links)},{n_paths},gt.collection._graph()\n'
            )

ns_network_info = gt.collection.ns_info

for name in ns_network_info.keys():
    if len(ns_network_info[name]['nets']) == 1:
        n_nodes = ns_network_info[name]['analyses']['num_vertices']
        if (n_nodes < 2000) and (n_nodes
                                 > 1) and (saved_network_names.get(name)
                                           is None):
            print(name, n_nodes)
            g = gt.collection.ns[name]
            nodes, links, n_paths = process_g(g)

            saved_network_names[name] = True
            with open(f'./data/{name}.json', 'w') as f:
                json.dump({'nodes': nodes, 'links': links}, f)
            with open(network_list_csv, 'a') as f:
                f.write(
                    f'{name},{len(nodes)},{len(links)},{n_paths},gt.collection.ns\n'
                )

# network_names = [
#     'adjnoun', 'celegansneural', 'dolphins', 'football', 'karate', 'lesmis',
#     'netscience', 'polblogs', 'polbooks', 'serengeti-foodweb', 'as-22july06',
#     'astro-ph', 'cond-mat', 'cond-mat-2003', 'cond-mat-2005', 'email-Enron',
#     'hep-th', 'pgp-strong-2009', 'power'
# ]
## most of the above ones have overlaps with gt.collection.ns except for serengeti-foodweb
network_names = ['serengeti-foodweb']
for name in network_names:
    g = gt.collection.data[name]
    n_nodes = len(g.get_vertices())
    if (n_nodes < 2000) and (n_nodes > 1) and (saved_network_names.get(name)
                                               is None):
        print(name, n_nodes)
        nodes, links, n_paths = process_g(g)

        saved_network_names[name] = True
        with open(f'./data/{name}.json', 'w') as f:
            json.dump({'nodes': nodes, 'links': links}, f)
        with open(network_list_csv, 'a') as f:
            f.write(
                f'{name},{len(nodes)},{len(links)},{n_paths},gt.collection.data\n'
            )
