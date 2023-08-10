'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression

from sensing_network.resistor_link_selection import select_resistor_links
from sensing_network.network_layout import layout
from sensing_network.layout_adjustment import LayoutAdjustment
from sensing_network.resistor_path_generation import ResistorPathGenerator

if __name__ == '__main__':
    if not os.path.exists('./result/'):
        os.makedirs('./result')

    # 1. Acheivable resitance for different volumes
    # fix the angle and radius but change length
    link_radius = 3.0
    node_radius = 6.0

    link_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    total_resists = []
    for link_length in link_lengths:
        node1_pos = np.array((0.0, 0.0, 0.0))
        x = (link_length + node_radius * 2) / math.sqrt(2)
        node2_pos = np.array((x, 0, x))

        # use very large resitance to evaluate the achievable limit
        resistance = 5000e3

        generator = ResistorPathGenerator(resistance,
                                          node1_pos=node1_pos,
                                          node2_pos=node2_pos,
                                          link_radius=link_radius,
                                          node_radius=node_radius,
                                          vertical_step=0.6,
                                          path_margin=0.55,
                                          vertical_resistivity=2100.0,
                                          horizontal_resistivity=890.0)
        _, _, total_resist = generator.generate_path()
        print(total_resist)
        total_resists.append(total_resist)

    reg = LinearRegression(fit_intercept=True).fit(
        np.array(link_lengths)[:, None], total_resists)
    xseq = np.linspace(0, np.array(link_lengths).max(), num=len(link_lengths))

    fig, ax = plt.subplots(figsize=(3, 2.1))
    ax.scatter(link_lengths, total_resists)
    ax.plot(xseq, reg.coef_ * xseq + reg.intercept_, color='k', lw=1)
    plt.xlabel('Cylinder axis length (mm)')
    plt.ylabel(r'Max resistance ($\Omega$)')
    plt.ylim([0, 1000000])
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('./result/resistance_and_volume.pdf')
    plt.show()

    # 2. required resitance for different networks
    from scipy.optimize import minimize_scalar
    import math

    def func(r, min_diff=2.1e-6):
        return abs(2.5 - 5.0 * (1.0 - math.exp(-min_diff / (100.0e-12 * r))))

    res = minimize_scalar(func, bounds=(1e3, 1e10))
    required_resist = res.x

    # 3. identify corresponding physical size
    link_length = (required_resist -
                   reg.intercept_) / reg.coef_ + node_radius * 2

    network_list = pd.read_csv('./networks.csv')
    network_list = network_list[network_list['n_nodes'] < 100]
    network_list = network_list.sort_values(by=['n_nodes', 'n_links'])
    print(network_list)

    networks = []
    for name, n_paths in zip(network_list['name'], network_list['n_paths']):
        with open(f'./data/{name}.json') as f:
            data = json.load(f)
            data['name'] = name
            data['n_paths'] = n_paths
            networks.append(data)

    with open('./network_size.csv', 'w') as f:
        f.write('name,n_nodes,n_links,size\n')

    for network in networks:
        name = network['name']
        nodes = network['nodes']
        links = network['links']
        print('======')
        print(name, len(nodes), len(links))
        print('======')

        resistor_links, _ = select_resistor_links(nodes, links)

        P = layout(nodes, links, mean_link_length=1)
        min_link_length = np.infty
        for s, t in resistor_links:
            dist = np.linalg.norm(P[s] - P[t])
            if dist < min_link_length:
                min_link_length = dist

        P *= link_length / min_link_length

        la = LayoutAdjustment(link_radius=0,
                              node_radius=0,
                              ref_node_positions=P,
                              non_intersect_prior_links=None,
                              n_components=3,
                              batch_size=len(links),
                              loss_weights={'layout_change': 1})
        la.fit(nodes, links, max_epochs=200)

        PP = la.transform(nodes)
        resistor_link_lengths = np.array(
            [np.linalg.norm(PP[s] - PP[t]) for s, t in resistor_links])

        scaling_factor = link_length / resistor_link_lengths.min()
        la = LayoutAdjustment(link_radius=link_radius / scaling_factor,
                              node_radius=node_radius / scaling_factor,
                              ref_node_positions=P,
                              non_intersect_prior_links=resistor_links,
                              n_components=3,
                              batch_size=len(links),
                              encoder=la.encoder,
                              loss_weights={
                                  'layout_change': 1,
                                  'prior_link_intersect': 3,
                                  'link_intersect': 0,
                                  'node_intersect': 3,
                                  'link_length_variation': 0
                              })
        la.fit(nodes, links, max_epochs=200)
        PP = la.transform(nodes)
        resistor_link_lengths = np.array(
            [np.linalg.norm(PP[s] - PP[t]) for s, t in resistor_links])
        PP *= link_length / resistor_link_lengths.min()

        D = cdist(PP, PP)
        max_dist = D.max() + node_radius * 2
        print(max_dist)

        with open('./network_size.csv', 'a') as f:
            f.write(f'{name},{len(nodes)},{len(links)},{max_dist}\n')

    df = pd.read_csv('./network_size.csv')

    # filt = df['size'] < 500
    filt = df['n_nodes'] < 50
    x = df['n_nodes'][filt]
    y = df['size'][filt]
    z = df['n_links'][filt]

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    pcm = ax.scatter(x,
                     y,
                     c=z,
                     cmap='YlOrRd',
                     edgecolors='#aaaaaa',
                     lw=0.5,
                     vmin=0)
    plt.xlabel(r'$N$')
    plt.ylabel('Max dim length (mm)')
    plt.axhline(y=250, color='#888888', linestyle='--', lw=1)
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.colorbar(pcm, ax=ax)
    plt.savefig('./result/network_size_by_node.pdf')
    plt.show()