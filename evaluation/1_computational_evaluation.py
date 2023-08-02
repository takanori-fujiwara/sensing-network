import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from sensing_network.resistor_link_selection import select_resistor_links, to_paths
from sensing_network.resistance_optimization import ResistanceOptimization
from sensing_network.network_layout import layout
from sensing_network.layout_adjustment import LayoutAdjustment

if __name__ == '__main__':
    network_list = pd.read_csv('./networks.csv')
    network_list = network_list[network_list['n_nodes'] < 100]
    network_list = network_list.sort_values(by=['n_nodes', 'n_links'])

    networks = []
    for name, n_paths in zip(network_list['name'], network_list['n_paths']):
        with open(f'./data/{name}.json') as f:
            data = json.load(f)
            data['name'] = name
            data['n_paths'] = n_paths
            networks.append(data)

    resistance_range = [50.0e3, 300.0e3]
    mean_link_length = 50
    node_radius = 8.5
    link_radius = 3.15
    resistance_optimization_kwargs = {
        'buget_for_single_iter': 5000,
        'max_iterations': 10000,
        'convergence_thres': 0.01,
        'convergence_judge_buffer_size': 500
    }
    layout_adjustment_kwargs = {
        'layout_learning_epoch': 200,
        'main_learning_epoch': 200,
        'loss_weights': {
            'layout_change': 1,
            'prior_link_intersect': 3,
            'link_intersect': 1,
            'node_intersect': 3,
            'link_length_variation': 1
        }
    }

    if not os.path.exists('./result/'):
        os.makedirs('./result')
    outfile_path = f'./result/computational_evaluation.csv'

    with open(outfile_path, 'w') as f:
        f.write(
            'name,n_nodes,n_links,n_paths,resistor_selection,resistance_optimization,initial_layout,copy_layout,adjust_layout\n'
        )

    # run one time extra to avoid the launching time for pytoch
    simplest_network = {
        'name': 'dummy',
        'nodes': [0, 1, 2, 3],
        'links': [[0, 1], [0, 2], [1, 2], [1, 3]],
        'n_paths': 1
    }
    for network in [simplest_network] + networks:
        name = network['name']
        nodes = network['nodes']
        links = network['links']
        n_links = len(links)
        n_paths = network['n_paths']
        print(name, len(nodes), n_links, n_paths)

        start = time.time()
        resistor_links, node_order = select_resistor_links(nodes, links)
        t_resistor_selection = time.time() - start

        n_paths = len(to_paths(resistor_links))
        in_node = node_order[0]
        out_node = node_order[-1]

        if n_paths >= 5:
            t_resistance_optimization = np.nan
        else:
            start = time.time()
            ropt = ResistanceOptimization(nodes=nodes,
                                          links=resistor_links,
                                          in_node=in_node,
                                          out_node=out_node,
                                          capacitance=100e-12,
                                          voltage_thres=2.5,
                                          n_jobs=-1,
                                          verbose=False,
                                          logging=False)
            resistances, min_sq_diff = ropt.optimize(
                init_resistances=None,
                resistance_range=resistance_range,
                **resistance_optimization_kwargs)
            t_resistance_optimization = time.time() - start

        opt_log = ropt.opt_log

        start = time.time()
        node_positions = layout(nodes,
                                links,
                                mean_link_length=mean_link_length)
        t_initial_layout = time.time() - start

        la = LayoutAdjustment(link_radius=0,
                              node_radius=0,
                              ref_node_positions=node_positions,
                              non_intersect_prior_links=None,
                              n_components=3,
                              batch_size=len(links),
                              loss_weights={'layout_change': 1})

        if n_links > 300:
            t_copy_layout = np.nan
            t_layout_adjustment = np.nan
        else:
            start = time.time()
            la.fit(
                nodes,
                links,
                max_epochs=layout_adjustment_kwargs['layout_learning_epoch'])
            # handle the fact that node positions from the above process have a different scale
            adjusted_node_positions = la.transform(nodes)
            adjusted_link_lengths = np.array([
                np.linalg.norm(adjusted_node_positions[s] -
                               adjusted_node_positions[t]) for s, t in links
            ])
            scaling_factor = mean_link_length / adjusted_link_lengths.mean()
            t_copy_layout = time.time() - start

            la = LayoutAdjustment(
                link_radius=link_radius / scaling_factor,
                node_radius=node_radius / scaling_factor,
                ref_node_positions=node_positions,
                non_intersect_prior_links=resistor_links,
                n_components=3,
                batch_size=len(links),
                encoder=la.encoder,
                loss_weights=layout_adjustment_kwargs['loss_weights'])

            start = time.time()
            la.fit(nodes,
                   links,
                   max_epochs=layout_adjustment_kwargs['main_learning_epoch'])
            adjusted_node_positions = la.transform(nodes)
            adjusted_link_lengths = np.array([
                np.linalg.norm(adjusted_node_positions[s] -
                               adjusted_node_positions[t]) for s, t in links
            ])
            adjusted_node_positions *= mean_link_length / adjusted_link_lengths.mean(
            )
            t_layout_adjustment = time.time() - start

        print(
            f'{name},{len(nodes)},{len(links)},{n_paths},{t_resistor_selection},{t_resistance_optimization},{t_initial_layout},{t_copy_layout},{t_layout_adjustment}\n'
        )
        with open(outfile_path, 'a') as f:
            f.write(
                f'{name},{len(nodes)},{len(links)},{n_paths},{t_resistor_selection},{t_resistance_optimization},{t_initial_layout},{t_copy_layout},{t_layout_adjustment}\n'
            )

    # plot for resistnce optimization
    outfile_path = f'./result/computational_evaluation_vis2023.csv'
    df = pd.read_csv(outfile_path)
    vmax = 75  # df['n_nodes'].max()

    filt = np.invert(df['resistance_optimization'].isna())
    x = df['n_paths'][filt]
    # x = df['n_links'][filt]
    y = df['resistance_optimization'][filt]
    z = df['n_nodes'][filt]

    fig, ax = plt.subplots(figsize=(2.75, 2.1))
    ax.scatter(x, y, c=z, cmap='YlGnBu', vmin=0, vmax=vmax)
    # ax.plot(xseq, a + b * xseq, color="k", lw=1)
    plt.xlabel(r'$B$')
    plt.ylabel('Completion time (s)')
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('./result/resist_optim.pdf')
    plt.show()

    # plot for network layout adjustment
    filt = np.invert(df['adjust_layout'].isna())
    x = df['n_links'][filt]
    y = df['copy_layout'][filt] + df['adjust_layout'][filt]
    z = df['n_nodes'][filt]

    fig, ax = plt.subplots(figsize=(2.75, 2.1))
    pcm = ax.scatter(x, y, c=z, cmap='YlGnBu', vmin=0, vmax=vmax)
    # ax.plot(xseq, a + b * xseq, color="k", lw=1)
    plt.xlabel(r'$L$ (# of links)')
    plt.ylabel('Completion time (s)')
    plt.tight_layout()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.colorbar(pcm, ax=ax)
    plt.savefig('./result/layout_adj.pdf')
    plt.show()

    network_list = pd.read_csv('./networks.csv')
    networks = []
    for name, n_paths in zip(network_list['name'], network_list['n_paths']):
        with open(f'./data/{name}.json') as f:
            data = json.load(f)
            data['name'] = name
            data['n_paths'] = n_paths
            networks.append(data)

    target_network_names = ['sa_companies', 'bison', 'law_firm']
    for network in networks:
        name = network['name']
        nodes = network['nodes']
        links = network['links']
        n_links = len(links)
        n_paths = network['n_paths']
        if name in target_network_names:
            print(name, len(nodes), n_links, n_paths)
            resistor_links, node_order = select_resistor_links(nodes, links)
            in_node = node_order[0]
            out_node = node_order[-1]

            ropt = ResistanceOptimization(nodes=nodes,
                                          links=resistor_links,
                                          in_node=in_node,
                                          out_node=out_node,
                                          capacitance=100e-12,
                                          voltage_thres=2.5,
                                          n_jobs=-1,
                                          verbose=True,
                                          logging=True)
            resistances, min_sq_diff = ropt.optimize(
                init_resistances=None,
                resistance_range=resistance_range,
                **resistance_optimization_kwargs)
            opt_log = ropt.opt_log

            with open(f'./result/{name}_ropt.json', 'w') as f:
                json.dump(opt_log, f)

    fig, ax = plt.subplots(figsize=(2.3, 2.1))
    n_iters = 340
    for name in target_network_names:
        if os.path.exists(f'./result/{name}_ropt.json'):
            with open(f'./result/{name}_ropt.json', 'r') as f:
                df = pd.DataFrame(json.load(f))
                x = df['iteration'][:n_iters]
                y = df['value'][:n_iters]**0.5

            for network in networks:
                if name == network['name']:
                    z = len(network['nodes'])
                    print(name, z)
                    break

            color = get_cmap('YlGnBu')(z / vmax)
            ax.plot(x, y, color=color)
    plt.xlabel('Iteration')
    plt.ylabel(r'Min time delay diff ($\mu$s)')
    plt.tight_layout()
    plt.ylim([0.0, 8.0e-6])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig('./result/resist_optim_iter.pdf')
    plt.show()