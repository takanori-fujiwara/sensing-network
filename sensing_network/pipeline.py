'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import numpy as np

from sensing_network.network_layout import layout
from sensing_network.resistor_link_selection import select_resistor_links
from sensing_network.resistance_optimization import ResistanceOptimization
from sensing_network.layout_adjustment import LayoutAdjustment
from sensing_network.resistor_path_generation import generate_all_paths


def default_pipeline(
        nodes,
        links,
        node_radius=8.5,
        link_radius=3.15,
        mean_link_length=40,
        init_resistances=None,
        resistance_range=[50.0e3, 300.0e3],
        layout_method=layout,
        resistor_link_selection_kwargs={
            'link_weights': None,
            # the args below are used only when link_weights: 'length'
            'resistor_length_range': (30, 100),
            'resistor_length_outrange_penalty': 100
        },
        resistance_optimization_kwargs={
            'buget_for_single_iter': 5000,
            'max_iterations': 10000,
            'convergence_thres': 0.01,
            'convergence_judge_buffer_size': 500
        },
        layout_adjustment_kwargs={
            'layout_learning_epoch': 200,
            'main_learning_epoch': 200,
            'loss_weights': {
                'layout_change': 1,
                'prior_link_intersect': 3,
                'link_intersect': 1,
                'node_intersect': 3,
                'link_length_variation': 1
            }
        },
        resistor_path_generation_kwargs={
            'vertical_step': 0.6,
            'path_margin': 0.55,
            'vertical_box_width': 0.8,
            'vertical_box_additional_height': 0.4,
            'min_node_link_overlap': 3.0,
            'vertical_resistivity': 2100.0,
            'horizontal_resistivity': 890.0
        },
        use_out_node=True,
        skip_resistance_optimization=False,
        verbose=True):
    # 1. lay out a network in a virtual world
    node_positions = layout_method(nodes, links)
    link_lengths = np.array([
        np.linalg.norm(node_positions[s] - node_positions[t]) for s, t in links
    ])
    node_positions *= mean_link_length / link_lengths.mean()
    if verbose:
        print('laid out a network')

    # 2. select links used as resistors
    if resistor_link_selection_kwargs['link_weights'] == 'length':
        length_range = resistor_link_selection_kwargs['resistor_length_range']
        outrange_penalty = resistor_link_selection_kwargs[
            'resistor_length_outrange_penalty']
        # link weights based on the distance between source and target nodes
        link_weights = np.array([
            np.linalg.norm(node_positions[s] - node_positions[t])
            for s, t in links
        ])

        link_weights[(link_weights < length_range[0]) +
                     (link_weights > length_range[1])] = outrange_penalty
    else:
        link_weights = resistor_link_selection_kwargs['link_weights']

    resistor_links, node_order = select_resistor_links(
        nodes, links, link_weights=link_weights)
    in_node = node_order[0]
    if use_out_node:
        out_node = node_order[-1]
    else:
        out_node = None

    resistor_links = [[s, t] if s < t else [t, s] for s, t in resistor_links]
    resistor_links.sort()

    if verbose:
        print('selected resitor links')

    # 3. select resistance for each resistor
    resistances = None
    if init_resistances is not None:
        resistances = np.array(init_resistances)

    if not skip_resistance_optimization:
        ropt = ResistanceOptimization(nodes=nodes,
                                      links=resistor_links,
                                      in_node=in_node,
                                      out_node=out_node,
                                      capacitance=100e-12,
                                      voltage_thres=2.5,
                                      n_jobs=-1,
                                      verbose=True)

        resistances, min_sq_diff = ropt.optimize(
            init_resistances=resistances,
            resistance_range=resistance_range,
            **resistance_optimization_kwargs)

    if verbose:
        print('optimized resitances')

    # 4. lay out a network in a physical world
    # 4-1. mimic laid-out network positions first
    la = LayoutAdjustment(link_radius=0,
                          node_radius=0,
                          ref_node_positions=node_positions,
                          non_intersect_prior_links=None,
                          n_components=3,
                          batch_size=len(links),
                          loss_weights={'layout_change': 1})
    la.fit(nodes,
           links,
           max_epochs=layout_adjustment_kwargs['layout_learning_epoch'])

    # handle the fact that node positions from the above process have a different scale
    adjusted_node_positions = la.transform(nodes)
    adjusted_link_lengths = np.array([
        np.linalg.norm(adjusted_node_positions[s] - adjusted_node_positions[t])
        for s, t in links
    ])
    scaling_factor = mean_link_length / adjusted_link_lengths.mean()

    # then learn other things too.
    la = LayoutAdjustment(
        link_radius=link_radius / scaling_factor,
        node_radius=node_radius / scaling_factor,
        ref_node_positions=node_positions,
        non_intersect_prior_links=resistor_links,
        n_components=3,
        batch_size=len(links),
        encoder=la.encoder,
        loss_weights=layout_adjustment_kwargs['loss_weights'])
    la.fit(nodes,
           links,
           max_epochs=layout_adjustment_kwargs['main_learning_epoch'])

    adjusted_node_positions = la.transform(nodes)
    adjusted_link_lengths = np.array([
        np.linalg.norm(adjusted_node_positions[s] - adjusted_node_positions[t])
        for s, t in links
    ])
    adjusted_node_positions *= mean_link_length / adjusted_link_lengths.mean()

    if verbose:
        print('adjusted node positions')

    # 5: generate resistor paths
    if (resistances is not None) and (len(resistances) == len(resistor_links)):
        h_paths, v_boxes = generate_all_paths(
            node_positions=adjusted_node_positions,
            resistor_links=resistor_links,
            resistances=resistances,
            link_radius=link_radius,
            node_radius=node_radius,
            **resistor_path_generation_kwargs)
    else:
        h_paths = []
        v_boxes = []
    if verbose:
        print('generated resistor paths')

    return {
        'nodes': nodes,
        'links': links,
        'node_positions': adjusted_node_positions,
        'node_radius': node_radius,
        'link_radius': link_radius,
        'resistor_links': resistor_links,
        'resistances': resistances,
        'in_node': in_node,
        'out_node': out_node,
        'resistors_h_paths': h_paths,
        'resistors_v_boxes': v_boxes
    }


if __name__ == '__main__':
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    info = default_pipeline(nodes, links)
    print(info)
