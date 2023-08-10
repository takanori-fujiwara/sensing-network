'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import json
import os
import re

import numpy as np
import networkx as nx
import lcapy
import pyvista as pv

from functools import reduce


def to_nxgraph(nodes, links, link_weights=None):
    '''
    Convert nodes and links to NetworkX's graph
    '''
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(links)
    if link_weights is not None:
        weight_dict = {}
        for l, w in zip(links, link_weights):
            weight_dict[tuple(l)] = w
        nx.set_edge_attributes(G, weight_dict, 'weight')

    return G


def line_to_cylinder(p0, p1, radius, resolution=100, capping=True):
    return pv.Cylinder(radius=radius,
                       center=(p1 + p0) / 2,
                       direction=p1 - p0,
                       height=np.linalg.norm(p1 - p0),
                       resolution=resolution,
                       capping=capping).triangulate()


def line_to_cone(p0, p1, radius, resolution=100, capping=True):
    return pv.Cone(radius=radius,
                   center=(p1 + p0) / 2,
                   direction=p1 - p0,
                   height=np.linalg.norm(p1 - p0),
                   resolution=resolution,
                   capping=capping).triangulate()


def line_to_box(p0, p1, extrude_dir1, extrude_dir2, width, capping=True):
    u1 = extrude_dir1 / np.linalg.norm(extrude_dir1)
    u2 = extrude_dir2 / np.linalg.norm(extrude_dir2)
    p0_ = p0 - (u1 + u2) * width / 2
    p1_ = p1 - (u1 + u2) * width / 2
    return pv.Line(p0_, p1_).extrude(u1 * width, capping=capping).extrude(
        u2 * width, capping=capping).triangulate()


def combine_polydata_objects(objects):

    def combine(obj1, obj2):
        return obj1.append_polydata(obj2)

    return reduce(combine, objects)


def output_to_stl(nodes,
                  links,
                  node_positions,
                  node_radius=12.5,
                  link_radius=3.15,
                  resistor_links=[],
                  resistances=[],
                  in_node=None,
                  out_node=None,
                  resistors_h_paths=[],
                  resistors_v_boxes=[],
                  sphere_resolution=60,
                  link_cone_radius=None,
                  resistors_h_paths_width=0.5,
                  resistors_v_boxes_width=1.0,
                  in_out_nodes_cylinder_width=2,
                  in_out_nodes_cylinder_height=6,
                  outfile_path=None):
    '''
    Output STL files with all necessary meshes to build a sensing physicalized network
    '''
    node_positions_ = np.array(node_positions)[np.argsort(nodes)]

    # nodes
    node_objects = [
        pv.Sphere(radius=node_radius,
                  center=pos,
                  theta_resolution=sphere_resolution,
                  phi_resolution=sphere_resolution) for pos in node_positions_
    ]

    # links
    link_objects = [
        line_to_cylinder(node_positions_[s], node_positions_[t], link_radius)
        for s, t in links
    ]
    # add cone shapes for structural support
    if link_cone_radius is None:
        link_cone_radius = min(link_radius * 2, node_radius)
    if link_cone_radius > link_radius:
        for s, t in links:
            pos1 = node_positions_[s]
            pos2 = node_positions_[t]
            center_pos = (pos1 + pos2) / 2
            link_objects += [
                line_to_cone(pos1, center_pos, link_cone_radius),
                line_to_cone(pos2, center_pos, link_cone_radius)
            ]

    # resisor paths
    # 1. horizontal paths (thin)
    h_path_objects = []
    for h_paths in resistors_h_paths:
        for h_path in h_paths:
            h_path = np.array(h_path)
            for i in range(1, len(h_path)):
                mag = np.linalg.norm(h_path[i] - h_path[i - 1])
                if mag > 0:
                    u = (h_path[i] - h_path[i - 1]) / mag
                    u_z = np.array([0, 0, 1])
                    u_xy = np.cross(u, u_z)
                    pos1 = h_path[i - 1] - u * resistors_h_paths_width / 2
                    pos2 = h_path[i] + u * resistors_h_paths_width / 2
                    h_path_object = line_to_box(pos1, pos2, u_xy, u_z,
                                                resistors_h_paths_width)
                    h_path_objects.append(h_path_object)

    # 2. vertical paths (thicker)
    v_path_objects = []
    for v_boxes, (s, t) in zip(resistors_v_boxes, resistor_links):
        for v_box in v_boxes:
            box_center = np.array(v_box).mean(axis=0)
            pos1 = np.array([box_center[0], box_center[1], v_box[0][2]])
            pos2 = np.array([box_center[0], box_center[1], v_box[1][2]])
            if np.linalg.norm(pos1 - pos2) > 0:
                vec_nodes = node_positions_[t] - node_positions_[s]
                u_z = np.array([0, 0, 1])
                u_xy1 = np.array([vec_nodes[0], vec_nodes[1], 0])
                u_xy2 = -np.cross(u_z, u_xy1)
                v_path_object = line_to_box(pos1, pos2, u_xy1, u_xy2,
                                            resistors_v_boxes_width)
                v_path_objects.append(v_path_object)

    # 3. input and output points
    in_out_node_objects = []
    for in_out_node in [in_node, out_node]:
        if in_out_node is not None:
            # make a cylinder along a direction opposite from connected links
            related_links = []
            for s, t in links:
                if in_out_node == s:
                    related_links.append([s, t])
                elif in_out_node == t:
                    related_links.append([t, s])

            mean_vec = np.array([
                node_positions_[t] - node_positions_[s]
                for s, t in related_links
            ]).mean(axis=0)
            if (np.linalg.norm(mean_vec) == 0) or (np.isnan(
                    np.linalg.norm(mean_vec))):
                mean_vec = np.array([1, 0, 0])
            mean_vec /= np.linalg.norm(mean_vec)

            pos1 = node_positions_[in_out_node]
            pos2 = node_positions_[in_out_node] - (
                node_radius + in_out_nodes_cylinder_height) * mean_vec
            in_out_node_object = line_to_cylinder(pos1, pos2,
                                                  in_out_nodes_cylinder_width)
            in_out_node_objects.append(in_out_node_object)

    # 4. combine all
    resistor_objects = h_path_objects + v_path_objects + in_out_node_objects

    node_stl = combine_polydata_objects(node_objects)
    link_stl = combine_polydata_objects(link_objects)
    resistor_stl = combine_polydata_objects(resistor_objects)

    if outfile_path is not None:
        dirname = os.path.dirname(outfile_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        basename = os.path.basename(outfile_path).split('.')[0]
        node_stl.save(f'{dirname}/{basename}.node.stl')
        link_stl.save(f'{dirname}/{basename}.link.stl')
        resistor_stl.save(f'{dirname}/{basename}.resistor.stl')

    return node_stl, link_stl, resistor_stl


def output_to_json(nodes,
                   links,
                   node_positions,
                   node_radius=12.5,
                   link_radius=3.15,
                   resistor_links=[],
                   resistances=[],
                   in_node=None,
                   out_node=None,
                   resistors_h_paths=[],
                   resistors_v_boxes=[],
                   outfile_path=None):
    '''
    Output json with all information necessary to build a sensing physicalized network
    '''

    def convert_to_list(arr):
        return arr if isinstance(arr, list) else arr.tolist()

    info = {
        'nodes': convert_to_list(nodes),
        'links': [convert_to_list(link) for link in links],
        'node_positions': [convert_to_list(p) for p in node_positions],
        'node_radius': node_radius,
        'link_radius': link_radius,
        'in_node': in_node,
        'out_node': out_node,
        'resistor_links': [convert_to_list(link) for link in resistor_links],
        'resistances': convert_to_list(resistances),
        'resistors_h_paths': resistors_h_paths,
        'resistors_v_boxes': resistors_v_boxes
    }

    if outfile_path is not None:
        if not os.path.exists(os.path.dirname(outfile_path)):
            os.makedirs(os.path.dirname(outfile_path))
        with open(outfile_path, 'w') as f:
            json.dump(info, f)

    return info


def output_to_3dforce_json(nodes,
                           links,
                           node_positions,
                           node_radius=12.5,
                           link_radius=3.15,
                           outfile_path=None):
    '''
    Output json to draw a 3D network with 3d-force-graph (https://github.com/vasturiano/3d-force-graph)
    '''

    def convert_to_list(arr):
        return arr if isinstance(arr, list) else arr.tolist()

    network = {
        'nodes': [],
        'links': [],
        'nodeRadius': node_radius,
        'linkRadius': link_radius
    }
    for id, p in zip(nodes, node_positions):
        network['nodes'].append({
            'id': f'{id}',
            'fx': convert_to_list(p[0]),
            'fy': convert_to_list(p[1]),
            'fz': convert_to_list(p[2])
        })
    for link in links:
        network['links'].append({
            'source': f'{link[0]}',
            'target': f'{link[1]}'
        })

    if outfile_path is not None:
        if not os.path.exists(os.path.dirname(outfile_path)):
            os.makedirs(os.path.dirname(outfile_path))
        with open(outfile_path, 'w') as f:
            json.dump(network, f)

    return network


def to_lcapy_circuit(nodes,
                     links,
                     resistances,
                     in_node=None,
                     out_node=None,
                     touch_node=None,
                     additional_resistance=1.0e6,
                     touch_capacitance=100.0e-12,
                     arduino_inner_resistance=100.0e6,
                     voltage_value=5.0):
    '''
    Generate Lcapy's circuit (https://lcapy.readthedocs.io/en/latest/)
    '''
    ground = 0
    cct = lcapy.Circuit()

    if in_node is None:
        in_node = 0
    if out_node is None:
        out_node = in_node

    def to_cct_node(network_node, increment=1):
        # 0 is for the ground and we cannot use 0 for node id
        return network_node + increment

    # resistors corresponding to network links
    for i, (l, r) in enumerate(zip(links, resistances)):
        # args are name, source node, target node, and value
        cct.add(f'R{i} {to_cct_node(l[0])} {to_cct_node(l[1])}, {r}')

    # arduino inner resistor
    cct.add(
        f'Rarduino {to_cct_node(out_node)} {ground} {arduino_inner_resistance}'
    )

    # additional resistor
    cct.add(f'Rin voltage {to_cct_node(in_node)} {additional_resistance}')

    # voltage
    cct.add(f'Vin voltage {ground} step {voltage_value}')

    if touch_node is not None:
        # capacitor
        cct.add(
            f'Ctouch {ground} {to_cct_node(touch_node)} {touch_capacitance}')

    return cct


class LcapyCircuitSimplifier():
    '''
    For simplification of Lcapy circuit
    '''

    def __init__(self, simplified_symbol_prefix='P'):
        self.simplified_symbol_prefix = simplified_symbol_prefix
        self.resistance_name_to_expr = {}
        self.expr_to_resistance_name = {}

    def simplify(self,
                 cct,
                 remove_wires=False,
                 array_style=True,
                 no_simplification=False):
        if no_simplification:
            cct_ = cct
        else:
            cct_ = cct.simplify_series().remove_dangling_wires()
            if remove_wires:
                cct_ = self.remove_wires(cct_)

        simlified_cct = ''
        n_combined_resistors = len(self.resistance_name_to_expr)
        for line in str(cct_).split('\n'):
            c, s, t = line.split(' ')[:3]
            if c.startswith('R'):
                resistance = ' '.join(line.split(' ')[3:])

                if resistance.startswith('{'):
                    sub_resistances = resistance[1:-1].split(' + ')
                    expr = ''
                    for sub_r in sub_resistances:
                        if not expr == '':
                            expr += '+'

                        if array_style:
                            char_and_idx = re.findall(r"[^\W\d_]+|\d+", sub_r)
                            if not char_and_idx[0].isnumeric():
                                expr += f'r[{char_and_idx[1]}]'  # e.g., r[3]
                            else:
                                expr += f'{sub_r}'  # e.g., 10000
                        else:
                            expr += f'{sub_r}'

                    if expr in self.expr_to_resistance_name:
                        resistance_name = self.expr_to_resistance_name[expr]
                    else:
                        n_combined_resistors += 1
                        resistance_name = f'{self.simplified_symbol_prefix}{n_combined_resistors}'
                        self.resistance_name_to_expr[resistance_name] = expr
                        self.expr_to_resistance_name[expr] = resistance_name

                    simlified_cct += f'{c} {s} {t} {resistance_name}\n'
                else:
                    if array_style:
                        char_and_idx = re.findall(r"[^\W\d_]+|\d+", resistance)
                        if not char_and_idx[0].isnumeric():
                            expr = f'r[{char_and_idx[1]}]'  # e.g., r[3]
                        else:
                            expr = resistance  # e.g., 10000
                    else:
                        expr = resistance

                    self.resistance_name_to_expr[resistance] = expr
                    self.expr_to_resistance_name[expr] = resistance

                    simlified_cct += line + '\n'
            else:
                simlified_cct += line + '\n'

        simlified_cct = lcapy.Circuit(simlified_cct)

        return simlified_cct

    def remove_wires(self, cct):
        # TODO: this functon assumes the lcapy circuit is made by to_lcapy_circuit
        simplified_cct = cct.remove_dangling_wires()
        wire_source_to_dest = {}
        non_wires = []
        simplified_cct_str = ''

        in_wire_path = False
        for line in str(simplified_cct).split('\n'):
            c, s, t = line.split(' ')[:3]
            if c == 'W':  # wire
                if not in_wire_path:
                    wire_source = s
                wire_source_to_dest[wire_source] = t
                in_wire_path = True
            else:
                in_wire_path = False
                non_wires.append(line)
            for line in non_wires:
                split_line = line.split(' ')
                c, s, t = split_line[:3]
                args = " ".join(split_line[3:])
                if t in wire_source_to_dest:
                    t = wire_source_to_dest[t]
                simplified_cct_str += f'{c} {s} {t} {args}\n'

            simplified_cct = lcapy.Circuit(simplified_cct_str)

        return simplified_cct


###
### functions below are to generate ngspice files
###
def to_ngspice_circuit_text(nodes,
                            links,
                            resistances,
                            outfile_path=None,
                            in_node=None,
                            out_node=None,
                            touch_node=None,
                            additional_resistance=1.0e6,
                            touch_capacitance=100e-12,
                            arduino_inner_resistance=100e6,
                            voltage_value=5):
    ## this is for use with ngspice
    ## to run generated net file need to install ngspice (e.g., brew install ngspice)
    ## then, ngspice -b out.cir
    cir = to_ngspice_circuit(nodes,
                             links,
                             resistances,
                             in_node=in_node,
                             out_node=out_node,
                             touch_node=touch_node,
                             additional_resistance=additional_resistance,
                             touch_capacitance=touch_capacitance,
                             arduino_inner_resistance=arduino_inner_resistance,
                             voltage_value=voltage_value)

    if outfile_path is not None:
        if not os.path.exists(os.path.dirname(outfile_path)):
            os.makedirs(os.path.dirname(outfile_path))
        with open(outfile_path, 'w') as f:
            f.write(cir.str())

    return cir.str()


def to_ngspice_circuit(nodes,
                       links,
                       resistances,
                       in_node=None,
                       out_node=None,
                       touch_node=None,
                       additional_resistance=1e6,
                       touch_capacitance=100e-12,
                       arduino_inner_resistance=100e6,
                       voltage_value=5):
    from PySpice.Spice.Netlist import Circuit

    def to_circuit_node(network_node, increment=1):
        # pyspice allocate 0 to the ground and we cannot use 0 for node id
        return network_node + increment

    if in_node is None:
        in_node = 0
    if out_node is None:
        out_node = in_node

    cir = Circuit('')

    # resistors corresponding to network links
    for i, (l, r) in enumerate(zip(links, resistances)):
        # args are name, source node, target node, and value
        cir.R(i, to_circuit_node(l[0]), to_circuit_node(l[1]), r)

    # arduino inner resistor
    cir.R('arduino', to_circuit_node(out_node), cir.gnd,
          arduino_inner_resistance)
    # additional resistor (TODO: this might not be needed)
    cir.R('additional', 'voltage', to_circuit_node(in_node),
          additional_resistance)
    # voltage
    cir.V('in', 'voltage', cir.gnd, f'PWL(0 0 1n {voltage_value})')

    if touch_node is not None:
        # capacitor
        cir.C('touch', to_circuit_node(touch_node), cir.gnd, touch_capacitance)

    return cir


###
### functions below are to generate lgspice drawing file
###
def to_ltspice_drawing_text(nodes,
                            links,
                            resistances,
                            outfile_path=None,
                            scale_factor=5,
                            in_node=None,
                            out_node=None,
                            touch_node=None,
                            additional_resistance='1Meg',
                            touch_capacitance='100p',
                            arduino_inner_resistance='100Meg',
                            voltage_value='5'):
    version = 4
    sheet_size = [880, 680]
    R_width = 32  # LTSpice's default
    R_height = 96  # LTSpice's default
    R_height_margin = 16  # Idk why some slight positional diff exists
    capacitor_width = 32  # LTSpice's default
    capacitor_height = 64  # LTSpice's default

    if in_node is None:
        in_node = 0
    if out_node is None:
        out_node = in_node

    header_text = f'Version {version}\nSHEET 1 {sheet_size[0]} {sheet_size[1]}\n'

    resistor_positions = _layout_resistors(nodes, links)
    Rs, Rs_text = _to_ltspice_resistors(resistances, resistor_positions)
    Ns, Ws, Ws_text = _to_ltspice_nodes_wires(links, Rs)

    in_N = Ns[in_node]
    out_N = Ns[out_node]
    touch_N = Ns[touch_node]

    min_x, max_x, min_y, max_y = _get_pos_minmax(Rs)
    max_y += R_height
    max_x += R_width // 2
    min_x -= R_width // 2

    # voltage
    V_x = min_x - 200
    V_y = (min_y + max_y) // 2
    V_text = f'SYMBOL voltage {V_x} {V_y} R0\n'
    V_text += f'SYMATTR InstName Vin\n'
    V_text += f'SYMATTR Value PWL(0 0 1n {voltage_value})\n'

    # arduino inner resistor
    arduino_R_x = (min_x + max_x) // 2
    arduino_R_y = max_y + 100
    Rs_text += f'SYMBOL res {arduino_R_x} {arduino_R_y} R0\n'
    Rs_text += f'SYMATTR InstName Rarduino\n'
    Rs_text += f'SYMATTR Value {arduino_inner_resistance}\n'

    # additional resistor
    additional_R_x = (min_x + max_x) // 2
    additional_R_y = min_y - 100 - R_height
    Rs_text += f'SYMBOL res {additional_R_x} {additional_R_y} R0\n'
    Rs_text += f'SYMATTR InstName RAdditional\n'
    Rs_text += f'SYMATTR Value {additional_resistance}\n'

    # wire to measure V_out
    W_out_x1 = arduino_R_x + R_width // 2
    W_out_y1 = arduino_R_y + R_height_margin
    W_out_x2 = arduino_R_x + R_width // 2 + 100
    W_out_y2 = arduino_R_y + R_height_margin
    Ws_text += f'WIRE {W_out_x1} {W_out_y1} {W_out_x2} {W_out_y2}\n'
    Ws_text += f'FLAG {W_out_x2} {W_out_y2} Out\n'

    # wire to ground
    W_gnd_y1 = arduino_R_y + R_height
    W_gnd_y2 = arduino_R_y + R_height + 100
    Ws_text += f'WIRE {V_x} {W_gnd_y1} {V_x} {W_gnd_y2}\n'
    Ws_text += f'FLAG {V_x} {W_gnd_y2} 0\n'

    # wire to connct ground and arduino resistor bottom
    Ws_text += f'WIRE {V_x} {W_gnd_y1} {W_out_x1} {arduino_R_y + R_height}\n'

    # wire to connect voltage and ground
    Ws_text += f'WIRE {V_x} {V_y} {V_x} {W_gnd_y1}\n'

    # wires to connect voltage and additonal resistor
    Ws_text += f'WIRE {V_x} {V_y} {V_x} {additional_R_y + R_height_margin}\n'
    Ws_text += f'WIRE {V_x} {additional_R_y + R_height_margin} {additional_R_x + R_width // 2} {additional_R_y + R_height_margin}\n'

    # wire to connect additional resistor and in_node
    Ws_text += f'WIRE {additional_R_x + R_width // 2} {additional_R_y + R_height} {in_N["x"]} {in_N["y"]}\n'

    # wire to connect arduino resistor and out_node
    Ws_text += f'WIRE {W_out_x1} {W_out_y1} {out_N["x"]} {out_N["y"]}\n'

    # capacitor to mimic touch
    C_text = ''
    if touch_node is not None:
        C_x = max_x + 100
        C_y = arduino_R_y + R_height - capacitor_height
        C_text += f'SYMBOL cap {C_x} {C_y} R0\n'
        C_text += f'SYMATTR InstName Ctouch\n'
        C_text += f'SYMATTR Value {touch_capacitance}\n'
        # connection to ground level wire
        Ws_text += f'WIRE {C_x + capacitor_width // 2} {W_gnd_y1} {arduino_R_x + capacitor_width // 2} {W_gnd_y1}\n'
        # connection to touch node point
        Ws_text += f'WIRE {C_x + capacitor_width // 2} {C_y} {touch_N["x"]} {touch_N["y"]}\n'

    # analysis_text = f'TEXT {min_x - 400} {min_y} LEFT 2 !.tran 1m\n'
    # analysis_text += f'TEXT {min_x - 400} {min_y + 50} LEFT 2 !.measure TRAN t_thres time when V(OUT)=2.5\n'

    ltspice_text = header_text + Rs_text + Ws_text + V_text + C_text

    if outfile_path is not None:
        if not os.path.exists(os.path.dirname(outfile_path)):
            os.makedirs(os.path.dirname(outfile_path))
        with open(outfile_path, 'w') as f:
            f.write(ltspice_text)

    return ltspice_text


def _flip_nodes_links(nodes, links):
    fnodes = []  # fnodes: nodes in the flipped network
    flinks = []
    fnode_to_rowcol = {}
    rowcol_to_fnode = {}

    n = len(nodes)
    m = len(links)
    A = np.zeros((n, n), dtype=bool)
    for link in links:
        source, target = link
        A[source, target] = 1
        A[target, source] = 1

        fnode = len(fnodes)
        fnodes.append(fnode)
        fnode_to_rowcol[fnode] = [min(source, target), max(source, target)]
        rowcol_to_fnode[f'{source}-{target}'] = fnode
        rowcol_to_fnode[f'{target}-{source}'] = fnode

    fA = np.zeros((m, m), dtype=bool)
    for fsource in fnodes:
        r, c = fnode_to_rowcol[fsource]
        for c_ in np.where(A[r, :])[0]:
            ftarget = rowcol_to_fnode[f'{r}-{c_}']
            if (not fsource == ftarget) and (not fA[fsource, ftarget]):
                fA[fsource, ftarget] = True
                fA[ftarget, fsource] = True
                flinks.append([fsource, ftarget])
        for r_ in np.where(A[:, c])[0]:
            ftarget = rowcol_to_fnode[f'{r_}-{c}']
            if (not fsource == ftarget) and (not fA[fsource, ftarget]):
                fA[fsource, ftarget] = True
                fA[ftarget, fsource] = True
                flinks.append([fsource, ftarget])

    return fnodes, flinks


def _layout_resistors(nodes, links):
    ## to use graphviz (python), run commands like below
    ## brew install graphviz
    ## pip3 install \
    ##     --global-option=build_ext \
    ##     --global-option="-I$(brew --prefix graphviz)/include/" \
    ##     --global-option="-L$(brew --prefix graphviz)/lib/" \
    ##     pygraphviz
    ## pip3 install graphviz
    import graphviz

    # This is to avoid overlapping resistors due to LTSpice's spec
    fnodes, flinks = _flip_nodes_links(nodes, links)
    g = graphviz.Digraph('test',
                         engine='dot',
                         graph_attr={
                             'splines': 'line',
                             'nodesep': '0.8'
                         },
                         node_attr={
                             'shape': 'box',
                             'fixedsize': 'true',
                             'width': '0.1',
                             'height': '0.3'
                             ''
                         },
                         edge_attr={
                             'arrowhead': 'none',
                             'headport': 'n',
                             'tailport': 's'
                         })
    g.attr(size='10,10')
    for flink in flinks:
        s, t = flink
        g.edge(f'{s}', f'{t}')  # graphviz can only use strings
    dot_text = g.pipe(format='dot')

    # need to output dot format file to get node positions, etc
    dot_file_name = 'tmp.dot'
    with open(dot_file_name, 'wb') as f:
        f.write(dot_text)

    resistors = nx.drawing.nx_agraph.read_dot(dot_file_name).nodes
    os.remove(dot_file_name)

    pos = [resistors[key]['pos'].split(',') for key in resistors]

    return np.array(pos, dtype=float)


def _get_pos_minmax(Rs):
    min_x = np.iinfo(np.int32).max
    max_x = np.iinfo(np.int32).min
    min_y = np.iinfo(np.int32).max
    max_y = np.iinfo(np.int32).min
    for R in Rs:
        if min_x > R['x']:
            min_x = R['x']
        if max_x < R['x']:
            max_x = R['x']
        if min_y > R['y']:
            min_y = R['y']
        if max_y < R['y']:
            max_y = R['y']

    return min_x, max_x, min_y, max_y


def _to_ltspice_resistors(
        resistances,
        resistor_positions,
        scale_factor=5,
        resistor_width=32,  # LTSpice's default
        resistor_height=96  # LTSpice's default
):
    Rs = []
    for xy, resistance in zip(resistor_positions, resistances):
        # LTSpice allows only integers and y-coord is reverse
        Rs.append({
            'name': f'R{len(Rs)}',
            'x': int(xy[0] * scale_factor),
            'y': -int(xy[1] * scale_factor),
            'width': resistor_width,
            'height': resistor_height,
            'value': resistance
        })

    Rs_text = ''
    for R in Rs:
        Rs_text += f'SYMBOL res {R["x"]} {R["y"]} R0\n'
        Rs_text += f'SYMATTR InstName {R["name"]}\n'
        Rs_text += f'SYMATTR Value {R["value"]}\n'

    return Rs, Rs_text


def _to_ltspice_nodes_wires(links,
                            ltspice_resistors,
                            resistor_height_margin=16):
    # for wires, use original graph's links to connect resistors
    # (resistor's top and bottom can be considered source and target respectively)
    Ws = []
    Ns = {}  # circuit nodes
    for link, R in zip(links, ltspice_resistors):
        source, target = link
        if not source in Ns:
            Ns[source] = {
                'x': R['x'] + R['width'] // 2,
                'y': R['y'] + resistor_height_margin
            }
        else:
            # wire connecting resistor and source node point
            Ws.append({
                'x1': Ns[source]['x'],
                'y1': Ns[source]['y'],
                'x2': R['x'] + R['width'] // 2,
                'y2': R['y'] + resistor_height_margin
            })

        if not target in Ns:
            Ns[target] = {
                'x': R['x'] + R['width'] // 2,
                'y': R['y'] + R['height']
            }
        else:
            # wire connecting resistor and target node point
            Ws.append({
                'x1': R['x'] + R['width'] // 2,
                'y1': R['y'] + R['height'],
                'x2': Ns[target]['x'],
                'y2': Ns[target]['y']
            })

    Ws_text = ''
    for w in Ws:
        Ws_text += f'WIRE {w["x1"]} {w["y1"]} {w["x2"]} {w["y2"]}\n'

    return Ns, Ws, Ws_text


if __name__ == '__main__':
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [1, 2], [1, 3], [2, 3]]
    resistances = [1e5] * len(links)
    in_node = 0
    out_node = 3

    lcapy_circuit = to_lcapy_circuit(nodes,
                                     links,
                                     resistances=resistances,
                                     in_node=in_node,
                                     out_node=out_node)
    print(lcapy_circuit)

    ## ngspice example
    # ngspice_circuit = to_ngspice_circuit_text(nodes,
    #                                           links,
    #                                           resistances,
    #                                           outfile='out.cir',
    #                                           in_node=in_node,
    #                                           touch_node=1)
    # print(ngspice_circuit)

    ## ltspice example
    # ltspice_drawing = to_ltspice_drawing_text(nodes,
    #                                           links,
    #                                           resistances,
    #                                           outfile='out.asc',
    #                                           in_node=in_node,
    #                                           out_node=out_node,
    #                                           touch_node=1)
