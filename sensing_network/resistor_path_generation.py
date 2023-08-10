'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import numpy as np


class ResistorPathGenerator():

    def __init__(self,
                 resistance,
                 node1_pos,
                 node2_pos,
                 link_radius,
                 node_radius,
                 vertical_step=0.6,
                 path_margin=0.55,
                 vertical_box_width=0.8,
                 vertical_box_additional_height=0.4,
                 min_node_link_overlap=3.0,
                 vertical_resistivity=2100.0,
                 horizontal_resistivity=890.0,
                 verbose=False):
        self.resistance = resistance
        self.link_r = link_radius
        self.node_r = node_radius
        self.v_step = vertical_step
        self.margin = path_margin
        self.vbox_width = vertical_box_width
        self.vbox_additional_height = vertical_box_additional_height
        self.min_node_link_overlap = min_node_link_overlap
        self.v_resist = vertical_resistivity
        self.h_resist = horizontal_resistivity
        self.verbose = verbose

        self.s_pos, self.t_pos = self._select_source_target_positions(
            node1_pos, node2_pos)
        self._prepare_geom_info()
        self.start_z, self.end_z = self._find_start_end_zcoords()

    def _select_source_target_positions(self, node1_pos, node2_pos):
        if node1_pos[2] < node2_pos[2]:
            source_pos = node1_pos
            target_pos = node2_pos
        else:
            source_pos = node2_pos
            target_pos = node1_pos
        source_pos = np.array(source_pos)
        target_pos = np.array(target_pos)

        # avoid overlapping with nodes
        u = (target_pos - source_pos) / np.linalg.norm(target_pos - source_pos)
        source_pos += (self.node_r + self.margin) * u
        target_pos -= (self.node_r + self.margin) * u

        return source_pos, target_pos

    def _prepare_geom_info(self):
        self.v_dist = np.linalg.norm(self.t_pos[2] - self.s_pos[2])
        self.h_dist = np.linalg.norm(self.t_pos[:2] - self.s_pos[:2])
        self.dist = np.linalg.norm(self.t_pos - self.s_pos)

        # info of ellipse appeared when cutting a link along a horizontal direciton
        self.ellipse_a = self.link_r  # shorter side
        self.ellipse_b = self.ellipse_a * self.dist / self.v_dist if self.v_dist > 0 else None  # longer side

        # info of a rectangle inside of the ellipse, which has the maximum area
        self.rect_a = self.ellipse_a * np.sqrt(2.0)
        if self.ellipse_b is not None:
            self.rect_b = self.ellipse_b * np.sqrt(2.0)
        if self.v_dist == 0:
            self.rect_b = self.dist

        # to know how we should move x and y coodinates
        b_unit_vec = np.array([1.0, 0.0, 0.0])
        if self.h_dist > 0:
            b_unit_vec[:2] = self.t_pos[:2] - self.s_pos[:2]
            b_unit_vec /= self.h_dist
        a_unit_vec = np.zeros_like(b_unit_vec)
        a_unit_vec[0] = -b_unit_vec[1]
        a_unit_vec[1] = b_unit_vec[0]
        self.u_a = a_unit_vec
        self.u_b = b_unit_vec

        # to know when we can use full ellipse
        self.s_zbounds = np.array([
            self.s_pos[2] -
            self.link_r * self.dist / max(self.h_dist, self.v_dist),
            self.s_pos[2] +
            self.link_r * self.dist / max(self.h_dist, self.v_dist)
        ])
        self.t_zbounds = np.array([
            self.t_pos[2] -
            self.link_r * self.dist / max(self.h_dist, self.v_dist),
            self.t_pos[2] +
            self.link_r * self.dist / max(self.h_dist, self.v_dist)
        ])

        # u = (self.t_pos - self.s_pos) / self.dist
        # self.s_zbounds = np.array([(self.s_pos - self.node_r * u)[2] -
        #                            self.link_r * self.dist / self.h_dist,
        #                            (self.s_pos - self.node_r * u)[2] +
        #                            self.link_r * self.dist / self.h_dist])
        # self.t_zbounds = np.array([(self.t_pos + self.node_r * u)[2] -
        #                            self.link_r * self.dist / self.h_dist,
        #                            (self.t_pos + self.node_r * u)[2] +
        #                            self.link_r * self.dist / self.h_dist])

        return self

    def _find_start_end_zcoords(self):
        min_z = self.s_pos[2] - self.v_dist * (self.link_r -
                                               self.margin) / self.dist
        max_z = self.t_pos[2] + self.v_dist * (self.link_r -
                                               self.margin) / self.dist
        min_z = self.s_pos[2] - self.v_dist * (self.link_r -
                                               self.margin) / self.dist
        max_z = self.t_pos[2] + self.v_dist * (self.link_r -
                                               self.margin) / self.dist

        # to handle the case a link is close to horizontal
        # (2 * self.margin: taking a slightly wider margin)
        min_z = min(min_z, self.s_pos[2] - (self.link_r - 2 * self.margin))
        max_z = max(min_z, self.t_pos[2] + (self.link_r - 2 * self.margin))

        start_z = (min_z // self.v_step) * self.v_step
        end_z = (max_z // self.v_step) * self.v_step
        start_z += self.v_step
        # if start_z < 0:
        #     start_z += self.v_step
        # if end_z < 0:
        #     end_z += self.v_step

        return start_z, end_z

    def _path_start_end_for_even_layers(self, layer_center):
        # find start and end in a local coordinate of each layer
        # i = 0, 2, 4, 6, 8...
        center_z = layer_center[2]
        start = np.array((0, -self.rect_b / 2))
        end = np.array((0, self.rect_b / 2))

        if (center_z < self.s_zbounds[1]) and (center_z > self.t_zbounds[0]):
            u = (self.t_pos - self.s_pos) / self.dist
            u_orth = np.array((u[2], -u[0], -u[2]))

            s_face_pos = self.s_pos + u_orth * (
                center_z - self.s_pos[2]) * self.v_dist / self.h_dist
            s_face_pos[2] = center_z
            t_face_pos = self.t_pos + u_orth * (
                center_z - self.t_pos[2]) * self.v_dist / self.h_dist
            t_face_pos[2] = center_z

            if np.dot(t_face_pos[:2] - layer_center[:2], u[:2]) > 0:
                end[1] = np.linalg.norm(t_face_pos[:2] - layer_center[:2])
            else:
                end[1] = -np.linalg.norm(t_face_pos[:2] - layer_center[:2])
            start[1] = end[1] - np.linalg.norm(t_face_pos[:2] - s_face_pos[:2])
            # start[1] = end[1] - np.linalg.norm(
            #     self.t_pos - self.s_pos) * self.dist / self.h_dist

            if (self.v_dist > 0) and (center_z - self.t_zbounds[0]
                                      < self.margin):
                end[1] -= self.margin * self.h_dist / self.v_dist
            if (self.v_dist > 0) and (self.s_zbounds[1] - center_z
                                      < self.margin):
                start[1] += self.margin * self.h_dist / self.v_dist
            # if self.v_dist > 0:
            #     end[1] -= self.margin * self.h_dist / self.v_dist
            #     start[1] += self.margin * self.h_dist / self.v_dist

        elif center_z < self.s_zbounds[1]:
            till_s_face = self.rect_b * (center_z - self.s_zbounds[0]) / (
                self.s_zbounds[1] - self.s_zbounds[0])
            start[1] = end[1] - till_s_face
        elif center_z > self.t_zbounds[0]:
            till_t_face = self.rect_b * (center_z - self.t_zbounds[0]) / (
                self.t_zbounds[1] - self.t_zbounds[0])
            end[1] = end[1] - till_t_face
            # till_t_face = self.rect_b * (self.t_zbounds[1] - center_z) / (
            #     self.t_zbounds[1] - self.t_zbounds[0])
            # end[1] = start[1] - till_t_face

        start[1] += self.margin
        end[1] -= self.margin

        return np.array(start), np.array(end)

    def _path_start_end_for_odd_layers(self, layer_center, prev_end,
                                       next_start):
        # i = i, 3, 5, 7, 9...
        center_z = layer_center[2]
        h_for_v_step = 0
        if self.v_dist > 0:
            h_for_v_step = self.v_step * self.h_dist / self.v_dist

        start = prev_end - np.array((0, h_for_v_step))
        if next_start is not None:
            end = next_start + np.array((0, h_for_v_step))
        else:
            end = start

        return start, end

    def _generate_layer_path(self, start, end, zigzag_a):
        path = [start]

        sign_a = 1
        sign_b = 1 if end[1] > start[1] else -1
        while sign_b * (end[1] - path[-1][1]) > self.margin * 2:
            p0 = path[-1]
            p1 = p0 + sign_a * np.array((zigzag_a / 2, 0))
            p2 = p1 + sign_b * np.array((0, self.margin * 2))
            p3 = p2 - sign_a * np.array((zigzag_a / 2, 0))

            path += [p1, p2, p3]
            sign_a *= -1

        if len(path) == 1:
            path.append((start + end) / 2)
        path.append(end)

        return np.array(path)

    def _generate_path_from_node(self, layer_paths):
        layer_path_start = layer_paths[0][0]
        layer_path_end = layer_paths[-1][-1]

        u = (self.t_pos - self.s_pos) / self.dist
        s_pos_ = self.s_pos - self.min_node_link_overlap * u
        t_pos_ = self.t_pos + self.min_node_link_overlap * u
        s_pos_[2] = (s_pos_[2] // self.v_step - 1) * self.v_step
        t_pos_[2] = (t_pos_[2] // self.v_step + 1) * self.v_step

        s_v = layer_path_start[2] - s_pos_[2]
        s_h = np.linalg.norm(layer_path_start[:2] - s_pos_[:2])
        s_dh = s_h
        if self.v_step > 0:
            s_dh = s_h / (1 + abs(s_v) / self.v_step)
        s_u_h = np.zeros_like(u)
        if s_h > 0:
            s_u_h[:2] = (layer_path_start[:2] - s_pos_[:2]) / s_h
        s_u_v = np.array((0, 0, 1)) if s_v > 0 else np.array((0, 0, -1))

        # adding a curve in Rhino needs more than 2 points, so hlines have 3 points here
        s_hlines = [[s_pos_, s_pos_ + s_dh * s_u_h / 2, s_pos_ + s_dh * s_u_h]]
        s_vlines = []
        for _ in range(int(abs(s_v) / self.v_step) + 1):
            p0 = s_hlines[-1][-1]
            p1 = p0 + self.v_step * s_u_v
            p2 = p1 + s_dh * s_u_h
            s_vlines.append([p0, p1])
            s_hlines.append([p1, (p1 + p2) / 2, p2])

        t_v = t_pos_[2] - layer_path_end[2]
        t_h = np.linalg.norm(t_pos_[:2] - layer_path_end[:2])
        t_dh = t_h
        if self.v_step > 0:
            t_dh = t_h / (1 + abs(t_v) / self.v_step)
        t_u_h = np.zeros_like(u)
        if t_h > 0:
            t_u_h[:2] = (t_pos_[:2] - layer_path_end[:2]) / t_h
        t_u_v = np.array((0, 0, 1)) if t_v > 0 else np.array((0, 0, -1))

        t_hlines = [[
            layer_path_end, layer_path_end + t_dh * t_u_h / 2,
            layer_path_end + t_dh * t_u_h
        ]]
        t_vlines = []
        for _ in range(int(abs(t_v) / self.v_step) + 1):
            p0 = t_hlines[-1][-1]
            p1 = p0 + self.v_step * t_u_v
            p2 = p1 + t_dh * t_u_h
            t_vlines.append([p0, p1])
            t_hlines.append([p1, (p1 + p2) / 2, p2])

        s_hlines = np.array(s_hlines).tolist()
        s_vlines = np.array(s_vlines).tolist()
        t_hlines = np.array(t_hlines).tolist()
        t_vlines = np.array(t_vlines).tolist()

        return s_hlines, s_vlines, t_hlines, t_vlines

    def _compute_total_h_resist(self, layer_paths):
        total_h_length = 0
        for path in layer_paths:
            for i in range(len(path) - 1):
                p0 = path[i]
                p1 = path[i + 1]
                total_h_length += np.linalg.norm(p1 - p0)
        total_h_resist = self.h_resist * total_h_length

        return total_h_resist

    def _compress_h_path(self, h_path):
        compressed_h_path = []
        for p in h_path:
            if (len(compressed_h_path) == 0):
                compressed_h_path.append(p)
            else:
                if not np.all(p == compressed_h_path[-1]):
                    compressed_h_path.append(p)
        return compressed_h_path

    def generate_path(self, convert_to_v_box=True):
        layer_zs = np.arange(self.start_z, self.end_z + self.v_step,
                             self.v_step)

        layer_centers = []
        for z in layer_zs:
            center = (self.s_pos + self.t_pos) / 2
            if self.v_dist > 0:
                center = center + self.u_b * self.h_dist * (
                    z - center[2]) / self.v_dist
            center[2] = z
            layer_centers.append(center)

        starts_ends = [None] * len(layer_centers)
        # prepare even layers first
        for i, center in enumerate(layer_centers):
            if i % 2 == 0:
                starts_ends[i] = self._path_start_end_for_even_layers(center)
        # prepare odd layers
        for i, center in enumerate(layer_centers):
            if i % 2 == 1:
                prev_end = starts_ends[i - 1][1]
                next_start = None
                if i < len(layer_centers) - 1:
                    next_start = starts_ends[i + 1][0]
                starts_ends[i] = self._path_start_end_for_odd_layers(
                    center, prev_end, next_start)

        total_v_resist = self.v_dist * self.v_resist
        aiming_total_h_resist = self.resistance - total_v_resist

        # compute delta of total_h_resist when zigzag_a increases 1
        total_h_resist0 = self._compute_total_h_resist(
            [self._generate_layer_path(s, e, 0) for s, e in starts_ends])
        total_h_resist1 = self._compute_total_h_resist(
            [self._generate_layer_path(s, e, 1) for s, e in starts_ends])
        delta_total_h_resist = total_h_resist1 - total_h_resist0

        zigzag_a = (aiming_total_h_resist -
                    total_h_resist0) / delta_total_h_resist
        zigzag_a = min(zigzag_a, self.rect_a - self.margin)
        if zigzag_a < 0.5:
            if (zigzag_a - 0.0) < (0.5 - zigzag_a):
                zigzag_a = 0
            else:
                zigzag_a = 0.5

        if self.verbose:
            print('zigzag_a', zigzag_a)

        layer_paths = [
            self._generate_layer_path(s, e, zigzag_a) for s, e in starts_ends
        ]
        total_h_resist = self._compute_total_h_resist(layer_paths)

        total_resist = total_h_resist + total_v_resist
        if self.verbose and (abs(self.resistance - total_resist) > 500):
            print('specified resistance was not able to be produced.')
            print(f'aimed: {self.resistance}, actual: {total_resist}')

        h_paths = []
        for center, path in zip(layer_centers, layer_paths):
            h_path = []
            for p in path:
                pos_from_center = p[0] * self.u_a + p[1] * self.u_b
                h_path.append((pos_from_center + center).tolist())

            h_path = self._compress_h_path(h_path)
            if len(h_path) > 2:
                h_paths.append(h_path)

        v_lines = []
        for i in range(len(h_paths) - 1):
            current_layer_end = h_paths[i][-1]
            higher_layer_start = h_paths[i + 1][0]
            v_lines.append([current_layer_end, higher_layer_start])

        s_hlines, s_vlines, t_hlines, t_vlines = self._generate_path_from_node(
            h_paths)
        h_paths = s_hlines + h_paths + t_hlines
        v_lines = s_vlines + v_lines + t_vlines

        if convert_to_v_box:
            # these adjustments are to make more space with h paths
            v_line_adjusts = [np.zeros_like(self.u_b)] * len(s_vlines)
            for i in range(len(h_paths) - 1):
                if i % 2 == 0:
                    v_line_adjusts.append(self.u_b)
                else:
                    v_line_adjusts.append(-self.u_b)
            v_line_adjusts += [np.zeros_like(self.u_b)] * len(t_vlines)
            v_lines = self.vertical_lines_to_boxes(
                v_lines,
                v_line_adjusts,
                width=self.vbox_width,
                additional_height=self.vbox_additional_height)

        return h_paths, v_lines, total_resist

    def vertical_line_to_box(self, line, pos_adjust, width, additional_height):
        line = np.array(line) + pos_adjust * width / 6
        p0 = line.min(axis=0)
        p1 = line.max(axis=0)

        diff = np.array([1, 0, 0]) * width / 2 + np.array([0, 1, 0
                                                           ]) * width / 2
        p0 -= np.abs(diff)
        p1 += np.abs(diff)
        p0[2] -= additional_height / 2
        p1[2] += additional_height / 2

        return [p0.tolist(), p1.tolist()]

    def vertical_lines_to_boxes(self,
                                lines,
                                pos_adjusts,
                                width=0.8,
                                additional_height=0.4):
        return [
            self.vertical_line_to_box(line,
                                      p_adjust,
                                      width=width,
                                      additional_height=additional_height)
            for line, p_adjust in zip(lines, pos_adjusts)
        ]


def generate_all_paths(node_positions,
                       resistor_links,
                       resistances,
                       link_radius=3.15,
                       node_radius=8.5,
                       vertical_step=0.6,
                       path_margin=0.55,
                       vertical_box_width=0.8,
                       vertical_box_additional_height=0.4,
                       min_node_link_overlap=5.0,
                       vertical_resistivity=2100.0,
                       horizontal_resistivity=890.0):
    all_h_paths = []
    all_v_boxes = []
    for resistor_link, resistance in zip(resistor_links, resistances):
        node1_pos = node_positions[resistor_link[0]]
        node2_pos = node_positions[resistor_link[1]]

        generator = ResistorPathGenerator(
            resistance,
            node1_pos=node1_pos,
            node2_pos=node2_pos,
            link_radius=link_radius,
            node_radius=node_radius,
            vertical_step=vertical_step,
            path_margin=path_margin,
            vertical_box_width=vertical_box_width,
            vertical_box_additional_height=vertical_box_additional_height,
            min_node_link_overlap=min_node_link_overlap,
            vertical_resistivity=vertical_resistivity,
            horizontal_resistivity=horizontal_resistivity)
        h_paths, v_boxes, total_resist = generator.generate_path()

        adjusted_v_step = vertical_step * 2
        while abs(resistance -
                  total_resist) > 500 and adjusted_v_step < link_radius:
            # try doubled vertical step
            generator = ResistorPathGenerator(
                resistance,
                node1_pos=node1_pos,
                node2_pos=node2_pos,
                link_radius=link_radius,
                node_radius=node_radius,
                vertical_step=adjusted_v_step,
                path_margin=path_margin,
                vertical_box_width=vertical_box_width,
                vertical_box_additional_height=vertical_box_additional_height,
                min_node_link_overlap=min_node_link_overlap,
                vertical_resistivity=vertical_resistivity,
                horizontal_resistivity=horizontal_resistivity)
            h_paths_, v_boxes_, total_resist_ = generator.generate_path()
            adjusted_v_step *= 2

            if abs(resistance - total_resist_) < abs(resistance -
                                                     total_resist):
                h_paths = h_paths_
                v_boxes = v_boxes_
                total_resist = total_resist_

        if abs(resistance - total_resist) > 1000:
            print(
                f'For Link {resistor_link}, specified resistance was not able to be produced.'
            )
            print(f'aimed: {resistance}, actual: {total_resist}')

        all_h_paths.append(h_paths)
        all_v_boxes.append(v_boxes)

    return all_h_paths, all_v_boxes


if __name__ == '__main__':

    # positions are in mm
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    node_positions = [
        [21.90741539001465, -9.782852172851562, -6.341604232788086],
        [-18.3528995513916, -15.735390663146973, -5.31614351272583],
        [-0.02511332742869854, 4.449100017547607, 23.544418334960938],
        [-3.5293960571289062, 21.069141387939453, -11.886672019958496]
    ]
    node_radius = 8.5
    link_radius = 3.15
    in_node = 0
    out_node = 3
    resistor_links = [[0, 1], [1, 2], [2, 3]]
    resistances = [300000.0, 299307.6283351532, 298726.9327050891]

    all_h_paths, all_v_boxes = generate_all_paths(
        node_positions=node_positions,
        resistor_links=resistor_links,
        resistances=resistances,
        node_radius=node_radius,
        link_radius=link_radius)

    print('horizontal paths:', all_h_paths)
    print('vertical boxes:', all_v_boxes)
