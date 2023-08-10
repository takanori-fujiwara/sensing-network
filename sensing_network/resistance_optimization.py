'''
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Copyright (c) 2023, Takanori Fujiwara and S. Sandra Bae
All rights reserved.
'''

import time
import re

import numpy as np
import sympy as sy
from scipy import optimize
from pathos.multiprocessing import ProcessPool as Pool

from sensing_network.convert_utils import to_lcapy_circuit, LcapyCircuitSimplifier


class ResistanceOptimization():

    def __init__(self,
                 nodes,
                 links,
                 in_node=None,
                 out_node=None,
                 capacitance=100e-12,
                 voltage_thres=2.5,
                 n_jobs=-1,
                 circuit_simplification=True,
                 verbose=True,
                 logging=False):
        self.nodes = nodes
        self.links = links
        self.in_node = 0 if in_node is None else in_node
        self.out_node = out_node
        self.capacitance = capacitance
        self.voltage_thres = voltage_thres
        self.n_jobs = n_jobs
        self.circuit_simplification = circuit_simplification
        self.verbose = verbose
        self.logging = logging
        self.opt_log = []

        self._ground = 0
        self._rnames = [f'r{i}' for i in range(len(self.links))]
        self._circuit_simplifier = LcapyCircuitSimplifier()

        self._iter_count = 0
        self._info = None  # TODO: find better name
        self._sq_diff_func_symbols = None
        self._lambdified_sq_diff_funcs = None
        self._lambdified_sq_diff_func_jacobians = None

        # initial processes
        self._generate_circuits()
        self._prepare_transient_funcs()
        self._solve_equations()
        self._prepare_funcs_for_optimization()

    def _to_circuit_node(self, network_node, increment=1):
        #  0 is for the ground and we cannot use 0 for node id
        return network_node + increment

    def _generate_circuits(self):
        self._info = []
        for touch_node in self.nodes:
            # use symbols instead of resistance values to utilize lcapy/sympy
            cct = to_lcapy_circuit(self.nodes,
                                   self.links,
                                   resistances=self._rnames,
                                   in_node=self.in_node,
                                   out_node=self.out_node)
            cct.add(
                f'Ctouch {self._to_circuit_node(touch_node)} {self._ground} {self.capacitance}'
            )

            # Note: the simplification below is very important
            # Without simplification transient_response() requires very long time computation
            # especially when there are many resistors.
            cct = self._circuit_simplifier.simplify(
                cct, no_simplification=(not self.circuit_simplification))
            self._info.append({
                'circuit': cct,
            })

        return self._info

    def _prepare_transient_funcs(self):
        for info in self._info:
            info['transient_func'] = info[
                'circuit'].Ctouch.V.transient_response().sympy

    def _solve_equations(self):

        def eval_one_circuit(transient_func, voltage_thres=self.voltage_thres):
            eq = sy.Eq(transient_func, voltage_thres)
            for symbol in transient_func.free_symbols:
                if symbol.name == 't':
                    tsymbol = symbol
            t_thres = sy.solveset(eq, tsymbol, domain=sy.core.S.Reals)

            return t_thres

        funcs = [info['transient_func'] for info in self._info]

        if self.n_jobs == 1:
            t_thresholds = [eval_one_circuit(f) for f in funcs]
        elif self.n_jobs > 1:
            with Pool(nodes=self.n_jobs) as p:
                t_thresholds = p.map(eval_one_circuit, funcs)
        else:
            with Pool() as p:
                t_thresholds = p.map(eval_one_circuit, funcs)

        for t_thres, info in zip(t_thresholds, self._info):
            # TODO: maybe this part can cause a problem.
            # Check whether this process is fine for any cases
            if isinstance(t_thres, sy.sets.sets.Union):
                # first args[1]: take intersection for (0, inf) from Union(Intersection(0 and answer), Intersection((0, inf) and answer))
                # second args[-1]: take FiniteSet for answer from Intersection((0, inf) and answer))
                # third args[0]: as the FiniteSet only contains one answer, convert it to sympy.core.mul.Mul
                info['time_threshold'] = t_thres.args[1].args[-1].args[0]
            elif isinstance(t_thres, sy.sets.sets.Intersection):
                info['time_threshold'] = t_thres.args[-1].args[0]
            else:
                info['time_threshold'] = t_thres.args[0]

    def _to_func_taking_resistances(self, f):
        symbols = list(f.free_symbols)
        lam_f = sy.lambdify(symbols, f)

        symbol_name_to_expr_str = {}
        for symbol in symbols:
            expr_str = self._circuit_simplifier.resistance_name_to_expr[
                symbol.name]
            symbol_name_to_expr_str[symbol.name] = expr_str
        expr_strs = list(symbol_name_to_expr_str.values())

        def func(resistances):
            nonlocal expr_strs
            return lam_f(*[(lambda r: eval(expr_str))(resistances)
                           for expr_str in expr_strs])

        return func

    def _prepare_funcs_for_optimization(self):
        fs = [info['time_threshold'] for info in self._info]

        sq_diff_funcs = []
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                sq_diff_funcs.append((fs[i] - fs[j])**2)

        sq_diff_func_symbols = []
        sq_diff_func_jacobians = []
        for f in sq_diff_funcs:
            symbols = list(f.free_symbols)
            sq_diff_func_symbols.append(symbols)
            sq_diff_func_jacobians.append(sy.Matrix([f]).jacobian(symbols))

        self._sq_diff_func_symbols = sq_diff_func_symbols
        self._lambdified_sq_diff_funcs = [
            self._to_func_taking_resistances(f) for f in sq_diff_funcs
        ]

        self._lambdified_sq_diff_func_jacobians = [
            self._to_func_taking_resistances(f) for f in sq_diff_func_jacobians
        ]

    def optimize(self,
                 init_resistances=None,
                 resistance_range=[50.0e3, 300.0e3],
                 buget_for_single_iter=5000,
                 max_iterations=10000,
                 convergence_thres=0.01,
                 convergence_judge_buffer_size=500):
        if init_resistances is None:
            init_resistances = resistance_range[0] + np.random.rand(
                len(self.links)) * (resistance_range[1] - resistance_range[0])

        resistances = init_resistances
        best_val = 0.0
        optimized_resistances = resistances.copy()
        val_for_covergence_judge = best_val
        remaining_covergence_buffer = convergence_judge_buffer_size

        for i in range(max_iterations):
            # stop optimization if there is no improvement more than convergence_thres
            # within a specified buffer size.
            if remaining_covergence_buffer == 0:
                break
            remaining_covergence_buffer -= 1

            # find minimum diff pair of t_thresholds (i.e., less diff when touching these two nodes)
            sq_diffs = np.array(
                [f(resistances) for f in self._lambdified_sq_diff_funcs])
            min_sq_diff_func_idx = np.argmin(sq_diffs)
            min_sq_diff = sq_diffs[min_sq_diff_func_idx]

            if min_sq_diff > best_val:
                best_val = min_sq_diff
                optimized_resistances = resistances.copy()

            if best_val > val_for_covergence_judge * (1.0 + convergence_thres):
                remaining_covergence_buffer = convergence_judge_buffer_size
                val_for_covergence_judge = best_val

            self._iter_count = i
            if self.verbose:
                print(
                    f'Iter {self._iter_count}: {min_sq_diff}. Best: {best_val}'
                )
            if self.logging:
                self.opt_log.append({
                    'iteration': self._iter_count,
                    'value': best_val
                })

            symbols = self._sq_diff_func_symbols[min_sq_diff_func_idx]
            lam_jacobians = self._lambdified_sq_diff_func_jacobians[
                min_sq_diff_func_idx]
            grads = lam_jacobians(resistances)[0]

            no_grads = False
            if np.isnan(grads.sum()) or grads.sum() == 0:
                grads = np.random.rand(len(self.links))
                no_grads = True

            # evenly distribute grads to resistances in a combined resistance
            # (or when there is only one resistance, it gets grad as is)
            deltas = np.zeros_like(resistances)
            for gradient, symbol in zip(grads, symbols):
                # this converts 'r[3] + r[5] + 1000' => ['r[3]', 'r[5]', '1000']
                sub_resistances = self._circuit_simplifier.resistance_name_to_expr[
                    symbol.name].split('+')

                # if there are 5 resitances, and first and third are used in the combined resistance
                # => related = [True, False, True, False, False]
                related = np.zeros_like(resistances, dtype=bool)
                for sub_r in sub_resistances:
                    # this regex getting the number in []. e.g., r[12] => '12'
                    match_obj = re.search(r"\[(\w+)\]", sub_r)
                    if match_obj is not None:
                        # get the number and convert it to int
                        resistance_index = int(match_obj.group(1))
                        related[resistance_index] = True
                deltas[related] += gradient / related.sum()

            if not no_grads:
                deltas *= buget_for_single_iter / np.linalg.norm(deltas)
            else:
                deltas *= buget_for_single_iter / 1000 / np.linalg.norm(deltas)
            resistances += deltas
            resistances = np.clip(resistances, resistance_range[0],
                                  resistance_range[1])

        return optimized_resistances, best_val

    def _to_dists(self, vec, diagonal_value=np.finfo(np.float64).max):
        '''compute pairwise distance for each pair of elements in vector'''
        A = np.tile(vec[:, np.newaxis], len(vec))
        D = (A - A.T)**2
        np.fill_diagonal(D, diagonal_value)

        return D

    def evaluate_with_time_threshold_sym_funcs(self, resistances):
        # Note: when 20 links without simplfication of a circuit, computation breakdown as follows
        # subs(): 1.5726001262664795 (single core)
        # transient_response(): 11.333814859390259 (single core)
        # Eq(): 2.984656810760498(single core) => multicore 0.5
        # So, want to compute transient_response in parallel but, probably lcapy's implementation is weird
        # and "cannot pickle '_global_parameters' object" error occured

        self._iter_count += 1
        if self.verbose is True:
            start = time.time()

        t_thresholds = []
        for info in self._info:
            sym_t_thres = info['time_threshold']  # still symbolic

            symbol_to_val = {}
            for symbol in sym_t_thres.free_symbols:
                expr_str = self._circuit_simplifier.resistance_name_to_expr[
                    symbol.name]
                val = (lambda r: eval(expr_str))(resistances)
                symbol_to_val[symbol] = val

            t_thres = sym_t_thres.subs(symbol_to_val)
            t_thresholds.append(t_thres)

        t_thresholds = np.array(t_thresholds)

        result = np.min(self._to_dists(t_thresholds))

        if self.verbose is True:
            print(
                f'{self._iter_count}th eval. time: {time.time() - start}, result: {result}'
            )
        if self.logging:
            self.opt_log.append({
                'iteration': self._iter_count,
                'value': result
            })

        return result

    def optimize_without_using_grad(self,
                                    resistance_bounds=[0.5e5, 3.0e5],
                                    resistance_precision=1e4,
                                    n_sampling_points=100,
                                    iters=3,
                                    minimizer_kwargs={
                                        'method': 'COBYLA',
                                        'options': {
                                            'maxiter': 30,
                                            'disp': False
                                        }
                                    }):
        if self.verbose:
            print('optimizing resistances...')
        self._iter_count = 0
        cost_func = lambda resistances: -self.evaluate_with_time_threshold_sym_funcs(
            resistances)

        result = optimize.shgo(cost_func,
                               bounds=[resistance_bounds] * len(self.links),
                               n=n_sampling_points,
                               iters=iters,
                               minimizer_kwargs=minimizer_kwargs)
        resistances = result.x
        cost = result.fun

        if resistance_precision is not None:
            resistances = (resistances /
                           resistance_precision).round() * resistance_precision
            cost = cost_func(resistances)

        best_val = -cost
        return resistances, best_val

    def evaluate_resitances_without_involving_symbols(self,
                                                      resistances,
                                                      return_related_info=True
                                                      ):
        # this method is more for debugging but slow when there are many resistors
        circuits = []
        for touch_node in self.nodes:
            # use symbols instead of resistance values to utilize lcapy/sympy
            cct = to_lcapy_circuit(self.nodes,
                                   self.links,
                                   resistances=resistances,
                                   in_node=self.in_node,
                                   out_node=self.out_node)
            cct.add(
                f'Ctouch {self._to_circuit_node(touch_node)} {self._ground} {self.capacitance}'
            )

            circuits.append(cct)

        transient_funcs = [
            cct.Ctouch.V.transient_response().sympy for cct in circuits
        ]

        t_thresholds = []
        for f in transient_funcs:
            eq = sy.Eq(f, self.voltage_thres)
            for symbol in f.free_symbols:
                if symbol.name == 't':
                    tsymbol = symbol
            t_thres = sy.solveset(eq, tsymbol, domain=sy.core.S.Reals).args[0]
            t_thresholds.append(t_thres)
        t_thresholds = np.array(t_thresholds)
        print(t_thresholds)
        min_sq_diff = np.min(self._to_dists(t_thresholds))

        if return_related_info:
            return min_sq_diff, t_thresholds, circuits
        else:
            return min_sq_diff

    def _insert_resistance_values(self, f, resistances):
        if not isinstance(f, sy.core.mul.Mul):
            return f
        else:
            f_ = f.copy()
            symbols = list(f_.free_symbols)

            for symbol in symbols:
                if symbol.name in self._circuit_simplifier.resistance_name_to_expr:
                    expr_str = self._circuit_simplifier.resistance_name_to_expr[
                        symbol.name]
                    val = (lambda r: eval(expr_str))(resistances)
                    f_ = f_.subs(symbol, val)

            return f_

    def evaluate_resitances(self,
                            resistances,
                            from_solved_eq=True,
                            return_related_info=True):
        if from_solved_eq:
            t_thresholds = [
                self._insert_resistance_values(info['time_threshold'],
                                               resistances)
                for info in self._info
            ]
        else:
            # this method is more for debugging to check solving eq works in an intented way
            t_thresholds = []
            for info in self._info:
                f = self._insert_resistance_values(info['transient_func'],
                                                   resistances)
                eq = sy.Eq(f, self.voltage_thres)
                for symbol in f.free_symbols:
                    if symbol.name == 't':
                        tsymbol = symbol
                t_thres = sy.solveset(eq, tsymbol,
                                      domain=sy.core.S.Reals).args[0]
                t_thresholds.append(t_thres)

        t_thresholds = np.array(t_thresholds)
        min_sq_diff = np.min(self._to_dists(t_thresholds))

        if return_related_info:
            return min_sq_diff, t_thresholds
        else:
            return min_sq_diff


if __name__ == '__main__':
    nodes = [0, 1, 2, 3]
    links = [[0, 1], [1, 2], [2, 3]]
    in_node = 0
    out_node = None

    ropt = ResistanceOptimization(nodes,
                                  links,
                                  in_node,
                                  out_node=out_node,
                                  circuit_simplification=False)

    # this can be super slow if there are many nodes, links
    # resistances, min_sq_diff = ropt.optimize_without_using_grad()
    # (min_sq_diff, t_thresholds, circuits
    #  ) = ropt.evaluate_resitances_without_involving_symbols(resistances)
    # print('resistances:', resistances)
    # print('time thresholds:', t_thresholds)

    # while the above needs 900sec and does not show a better result
    # the below needs 0.5 sec and shows a much better result
    resistances, min_sq_diff = ropt.optimize()
    min_sq_diff, t_thresholds = ropt.evaluate_resitances(resistances)
    print('resistances:', resistances)
    print('time thresholds:', t_thresholds)
    # min_sq_diff, t_thresholds = ropt.evaluate_resitances(resistances,
    #                                                      from_solved_eq=False)
    # print('time thresholds:', t_thresholds)