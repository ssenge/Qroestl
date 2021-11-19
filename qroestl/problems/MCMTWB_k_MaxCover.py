import copy
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
import time
from datetime import datetime
from operator import mul
from random import randint, choice
from typing import List, Optional

import dimod
import docplex
import networkx as nx
import numpy as np
from dimod import ConstrainedQuadraticModel, Binary
from matplotlib import pyplot as plt
from qiskit_optimization import QuadraticProgram
import gurobipy as gp
from gurobipy import GRB
from qiskit_optimization.converters import QuadraticProgramToQubo

from qroestl.backends.Cplex import CplexModelConvertible
from qroestl.backends.Gurobi import GurobiModelConvertible
from qroestl.backends.Ocean import OceanCQMToBQMConvertible, OceanCQMConvertible
from qroestl.model import Model
from qroestl.model.Model import Solution, Approach, Optimizer
from qroestl.backends.Qiskit import QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible
from qroestl.utils import Utils
from qroestl.utils.Utils import unique, flatten2d

TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List[int]]


@dataclass
class Problem(Model.Problem[TCandidate]):
    nU: int  # number of universe elements
    nS: int  # number of sets
    k: int  # max number of sets to be selected
    C: List[List[int]]  # covered universe elements (per set)
    R: Optional[List[int]] = None  # required covers (per universe element)
    T: Optional[List[int]] = None  # type (per set)
    W: Optional[List[float]] = None  # weight (per universe element)
    P: Optional[List[float]] = None  # price (per set)
    B: Optional[float] = None  # max budget
    name: str = 'Multi Cover Multi Type Weighted Budgeted k-Max Set Cover'

    def __post_init__(self) -> None:
        if not self.R:
            self.R = [0] * self.nU
        if not self.T:
            self.T = [0] * self.nS
        if not self.W:
            self.W = [1] * self.nU
        if not self.P:
            self.P = [1] * self.nS
        if not self.B:
            self.B = 1000000
        self._validate()
        self.U = list(range(self.nU))
        self.S = list(range(self.nS))

        self.s_us = dict(zip(self.S, self.C))
        self.u_ss = defaultdict(list)
        for s, us in enumerate(self.C):
            for u in us:
                self.u_ss[u].append(s)

        self.u_ts = {u: list(set([self.T[s] for s in self.u_ss[u]])) for u in self.U}  # u -> types (of sets that cover u)
        self.W_offset = [w * (self.k + 1) for w in
                         self.W]  # offsetting weights (by linear multiplicative spread) to allow for negative coefficients for set variables without affecting the optimal solution
        # self.P_normalized = [self.k * (self.P[s] / sum(self.P)) for s in self.S]
        # self.B_normalized = sum(self.P_normalized)

    def __str__(self):
        return f'nU={self.nU} nS={self.nS} k={self.k} B={self.B}\nW={np.array(self.W)}\nR={np.array(self.R)}\nT={np.array(self.T)}\nP={np.array(self.P)}\nC={np.array(self.C)}'

    def stats(self):
        avg_set_size = sum([len(us) for us in self.C]) / self.nS
        avg_hits = sum(u for u in self.U if self.u_ss[u]) / self.nU
        total_sus = 0
        for s in self.S:
            sus = [u for u in self.s_us[s] if len(self.u_ss[u]) == 1]
            if len(sus) >= 2:
                total_sus += len(sus)
        return avg_set_size, avg_hits, total_sus

    def collapse(self):

        U = self.U.copy()
        C = copy.deepcopy(self.C)

        U_not_hit = [u for u in U if len(self.u_ss[u]) == 0]

        u_r = {u: self.R[u] for u in U}  # switch to maps
        u_w = {u: self.W[u] for u in U}
        for s in self.S:
            sus = [u for u in self.s_us[s] if len(self.u_ss[u]) == 1]
            if len(sus) >= 2:
                U = [u for u in self.U if u not in sus]
                pos = 0 if len(U) == 0 else max(U) + 1
                u_w[pos] = sum(u_w[u] for u in sus)
                u_r[pos] = max(u_r[u] for u in sus)
                U.append(pos)
                C[s] = [u for u in self.s_us[s] if u not in sus] + [pos]
        if not U == self.U:
            # switch back to arrays
            U_map = {u: i for i, u in enumerate(U)}
            for s in self.S:
                self.C[s] = [U_map[u] for u in C[s]]
            U = [u for u in range(len(U_map.keys()))]
            R = [0] * len(U)
            W = [1] * len(U)
            for old_pos, new_pos in U_map.items():
                R[new_pos] = u_r[old_pos]
                W[new_pos] = u_w[old_pos]
            return Problem(nU=len(U), nS=self.nS, k=self.k, C=C, R=R, T=self.T, W=W, P=self.P, B=self.B)
        return self

    def _validate(self):
        # assert self.nU == len(self.U)
        assert self.nU == len(self.W)
        assert self.nU == len(self.R)
        # assert self.nS == len(self.S)
        assert self.nS == len(self.T)
        assert self.nS == len(self.P)
        assert self.B >= 0

    def feasible(self, c: TCandidate) -> bool:
        # This is a (slightly) speed optimized (and hence ugly ;-)) version (BruteForce heavily relies on this, so unfortunately a fast version is required)
        if len(c) > self.k or sum(self.P[s] for s in c) > self.B:
            return False
        # hits = lambda u: np.intersect1d(self.u_ss[u], c)
        # hits = lambda u: [ce for ce in c if ce in self.u_ss[u]]

        # Skip multi-cover as this superseded by type-cover
        # multi_covered = np.all(list(map(lambda u: self.R[u] <= len(hits(u)), self.U)))

        # types = lambda u: list(set(map(lambda s: self.T[s], hits(u))))
        types = lambda u: set([self.T[s] for s in c if s in self.u_ss[u]])  # hits(u)])
        U = [u for u in self.U if self.R[u] > 0]

        # type_covered = np.all(list(map(lambda u: self.R[u] <= len(types(u)), U)))
        for u in U:  # ugly but faster due to premature exit
            if self.R[u] > len(types(u)):
                return False
        return True
        # return len(c) <= self.k and multi_covered and type_covered

    def value(self, c: TCandidate) -> float:
        return sum(self.W[u] for u in set(us for s in c for us in self.s_us[s]))

    def all_solutions(self) -> List[TCandidate]:
        return Utils.powerset(self.S)

    def visualize(self, selected=[], save=False, k_sol=None, obj_sol=None, w_sol=None, p_sol=None):
        # selected = selected or self.U
        ss = [f's{s}' for s in self.S]
        us = [f'u{u}' for u in self.U]
        cs = [[us[c] for c in cover] for cover in self.C]
        es = Utils.flatten2d([[(s, u) for u in us] for s, us in zip(ss, cs)])
        G = nx.Graph()
        G.add_edges_from(es)
        top_pos = {u: (i / len(us), 1) for i, u in enumerate(us)}
        bottom_pos = {s: (i / len(ss), -1) for i, s in enumerate(ss)}
        s_colors = {f's{s}': ('red' if s in selected or len(selected) == 0 else 'grey') for s in self.S}
        u_colors = {f'u{u}': ('blue' if any(x in selected for x in self.u_ss[u]) or len(selected) == 0 else 'grey') for u in self.U}
        pos = {**top_pos, **bottom_pos}
        node_colors = [(u_colors[n] if str(n).startswith('u') else s_colors[n]) for n in G.nodes()]
        edge_colors = ['green' if int((a if str(a).startswith('s') else b)[1:]) in selected else 'grey' for a, b in G.edges()]
        edge_widths = [5 if int((a if str(a).startswith('s') else b)[1:]) in selected else 1 for a, b in G.edges()]
        nx.draw(G, pos=pos, node_color=node_colors, edge_color=edge_colors, width=edge_widths)
        pos_sets_ = pos[f's{0}']
        x_values, y_values = zip(*pos.values())
        y_max = max(y_values)
        y_min = min(y_values)
        y_margin = (y_max - y_min) * 0.25
        x_max = max(x_values)
        x_min = min(x_values)
        x_margin = (x_max - x_min) * 0.45
        pos_sets = {f's{0}': (pos_sets_[0] - 0.3, pos_sets_[1])}
        pos_k = {f's{0}': (pos_sets_[0] - 0.3, pos_sets_[1] + 0.2)}
        pos_B = {f's{0}': (pos_sets_[0] - 0.3, pos_sets_[1] + 0.32)}
        pos_T = {f's{0}': (pos_sets_[0] - 0.3, pos_sets_[1] - 0.2)}
        pos_P = {f's{0}': (pos_sets_[0] - 0.3, pos_sets_[1] - 0.32)}
        pos_universe_ = pos[f'u{0}']
        pos_universe = {f'u{0}': (pos_universe_[0] - 0.3, pos_universe_[1])}
        pos_R = {f's{0}': (pos_universe_[0] - 0.3, pos_universe_[1] + 0.2)}
        pos_W = {f's{0}': (pos_universe_[0] - 0.3, pos_universe_[1] + 0.32)}
        pos_sol_ = (x_max, -1)  # pos[f's{len(ss)-1}']
        pos_sol = {f's{len(ss) - 1}': (pos_sol_[0] + 0.07, pos_sol_[1] + 0.56)}
        pos_sol_k = {f's{len(ss) - 1}': (pos_sol_[0] + 0.07, pos_sol_[1] + 0.44)}
        pos_sol_p = {f's{len(ss) - 1}': (pos_sol_[0] + 0.07, pos_sol_[1] + 0.32)}
        pos_sol_obj = {f's{len(ss) - 1}': (pos_sol_[0] + 0.07, pos_sol_[1] + 0.2)}
        nx.draw_networkx_labels(G, pos=pos_k, labels={f's{0}': f'k={self.k}'}, font_color='black', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos_B, labels={f's{0}': f'B={self.B}'}, font_color='black', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos_T, labels={f's{0}': f'T='}, font_color='black', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos_P, labels={f's{0}': f'P='}, font_color='black', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos_R, labels={f's{0}': f'R='}, font_color='black', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos_W, labels={f's{0}': f'W='}, font_color='black', horizontalalignment='left')
        if k_sol or obj_sol or w_sol or p_sol:
            nx.draw_networkx_labels(G, pos=pos_sol, labels={f's{len(ss) - 1}': f'Solution:'}, font_color='black', font_weight='bold', horizontalalignment='left')
            if k_sol:
                nx.draw_networkx_labels(G, pos=pos_sol_k, labels={f's{len(ss) - 1}': f'k={k_sol}'}, font_color='black', horizontalalignment='left')
            if p_sol:
                nx.draw_networkx_labels(G, pos=pos_sol_p, labels={f's{len(ss) - 1}': f'P={p_sol}'}, font_color='black', horizontalalignment='left')
            if obj_sol:
                nx.draw_networkx_labels(G, pos=pos_sol_obj, labels={f's{len(ss) - 1}': f'obj={obj_sol}'}, font_color='black', horizontalalignment='left')

        nx.draw_networkx_labels(G, pos=pos_sets, labels={f's{0}': 'Sets'}, font_color='red', font_weight='bold', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos_universe, labels={f'u{0}': 'Universe'}, font_color='blue', font_weight='bold', horizontalalignment='left')
        nx.draw_networkx_labels(G, pos=pos, labels={f'u{k}': k for k in self.U}, font_color='white', font_weight='bold')
        nx.draw_networkx_labels(G, pos=pos, labels={f's{k}': k for k in self.S}, font_color='white', font_weight='bold')
        nx.draw_networkx_labels(G, pos={k: (v[0] + 0.00, v[1] + 0.2) for k, v in pos.items()}, labels={f'u{k}': f'{self.R[k]}' for k in self.U}, horizontalalignment='center')
        nx.draw_networkx_labels(G, pos={k: (v[0] + 0.00, v[1] + 0.32) for k, v in pos.items()}, labels={f'u{k}': f'{self.W[k]}' for k in self.U}, horizontalalignment='center')
        nx.draw_networkx_labels(G, pos={k: (v[0] + 0.00, v[1] - 0.2) for k, v in pos.items()}, labels={f's{k}': f'{self.T[k]}' for k in self.S}, horizontalalignment='center')
        nx.draw_networkx_labels(G, pos={k: (v[0] + 0.00, v[1] - 0.32) for k, v in pos.items()}, labels={f's{k}': f'{self.P[k]}' for k in self.S}, horizontalalignment='center')

        # x_values, y_values = zip(*pos.values())
        # y_max = max(y_values)
        # y_min = min(y_values)
        # y_margin = (y_max - y_min) * 0.25
        # x_max = max(x_values)
        # x_min = min(x_values)
        # x_margin = (x_max - x_min) * 0.35
        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.xlim(x_min - x_margin, x_max + x_margin)
        if save:
            plt.savefig(f'Graph_{str(time.strftime("%Y-%m-%d_%H-%M-%S"))}.png', format="PNG")
        plt.show()


@dataclass
class Standard(Model.Approach[Problem], QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible, CplexModelConvertible, GurobiModelConvertible, OceanCQMConvertible,
               OceanCQMToBQMConvertible):
    name: str = "MCMTWB-k-MaxCover Standard"

    def to_qiskit_qp(self, p: Problem) -> QuadraticProgram:
        uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]

        m = QuadraticProgram(self.name)
        m.binary_var_list(p.S, name='s')
        m.binary_var_list(p.U, name='u')
        m.binary_var_list(uts, name='')
        # m.maximize(linear=([-1] * p.nS) + p.W_offset + ([0] * len(uts)))  # note: all vars appear in obj (in the order as added above, hence, zero coeffs are required)
        m.minimize(linear=([1] * p.nS) + ([-w for w in p.W_offset]) + ([0] * len(uts)))
        # m.maximize(linear=([-1] * p.nS) + [1] * p.nU)

        # m.maximize(linear=([-p for p in p.P_normalized] + p.W_offset + ([0] * len(uts))), constant=p.B_normalized)  # note: all vars appear in obj (in the order as added above, hence, zero coeffs are required)
        m.linear_constraint(linear={f's{s}': 1 for s in p.S}, sense='<=', rhs=p.k, name='max k sets')  # equality constraint instead of less than as the used in a loop over k
        # m.linear_constraint(linear={f's{s}': p.P[s] for s in p.S}, sense='<=', rhs=p.B, name='max budget')
        for u in p.U:
            m.linear_constraint(linear={**{f's{s}': 1 for s in p.S if u in p.s_us[s]}, **{f'u{u}': -1}}, sense='>=', rhs=0, name=f'cover for u{u}')
            # This implements a "must multi cover" policy
            # "Should multi cover": u{u} coefficient needs to be -1*self.SC[u] above (and the 'multi cover u{u}' constraint below can be removed)
            if (c := p.R[u]) > 0:
                m.linear_constraint(linear={**{f's{s}': 1 for s in p.S if u in p.s_us[s]}}, sense='>=', rhs=c, name=f'multi cover for u{u}')
                m.linear_constraint(linear={f'u{u}t{t}': 1 for t in p.u_ts[u]}, sense='>=', rhs=c, name=f'min {c} types for u{u}')
                for t in p.u_ts[u]:
                    m.linear_constraint(
                        linear={**{f's{s}': -1 for s in p.u_ss[u] if p.T[s] == t}, **{f'u{u}t{t}': 1}}, sense='<=', rhs=0, name=f'type cover u{u}t{t}')

        # H, offset = QuadraticProgramToQubo().convert(m).to_ising()
        # qubo = QuadraticProgramToQubo().convert(m)
        # print(f'#Vars: {m.get_num_binary_vars()} -> {qubo.get_num_binary_vars()}')
        return m

    def to_ocean_cqm(self, p: Problem):
        s_sb = {s: Binary(f's{s}') for s in p.S}
        sbs = s_sb.values()
        u_ub = {u: Binary(f'u{u}') for u in p.U}
        ubs = u_ub.values()
        u_utbs = {u: [Binary(f'u{u}t{t}') for t in p.u_ts[u]] for u in p.U}
        u_t_utb = {u: {t: Binary(f'u{u}t{t}') for t in p.u_ts[u]} for u in p.U}

        m = ConstrainedQuadraticModel()
        m.set_objective(-1 * (sum(map(mul, ubs, p.W_offset)) - sum(sbs)))  # min
        # m.set_objective(sum(map(mul, ubs, p.W_offset)))
        m.add_constraint(sum(sbs) <= p.k, "max k sets")
        m.add_constraint(sum(map(mul, sbs, p.P)) <= p.B, 'max budget')
        for u in p.U:
            m.add_constraint(sum([s_sb[s] for s in p.u_ss[u]]) - u_ub[u] >= 0, f'cover for u{u}')
            if (c := p.R[u]) > 0:
                m.add_constraint(sum([s_sb[s] for s in p.u_ss[u]]) >= c, f'multi cover for u{u}')
                m.add_constraint(sum(u_utbs[u]) >= c, f'min {c} types for u{u}')
                for t in p.u_ts[u]:
                    m.add_constraint(-sum([s_sb[s] for s in p.u_ss[u] if p.T[s] == t]) + u_t_utb[u][t] <= 0, f'type cover u{u}t{t}')
        return m

    def to_ocean_bqm_from_cqm(self, p: Problem):
        return dimod.constrained.cqm_to_bqm(self.to_ocean_cqm(p))

    def to_cplex_model(self, p: Problem):
        uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]

        m = docplex.mp.model.Model(name=self.name)
        s_sb = m.binary_var_dict(p.S, name='s')
        u_ub = m.binary_var_dict(p.U, name='u')
        ut_utbs = m.binary_var_dict(uts, name='')

        m.maximize(m.sum(m.scal_prod(list(u_ub.values()), p.W_offset)) - m.sum(s_sb))
        m.add(m.sum(s_sb) <= p.k, 'k sets')
        m.add(m.sum(m.scal_prod(list(s_sb.values()), p.P)) <= p.B, 'max budget')
        for u in p.U:
            m.add(m.sum([s_sb[s] for s in p.u_ss[u]]) - u_ub[u] >= 0, f'cover for u{u}')
            if (c := p.R[u]) > 0:
                m.add(m.sum(s_sb[s] for s in p.u_ss[u]) >= c, f'multi cover for u{u}')
                m.add(m.sum(ut_utbs[f'u{u}t{t}'] for t in p.u_ts[u]) >= c, f'min {c} types for u{u}')
                for t in p.u_ts[u]:
                    m.add(-m.sum([s_sb[s] for s in p.u_ss[u] if p.T[s] == t]) + ut_utbs[f'u{u}t{t}'] <= 0, f'type cover u{u}t{t}')
        return m

    def to_gurobi_model(self, p: Problem):
        uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]

        m = gp.Model(self.name)
        s_sb = m.addVars(p.S, obj=-1, name='s', vtype=GRB.BINARY)
        u_ub = m.addVars(p.U, obj=p.W_offset, name='u', vtype=GRB.BINARY)
        ut_utbs = m.addVars(uts, name='', vtype=GRB.BINARY)
        # m.setObjective(u_ub.sum(), sense=GRB.MAXIMIZE) #- s_sb.sum()
        m.ModelSense = GRB.MAXIMIZE  # use instead of setObj() as coeffs were already set above
        m.addConstr(s_sb.sum() <= p.k, "max k sets")
        m.addConstr(gp.quicksum([s_sb[s] * p.P[s] for s in p.S]) <= p.B, 'max budget')
        for u in p.U:
            m.addConstr(gp.quicksum([s_sb[s] for s in p.u_ss[u]]) - u_ub[u] >= 0, f'cover for u{u}')
            if (c := p.R[u]) > 0:
                m.addConstr(gp.quicksum([s_sb[s] for s in p.u_ss[u]]) >= c, f'multi cover for u{u}')
                m.addConstr(gp.quicksum(ut_utbs[f'u{u}t{t}'] for t in p.u_ts[u]) >= c, f'min {c} types for u{u}')
                for t in p.u_ts[u]:
                    m.addConstr(-gp.quicksum([s_sb[s] for s in p.u_ss[u] if p.T[s] == t]) + ut_utbs[f'u{u}t{t}'] <= 0, f'type cover u{u}t{t}')
        m.write('out.lp')
        return m

    def extract(self, result_dict):
        return sorted({k: v for k, v in result_dict.items() if k.startswith('s')}.values())


@dataclass
class Greedy(Optimizer[TCandidate, Problem]):
    name: str = "Greedy"

    # A large number used for weighting required universe elements.
    # Should be larger than largest_weight * highest_number_of_covered_elements_by_one_set
    weight_offset: int = 10000

    def optimize_(self, p: Problem, p_conv, a: Approach, s_=Solution[TCandidate, Problem]()) -> Solution[TCandidate, Problem]:

        weights = lambda: {u: self.weight_offset + p.W[u] if R[u] > 0 else (p.W[u] if R[u] == 0 else 0) for u in p.U}
        indicator = lambda u, s: 0 if p.T[s] in T[u] else 1
        dominated = lambda max_s: np.any([set(p.s_us[max_s]).issubset(p.s_us[s]) for s in sol if p.T[s] == p.T[max_s]])

        sol = []
        S = np.copy(p.S)
        R = np.copy(p.R)
        T = defaultdict(list)
        start = datetime.now()
        for k in range(p.k):
            # Speed-up 1: if nothing to do, break
            if len(S) == 0 or np.all(R < 0): break
            # Speed-up 2 (rarely applicable): if only one set left, choose it and break
            if len(S) == 1:
                if np.any(R >= 0) or not dominated(S[0]):  # however, there are two exceptions
                    sol.append(S[0])
                break
            # Speed-up 3: if cover requirements exist, consider only sets that could contribute
            S_ = S
            if np.any(R > 0):
                S_ = [s for s in S if np.any(np.take(R, p.s_us[s]) > 0)]

            cost = lambda s_: sum(p.P[s] for s in sol + [s_])
            ws = weights()
            d = {s: sum(ws[u] * indicator(u, s) for u in p.s_us[s]) for s in S_ if cost(s) <= p.B}
            if len(d) == 0: break  # no fitting found due to budget constraint
            max_s = max(d, key=d.get)
            S = np.setdiff1d(S, max_s)
            R = np.add(R, [-1 if u in p.s_us[max_s] else 0 for u in p.U])
            # Speed-up 4: if no improvement, break (warning: this does not work in some edge cases)
            # if p.cost(sol) == p.cost(sol+[max_s]): break

            # if a set is dominated by another and no type covers are required any longer, break
            if np.all(R <= 0) and dominated(max_s): break

            sol.append(max_s)
            for u in p.s_us[max_s]:  # update covers
                T[u].append(p.T[max_s])

            if sum(p.P[s] for s in sol) >= p.B: break  # budget constraint # can probably be removed

        return s_.eval(p, sol), datetime.now() - start


def to_consecutive(k, S, R, T, W, P, B) -> Problem:
    uid = defaultdict(lambda: len(uid))
    return Problem(nU=len(unique(flatten2d(S))), nS=len(S), k=k, R=R, T=T, W=W, P=P, B=B, C=[[uid[u] for u in s] for s in S])


def gen_syn_one_to_one(nS: int, k) -> Problem:
    return Problem(nS, nS, k=k, R=[1] * nS, T=[0] * nS, W=[1] * nS, C=[[s] for s in range(nS)])
    # return Problem(nS, nS, k=k, R=[1] * nS, T=list(range(nS)), W=[1] * nS, C=[[s] for s in range(nS)])


def gen_syn_fully_connected(nS: int, k) -> Problem:
    return Problem(nS, nS, k=k, R=[1] * nS, T=[0] * nS, W=[1] * nS, C=[list(range(nS)) for s in range(nS)])
    # return Problem(nS, nS, k=k, R=[1] * nS, T=list(range(nS)), W=[1] * nS, C=[list(range(nS)) for s in range(nS)])


def gen_syn_random(max_nU: int, max_nS: int, k: int) -> Problem:
    nU = max(2, randint(1, max_nU))
    nS = randint(1, max_nS)
    return Problem(nU, nS,
                   k=k,
                   R=[0] * nU,
                   T=list(range(nS)),
                   W=list(random.choices(range(1, nU), k=max(nU, min(nS, nU)))),
                   C=[random.sample(range(nU), k=randint(1, nU)) for _ in range(nS)])


def gen_syn_random_coverages(nU, nS, k, T):
    if nU < 1:
        return None
    if nS <= 1:
        nS = max(1, int(nS * nU))
    if k <= 1:
        k = max(1, int((k * nS) / 10))
    if T < 0:
        T = list(range(nS))
    else:
        T = [T] * nS

    C = defaultdict(list)
    for i in range(nU):
        C_s = random.sample(range(nS), k=randint(1, nS))
        for u in C_s:
            C[u].append(i)
    return Problem(nU, nS, k=k, R=[0] * nU, T=T, W=list(random.choices(range(1, nU), k=max(nU, min(nS, nU)))), C=C.values())
