import itertools
import math
import random
from dataclasses import dataclass
from operator import mul
from typing import List, Optional, Dict

import dimod
import docplex
import gurobipy as gp
import numpy as np
from dimod import ConstrainedQuadraticModel, Binary
from gurobipy import GRB
from qiskit_optimization import QuadraticProgram

from qroestl.backends.Cplex import CplexModelConvertible
from qroestl.backends.Gurobi import GurobiModelConvertible
from qroestl.backends.Ocean import OceanCQMToBQMConvertible, OceanCQMConvertible
from qroestl.backends.Qiskit import QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible
from qroestl.model import Model
from qroestl.problems import MCMTWB_k_MaxCover
from qroestl.utils import Utils
from qroestl.utils.Utils import unique, flatten2d

TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List[int]]


@dataclass
class Problem(Model.Problem[TCandidate]):
    """
    Sample how to instantiate a problem (see a description of the variables in the source code below):
    MPCMTWB_k_MaxCover.Problem(nU=2, nS=3, k=2, R=[2, 0], T=[0, 1, 0], W=[1, 1], UQ=[50, 50], SQ=[50, 50, 52],
                                 C=[
                                     # C1 -- required single coverages (in %)
                                     [
                                         {(0,): 60, (1,): 15, (2,): 20}, # u1
                                         {(2,): 20} # u2
                                     ],
                                     # C2 -- required double coverages
                                     [
                                         {(1, 2): 10}, # u1
                                         {} # u2
                                     ]
                                 ],
                                 RC=[
                                     # C1 -- provided single coverages (in %)
                                     [15, # u1
                                      0 # u2
                                      ],
                                     # C2 -- provided double converages (in %)
                                     [10, # u1
                                      0]  # u2
                                 ])
    """
    nU: int  # number of universe elements
    nS: int  # number of sets
    k: int  # max number of sets to be selected
    R: Optional[List[int]] = None  # required covers (per universe element)
    C: List[List[Dict[List[int], float]]] = None  # partially covered universe elements (per universe element)
    RC: Optional[List[float]] = None  # coverages (per set / per universe element)
    UQ: Optional[List[float]] = None  # required quality coverage (per universe element)
    SQ: Optional[List[float]] = None  # quality (per set)
    T: Optional[List[int]] = None  # type (per set)
    W: Optional[List[float]] = None  # weight (per universe element)
    P: Optional[List[float]] = None  # price (per set)
    B: Optional[float] = None  # max budget
    name: str = 'Multi Partial Cover Multi Type Weighted Budgeted k-Max Set Cover'

    def __post_init__(self) -> None:

        # In the following some default values are set (if requried):
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

        # Additional parameters
        self.alpha = 1  # coefficient for the single coverage term
        self.beta = 1  # coefficient for the double coverage term

        self.sC_sum = {s: 0 for s in self.S}
        for u_dict in self.C[0]:
            for sc in u_dict.items():
                self.sC_sum[sc[0][0]] = self.sC_sum[sc[0][0]] + sc[1]

        # Sets that cover at least one universe element
        # self.us_C1 = [u for u, r in enumerate(self.R) if r <= 1]
        # self.covers1 = [sc for u in us_C1 for sc in p.C[0][u].items()]
        self.covers1 = self.S
        # Sets that cover two universe element
        us_C2 = [u for u, r in enumerate(self.R) if r >= 2]
        self.covers2 = [sc for u in us_C2 for sc in self.C[1][u].items()]

        # Configuration for Big M transformation
        self.M1 = 110  # for single coverage
        self.M2 = self.M1  # for double coverage

        ### To speed up calculations below, let's prepare some look up dicts
        ### Terminology: x_y => a map from 'x' to 'y' (an additional 's' indicates plural, i.e. is a mapping to multiple elements (list))

        # Mapping from an universe elements (u) to sets (ss) where u is covered by one s in ss
        self.u_ss = {u: [s[0] for s in ss.keys()] for u, ss in enumerate(self.C[0])}
        # All universe elements covered by a set
        self.s_us = {s: [u for u, ss in self.u_ss.items() if s in ss] for s in self.S}
        # All set types covering an universe element
        self.u_ts = {u: list(set([self.T[s] for s in self.u_ss[u]])) for u in
                     self.U}  # u -> types (of sets that cover u)

        # offsetting weights (by linear multiplicative spread) to allow for negative coefficients for set variables without affecting the optimal solution
        self.W_offset = [w * (self.k + 1) for w in self.W]

        # Typically not required
        # self.P_normalized = [self.k * (self.P[s] / sum(self.P)) for s in self.S]
        # self.B_normalized = sum(self.P_normalized)

    def __str__(self):
        return f'nU={self.nU} nS={self.nS} k={self.k} B={self.B}\nW={np.array(self.W)}\nR={np.array(self.R)}\nT={np.array(self.T)}\nP={np.array(self.P)}\nC={np.array(self.C)}'

    def _validate(self):
        """ Doing some sanity checks. """
        # assert self.nU == len(self.U)
        assert self.nU == len(self.W)
        assert self.nU == len(self.R)
        # assert self.nS == len(self.S)
        assert self.nS == len(self.T)
        assert self.nS == len(self.P)
        assert self.B >= 0

    def feasible(self, c: TCandidate) -> bool:
        ''' Checks if a candidate solution is feasible, used e.g. in the brute force approach. '''
        # 1 - Check for max k
        if len(c) > self.k or sum(self.P[s] for s in c) > self.B:
            return False

        # 2 - Check for R cover
        U = [u for u in self.U if self.R[u] > 0]
        for u in U:
            if len([s for s in c if s in self.u_ss[u]]) < self.R[u]:
                return False

        # 3 - Check for type cover
        types = lambda u: set([self.T[s] for s in c if s in self.u_ss[u]])  # hits(u)])
        # type_covered = np.all(list(map(lambda u: self.R[u] <= len(types(u)), U)))
        for u in U:  # ugly but faster due to premature exit
            if self.R[u] > len(types(u)):  # This also checks for general >R[u] coverage, not only types
                return False

        # 4 - Check for multi type and min quality cover

        # Helper
        def is_covered_Cn(u, Cn):
            return np.any(list(map(
                # check if required cover (with minimum RC value) is given
                lambda ssv: ssv[1] >= self.RC[Cn][u] and set(ssv[0]).issubset(c),
                # extract all sets-value tuples (ssv) from Cn where n depends on the required cover
                # (assuming: tuples fulfill the multiple types criterion)
                # [ssv for ssvs in self.C[Cn] for ssv in ssvs.items()])))
                [ssv for ssv in self.C[Cn][u].items()])))

        def has_min_quality(u):
            return np.all(list(map(lambda s: self.SQ[s] >= self.UQ[u], [s for s in self.u_ss[u] if s in c])))

        # Check for violations of constraints (eager eval to exit prematurely if possible)
        covered_Cn = lambda Cn: not np.any(
            list(map(lambda u: not is_covered_Cn(u, Cn), [u for u in U if self.RC[Cn][u] > 0])))
        minq = lambda: np.all(list(map(lambda u: has_min_quality(u), self.U)))

        res = np.all([covered_Cn(Cn) for Cn in range(len(self.C))]) and minq()
        if res:
            return True
        return False

    def value(self, c: TCandidate) -> float:
        """ Returns the value of a solution candidate. """
        # Sum up all C1 values if the set is in the candidate
        # sum(c_dict[1] for c1 in self.C[0] for c_dict in c1.items() for s in c if c_dict[0][0] == s)
        return sum(c1[(s,)] for c1 in self.C[0] for s in c if (s,) in c1)
        # + sum((self.RQ[u] - self.SQ[s]) for s in c for u in self.s_us[s])

    def all_solutions(self) -> List[TCandidate]:
        """ Return all possible solutions. """
        return Utils.powerset(self.S)


@dataclass
class Standard(Model.Approach[Problem], QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible,
               GurobiModelConvertible, CplexModelConvertible, OceanCQMConvertible, OceanCQMToBQMConvertible):
    """ Standard solution approach. """

    name: str = "MPCMTWB-k-MaxCover Standard"

    def to_qiskit_qp(self, p: Problem) -> QuadraticProgram:
        """ Returns a Qiskit QuadraticProgram representation of the optimization problem.  """

        ##### The general outline for the LP is described here, in the below ports to other SDKs, the same structure will be used

        m = QuadraticProgram(self.name)
        m.binary_var_list(p.S, name='s')
        m.binary_var_list(p.U, name='u')
        uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]
        m.binary_var_list(uts, name='')

        m.maximize(linear={f's{s}': p.alpha * p.sC_sum[s] for s in p.covers1},
                   # *-1 {f's{s[0]}': alpha * c for s, c in covers1},  # *-1
                   quadratic={**{(f's{s1_s2[0]}', f's{s1_s2[1]}'): p.beta * c_s1_s2 for s1_s2, c_s1_s2 in p.covers2},
                              # **{(f's{s1_s2[0]}', f's{s1_s2[1]}'): 1 for s1_s2, c_s1_s2 in covers2}
                              },  ## theoretically, the quadratic part can be skipped as constraints enfore the double coverage
                   constant=0)

        m.linear_constraint(linear={f's{s}': 1 for s in p.S}, sense='<=', rhs=p.k, name='max_k_sets')
        m.linear_constraint(linear={f's{s}': p.P[s] for s in p.S}, sense='<=', rhs=p.B, name='max_budget')

        for u in p.U:
            if (min_ss := p.R[u]) > 0:
                m.linear_constraint(linear={s: 1 for s in p.u_ss[u]}, sense='>=', rhs=min_ss,
                                    name=f'min_{min_ss}_sets_for_u{u}')
                m.linear_constraint(linear={f'u{u}t{t}': 1 for t in p.u_ts[u]}, sense='>=', rhs=min_ss,
                                    name=f'min_{min_ss}_types_for_u{u}')
                for t in p.u_ts[u]:
                    m.linear_constraint(
                        linear={**{f's{s}': -1 for s in p.u_ss[u] if p.T[s] == t}, **{f'u{u}t{t}': 1}}, sense='<=',
                        rhs=0, name=f'type_cover_u{u}t{t}')

        for u in p.U:
            pc1_vars = []
            if (min_pc1 := p.RC[0][u]) > 0:
                for s in [s for s in p.u_ss[u] if p.C[0][u][(s,)] >= min_pc1]:
                    pc1_vars.append(f'pc1_u{u}_s{s}')
                    m.binary_var(name=f'pc1_u{u}_s{s}')
                    m.linear_constraint(linear={**{f's{s}': -p.C[0][u][(s,)]}, **{f'pc1_u{u}_s{s}': p.M1}}, sense='<=',
                                        rhs=-min_pc1 + p.M1, name=f'min_C1_for_u{u}_by_s{s}')
                m.linear_constraint(linear={var: 1 for var in pc1_vars}, sense='>=', rhs=1,
                                    name=f'at_least_one_C1_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc1}%')

        for u in p.U:
            pc2_vars = []
            if (min_pc2 := p.RC[1][u]) > 0:
                for s1s2 in [s1s2 for s1s2 in p.C[1][u] if p.C[1][u][s1s2] >= min_pc2]:
                    pc2_vars.append(f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    m.binary_var(name=f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    m.binary_var(name=f's{s1s2[0]}_s{s1s2[1]}')
                    m.linear_constraint(
                        linear={**{f's{s1s2[0]}': -p.C[1][u][s1s2]}, **{f's{s1s2[1]}': -p.C[1][u][s1s2]},
                                **{f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}': p.M2}}, sense='<=',
                        rhs=-p.C[1][u][s1s2] + p.M2 - 1,
                        name=f'min_C2_for_u{u}_by_s{s1s2[0]}_s{s1s2[1]}')
                m.linear_constraint(linear={var: 1 for var in pc2_vars}, sense='>=', rhs=1,
                                    name=f'at_least_one_C2_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc2}%')  # len(pc1_vars)-1

        for u in p.U:
            if len(p.u_ss[u]) > 0:
                m.linear_constraint(linear={**{f's{s}': p.SQ[s] for s in p.u_ss[u]}, **{f'u{u}': -p.UQ[u]}}, sense='>=',
                                    rhs=0, name=f'min_quality_for_u{u}')

        ### Quick hack for file export and QUBO stats
        # mp = m.export_mp()
        # mp.export_as_mps('qiskit.mps')
        # m.write_to_lp_file('qiskit.lp')
        #
        # qubo = QuadraticProgramToQubo().convert(m)
        # print(f'#Vars MP: {m.get_num_vars()} -> #Vars QUBO: {qubo.get_num_vars()}')
        # print(f'#LC MP: {m.get_num_linear_constraints()} -> #LC QUBO: {qubo.get_num_linear_constraints()}')
        # print(f'#QC MP: {m.get_num_quadratic_constraints()} -> #QC QUBO: {qubo.get_num_quadratic_constraints()}')
        # # print(qubo)
        # qubo.export_mp().export_as_mps('qubo.mps')

        return m

    ### The following methods just re-cast the problem formulation to different libraries

    def to_gurobi_model(self, p: Problem):
        m = gp.Model(self.name)
        # m = gp.read('qubo.mps')
        m.ModelSense = GRB.MAXIMIZE

        s_sb = m.addVars(p.S, obj=p.sC_sum, name='s', vtype=GRB.BINARY)
        u_ub = m.addVars(p.U, obj=p.W_offset, name='u', vtype=GRB.BINARY)
        uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]
        ut_utbs = m.addVars(uts, name='', vtype=GRB.BINARY)
        m.setObjective(gp.quicksum([p.alpha * p.sC_sum[s] * s_sb[s] for s in
                                    p.covers1]))  # + gp.quicksum([p.beta * s_sb[s1_s2[0]] * s_sb[s1_s2[1]] * c_s1_s2 for s1_s2, c_s1_s2 in p.covers2]))
        m.addConstr(s_sb.sum() <= p.k, "max k sets")
        m.addConstr(gp.quicksum([s_sb[s] * p.P[s] for s in p.S]) <= p.B, 'max budget')

        for u in p.U:
            if (min_ss := p.R[u]) > 0:
                m.addConstr(gp.quicksum([s_sb[s] for s in p.u_ss[u]]) >= min_ss, f'min_{min_ss}_sets_for_u{u}')
                m.addConstr(gp.quicksum(ut_utbs[f'u{u}t{t}'] for t in p.u_ts[u]) >= min_ss,
                            f'min_{min_ss}_types_for_u{u}')
                for t in p.u_ts[u]:
                    m.addConstr((gp.quicksum([-s_sb[s] for s in p.u_ss[u] if p.T[s] == t])) + ut_utbs[f'u{u}t{t}'] <= 0,
                                f'type_cover_u{u}t{t}')

        for u in p.U:
            pc1_vars = []
            pc1_vars_b = []
            if (min_pc1 := p.RC[0][u]) > 0:
                for s in [s for s in p.u_ss[u] if p.C[0][u][(s,)] >= min_pc1]:
                    pc1_vars.append(f'pc1_u{u}_s{s}')
                    pc1_var = m.addVar(name=f'pc1_u{u}_s{s}', vtype=GRB.BINARY)
                    pc1_vars_b.append(pc1_var)
                    m.addConstr((-p.C[0][u][(s,)] * s_sb[s] + p.M1 * pc1_var) <= -min_pc1 + p.M1,
                                f'min_C1_for_u{u}_by_s{s}')
                m.addConstr(gp.quicksum(pc1_vars_b) >= 1,
                            f'at_least_one_C1_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc1}%')

        for u in p.U:
            pc2_vars = []
            pc2_vars_b = []
            if (min_pc2 := p.RC[1][u]) > 0:
                for s1s2 in [s1s2 for s1s2 in p.C[1][u] if p.C[1][u][s1s2] >= min_pc2]:
                    pc2_vars.append(f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    pc2_var = m.addVar(name=f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}', vtype=GRB.BINARY)
                    pc2_vars_b.append(pc2_var)
                    m.addConstr(
                        (-p.C[1][u][s1s2] * s_sb[s1s2[0]] + (-p.C[1][u][s1s2] * s_sb[s1s2[1]]) + p.M2 * pc2_var) <= -
                        p.C[1][u][s1s2] + p.M2 - 1, f'min_C2_for_u{u}_by_s{s1s2[0]}_s{s1s2[1]}')
                m.addConstr(gp.quicksum(pc2_vars_b) >= 1,
                            f'at_least_one_C2_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc2}%')

        for u in p.U:
            if len(p.u_ss[u]) > 0:
                m.addConstr(
                    (gp.quicksum([s_sb[s] * p.SQ[s] for s in p.u_ss[u]]) + gp.quicksum([-p.UQ[u] * u_ub[u]])) >= 0,
                    f'min_quality_for_u{u}')
        # m.write('qurobi.lp')
        return m

    def to_cplex_model(self, p: Problem):
        m = docplex.mp.model.Model(name=self.name)
        s_sb = m.binary_var_dict(p.S, name='s')
        u_ub = m.binary_var_dict(p.U, name='u')
        uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]
        ut_utbs = m.binary_var_dict(uts, name='')

        m.maximize(m.sum([p.alpha * p.sC_sum[s] * s_sb[s] for s in p.covers1]))
        m.add(m.sum(s_sb) <= p.k, 'k sets')
        m.add(m.sum(m.scal_prod(list(s_sb.values()), p.P)) <= p.B, 'max budget')

        for u in p.U:
            if (min_ss := p.R[u]) > 0:
                m.add(m.sum([s_sb[s] for s in p.u_ss[u]]) >= min_ss, f'min_{min_ss}_sets_for_u{u}')
                m.add(m.sum(ut_utbs[f'u{u}t{t}'] for t in p.u_ts[u]) >= min_ss, f'min_{min_ss}_types_for_u{u}')
                for t in p.u_ts[u]:
                    m.add((m.sum([-s_sb[s] for s in p.u_ss[u] if p.T[s] == t])) + ut_utbs[f'u{u}t{t}'] <= 0,
                          f'type_cover_u{u}t{t}')

        for u in p.U:
            pc1_vars = []
            pc1_vars_b = []
            if (min_pc1 := p.RC[0][u]) > 0:
                for s in [s for s in p.u_ss[u] if p.C[0][u][(s,)] >= min_pc1]:
                    pc1_vars.append(f'pc1_u{u}_s{s}')
                    pc1_var = m.binary_var(f'pc1_u{u}_s{s}')
                    pc1_vars_b.append(pc1_var)
                    m.add((-p.C[0][u][(s,)] * s_sb[s] + p.M1 * pc1_var) <= -min_pc1 + p.M1, f'min_C1_for_u{u}_by_s{s}')
                m.add(m.sum(pc1_vars_b) >= 1,
                      f'at_least_one_C1_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc1}%')

        for u in p.U:
            pc2_vars = []
            pc2_vars_b = []
            if (min_pc2 := p.RC[1][u]) > 0:
                for s1s2 in [s1s2 for s1s2 in p.C[1][u] if p.C[1][u][s1s2] >= min_pc2]:
                    pc2_vars.append(f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    pc2_var = m.binary_var(
                        name=f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    pc2_vars_b.append(pc2_var)
                    m.add(
                        (-p.C[1][u][s1s2] * s_sb[s1s2[0]] + (-p.C[1][u][s1s2] * s_sb[s1s2[1]]) + p.M2 * pc2_var) <= -
                        p.C[1][u][s1s2] + p.M2 - 1, f'min_C2_for_u{u}_by_s{s1s2[0]}_s{s1s2[1]}')
                m.add(m.sum(pc2_vars_b) >= 1,
                      f'at_least_one_C2_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc2}%')

        for u in p.U:
            if len(p.u_ss[u]) > 0:
                m.add(
                    (m.sum([s_sb[s] * p.SQ[s] for s in p.u_ss[u]]) + m.sum([-p.UQ[u] * u_ub[u]])) >= 0,
                    f'min_quality_for_u{u}')
        # m.export_as_lp('cplex.lp')
        return m

    def to_ocean_cqm(self, p: Problem):
        m = ConstrainedQuadraticModel()
        s_sb = {s: Binary(f's{s}') for s in p.S}
        sbs = s_sb.values()
        u_ub = {u: Binary(f'u{u}') for u in p.U}
        u_t_utb = {u: {t: Binary(f'u{u}t{t}') for t in p.u_ts[u]} for u in p.U}

        m.set_objective(-1 * (sum([p.alpha * p.sC_sum[s] * s_sb[s] for s in p.covers1])))

        m.add_constraint(sum(sbs) <= p.k, "max k sets")
        m.add_constraint(sum(map(mul, sbs, p.P)) <= p.B, 'max budget')

        for u in p.U:
            if (min_ss := p.R[u]) > 0:
                m.add_constraint(sum([s_sb[s] for s in p.u_ss[u]]) >= min_ss, f'min_{min_ss}_sets_for_u{u}')
                m.add_constraint(sum(u_t_utb[u][t] for t in p.u_ts[u]) >= min_ss, f'min_{min_ss}_types_for_u{u}')
                for t in p.u_ts[u]:
                    m.add_constraint((sum([-s_sb[s] for s in p.u_ss[u] if p.T[s] == t])) + u_t_utb[u][t] <= 0,
                                     f'type_cover_u{u}t{t}')

        for u in p.U:
            pc1_vars = []
            pc1_vars_b = []
            if (min_pc1 := p.RC[0][u]) > 0:
                for s in [s for s in p.u_ss[u] if p.C[0][u][(s,)] >= min_pc1]:
                    pc1_vars.append(f'pc1_u{u}_s{s}')
                    pc1_var = Binary(f'pc1_u{u}_s{s}')
                    pc1_vars_b.append(pc1_var)
                    m.add_constraint((-p.C[0][u][(s,)] * s_sb[s] + p.M1 * pc1_var) <= -min_pc1 + p.M1,
                                     f'min_C1_for_u{u}_by_s{s}')
                m.add_constraint(sum(pc1_vars_b) >= 1,
                                 f'at_least_one_C1_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc1}%')

        for u in p.U:
            pc2_vars = []
            pc2_vars_b = []
            if (min_pc2 := p.RC[1][u]) > 0:
                for s1s2 in [s1s2 for s1s2 in p.C[1][u] if p.C[1][u][s1s2] >= min_pc2]:
                    pc2_vars.append(f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    pc2_var = Binary(f'pc2_u{u}_s{s1s2[0]}_s{s1s2[1]}')
                    pc2_vars_b.append(pc2_var)
                    m.add_constraint(
                        (-p.C[1][u][s1s2] * s_sb[s1s2[0]] + (-p.C[1][u][s1s2] * s_sb[s1s2[1]]) + p.M2 * pc2_var) <= -
                        p.C[1][u][s1s2] + p.M2 - 1, f'min_C2_for_u{u}_by_s{s1s2[0]}_s{s1s2[1]}')
                m.add_constraint(sum(pc2_vars_b) >= 1,
                                 f'at_least_one_C2_set_with_sufficient_partial_coverage_for_u{u}_with_{min_pc2}%')

        for u in p.U:
            if len(p.u_ss[u]) > 0:
                m.add_constraint(
                    (sum([s_sb[s] * p.SQ[s] for s in p.u_ss[u]]) + sum([-p.UQ[u] * u_ub[u]])) >= 0,
                    f'min_quality_for_u{u}')
        return m

    def to_ocean_bqm_from_cqm(self, p: Problem):
        return dimod.constrained.cqm_to_bqm(self.to_ocean_cqm(p))

    def extract(self, result_dict):
        return sorted({k: v for k, v in result_dict.items() if k.startswith('s')}.values())


### Different generators for synthentic data

def gen(nU, nS_per_U, k):
    nS = nU * nS_per_U
    RC = [[0] * nU, [0] * nU]
    C = [[]]
    for u in range(nU):
        C[0].append({(s,): 25 for s in range(nS_per_U)})
    return Problem(nU=nU, nS=nS, k=k, R=[0] * nU, C=C, RC=RC, UQ=[0] * nU, SQ=[1] * nS, W=[1] * nU, T=range(1, nS + 1),
                   P=[1] * nS, B=10000)


def gen2(nU, nS_per_U, k):
    nS = nU * nS_per_U
    RC = [[90] * nU, [80] * nU]
    C = [[], []]
    for u in range(nU):
        # C[0].append({(s,): 25 for s in range(nS_per_U)})
        ss = range(max(0, u * nS_per_U - 1),
                   (u * nS_per_U) + nS_per_U)  # list(set(random.sample(range(nS), k=nS_per_U)))
        C[0].append({(s,): random.randint(90, 100) for s in ss})
        sss = list(set(random.sample(ss, k=int(len(ss) / 3) * 2)))
        # print("SSS=", sss)
        C[1].append(
            {(s1, s2): random.randint(80, min(C[0][u][(s1,)], C[0][u][(s2,)])) for s1, s2 in zip(*[iter(sss)] * 2)})
    return Problem(nU=nU, nS=nS, k=k, R=[2] * nU, C=C, RC=RC, UQ=[1] * nU, SQ=[1] * nS, W=[1] * nU,
                   T=[i % 20 for i in range(nS)],
                   # range(1, nS + 1),
                   P=[1] * nS, B=10000)


def gen_partial_coverage_for_R_model(nU, nS_per_U, minC1=90, minC2=80, overlap_ratio=1 / 3, nS100=0):
    # nS = nU * nS_per_U
    # RC = [[minC1] * nU, [minC2] * nU]
    C = [[], []]
    for u in range(nU):
        # C[0].append({(s,): 25 for s in range(nS_per_U)})
        ss = list(
            range(max(0, u * nS_per_U), (u * nS_per_U) + nS_per_U))  # list(set(random.sample(range(nS), k=nS_per_U)))
        # C[0].append({(ss[0],): 100, (ss[1],): 100} | {(s,): random.randint(minC1, 100) for s in ss[2:]})
        C[0].append({(s,): random.randint(minC1, 100) for s in ss})
        sss = list(set(random.sample(ss, k=int(overlap_ratio * len(ss)) * 2)))
        C[1].append(
            {(s1, s2): random.randint(minC2, min(C[0][u][(s1,)], C[0][u][(s2,)])) for s1, s2 in zip(*[iter(sss)] * 2)})
    return C


def gen_partial_coverage_for_U_model(nU, nS_per_U, R_width, R_height, C):
    volume = R_width * R_height
    U_cnt = 0
    nS = nU * nS_per_U
    Ss = {s: [] for s in range(nS)}
    Ss_gap = {s: 0 for s in range(nS)}
    U_R = {}
    for coverages in C[0]:
        for cover in coverages.items():
            s, c_percent = cover
            Ss_gap[s[0]] = Ss_gap[s[0]] + c_percent

    def gen_Us(s1, s2=None, percent=1, required_cover=0):
        nonlocal U_cnt
        nonlocal U_R
        new_Us_abs = max(1, math.floor((percent / 100) * volume))
        new_Us = list(range(U_cnt, U_cnt + new_Us_abs))
        Ss[s1] = Ss[s1] + new_Us
        Ss_gap[s1] = Ss_gap[s1] - percent  # len(new_Us)
        U_cnt += len(new_Us)
        # assert Ss_gap[s1] >= 0
        if s2:
            Ss[s2] = Ss[s2] + new_Us
            Ss_gap[s2] = Ss_gap[s2] - percent  # len(new_Us)
        #    assert Ss_gap[s2] >= 0
        # U_R = U_R | {u: required_cover for u in new_Us}
        for u in new_Us:
            U_R[u] = required_cover

    for coverages in C[1]:
        # max_c_percent = max(coverages.values())
        argmax = np.argmax([Ss_gap[cover[0][0]] + Ss_gap[cover[0][1]] for cover in coverages.items()])
        # print([Ss_gap[cover[0][0]] + Ss_gap[cover[0][1]] for cover in coverages.items()], argmax)
        for i, cover in enumerate(coverages.items()):
            ss, c_percent = cover
            gen_Us(ss[0], ss[1], c_percent, required_cover=2 if i == argmax else 0)
    for coverages in C[0]:
        for cover in coverages.items():
            s, _ = cover
            if (gap_percent := Ss_gap[s[0]]) > 0:
                gen_Us(s[0], percent=gap_percent)

    return list(Ss.values()), U_R


def gen_partial(nR, nS_per_R, k, R_width=1, R_height=1, nT=20, minC1=90, minC2=80, overlap_ratio=1 / 3):
    C_R = gen_partial_coverage_for_R_model(nR, nS_per_R, minC1, minC2, overlap_ratio)

    C_U, U_R = gen_partial_coverage_for_U_model(nR, nS_per_R, R_width, R_height, C_R)

    nS = nR * nS_per_R

    T = [i % nT for i in range(nS)]
    P = [1] * nS
    B = 100000

    nU = len(unique(flatten2d(C_U)))
    W = [1] * nU
    V = flatten2d(c.values() for c in C_R[0])
    R = [r[1] for r in sorted(U_R.items(), key=lambda it: it[0])]  # [2] * nU
    U = MCMTWB_k_MaxCover.Problem(nU=nU, nS=nS, k=k, C=C_U, R=R, W=W, T=T, P=P, B=B, V=V)
    RC = [[minC1] * nR, [minC2] * nR]
    W = [1] * nR

    R = Problem(nU=nR, nS=nS, k=k, R=([2] * nR), C=C_R, RC=RC, UQ=[1] * nR, SQ=[1] * nS, W=W, T=T, P=P, B=B)

    return U, R


def gen_2d_random_boxes(nR, R_width, R_height, nS_per_R, U_per_S, S100_flag=False, offset=0):
    Rs = []
    for Ri in range(nR):
        Ss = []
        S100_max = -1
        if S100_flag:
            nE = (nR * nS_per_R * U_per_S) + 2
            # S100 = [xy for xy in range(offset + (Ri * R_width * R_height), offset + (((Ri + 1) * R_width * R_height) + 1))]
            S100 = [xy for xy in range(offset + (Ri * nE), offset + (((Ri + 1) * nE) + 0))]
            Ss.append(S100)
            Ss.append(S100)
            S100_max = max(S100)
        Rs.append(Ss)
        for Si in range(nS_per_R):
            cntU = 0
            R_offset = offset + (R_width * R_height * Ri)
            a = random.randint(1 + R_offset, R_width - 1 + R_offset)
            b = random.randint(a + 1, R_width + R_offset)
            # h = random.randint(1 + R_offset, R_height + R_offset -1)
            h = random.randint(1, R_height - 1)
            # if h*(b-a) < U_per_S:  # extend square if required
            #    a = 1
            #    b = int(math.ceil(math.sqrt(U_per_S)))
            #    h = int(math.ceil(math.sqrt(U_per_S)))
            S = []
            Ss.append(S)
            for hi in range(h):
                for xy in range(a + (hi * R_width), 1 + b + (hi * R_width)):
                    S.append(xy)
                    cntU += 1
                    if cntU == U_per_S:
                        break
                else:
                    continue
                break

    return Rs


def gen_U(Rs, k):
    S = list(itertools.chain(*Rs))
    return MCMTWB_k_MaxCover.to_consecutive(k, S, R=[2] * len(unique(flatten2d(S))), T=range(len(S)))


def gen_R(Rs):
    pass


def gen_Q1(nR, R_width, R_height, nS_per_R, U_per_S, k):
    Rs1 = gen_2d_random_boxes(nR, R_width, R_height, nS_per_R, U_per_S, S100_flag=False)
    S1 = list(itertools.chain(*Rs1))
    max_S1 = max(flatten2d(S1))
    Rs2 = gen_2d_random_boxes(nR, R_width, R_height, nS_per_R, U_per_S, S100_flag=False, offset=max_S1 + 1)
    S2 = list(itertools.chain(*Rs2))

    nU = nR  # 2 * nR
    nS = len(S2)  # len(S1) + len(S2)
    # RC = [([0] * nR) + ([90] * nR), ([0] * nR) + ([80] * nR)]
    RC = [([90] * nR), ([80] * nR)]
    C = [[], []]

    total_loop = 0
    total_identical = 0
    total_both_max = 0
    total_one_max = 0
    total_cut = 0
    zero_overlap = 0

    max_size = U_per_S  # (R_width * R_height) + 1

    def addC(S, offset=0, C1_limit=-1):
        # C[0].append({(offset + i,): int(100 * min(1, len(s) / (R_width * R_height))) for i, s in enumerate(S)})
        C[0].append(
            {(offset + i,): cover for i, s in enumerate(S) if (cover := int(100 * min(1, len(s) / max_size))) > 0})
        offset += 0  # len(S)
        C1 = {}

        C1_cnt = 0

        nonlocal total_loop
        nonlocal total_identical
        nonlocal total_both_max
        nonlocal total_one_max
        nonlocal total_cut
        nonlocal zero_overlap

        for i in range((len(S))):
            if C1_limit > 0 and C1_cnt > C1_limit:
                break
            for j in range(i + 1, len(S)):
                if C1_limit > 0 and C1_cnt > C1_limit:
                    break
                total_loop += 1
                if i == j:
                    total_identical += 1
                    continue
                # overlap = 0
                if len(S[i]) == max_size and len(S[j]) == max_size:
                    overlap = 1
                    total_both_max += 1
                elif len(S[i]) == max_size:
                    overlap = len(S[i]) / max_size
                    total_one_max += 1
                elif len(S[j]) == max_size:
                    overlap = len(S[j]) / max_size
                    total_one_max += 1
                else:
                    si = set(S[i])
                    sj = set(S[j])
                    overlap = min(1, len(list(si.intersection(sj))) / max_size)
                    total_cut += 1
                if (o := int(100 * overlap)) > 80:
                    C1[(offset + i, offset + j)] = o
                    C1_cnt += 1
                else:
                    zero_overlap += 1
        C[1].append(C1)
        # return len(S)

    offset = 0

    for rs in Rs2:
        addC(rs, offset=offset, C1_limit=int(nS_per_R / 3))
        offset += len(rs)

    # print(
    #     f"First conversion step done: #loop={total_loop} #same={total_identical} #bothmax={total_both_max} #onemax={total_one_max} #cut={total_cut} #zero_overlap={zero_overlap}")

    V = flatten2d(itm.values() for itm in C[0])
    U = MCMTWB_k_MaxCover.to_consecutive(k, S2, V=V, R=(([2] * len(unique(flatten2d(S2))))),
                                         T=[i % 20 for i in range(len(S2))])
    R = Problem(nU=nU, nS=nS, k=k, R=([2] * nR), C=C, RC=RC, UQ=[1] * nU, SQ=[1] * nS, W=[1] * nU,
                T=[i % 20 for i in range(nS)], P=[1] * nS, B=10000)
    return U, R
