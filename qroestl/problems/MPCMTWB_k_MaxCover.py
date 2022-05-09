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
from typing import List, Optional, Dict, Tuple

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
    C1: List[Dict[int, float]]  # partially covered universe elements (per universe element)
    C2: List[Dict[Tuple[int, int], float]]  # covered universe elements (per set)
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

        self.u_ss = {u: list(ss.keys()) for u, ss in enumerate(self.C1)}
        self.s_us = {s: [u for u, ss in self.u_ss.items() if s in ss] for s in self.S}

        # self.u_ts = {u: list(set([self.T[s] for s in self.u_ss[u]])) for u in self.U}  # u -> types (of sets that cover u)
        self.W_offset = [w * (self.k + 1) for w in
                         self.W]  # offsetting weights (by linear multiplicative spread) to allow for negative coefficients for set variables without affecting the optimal solution
        # self.P_normalized = [self.k * (self.P[s] / sum(self.P)) for s in self.S]
        # self.B_normalized = sum(self.P_normalized)

    def __str__(self):
        return f'nU={self.nU} nS={self.nS} k={self.k} B={self.B}\nW={np.array(self.W)}\nR={np.array(self.R)}\nT={np.array(self.T)}\nP={np.array(self.P)}\nC1={np.array(self.C1)}\nC2={np.array(self.C2)}'

    def _validate(self):
        # assert self.nU == len(self.U)
        assert self.nU == len(self.W)
        assert self.nU == len(self.R)
        # assert self.nS == len(self.S)
        assert self.nS == len(self.T)
        assert self.nS == len(self.P)
        assert self.B >= 0

    def feasible(self, c: TCandidate) -> bool:
        return True

    def value(self, c: TCandidate) -> float:
        c1_sum = sum(c1[s] for s in c for c1 in self.C1 if s in c1)
        # TODO: c2_sum
        return c1_sum # + c2_sum


    def all_solutions(self) -> List[TCandidate]:
        return Utils.powerset(self.S)


@dataclass
class Standard(Model.Approach[Problem], QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible):
    name: str = "MPCMTWB-k-MaxCover Heuristic1"

    def to_qiskit_qp(self, p: Problem) -> QuadraticProgram:
        # uts = [f'u{u}t{t}' for u in p.U if p.R[u] > 0 for t in p.u_ts[u]]

        alpha = 1  # TODO: parameter
        beta = 1  # TODO: parameter
        t = 0.1  # min coverage threshold, TODO: parameter
        us_C1 = [u for u, r in enumerate(p.R) if r <= 1]
        us_C2 = [u for u, r in enumerate(p.R) if r >= 2]

        covers1 = [sc for u in us_C1 for sc in p.C1[u].items()]
        covers2 = [sc for u in us_C2 for sc in p.C2[u].items()]

        m = QuadraticProgram(self.name)
        m.binary_var_list(p.S, name='s')
        m.binary_var_list(p.U, name='u')

        m.maximize(linear={f's{s}': alpha * c for s, c in covers1},  # *-1
                   quadratic={**{(f's{s1_s2[0]}', f's{s1_s2[1]}'): beta * c_s1_s2 for s1_s2, c_s1_s2 in covers2},
                              # **{(f's{s1_s2[0]}', f's{s1_s2[1]}'): 1 for s1_s2, c_s1_s2 in covers2}
                              },
                   constant=0
                   )

        m.linear_constraint(linear={f's{s}': 1 for s in p.S}, sense='<=', rhs=p.k, name='max k sets')  # equality constraint instead of less than as the used in a loop over k
        # m.linear_constraint(linear={f's{s}': p.P[s] for s in p.S}, sense='<=', rhs=p.B, name='max budget')
        for u in p.U:
            m.linear_constraint(linear={**{f's{s}': 1 for s in p.S if u in p.s_us[s]}, **{f'u{u}': -1}}, sense='>=', rhs=0,
                                name=f'cover for u{u}')

        ## The automatic Qiskit QUBO converter does not allow quadratic constraints but this would be the right formulation (untested)
        #     if (c := p.R[u]) > 0:
        #         m.quadratic_constraint(quadratic={(f's{s1_s2[0]}', f's{s1_s2[1]}'): c_s1_s2 for s1_s2, c_s1_s2 in p.C2[u].items()}, sense='>=', rhs=t, name=f'multi cover for u{u}')

        return m
