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

TCandidate = List[int]  # this is more the "logical" type, in reality it is np.array[List[int]]


@dataclass
class Problem(Model.Problem[TCandidate]):
    """
    Build quadratic mixed integer program
    """
    num_vars: int # number of variables
    matrix: List[List[int]] # quadratic factor matrix
    linear: Optional(List[int]) # linear coefficients
    constant: Optional(int) # constant offset
    upper_bounds: Optional(List[int]) # upper bounds of variables
    lower_bounds: Optional(List[int]) # lower bounds of variables
    integer: Optional(List[int]) # integer variables [0,1,0,1,1,1] (Bool Value)
    binary: Optional(List[int]) # binary variables [0,0,0,1,1,1] (Bool Value)
    name: str = 'Quadratic Mixed Integer Program'

    def __post_init__(self) -> None:
        if not self.upper_bounds:
            self.upper_bounds = [1] * self.n # upper bound 1
        if not self.lower_bounds:
            self.lower_bounds = [0] * self.n # lower bound 0
        if not self.binary:
            self.binary = [1] * self.n # every variable binary
        if not self.integer:
            self.integer = [1] * self.n # every variable integer
        if not self.constant:
            self.constant = 0 # every variable integer
        self._validate()
        self.index_list = list(range(self.num_vars))

    def __str__(self):
        return f'matrix={self.matrix}'

    def _validate(self):
        """ Doing some sanity checks. """
        pass

    def feasible(self, c: TCandidate) -> bool:
        ''' Checks if a candidate solution is feasible, used e.g. in the brute force approach. '''
        pass

    def value(self, c: TCandidate) -> float:
        """ Returns the value of a solution candidate. """
        return np.matmul(np.matmul(self.matrix, c) + self.linear, c)

    def all_solutions(self) -> List[TCandidate]:
        """ Return all possible solutions. """
        pass


@dataclass
class Standard(Model.Approach[Problem], QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible,
               GurobiModelConvertible, CplexModelConvertible, OceanCQMConvertible, OceanCQMToBQMConvertible):
    """ Standard solution approach. """

    name: str = "Quadratic Standard"

    def get_index_lists(p: Problem) -> tuple(List[int], List[int], List[int]):
        binary_index_list = [i for i, elem in enumerate(p.binary) if elem == 1]
        integer_index_list = [i for i, elem in enumerate(p.integer^p.binary) if elem == 1]
        continous_index_list = np.array(np.logical_not(np.logical_and(binary_index_list, integer_index_list)), dtype=int)
        return binary_index_list, integer_index_list, continous_index_list

    def to_qiskit_qp(self, p: Problem) -> QuadraticProgram:
        """ Returns a Qiskit QuadraticProgram representation of the optimization problem.  """

        ##### The general outline for the LP is described here, in the below ports to other SDKs, the same structure will be used
        m = QuadraticProgram(self.name)

        # get binary, integer, continuous variables
        binary_index_list, integer_index_list, continous_index_list = self.get_index_lists()
        m.binary_var_list(binary_index_list, lowerbound=p.lower_bounds[binary_index_list], upperbound=p.upper_bounds[binary_index_list], name='b')
        m.integer_var_list(integer_index_list, lowerbound=p.lower_bounds[integer_index_list], upperbound=p.upper_bounds[integer_index_list], name='i')
        m.continuous_var_list(continous_index_list, lowerbound=p.lower_bounds[continous_index_list], upperbound=p.upper_bounds[continous_index_list], name='c')

        # define objective function    # here not clear how qiskit deals with variables previously defined to be binary/integer?
        m.maximize(linear=p.linear,
                   quadratic=p.matrix,
                   constant=p.constant)

        # no constraints, only upper/lower bounds on variables

        # m.linear_constraint(linear=p.lower_bounds, sense='<=', rhs=p., name='max_k_sets')
        # m.linear_constraint(linear={f's{s}': p.P[s] for s in p.S}, sense='<=', rhs=p.B, name='max_budget')

        return m

    ### The following methods just re-cast the problem formulation to different libraries

    def extract(self, result_dict):
        return sorted({k: v for k, v in result_dict.items() if k.startswith('s')}.values())
