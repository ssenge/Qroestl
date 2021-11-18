# from dataclasses import dataclass
# from typing import List, Optional, Dict
# import numpy as np
# from qiskit_optimization import QuadraticProgram
# from qiskit_optimization.converters import QuadraticProgramToQubo
#
# from qroestl.model import Model
# from qroestl.model.Model import Backend, Solution
# from qroestl.backends.Qiskit import QiskitQPConvertible, QiskitQuboConvertible, QiskitOperatorConvertible
# from qroestl.utils import Utils
# from qroestl.problems import MaxCoverage
#
#
# # https://en.wikipedia.org/wiki/Maximum_coverage_problem
#
#
# #TCandidate = List[List[int]]  # this is more the "logical" type, in reality it is np.array[List[int]]
#
#
# @dataclass
# class Problem(MaxCoverage.Problem):#Model.Problem[TCandidate], QPConvertible, QuboConvertible, OperatorConvertible):
#     name: str = 'Max Set Coverage Heuristic 1'
#     k: int = 1
#     U: Optional[List[int]] = None
#     S: Optional[List[List[int]]] = None
#     W: Optional[List[int]] = None
#     UW: Optional[Dict[List, int]] = None
#
#     def __post_init__(self) -> None:
#         self.S = np.array(self.S+[[]], dtype=object)
#         self.S = self.S[:-1]
#         self.U = self.U if self.U is not None else list(dict.fromkeys(Utils.union_non_sorted(self.S)))
#         self.W = self.W if self.W is not None else [1]*len(self.U)
#         self.UW = self.UW if self.UW is not None else {s: w for s, w in zip(self.U, self.W)}
#         self.D = {(i,j):((len(np.union1d(si,sj))-len(np.intersect1d(si,sj)))/len(np.union1d(si,sj))) for i, si in enumerate(self.S) for j, sj in enumerate(self.S) if i is not j and i <= j}
#             #{(si,sj):len(np.union1d(si, sj))/(len(np.union1d(si,sj))-len(np.intersect1d(si, sj))) for si in self.S for sj in self.S}
#         i=0
#
#     def to_qiskit_qp(self) -> QuadraticProgram:
#         qp = QuadraticProgram('Maximum Set Coverage')
#         qp.binary_var_list(len(self.S), name='s')
#         #qp.binary_var_list(list(self.U), name='u')
#         qp.maximize(linear=([len(s) for s in self.S]), quadratic={(f's{i}', f's{j}'): -self.D[i,j] for i, si in enumerate(self.S) for j, sj in enumerate(self.S) if i is not j and i <= j})
#         qp.linear_constraint(linear={f's{i}': 1 for i in range(len(self.S))}, sense='<=', rhs=self.k, name='max k sets')
#         #for u in self.U:  # add cover constraints
#         #    qp.linear_constraint(linear={**{f's{i}': 1 for i in range(len(self.S)) if u in self.S[i]}, **{f'u{u}': -1}},
#         #                         sense='>=', rhs=0, name=f'cover y{u}')
#         print(qp.export_as_lp_string())
#         return qp
#
#     def to_qiskit_qubo(self) -> QuadraticProgram:
#         return QuadraticProgramToQubo().convert(self.to_qiskit_qp())
#
#     def to_qiskit_op(self) -> "Operator":
#         return self.to_qiskit_qubo().to_ising()[0]
#
#
#
#
#
#
#
#
