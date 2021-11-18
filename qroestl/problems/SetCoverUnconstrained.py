# from typing import List
#
# import numpy as np
# from qiskit_optimization import QuadraticProgram
#
# from qroestl.problems import SetCover
#
#
# class Problem(SetCover.Problem):
#     U: List[int] = None
#     S: List[List[int]] = None
#
#     def __post_init__(self) -> None:
#         self.U = np.array(self.U)
#         self.S = np.array(self.S+[[]], dtype=object)[:-1]
#
#     def to_qiskit_qp(self) -> QuadraticProgram:
#         qp = QuadraticProgram('Set Cover Unconstrained')
#         qp.binary_var_list(len(self.S), name='s')
#         qp.maximize(linear={'s0': -1, 's1': -1, 's0': 5, 's0': 6})##[-1,2]*len(self.S))  # add weights, TODO: currently only const = 1
#         #for u in self.U:  # add cover constraints
#         #    qp.linear_constraint(linear={f's{i}': 1 for i, s in enumerate(self.S) if u in s},
#         #                         sense='>=', rhs=1, name=f'cover u{u}')
#         print(qp.export_as_lp_string())
#         return qp