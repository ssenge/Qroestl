import unittest

from qroestl.backends import Qiskit, Ocean, Gurobi, Cplex
from qroestl.model import Model, BruteForce

from qroestl.problems import MCMTW_MaxCoverage, MCMTWB_k_MaxCover


class TestMCMT(unittest.TestCase):

    def run_test(self, algos, ps, aps):
        for a in algos:
            for p in ps:
                for ap in aps:
                    sol, val = a.optimize(p[0], ap).best
                    res = (sorted(sol) if sol else None, val)
                    self.assertEqual(res[1], p[1][1], f'{a} for {p} differs from expected result {p[1]}: {res} for algo {a} and approach {ap}')

    def test1(self):
        ps = [(MCMTWB_k_MaxCover.Problem(k=2, nS=3, nU=4, W=[1, 1, 1, 5], R=[0, 2, 0, 0], T=[1, 1, 2], C=[[0], [0, 1], [1, 2, 3]]), ([1, 2], 8)),
              (MCMTWB_k_MaxCover.Problem(k=1, nS=1, nU=2, W=[1, 1], R=[0, 1], T=[1], C=[[0, 1]]).collapse(), ([0], 2)),  # collapse test 1
              (MCMTWB_k_MaxCover.Problem(k=1, nS=2, nU=4, W=[1, 1, 1, 1], R=[0, 1, 0, 0], T=[1, 1], C=[[0, 1], [1, 2, 3]]), ([1], 3)),  # collapse test 2
              (MCMTWB_k_MaxCover.Problem(k=2, nS=5, nU=11, W=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], R=[0, 1, 1, 0, 0, 2, 1, 1, 0, 0, 0], T=[1, 1, 2, 1, 1], C=[[0, 1], [1, 2, 3, 4, 5], [5, 6, 7], [7, 8, 9, 10], [10]]), ([1, 2], 7)),  # collapse test 3
              (MCMTWB_k_MaxCover.Problem(k=3, nS=3, nU=4, W=[1, 1, 1, 10], R=[2, 2, 0, 0], T=[1, 2, 3], C=[[0], [0, 1], [1, 2, 3]]), ([0, 1, 2], 13)),
              (MCMTWB_k_MaxCover.Problem(k=3, nS=4, nU=8, W=[1, 1, 1, 10, 1, 1, 1, 1], R=[1, 2, 0, 0, 0, 0, 0, 0], T=[1, 2, 3, 4], C=[[0], [0, 1], [1, 2, 3], [4, 5, 6, 7]]), ([1, 2, 3], 17)),
              (MCMTWB_k_MaxCover.Problem(k=2, nS=3, nU=4, W=[1, 1, 1, 100], R=[2, 0, 0, 0], T=[1, 2, 1], C=[[0], [0, 1], [2, 3]]), ([0, 1], 2)),
              (MCMTWB_k_MaxCover.Problem(k=4, nS=5, nU=10, W=[1, 1, 1, 10, 1, 1, 1, 1, 100, 5], R=[1, 2, 0, 0, 0, 0, 0, 0, 0, 0], T=[1, 2, 3, 4, 5], C=[[0], [0, 1], [1, 2, 3], [4, 5, 6, 7], [8, 9]]), ([1, 2, 3, 4], 122)),
              (MCMTWB_k_MaxCover.Problem(k=2, nS=3, nU=3, W=[100, 1, 10], R=[2, 1, 0], T=[1, 1, 3], C=[[0, 2], [0, 1], [0, 2]]), ([1, 2], 111)),
              (MCMTWB_k_MaxCover.Problem(k=2, nS=3, nU=3, W=[1, 1, 10], R=[0, 0, 0], T=[1, 1, 1], C=[[0], [1], [2]]), ([1, 2], 11)),
              (MCMTWB_k_MaxCover.Problem(k=2, nS=3, nU=3, W=[1, 1, 10], R=[0, 0, 0], T=[1, 1, 1], B=20, P=[1, 1, 100], C=[[0], [1], [2]]), ([0, 1], 2)),
              (MCMTWB_k_MaxCover.Problem(k=3, nS=3, nU=4, W=[1, 1, 1, 1], R=[0, 2, 0, 0], T=[2, 1, 1], C=[[0], [0, 1], [2, 3, 1]]), (None, None)),
              (MCMTWB_k_MaxCover.Problem(k=2, nS=3, nU=3, W=[1, 1, 10], R=[0, 0, 0], T=[1, 1, 1], P=[1, 2, 3], B=3, C=[[0], [1], [2]]), ([2], 10)),
              (MCMTWB_k_MaxCover.to_consecutive(k=3, S=[[177, 194, 210, 211, 212], [178, 195, 211, 212, 213], [179, 196, 212, 213, 214], [177, 193, 194, 195, 210, 211, 212], [178, 194, 195, 196, 211, 212, 213], [179, 195, 196, 197, 212, 213, 214]],
                                                 T=[1, 1, 1, 2, 2, 2], R=[0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
                                                 W=[0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0]), ([0, 2, 4], 1.4))
              ]
        algos = [#Model.BruteForce(),
            MCMTWB_k_MaxCover.Greedy(),
            #Qiskit.CPLEX(),
            #Qiskit.NumpyExact(),
            #Ocean.HybridCQM()#Exact()
            #Cplex.CplexOptimizer(),
            #Gurobi.Optimizer()
        ]

        approaches = [MCMTWB_k_MaxCover.Standard(),
        ]
        self.run_test(algos, ps, approaches)


    def test2(self):
        # A case where the Greedy algorithm does not find an optimal solution
        p = MCMTWB_k_MaxCover.Problem(nU=6, nS=3, k=2, R=[0, 0, 0, 0, 0, 0], T=[0, 0, 0], W=[1, 1, 2, 2, 1, 1], C=[[2, 3], [0, 1, 2], [3, 4, 5]])
        sol, val = MCMTWB_k_MaxCover.Greedy().optimize(p).best
        self.assertEqual(val, 6, f'Greedy for {p} differs from expected result (6): {val}')
        sol, val = BruteForce().optimize(p).best
        self.assertEqual(val, 8, f'BruteForce for {p} differs from expected result (8): {val}')
