import unittest

from qroestl.backends import Qiskit, Ocean, Gurobi, Cplex
from qroestl.model import Model, BruteForce

from qroestl.problems import MPCMTWB_k_MaxCover


class TestMCMT(unittest.TestCase):

    def run_test(self, algos, ps, aps):
        for a in algos:
            for p in ps:
                for ap in aps:
                    sol, val = a.optimize(p[0], ap).best
                    res = (sorted(sol) if sol else None, val)
                    self.assertEqual(res[1], p[1][1], f'{a} for {p} differs from expected result {p[1]}: {res} for algo {a} and approach {ap}')

    def test1(self):
        ps = [(MPCMTWB_k_MaxCover.Problem(nU=2, nS=3, k=2, R=[2, 0], RC=[0.1, 0], T=[0, 1, 0], W=[1, 1],
                                          C=[
                                              [{(0,): 0.6, (1,): 0.15, (2,): 0.2}, {(2,): 0.2}],
                                              [{(1, 2): 0.1}, {}]
                                          ]), ([1, 2], 0.55)),
              ]
        algos = [Model.BruteForce(),
            #MPCMTWB_k_MaxCover.Greedy(),
            #Qiskit.CPLEX(),
            Qiskit.NumpyExact(),
            #Ocean.HybridCQM()#Exact()
            #Cplex.CplexOptimizer(),
            #Gurobi.Optimizer()
        ]

        approaches = [MPCMTWB_k_MaxCover.Standard(),
        ]
        self.run_test(algos, ps, approaches)
