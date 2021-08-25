import unittest

from qroestl.model import Model

from qroestl.problems import MultiCover


class TestMCMT(unittest.TestCase):

    def run_test(self, algos, ps):
        for a in algos:
            for p in ps:
                sol, val = a.solve(p[0]).best
                res = (sorted(sol) if sol else None, val)
                self.assertEquals(res, p[1], f'{a} for {p} differs from expected result {p[1]}: {res}')

    def test1(self):
        ps = [(MultiCover.Problem(k=2, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 5], C=[0, 2, 0, 0], T=[1, 1, 2]), ([1, 2], 8)),
              (MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 10], C=[2, 2, 0, 0], T=[1, 2, 3]), ([0, 1, 2], 13)),
              (MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8], [10, 11, 12, 13]], W=[1, 1, 1, 10, 1, 1, 1, 1], C=[1, 2, 0, 0, 0, 0, 0, 0], T=[1, 2, 3, 4]), ([1, 2, 3], 17)),
              (MultiCover.Problem(k=2, S=[[1], [2, 1], [3, 4]], W=[1, 1, 1, 100], C=[2, 0, 0, 0], T=[1, 2, 1]), ([0, 1], 2)),
              (MultiCover.Problem(k=4, S=[[7], [8, 7], [1, 2, 8], [10, 11, 12, 13], [14, 15]],W=[1, 1, 1, 10, 1, 1, 1, 1, 100, 5], C=[1, 2, 0, 0, 0, 0, 0, 0, 0, 0], T=[1, 2, 3, 4, 5]), ([1, 2, 3, 4], 122)),
              (MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 1], C=[0, 2, 0, 0], T=[2, 1, 1]), (None, None)),
              (MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 1], C=[0, 2, 0, 0], T=[2, 1, 1]), (None, None))
            ]
        algos = [Model.BruteForce(),
                 MultiCover.Greedy()
            ]
        self.run_test(algos, ps)
