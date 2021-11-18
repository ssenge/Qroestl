from datetime import datetime, timedelta

from qroestl.backends import Qiskit
from qroestl.model import Model
from qroestl.problems import MCMTWB_k_MaxCover
from run.Config import Config


limit = 60  # sec
nS = 10
k = 5
nS_grow_factor = 1.1
k_divisor = 10
ap = MCMTWB_k_MaxCover.Standard()
gen_p = lambda nS, k: MCMTWB_k_MaxCover.gen_syn_fully_connected(nS, k)#one_to_one(nS, k)
a = Qiskit.CPLEX()#Model.BruteForce()#MCMTWP_k_MaxCover.Greedy()


if __name__ == '__main__':
    delta = 0
    while delta <= limit:
        print(f'\nnS: {nS} k: {k}', end='')
        p = gen_p(nS, k)
        start = datetime.now()
        a.solve(p, ap)
        end = datetime.now()
        delta = (end - start).total_seconds()
        print(f' took {delta}', end='')
        nS = int(nS * nS_grow_factor)
        k = int(nS / k_divisor)
    print('... limit reached... terminate!')
