from datetime import datetime, timedelta

from qroestl.backends import Qiskit, Ocean, Gurobi
from qroestl.model import Model
from qroestl.problems import MCMTWB_k_MaxCover

import warnings
warnings.filterwarnings("ignore")

limit = 60  # sec
nS = 10
k = 5
nS_grow_factor = 1.1
k_divisor = 10
ap = MCMTWB_k_MaxCover.Standard()
gen_p = lambda nS, k: MCMTWB_k_MaxCover.gen_syn_fully_connected(nS, k)#one_to_one(nS, k)
o = Gurobi.Optimizer()


if __name__ == '__main__':
    delta = 0
    while delta <= limit:
        print(f'\n{nS}, {k}, ', end='')
        p = gen_p(nS, k)
        start = datetime.now()
        o.optimize(p, ap)
        end = datetime.now()
        delta = (end - start).total_seconds()
        print(f'{delta}', end='')
        nS = int(nS * nS_grow_factor)
        k = int(nS / k_divisor)
    print('... limit reached... terminate!')
