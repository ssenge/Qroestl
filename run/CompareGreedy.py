import time

from qroestl.backends import Gurobi, Ocean
from qroestl.problems import MCMTWB_k_MaxCover

import warnings
warnings.filterwarnings("ignore")

max_i = 500

a = MCMTWB_k_MaxCover.Standard()
greedy = Ocean.Tabu()#MCMTWB_k_MaxCover.Greedy()
gurobi = Gurobi.Optimizer()

errors = 0
total_obj_gurobi = 0
total_obj_greedy = 0
now = time.strftime("%Y-%m-%d_%H-%M-%S")
with open(f'results/greedy_comparison_{now}.csv', "wt") as fp:
    fp.write("# i, true obj, greedy obj, greedy error (abs), greedy error (%)\n")
    for i in range(2, max_i + 2):

        p = MCMTWB_k_MaxCover.gen_syn_random_coverages(nU=i, nS=0.1, k=1, T=1)
        sol_greedy = greedy.optimize(p, a)
        obj_greedy = sol_greedy.best[1]
        obj_greedy = obj_greedy if obj_greedy else 0
        sol_gurobi = gurobi.optimize(p, a)
        obj_gurobi = sol_gurobi.best[1]
        total_obj_greedy += obj_greedy
        total_obj_gurobi += obj_gurobi
        delta = obj_gurobi - obj_greedy
        delta_per = (delta * 100) / obj_gurobi
        print(f'{i - 2} ->  nU: {p.nU} nS: {p.nS} k: {p.k} true_obj: {obj_gurobi}', end='')
        if delta > 0:
            errors += 1
            print(f' greedy_obj: {obj_greedy} delta: {delta} ({delta_per}%) error: {errors}')
            # print(p)
        else:
            print()
        fp.write(f'{i}, {obj_gurobi}, {obj_greedy}, {delta}, {delta_per}\n')

total_delta = total_obj_gurobi - total_obj_greedy
print(f'Done: {errors} errors out of {max_i} = {(errors * 100) / max_i}% delta: {total_delta} ({(total_delta * 100) / total_obj_gurobi})%')
