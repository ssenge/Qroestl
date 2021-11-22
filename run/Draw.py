from qroestl.backends import Gurobi
from qroestl.problems import MCMTWB_k_MaxCover

# p = MCMTWP_k_MaxCover.gen_syn_random_coverages(nU=10, nS=0.1, k=1, T=1)
p = MCMTWB_k_MaxCover.Problem(nU=4, nS=3, k=2, R=[0, 0, 0, 0], T=[0, 0, 0], W=[1, 2.1, 1, 1], C=[[0], [1], [2, 3]])
# p = MCMTWP_k_MaxCover.Problem(nU=4, nS=3, k=2, R=[0, 0, 0, 0], T=[0, 0, 0], W=[1, 2, 2, 1], C=[ [0, 1], [1, 2], [2, 3]])
#p = MCMTWB_k_MaxCover.gen_syn_fully_connected(5, 1)
a = MCMTWB_k_MaxCover.Standard()
a.to_qiskit_qp(p)
greedy = MCMTWB_k_MaxCover.Greedy()
# gurobi = Gurobi.Optimizer()

p.visualize()
sol = greedy.optimize(p, a)
sets = sol.best[0]
obj = sol.best[1]
price = sum(p.P[s] for s in sets)
p.visualize(selected=sets, save=True, k_sol=len(sets), obj_sol=obj, p_sol=price)
