from qiskit.algorithms.optimizers import COBYLA

from qroestl.backends import Gurobi, Qiskit, Ocean, Cplex, Braket
from qroestl.model import Model
from qroestl.problems import MCMTWB_k_MaxCover, MPCMTWB_k_MaxCover, problem

tasks = [
    #(MCMTWB_k_MaxCover.gen_syn_fully_connected(20, 20), [MCMTWB_k_MaxCover.Standard()]),
    (MCMTWB_k_MaxCover.Problem(nU=4, nS=3, k=2, R=[0, 0, 0, 0], T=[0, 0, 0], W=[1, 2.1, 1, 1], C=[[0], [1], [2, 3]]), [MCMTWB_k_MaxCover.Heuristic1()]),
    # (MPCMTWB_k_MaxCover.Problem(nU=2, nS=3, k=2, R=[2, 0], RC=[0.1, 0], T=[0, 1, 0], W=[1, 1],
    #                             C=[  # List of dicts from tuple of sets to coverage
    #                                 [{(0,): 0.6, (1,): 0.15, (2,): 0.2}, {(2,): 0.2}],
    #                                 [{(1, 2): 0.1}, {}]
    #                             ]),
    #                             [MPCMTWB_k_MaxCover.Standard()]),
    # (problem.Problem(num_vars, matrix, linear, constant, upper_bounds, lower_bounds, integer, binary), [problem.Standard()])

]

optimizers = [
    ### Qroestl
    Model.BruteForce(),
    #MCMTWB_k_MaxCover.Greedy(),

    ### Qiskit
    Qiskit.NumpyExact(),
    # Qiskit.DWaveAnnealer(),  # make sure that DWave tools / configuration is in place
    # Qiskit.VQE(quantum_instance=qdev, kwargs={'optimizer': COBYLA(1)}),
    # Qiskit.QAOA(kwargs={'reps': 2, 'optimizer': COBYLA(2)}),

    ### DWave
    # Ocean.Exact(),
    # Ocean.Greedy(),
    # Ocean.Tabu(),
    # Ocean.BQM(),
    # Ocean.BQM_Clique(),
    # Ocean.HybridBQM(),
    # Ocean.HybridCQM(),

    ### Braket
    # Braket.DWave(),

    ### Cplex
    #Cplex.Optimizer(),  # make sure that Cplex is installed, this unfortunately rather complex and only works in Python 3.8 for me (not in

    ### Gurobi
    #Gurobi.Optimizer(),
]
