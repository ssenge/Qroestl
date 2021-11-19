from qiskit.algorithms.optimizers import COBYLA

from qroestl.backends import Gurobi, Qiskit, Ocean, Cplex, Braket
from qroestl.model import Model
from qroestl.problems import MCMTWB_k_MaxCover


#class TaskConfig:

tasks = [
    (MCMTWB_k_MaxCover.gen_syn_fully_connected(1, 1), [MCMTWB_k_MaxCover.Standard()]),
]

optimizers = [
    ### Qroestl
    # Model.BruteForce(),
    MCMTWB_k_MaxCover.Greedy(),

    ### Qiskit
    # Qiskit.NumpyExact(),
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
    Ocean.HybridCQM(),

    ### Braket
    # Braket.DWave(),

    ### Cplex
    # Cplex.Optimizer(),  # make sure that Cplex is installed, this unfortunately rather complex and only works in Python 3.8 for me (not in

    ### Gurobi
    # Gurobi.Optimizer(),
]
