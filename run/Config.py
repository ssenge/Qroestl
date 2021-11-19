from qroestl.backends import Gurobi, Qiskit, Ocean, Cplex
from qroestl.model import Model
from qroestl.problems import MCMTWB_k_MaxCover


class Config:
    # IonQ_token = 'put-your-token-here'
    # IBMQ: see utils.SaveIBMQToken.py
    # DWave: use the Dwave CLI tool to set token
    # Braket: use the AWS CLI tool to set token

    BRAKET_S3_BUCKET = ('amazon-braket-765de053a863', 'test')

    qdev = (
        # QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        # IBMQ.load_account().get_backend('ibmq_manila')  # make sure you have an IBMQ account and the APItoken is locally saved
        # IonQProvider(token=IonQ_token).get_backend("ionq_simulator")  # make sure you have an IonQ API token
        # IonQProvider(token=IonQ_token).get_backend("ionq_qpu")  # make sure you have an IonQ API token
        'arn:aws:braket:::device/qpu/d-wave/Advantage_system4'
    )

    tasks = [
        (MCMTWB_k_MaxCover.gen_syn_fully_connected(1, 1), [MCMTWB_k_MaxCover.Standard()]),
    ]

    optimizers = [
        ### Qroestl
        Model.BruteForce(),
        MCMTWB_k_MaxCover.Greedy(),

        ### Qiskit
        Qiskit.NumpyExact(),
        # Qiskit.DWaveAnnealer(),  # make sure that DWave tools / configuration is in place
        # Qiskit.VQE(quantum_instance=qdev, kwargs={'optimizer': COBYLA(1)}),
        # Qiskit.QAOA(kwargs={'reps': 2, 'optimizer': COBYLA(2)}),

        ### DWave
        Ocean.Exact(),
        Ocean.Greedy(),
        Ocean.Tabu(),
        # Ocean.BQM(),
        # Ocean.BQM_Clique(),
        # Ocean.HybridBQM(),
        # Ocean.HybridCQM(),

        ### Braket
        # Braket.DWave(),

        ### Cplex
        # Cplex.Optimizer(),  # make sure that Cplex is installed, this unfortunately rather complex and only works in Python 3.8 for me (not in

        ### Gurobi
        Gurobi.Optimizer(),
    ]
