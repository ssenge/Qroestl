from qiskit import BasicAer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit_ionq import IonQProvider
from qiskit import IBMQ

from model import Model
from backends import Qiskit
from problems import ExactCover, SetCover


class Config:
    IonQ_token = 'put-your-token-here'
    # IBMQ token: see utils.SaveIBMQToken.py

    qdev = (
        QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        # IBMQ.load_account().get_backend('ibmq_manila')  # make sure you have an IBMQ account and the api token is locally saved
        # IonQProvider(token=IonQ_token).get_backend("ionq_simulator")  # make sure you have an IonQ api token
        # IonQProvider(token=IonQ_token).get_backend("ionq_qpu")  # make sure you have an IonQ api token
    )

    p = (
        # SetCover.Generator.linear(20, 5)
        # SetCover.Problem([1, 2, 3, 4, 5, 6, 7], [[7, 1], [1, 2, 3], [4, 5], [6]])
        SetCover.Problem([1, 2], [[1], [2]])

        # ExactCover.Problem([1, 2, 3, 4, 5, 6, 7], [[7], [1, 2, 3], [4, 5], [6]])

        # MaxSetCoverageProblem(k=2, S=[[7], [1, 2], [3]])  # VQE: 152sec
        # SetCoverage.Problem(k=2, S=[[7], [1, 2]])
    )

    # Problem-specific algorithms
    sas = [
        SetCover.Greedy()
        # ExactCover.Greedy()
        # MaxSetCoverage.Greedy()
    ]

    # Generic algorithms
    gas = [
        Model.BruteForce(),
        Qiskit.NumpyExact(),
        # Qiskit.CPLEX(),  # make sure that CPLEX is installed
        # Qiskit.DWaveAnnealer(),  # make sure that DWave tools / configuration is in place
        Qiskit.VQE(quantum_instance=qdev, kwargs={'optimizer': COBYLA(1)}),
        # Qiskit.QAOA(quantum_instance=qdev, kwargs={'reps': 2, 'optimizer': COBYLA(1)})
    ]

    algos = sas + gas


