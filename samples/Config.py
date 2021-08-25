from qiskit import BasicAer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance

from qroestl.model import Model
from qroestl.backends import Qiskit
from qroestl.problems import SetCoverage, MultiCover


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
        # SetCover.Problem([1, 2], [[1], [2]])

        # ExactCover.Problem([1, 2, 3, 4, 5, 6, 7], [[7], [1, 2, 3], [4, 5], [6]])

        # SetCoverage.Problem(k=2, S=[[7], [1, 2], [3, 4]])
        # SetCoverage.Problem(k=2, S=[[7, 8, 2], [8, 7], [1, 2]], W=[1, 10, 1, 1])  # [1,2] -> 13 (tie)
        # SetCoverage.Problem(k=2, S=[[7], [8, 7], [1, 2, 8]], W=[40, 1, 1, 1])  # [0, 2] -> 43

        #MultiCover.Problem(k=2, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 5], C=[0, 2, 0, 0], T=[1, 1, 2])  # [1, 2] -> 8
        #MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 1], C=[0, 2, 0, 0], T=[2, 1, 1])  # NF
        # MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8]], W=[1, 1, 1, 10], C=[2, 2, 0, 0], T=[1, 2, 3])  # [0, 1, 2] -> 13
        #MultiCover.Problem(k=3, S=[[7], [8, 7], [1, 2, 8], [10, 11, 12, 13]],
        #                   W=[1, 1, 1, 10, 1, 1, 1, 1], C=[1, 2, 0, 0, 0, 0, 0, 0], T=[1, 2, 3, 4])  # [1, 2, 3] -> 17
        #MultiCover.Problem(k=2, S=[[1], [2, 1], [3, 4]], W=[1, 1, 1, 100], C=[2, 0, 0, 0], T=[1, 2, 1])  # [0, 1] -> 2
        #MultiCover.Problem(k=4, S=[[7], [8, 7], [1, 2, 8], [10, 11, 12, 13], [14, 15]],
        #                    W=[1, 1, 1, 10, 1, 1, 1, 1, 100, 5], C=[1, 2, 0, 0, 0, 0, 0, 0, 0, 0], T=[1, 2, 3, 4, 5]) # [1, 2, 3, 4] -> 122
        #MultiCover.Problem(k=4, S=[[7], [8, 7], [1, 2, 8], [10, 11, 12, 13], [14, 15]],
        #                    W=[1, 1, 1, 10, 1, 1, 1, 1, 100, 5], C=[1, 2, 0, 0, 0, 0, 0, 0, 0, 0], T=[1, 1, 1, 4, 5]) # NF
    )

    # Problem-specific algorithms
    sas = [
        # SetCover.Greedy()
        # ExactCover.Greedy()
        # SetCoverage.Greedy()
        MultiCover.Greedy()
    ]

    # Generic algorithms
    gas = [
        Model.BruteForce(),
        # Qiskit.NumpyExact(),
        Qiskit.CPLEX(),  # make sure that CPLEX is installed
        # Qiskit.DWaveAnnealer(),  # make sure that DWave tools / configuration is in place
        # Qiskit.VQE(quantum_instance=qdev, kwargs={'optimizer': COBYLA(1)}),
        # Qiskit.QAOA(quantum_instance=qdev, kwargs={'reps': 2, 'optimizer': COBYLA(1)})
    ]

    algos = sas + gas


