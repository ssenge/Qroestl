from dataclasses import dataclass
from functools import reduce

from pennylane import numpy as np
import pennylane as qml
from pennylane import qaoa
from networkx import Graph
import networkx as nx
from matplotlib import pyplot as plt
from qroestl.utils import Utils





@dataclass
class QAOA:
    pauli_str: str = None
    depth: int = 4
    steps: int = 100
    shots: int = 1000
    init_value = 0.5
    optimizer: 'Optimizer' = qml.GradientDescentOptimizer()
    #dev: 'Device' = None#qml.device("lightning.qubit", wires=wires)

    def __post_init__(self):
        self.coeffs, self.ops = self.parse_pauli_str(self.pauli_str)
        cols = []
        for col in self.ops:
            col_ops = []
            for i, op in enumerate(col):
                col_ops.append(qml.PauliZ(i) if op == 'Z' else qml.Identity(i))
            cols.append(col_ops)
        self.obs = [[reduce(lambda a, b: a @ b, col)] for col in cols]
        self.obs = Utils.flatten2d(self.obs)
        self.cost_h = qml.Hamiltonian(self.coeffs, self.obs)
        self.mixer_h = qaoa.x_mixer(self.wires)
        self.dev = qml.device("default.qubit", wires=self.wires)

    def parse_pauli_str(self, s):
        coeffs = []
        obs = []
        s = s.replace(' ', '')
        for l in s.splitlines():
            if len(l) > 0:
                r = l.split('*')
                coeffs.append(float(r[0]))
                obs.append(r[1])
        if len(obs) == 0:
            self.qubits = 0
        else:
            self.qubits = len(obs[0])
        self.wires = range(self.qubits)
        return coeffs, obs

    def op_to_ob(self, ops):
        cols = []
        for col in ops:
            col_ops = []
            for i, op in enumerate(col):
                col_ops.append(qml.PauliZ(i) if op == 'Z' else qml.Identity(i))
            cols.append(col_ops)
        obs = [[reduce(lambda a, b: a @ b, col)] for col in cols]
        obs = Utils.flatten2d(obs)
        return obs


    def optimize(self):
        def qaoa_layer(gamma, alpha):
            qaoa.cost_layer(gamma, self.cost_h)
            qaoa.mixer_layer(alpha, self.mixer_h)

        def circuit(params, **kwargs):
            for w in self.wires:
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, self.depth, params[0], params[1])


        @qml.qnode(self.dev)
        def cost_function(params):
            circuit(params)
            return qml.expval(self.cost_h)

        @qml.qnode(self.dev)
        def probability_circuit(gamma, alpha):
            circuit([gamma, alpha])
            return qml.probs(wires=self.wires)

        qubit_param = [self.init_value] * self.depth
        params = np.array([qubit_param] * self.qubits, requires_grad=True)
        for i in range(self.steps):
            params = self.optimizer.step(cost_function, params)
        probs = probability_circuit(params[0], params[1])
        return probs


pauli_str1 = '''
-5.25 * ZIII
+ 3.75 * IZII
+ 5.75 * IIZI
- 6.25 * ZIZI
+ 5.75 * IIIZ
- 6.25 * IZIZ
+ 40.5 * IIZZ
    '''

pauli_str2 = '''
3.75 * ZIII
- 5.25 * IZII
+ 5.75 * IIZI
- 6.25 * ZIZI
+ 5.75 * IIIZ
- 6.25 * IZIZ
+ 40.5 * IIZZ
    '''

a = QAOA(pauli_str1)
probs = a.optimize()

plt.style.use("seaborn")
plt.bar(range(2 ** len(a.wires)), probs)
plt.show()

m = np.argmax(probs)
print(m, bin(m))
l = [int(x) for x in bin(m)[2:]]
print(l[:2])