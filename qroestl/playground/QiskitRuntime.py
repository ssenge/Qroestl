import numpy as np
from qiskit import IBMQ
from qiskit.circuit.library import EfficientSU2

from qroestl.problems import MCMTWB_k_MaxCover
from qiskit_optimization.converters import QuadraticProgramToQubo
provider = IBMQ.load_account()
# Get a list of all backends that support runtime.
#runtime_backends = provider.backends(input_allowed='runtime')

#print(runtime_backends)

#program = provider.runtime.program('circuit-runner')
#print(program)

backend = provider.get_backend('simulator_mps') #ibmq_montreal


intermediate_info = {
    'nfev': [],
    'parameters': [],
    'energy': [],
    'stddev': []
}

def callback(nfev, parameters, energy, stddev):
    intermediate_info['nfev'].append(nfev)
    intermediate_info['parameters'].append(parameters)
    intermediate_info['energy'].append(energy)
    intermediate_info['stddev'].append(stddev)


measurement_error_mitigation = False

from qiskit_nature.runtime import VQEProgram

# in this first release, the optimizer must be specified as dictionary
optimizer = {'name': 'SPSA',
             'maxiter': 100}

from qiskit.opflow import Z, I


#p = MCMTWB_k_MaxCover.gen_syn_fully_connected(2, 2)
p = MCMTWB_k_MaxCover.gen_syn_one_to_one(2, 2)
a = MCMTWB_k_MaxCover.Standard()
p = a.to_qiskit_qp(p)
H, offset = QuadraticProgramToQubo().convert(p).to_ising()

num_qubits = H.num_qubits
print('#qubits: ', num_qubits)
hamiltonian = H#(Z ^ Z) ^ (I ^ (num_qubits - 2))
target_energy = -1
ansatz = EfficientSU2(num_qubits, reps=1, entanglement='linear', insert_barriers=True)

runtime_vqe = VQEProgram(ansatz=ansatz,
                         optimizer=optimizer,
                         #initial_point=initial_point,
                         provider=provider,
                         backend=backend,
                         shots=1024,
                         measurement_error_mitigation=measurement_error_mitigation,
                         callback=callback)

runtime_result = runtime_vqe.compute_minimum_eigenvalue(hamiltonian)
print('Eigenvalue:', runtime_result.eigenvalue)
print('Eigenvalue+offset:', runtime_result.eigenvalue+offset)
print('Target:', target_energy)
print('Mean error:', np.mean(intermediate_info['stddev']))
print(runtime_result.eigenstate)

max_value = max(runtime_result.eigenstate.values())
max_key = max(runtime_result.eigenstate, key=runtime_result.eigenstate.get)

print('Max value:', max_value)
print('Max key: ', max_key)