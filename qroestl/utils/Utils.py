
import warnings

from qiskit_optimization import QuadraticProgram

warnings.filterwarnings('ignore', category=DeprecationWarning)

from functools import reduce
from itertools import chain, combinations
import numpy as np
import qiskit

id = lambda x: x
where = lambda s, elem: np.nonzero(np.vectorize(lambda e: e == elem)(s))[0][0]
powerset = lambda s: map(list, chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))
union = lambda s, subset_idx = None: reduce(np.union1d, np.take(s, subset_idx if subset_idx else range(len(s)), axis=0)) if len(s) > 0 else []
union_non_sorted = lambda s: np.array(reduce(lambda a, b: a+b, s)) if len(s) > 0 else []
bits2idx = lambda n: lambda bits: [i for i, x in enumerate(bits[:n]) if x == 1]


def convert_qubo_to_legacy(qubo: QuadraticProgram) -> QuadraticProgram:
    from qiskit.optimization.converters import QuadraticProgramToQubo as ToQubo

    qp = qiskit.optimization.QuadraticProgram()
    qp.from_docplex(qubo.to_docplex())
    return ToQubo().convert(qp)






