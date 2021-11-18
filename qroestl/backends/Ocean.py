import datetime
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable

import dimod
import greedy
import numpy as np
from dimod import CQM, BQM
from dwave.system import LeapHybridCQMSampler, LeapHybridSampler, DWaveSampler, DWaveCliqueSampler, EmbeddingComposite
from tabu import TabuSampler
import dwave.inspector

from qroestl.model import Optimizer, Converter
from qroestl.utils import Utils

TCandidate = TypeVar('Solution Candidate', bound='Solution')
TProblem = TypeVar('Problem', bound='Problem')


class OceanCQMConvertible(Generic[TCandidate, TProblem]):
    @abstractmethod
    def to_ocean_cqm(self, p: TProblem) -> CQM:
        raise NotImplementedError


class OceanCQMConverter(Converter):
    def convert(self, p, a):
        if isinstance(a, OceanCQMConvertible):
            return a.to_ocean_cqm(p)
        else:
            raise ValueError('Approach not convertible to Ocean CQM')


class OceanBQMConvertible(Generic[TCandidate, TProblem]):
    @abstractmethod
    def to_ocean_bqm(self, p: TProblem) -> BQM:
        raise NotImplementedError


class OceanBQMConverter(Converter):
    def convert(self, p, a):
        if isinstance(a, OceanBQMConvertible):
            return a.to_ocean_bqm(p)
        else:
            raise ValueError('Approach not convertible to Ocean BQM')


class OceanCQMToBQMConvertible(Generic[TCandidate, TProblem], OceanBQMConvertible[TCandidate, TProblem]):
    pass


class OceanCQMToBQMConverter(Converter):
    def convert(self, p, a):
        if isinstance(a, OceanCQMToBQMConvertible):
            self.bqm, self.inverter = a.to_ocean_bqm_from_cqm(p)
            return self.bqm
        else:
            raise ValueError('Approach not convertible to Ocean BQM from CQM')

    def invert(self, s):
        return self.inverter(s)


@dataclass
class OceanOptimizer(Generic[TCandidate, TProblem], Optimizer[TCandidate, TProblem]):
    name: str = 'Qcean'
    samples: int = 1

    def sample(self, p):
        pass
    def time(self, result):
        return 0
    def extract(self, result_dict):
        pass

    def optimize_(self, p, p_conv, a, s):
        #import dwave.inspector  # do not remove this import, required for dwave.inspector.get_embedding, even though it is not used
        #result = self.sampler.sample(p_conv, num_reads=100)
        result = self.sample(p_conv)
        #embedding = result.info['embedding_context']['embedding']

        #Ocean.HybridBQM: result.info['run_time']  # microseconds?
        #Ocean.BQM: result.info['timing']['qpu_access_time'] + result.info['timing']['post_processing_overhead_time']  # microseconds
        #Ocean.BQM-Clique: result.info['timing']['qpu_access_time'] + result.info['timing']['post_processing_overhead_time']  # microseconds
        # Braket.BQM: result.info['additionalMetadata']['dwaveMetadata']['timing']['qpu_access_time'] + result.info['additionalMetadata']['dwaveMetadata']['timing']['post_processing_overhead_time']  # microseconds

        proc_time = self.time(result)
        #proc_time = result.info['timing']['qpu_access_time'] + result.info['timing']['post_processing_overhead_time']  # microseconds
        result = result.first.sample
        #print(f"Number of logical variables: {len(embedding.keys())}")
        #print(f"Number of physical qubits used in embedding: {sum(len(chain) for chain in embedding.values())}")
        #dwave.inspector.show(bqm=p_conv, sampleset=result)

        if isinstance(self.converter, OceanCQMToBQMConverter):
            result = self.converter.invert(result)
        result = a.extract(result)
        #result = sorted({k: v for k, v in result.items() if k.startswith('s')}.values())
        #else TODO
        return s.eval(p, Utils.bits2idx(len(p.S))(np.clip(np.rint(result), 0, 1))), datetime.timedelta(microseconds=proc_time)

@dataclass
class BQM(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMToBQMConverter()

    def sample(self, p):
        return self.sampler.sample(p, num_reads=self.samples)

    def time(self, result):
        return result.info['timing']['qpu_access_time'] + result.info['timing']['post_processing_overhead_time']  # microseconds

    def __post_init__(self) -> None:
        self.name: str = self.name + "-BQM"
        self.sampler = EmbeddingComposite(DWaveSampler())

@dataclass
class BQM_Clique(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMToBQMConverter()
    sample = BQM.sample
    time = BQM.time

    def __post_init__(self) -> None:
        self.name: str = self.name + "-BQM-Clique"
        self.sampler = DWaveCliqueSampler()

@dataclass
class HybridCQM(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMConverter()

    def __post_init__(self) -> None:
        self.name: str = self.name + "-HybridCQM"
        self.sampler = LeapHybridCQMSampler()

    def sample(self, p):
        return self.sampler.sample_cqm(p).filter(lambda d: d.is_feasible)

    def time(self, result):
        return result.info['run_time']

    # def optimize_(self, p, p_conv, a, s):
    #     result = self.sampler.sample_cqm(p_conv, time_limit=5).filter(lambda d: d.is_feasible)
    #     result = result.first.sample  # result.info['run_time']  # microseconds?
    #     return s.eval(p, Utils.bits2idx(len(p.S))(np.clip(np.rint(list(result.values())), 0, 1))), 0


@dataclass
class HybridBQM(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMToBQMConverter()

    def __post_init__(self) -> None:
        self.name: str = self.name + "-HybridBQM"
        self.sampler = LeapHybridSampler()

    time = HybridCQM.time

    def sample(self, p):
        return self.sampler.sample(p)



@dataclass
class Exact(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMToBQMConverter()

    def __post_init__(self) -> None:
        self.name: str = self.name + "-Exact"
        self.sampler = dimod.ExactSolver()

    def sample(self, p):
        return self.sampler.sample(p)


@dataclass
class Greedy(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMToBQMConverter()

    def __post_init__(self) -> None:
        self.name: str = self.name + "-Greedy"
        self.sampler = greedy.SteepestDescentSampler()

    sample = Exact.sample

@dataclass
class Tabu(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    converter: Converter = OceanCQMToBQMConverter()

    def __post_init__(self) -> None:
        self.name: str = self.name + "-Tabu"
        self.sampler = TabuSampler()

    sample = Greedy.sample
