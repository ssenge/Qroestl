from dataclasses import dataclass
from typing import Generic

from braket.ocean_plugin import BraketDWaveSampler
from dwave.system import EmbeddingComposite, DWaveSampler
import dwave.inspector
from qroestl.backends.Ocean import OceanOptimizer, OceanCQMToBQMConverter
from qroestl.model import TCandidate, TProblem, Converter
from run import BackendConfig


@dataclass
class DWave(Generic[TCandidate, TProblem], OceanOptimizer[TCandidate, TProblem]):
    name: str = "Braket-BQM"
    converter: Converter = OceanCQMToBQMConverter()

    def __post_init__(self) -> None:
        self.sampler = BraketDWaveSampler(BackendConfig.BRAKET_S3_BUCKET, BackendConfig.qdev)
        self.sampler = EmbeddingComposite(self.sampler)
