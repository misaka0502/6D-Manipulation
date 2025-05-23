import numpy as np

from furniture_bench.config import config
from furniture_bench.furniture.scans.parts.scanned_part import ScannedPart


class FactoryPeg(ScannedPart):
    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)