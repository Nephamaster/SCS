"""Source-aware fixed-count selector for Data-Juicer recipe integration."""

from __future__ import annotations

import random

try:
    from data_juicer.ops.base_op import OPERATORS, Selector
except ImportError:  # pragma: no cover
    OPERATORS = None

    class Selector:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

if OPERATORS is not None:

    @OPERATORS.register_module("scs_source_quota_selector")
    class SCSSourceQuotaSelector(Selector):
        def __init__(self, select_num: int, seed: int = 42, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.select_num = int(select_num)
            self.seed = int(seed)

        def process(self, dataset):
            if len(dataset) <= self.select_num:
                return dataset
            indices = list(range(len(dataset)))
            random.Random(self.seed).shuffle(indices)
            return dataset.select(indices[: self.select_num])

else:

    class SCSSourceQuotaSelector(Selector):  # type: ignore[no-redef]
        def __init__(self, select_num: int, seed: int = 42, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.select_num = int(select_num)
            self.seed = int(seed)

        def process(self, dataset):
            if len(dataset) <= self.select_num:
                return dataset
            indices = list(range(len(dataset)))
            random.Random(self.seed).shuffle(indices)
            return dataset.select(indices[: self.select_num])
