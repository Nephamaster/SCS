"""Source quota calculation for the frozen 300K candidate pool."""

from __future__ import annotations

import math
from typing import Mapping


def calculate_source_quotas(
    source_counts: Mapping[str, int], target_count: int = 300_000
) -> dict[str, int]:
    counts = {source: int(count) for source, count in source_counts.items() if count > 0}
    if not counts:
        raise ValueError("cannot calculate quotas for an empty source set")
    if sum(counts.values()) < target_count:
        raise ValueError(
            f"deduplicated data has {sum(counts.values())} samples, below target {target_count}"
        )

    base = {source: min(count, 500) for source, count in counts.items()}
    base_total = sum(base.values())
    if base_total > target_count:
        raise ValueError("base quota exceeds target count")

    quotas = dict(base)
    remaining = target_count - base_total
    capacities = {source: counts[source] - quotas[source] for source in counts}
    weights = {source: math.sqrt(capacity) for source, capacity in capacities.items()}

    while remaining > 0:
        active = [source for source, capacity in capacities.items() if capacity > 0]
        if not active:
            raise ValueError("quota allocation exhausted all source capacities")
        weight_sum = sum(weights[source] for source in active)
        proportional = {
            source: remaining * weights[source] / weight_sum for source in active
        }
        allocations = {
            source: min(math.floor(proportional[source]), capacities[source])
            for source in active
        }
        allocated = sum(allocations.values())
        capped = any(
            math.floor(proportional[source]) > capacities[source] for source in active
        )

        for source, amount in allocations.items():
            quotas[source] += amount
            capacities[source] -= amount
        remaining -= allocated

        if remaining == 0:
            break
        if not capped:
            # Only the largest fractional remainders remain. Allocate those
            # one at a time, with source name as the deterministic tie-breaker.
            ranked = sorted(
                active,
                key=lambda source: (-(
                    proportional[source] - math.floor(proportional[source])
                ), source),
            )
            for source in ranked:
                if remaining == 0:
                    break
                if capacities[source] > 0:
                    quotas[source] += 1
                    capacities[source] -= 1
                    remaining -= 1

    if sum(quotas.values()) != target_count:
        raise AssertionError("quota sum does not equal target count")
    if any(quotas[source] <= 0 or quotas[source] > counts[source] for source in quotas):
        raise AssertionError("quota violates source bounds")
    return quotas
