"""An exhaustive reference solver, used only by the tests.

It tries every possible way of splitting each ticker's required sale across that
ticker's lots and keeps the cheapest. The cost grows exponentially, so it is
useless in production - but it shares no logic with the linear program, which is
the point. When the two agree on hundreds of random portfolios, that is real
evidence rather than the same idea agreeing with itself.
"""

from __future__ import annotations

from itertools import product
from typing import Iterator, Sequence

from app.models import PricedLot, SellRequirement, TaxConfig
from app.optimizer import Allocation, lots_by_ticker
from app.tax import aggregates, tax_on


def splits(caps: list[int], target: int) -> Iterator[tuple[int, ...]]:
    """Every way to draw `target` shares from lots capped at `caps`."""
    if not caps:
        if target == 0:
            yield ()
        return
    head, *rest = caps
    for k in range(min(head, target) + 1):
        for tail in splits(rest, target - k):
            yield (k, *tail)


def solve_bruteforce(
    lots: Sequence[PricedLot], reqs: Sequence[SellRequirement], cfg: TaxConfig
) -> Allocation:
    index = lots_by_ticker(lots)

    per_ticker = [
        [(index[r.ticker], s) for s in splits(
            [lots[i].lot.quantity for i in index[r.ticker]], r.shares
        )]
        for r in reqs
    ]

    best: tuple[float, tuple[int, ...]] | None = None
    for combo in product(*per_ticker):
        shares = [0] * len(lots)
        for members, split in combo:
            for i, k in zip(members, split):
                shares[i] = k
        t = tax_on(*aggregates(lots, shares), cfg)
        if best is None or t < best[0] - 1e-9:
            best = (t, tuple(shares))

    assert best is not None
    return Allocation(best[1], "optimal", True)
