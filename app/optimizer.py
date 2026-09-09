"""Choosing which lots to sell.

Tax is the maximum of four linear formulas in the two realised-gain totals (see
tax.tax_pieces) and each lot adds gain_per_share * shares_sold to one of them,
so minimising tax subject to a fixed sell quantity per ticker is a linear
program with whole-number variables:

    minimise    t
    subject to  t >= a_k*A(x) + b_k*B(x) + c_k    for each of the four formulas
                sum of x over ticker T's lots = shares required from T
                0 <= x_i <= lot_i.quantity,  x_i a whole number

`t` stands in for the tax: you cannot write max(...) in a linear objective, so
force one extra variable above all four formulas and minimise it. HiGHS returns
a proved global optimum. Whole numbers are not a detail - the best answer often
lands part-way through a lot, where the exemption runs out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .models import PricedLot, SellRequirement, TaxConfig
from .tax import aggregates, tax_on, tax_pieces

Method = Literal["optimal", "fifo", "ltfo"]

# Big enough to absorb the solver's rounding, too small to buy a worse plan.
TIE_BREAK_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Allocation:
    """shares[i] is how many shares are sold out of lots[i]."""

    shares: tuple[int, ...]
    method: Method
    certified_optimal: bool

    def tax(self, lots: Sequence[PricedLot], cfg: TaxConfig) -> float:
        return tax_on(*aggregates(lots, self.shares), cfg)


def lots_by_ticker(lots: Sequence[PricedLot]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for i, lot in enumerate(lots):
        index.setdefault(lot.lot.ticker, []).append(i)
    return index


def _validate(lots: Sequence[PricedLot], reqs: Sequence[SellRequirement]) -> None:
    index = lots_by_ticker(lots)
    seen: set[str] = set()
    for r in reqs:
        if r.ticker in seen:
            raise ValueError(f"{r.ticker}: duplicate sell requirement")
        seen.add(r.ticker)
        held = sum(lots[i].lot.quantity for i in index.get(r.ticker, []))
        if r.shares > held:
            raise ValueError(
                f"{r.ticker}: need to sell {r.shares} shares but only {held} are held"
            )


def solve_optimal(
    lots: Sequence[PricedLot], reqs: Sequence[SellRequirement], cfg: TaxConfig
) -> Allocation:
    """The cheapest legal allocation, in two passes: minimise tax, then re-solve
    with the tax pinned there and break ties by leaving unrealised gains alone.

    A second pass rather than a weighted objective, because a tie-break weight
    big enough to matter on a portfolio worth lakhs could also override a real
    difference of a few rupees.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    _validate(lots, reqs)
    n = len(lots)
    index = lots_by_ticker(lots)

    # Decision variables: one per lot, plus t at the end.
    rows: list = []
    low: list[float] = []
    high: list[float] = []

    for r in reqs:  # sell exactly the required number of shares per ticker
        row = np.zeros(n + 1)
        for i in index.get(r.ticker, []):
            row[i] = 1.0
        rows.append(row)
        low.append(float(r.shares))
        high.append(float(r.shares))

    for a, b, c in tax_pieces(cfg):  # t >= a*A + b*B + c, rearranged
        row = np.zeros(n + 1)
        for i, lot in enumerate(lots):
            row[i] = (a if lot.bucket == "ST" else b) * lot.gain_per_share
        row[n] = -1.0
        rows.append(row)
        low.append(-np.inf)
        high.append(-c)

    matrix = np.array(rows)
    bounds = Bounds(
        lb=np.zeros(n + 1),
        ub=np.array([float(lot.lot.quantity) for lot in lots] + [np.inf]),
    )
    integrality = np.array([1] * n + [0])

    def run(objective, cap: float | None = None):
        m, lo, hi = matrix, list(low), list(high)
        if cap is not None:
            pin = np.zeros(n + 1)
            pin[n] = 1.0
            m = np.vstack([matrix, pin])
            lo, hi = lo + [-np.inf], hi + [cap]
        result = milp(
            c=objective,
            constraints=LinearConstraint(m, lo, hi),
            bounds=bounds,
            integrality=integrality,
        )
        if result.status != 0:
            raise RuntimeError(f"no solution found: {result.message}")
        return result

    minimise_tax = np.zeros(n + 1)
    minimise_tax[n] = 1.0
    first = run(minimise_tax)
    best_tax = float(first.fun)

    prefer_keeping_gains = np.array(
        [max(0.0, lot.gain_per_share) for lot in lots] + [0.0]
    )
    if prefer_keeping_gains.any():
        cap = best_tax + TIE_BREAK_TOLERANCE * max(1.0, abs(best_tax))
        chosen = run(prefer_keeping_gains, cap=cap).x
    else:
        chosen = first.x

    return Allocation(tuple(int(round(v)) for v in chosen[:n]), "optimal", True)


def solve_fifo(
    lots: Sequence[PricedLot], reqs: Sequence[SellRequirement], cfg: TaxConfig
) -> Allocation:
    """Oldest lot first: what a demat account does by default, so the gap
    against it is what picking lots is worth."""
    _validate(lots, reqs)
    index = lots_by_ticker(lots)
    shares = [0] * len(lots)
    for r in reqs:
        remaining = r.shares
        for i in sorted(index.get(r.ticker, []), key=lambda i: lots[i].lot.buy_date):
            if remaining <= 0:
                break
            take = min(remaining, lots[i].lot.quantity)
            shares[i] = take
            remaining -= take
    return Allocation(tuple(shares), "fifo", False)


def solve_ltfo(
    lots: Sequence[PricedLot], reqs: Sequence[SellRequirement], cfg: TaxConfig
) -> Allocation:
    """Least tax first out: per ticker, fill from the lot with the smallest
    gain_per_share * statutory rate. Ties break on the older lot.

    It prices each lot in isolation, so it cannot see that the exemption is one
    pool shared across the portfolio. No optimality guarantee.
    """
    _validate(lots, reqs)
    index = lots_by_ticker(lots)
    shares = [0] * len(lots)

    def tax_per_share(i: int) -> float:
        lot = lots[i]
        rate = cfg.stcg_rate if lot.bucket == "ST" else cfg.ltcg_rate
        return lot.gain_per_share * rate

    for r in reqs:
        remaining = r.shares
        order = sorted(
            index.get(r.ticker, []), key=lambda i: (tax_per_share(i), lots[i].lot.buy_date)
        )
        for i in order:
            if remaining <= 0:
                break
            take = min(remaining, lots[i].lot.quantity)
            shares[i] = take
            remaining -= take
    return Allocation(tuple(shares), "ltfo", False)


SOLVERS = {"optimal": solve_optimal, "fifo": solve_fifo, "ltfo": solve_ltfo}


def solve(
    lots: Sequence[PricedLot],
    reqs: Sequence[SellRequirement],
    cfg: TaxConfig,
    method: Method = "optimal",
) -> Allocation:
    return SOLVERS[method](lots, reqs, cfg)
