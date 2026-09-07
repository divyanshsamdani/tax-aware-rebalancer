"""Turning target weights into share counts. No tax logic here.

How many shares of a ticker must be sold depends only on its price and its
target weight, never on what was paid for it, so the quantities can be fixed
first and "which lots supply them" becomes a self-contained problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .models import Lot, SellRequirement

# Target weights must add up to 100% within this much (0.01%).
WEIGHT_TOLERANCE = 1e-4


def round_half_up(x: float) -> int:
    """Nearest whole share, halves away from zero. Built-in round() is banker's
    rounding, which sends 0.5 to 0 - not what "nearest share" means."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


@dataclass(frozen=True)
class Position:
    ticker: str
    quantity: int
    price: float

    @property
    def value(self) -> float:
        return self.quantity * self.price


def positions(lots: Sequence[Lot], prices: Mapping[str, float]) -> dict[str, Position]:
    """Aggregate lots into one position per ticker."""
    totals: dict[str, int] = {}
    for lot in lots:
        totals[lot.ticker] = totals.get(lot.ticker, 0) + lot.quantity
    missing = sorted(set(totals) - set(prices))
    if missing:
        raise ValueError(f"no current price supplied for: {', '.join(missing)}")
    return {t: Position(t, q, prices[t]) for t, q in totals.items()}


def weights(pos: Mapping[str, Position], cash: float = 0.0) -> dict[str, float]:
    total = sum(p.value for p in pos.values()) + cash
    return {t: p.value / total for t, p in pos.items()} if total else {}


def validate_targets(targets: Mapping[str, float]) -> None:
    total = sum(targets.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(
            f"target weights add up to {total * 100:.4f}%, expected 100%"
        )
    for ticker, w in targets.items():
        if w < 0:
            raise ValueError(f"{ticker}: target weight must not be negative")


def sell_requirements(
    pos: Mapping[str, Position], targets: Mapping[str, float], total_value: float
) -> list[SellRequirement]:
    """How many shares of each overweight ticker have to go. A held ticker absent
    from the targets has a 0% target and is sold off entirely."""
    out: list[SellRequirement] = []
    for ticker, p in pos.items():
        if p.price <= 0:
            raise ValueError(f"{ticker}: current price must be positive")
        target_value = targets.get(ticker, 0.0) * total_value
        shares = round_half_up((p.value - target_value) / p.price)
        shares = max(0, min(p.quantity, shares))
        if shares > 0:
            out.append(SellRequirement(ticker, shares))
    return out


def buy_plan(
    pos: Mapping[str, Position],
    targets: Mapping[str, float],
    sold: Mapping[str, int],
    total_value: float,
    proceeds: float,
) -> dict[str, int]:
    """Whole-share buys for underweight tickers, paid out of the proceeds.

    Sold tickers are skipped - the sale already put them at target. Largest
    shortfall first, capped by cash on hand, so the plan pays for itself.
    """
    shortfalls = {
        ticker: targets.get(ticker, 0.0) * total_value - p.value
        for ticker, p in pos.items()
        if ticker not in sold
    }
    budget = proceeds
    buys: dict[str, int] = {}
    for ticker in sorted(shortfalls, key=lambda t: -shortfalls[t]):
        price = pos[ticker].price
        n = min(int(shortfalls[ticker] // price), int(budget // price))
        if n > 0:
            buys[ticker] = n
            budget -= n * price
    return buys
