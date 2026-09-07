"""Data types and the tax rates. No logic beyond one-line derived values."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

# "ST" = short-term, "LT" = long-term.
Bucket = Literal["ST", "LT"]


@dataclass(frozen=True)
class TaxConfig:
    """Capital gains rates for listed Indian equity on which STT is paid.

    Set by the Finance (No. 2) Act 2024 for transfers on or after 23 July 2024
    and unchanged since: short-term 20% (s.111A), long-term 12.5% on the amount
    above Rs 1,25,000 a year (s.112A). Source: Income Tax Department,
    https://incometaxindia.gov.in/

    A dataclass rather than module constants so a test can vary a rate.
    """

    stcg_rate: float = 0.20
    ltcg_rate: float = 0.125
    ltcg_exemption: float = 125_000.0
    long_term_months: int = 12


@dataclass(frozen=True)
class Lot:
    """One purchase tranche. Frozen on purpose: a partial sale must leave the
    original lot and its buy date intact, so it produces a remainder instead."""

    lot_id: str
    ticker: str
    buy_date: date
    quantity: int
    buy_price: float

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"{self.lot_id}: quantity must be positive")
        if self.buy_price <= 0:
            raise ValueError(f"{self.lot_id}: buy price must be positive")


@dataclass(frozen=True)
class PricedLot:
    """A lot valued at today's price: what the optimiser actually works with."""

    lot: Lot
    bucket: Bucket
    price: float

    @property
    def gain_per_share(self) -> float:
        return self.price - self.lot.buy_price

    @property
    def classification(self) -> str:
        """STCG, STCL, LTCG or LTCL."""
        return f"{self.bucket}C{'L' if self.gain_per_share < 0 else 'G'}"


@dataclass(frozen=True)
class SellRequirement:
    """Shares of one ticker that the rebalance requires be sold."""

    ticker: str
    shares: int


@dataclass(frozen=True)
class Alternative:
    """A lot the engine could have sold instead, priced by re-running the tax
    calculation on the swap."""

    lot_id: str
    buy_date: date
    classification: str
    gain_per_share: float
    shares: int
    tax_delta: float
    note: str


@dataclass(frozen=True)
class LotSale:
    """One sell recommendation, with the reasoning behind it."""

    lot_id: str
    ticker: str
    buy_date: date
    holding: str
    classification: str
    shares_sold: int
    lot_quantity: int
    remaining_shares: int
    price: float
    cost_basis_per_share: float
    gain_per_share: float
    realized_gain: float
    marginal_tax_rate: float  # what the next rupee costs; see tax.effective_rate
    reason: str
    alternatives: list[Alternative] = field(default_factory=list)


@dataclass
class TaxBreakdown:
    """Every step of the set-off calculation, for the report."""

    stcg: float = 0.0
    stcl: float = 0.0
    ltcg: float = 0.0
    ltcl: float = 0.0
    stcl_used_against_stcg: float = 0.0
    stcl_used_against_ltcg: float = 0.0
    ltcl_used_against_ltcg: float = 0.0
    exemption_used: float = 0.0
    taxable_stcg: float = 0.0
    taxable_ltcg: float = 0.0
    total_tax: float = 0.0


@dataclass(frozen=True)
class Trade:
    """One executable instruction."""

    ticker: str
    action: Literal["SELL", "BUY"]
    shares: int
    price: float
    value: float


@dataclass(frozen=True)
class WeightRow:
    ticker: str
    target_pct: float
    before_pct: float
    after_pct: float


@dataclass
class Summary:
    total_tax: float = 0.0
    tax_if_fifo: float | None = None
    saving_vs_fifo: float | None = None
    shares_sold: int = 0
    shares_bought: int = 0
    gross_proceeds: float = 0.0
    cash_left_over: float = 0.0


@dataclass
class Plan:
    """The full rebalancing plan. This is also the API response body."""

    method: str
    certified_optimal: bool
    summary: Summary = field(default_factory=Summary)
    reasoning: list[str] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    lot_sales: list[LotSale] = field(default_factory=list)
    tax: TaxBreakdown = field(default_factory=TaxBreakdown)
    weights: list[WeightRow] = field(default_factory=list)
