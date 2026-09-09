"""Holding periods and the tax due on a set of realised gains.

Loss set-off plus the long-term exemption reduce to the maximum of four straight
lines in two numbers - net short-term gain A and net long-term gain B, with
s = 20%, l = 12.5%, E = Rs 1,25,000:

    tax(A, B) = max( 0,  s*A,  l*(A+B-E),  s*A + l*(B-E) )

That linearity is what lets optimizer.py treat lot selection as a linear program
rather than a search. test_tax.py checks the identity against `breakdown`.
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Sequence

from .models import Bucket, Lot, PricedLot, TaxBreakdown, TaxConfig


def add_months(d: date, months: int) -> date:
    """Add calendar months, clamping to month end: 31-Jan + 1 month is 28-Feb."""
    year, month = divmod(d.month - 1 + months, 12)
    year, month = d.year + year, month + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def classify(buy_date: date, sale_date: date, cfg: TaxConfig) -> Bucket:
    """Long-term only if held MORE than 12 months, so the anniversary itself is
    still short-term. Calendar months, not 365 days."""
    return "LT" if sale_date > add_months(buy_date, cfg.long_term_months) else "ST"


def months_held(buy_date: date, sale_date: date) -> int:
    m = (sale_date.year - buy_date.year) * 12 + (sale_date.month - buy_date.month)
    if sale_date < add_months(buy_date, m):
        m -= 1
    return max(0, m)


def holding_period(buy_date: date, sale_date: date) -> str:
    """'14 months', or a day count for anything under a month."""
    m = months_held(buy_date, sale_date)
    if m >= 1:
        return f"{m} month{'s' if m != 1 else ''}"
    return f"{(sale_date - buy_date).days} days"


def holding_label(buy_date: date, sale_date: date) -> str:
    return f"{holding_period(buy_date, sale_date)} held"


def price_lot(lot: Lot, price: float, sale_date: date, cfg: TaxConfig) -> PricedLot:
    return PricedLot(lot=lot, bucket=classify(lot.buy_date, sale_date, cfg), price=price)


def tax_pieces(cfg: TaxConfig) -> list[tuple[float, float, float]]:
    """The four (a, b, c) triples, where tax = max of a*A + b*B + c."""
    s, l, e = cfg.stcg_rate, cfg.ltcg_rate, cfg.ltcg_exemption
    return [
        (0.0, 0.0, 0.0),  # nothing is taxable
        (s, 0.0, 0.0),  # short-term taxable, long-term under the exemption
        (l, l, -l * e),  # leftover short-term loss spills into the long-term side
        (s, l, -l * e),  # both sides taxable
    ]


def tax_on(net_st: float, net_lt: float, cfg: TaxConfig) -> float:
    """Tax due on a net short-term and a net long-term realised gain."""
    return max(a * net_st + b * net_lt + c for a, b, c in tax_pieces(cfg))


def effective_rate(net_st: float, net_lt: float, bucket: Bucket, cfg: TaxConfig) -> float:
    """What one more rupee of gain on this side costs in this plan. Not the
    statutory rate: it depends on how much exemption and loss are left."""
    before = tax_on(net_st, net_lt, cfg)
    if bucket == "ST":
        return tax_on(net_st + 1.0, net_lt, cfg) - before
    return tax_on(net_st, net_lt + 1.0, cfg) - before


def aggregates(lots: Sequence[PricedLot], shares: Sequence[int]) -> tuple[float, float]:
    """(net short-term, net long-term) realised gain for a given allocation."""
    st = sum(l.gain_per_share * n for l, n in zip(lots, shares) if l.bucket == "ST")
    lt = sum(l.gain_per_share * n for l, n in zip(lots, shares) if l.bucket == "LT")
    return st, lt


def breakdown(
    lots: Sequence[PricedLot], shares: Sequence[int], cfg: TaxConfig
) -> TaxBreakdown:
    """The set-off calculation step by step. Same total as `tax_on`, but keeps
    every intermediate figure so the report can show its working."""
    b = TaxBreakdown()
    for lot, n in zip(lots, shares):
        if n <= 0:
            continue
        gain = lot.gain_per_share * n
        if lot.bucket == "ST":
            b.stcg += gain if gain >= 0 else 0.0
            b.stcl += -gain if gain < 0 else 0.0
        else:
            b.ltcg += gain if gain >= 0 else 0.0
            b.ltcl += -gain if gain < 0 else 0.0

    # 1. Long-term loss against long-term gain. It has no other use.
    b.ltcl_used_against_ltcg = min(b.ltcl, b.ltcg)
    long_term_left = b.ltcg - b.ltcl_used_against_ltcg

    # 2. Short-term loss against short-term gain (20%) before long-term (12.5%).
    b.stcl_used_against_stcg = min(b.stcl, b.stcg)
    b.taxable_stcg = b.stcg - b.stcl_used_against_stcg
    leftover_stcl = b.stcl - b.stcl_used_against_stcg
    b.stcl_used_against_ltcg = min(leftover_stcl, long_term_left)
    long_term_left -= b.stcl_used_against_ltcg

    # 3. The exemption, applied to what long-term gain remains.
    b.exemption_used = min(cfg.ltcg_exemption, long_term_left)
    b.taxable_ltcg = long_term_left - b.exemption_used

    b.total_tax = cfg.stcg_rate * b.taxable_stcg + cfg.ltcg_rate * b.taxable_ltcg
    return b
