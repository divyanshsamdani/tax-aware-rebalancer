"""The sentence attached to each lot in a plan.

Two rules work through a ticker one lot at a time, so their reasoning names the
place in that sequence and where the remainder spills. The third settles every
quantity at once and has no sequence to name, so it states the split and flags
the quantities that sit on a boundary.

Nothing here recomputes tax. The figures a reader needs are already structured
fields on the record, so the prose carries only what those cannot say.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from .models import LotSale, PricedLot, TaxBreakdown, TaxConfig
from .tax import aggregates, effective_rate, holding_label

def statutory_per_share(lot: PricedLot, cfg: TaxConfig) -> float:
    """Gain per share at the lot's own statutory rate: what LTFO ranks on."""
    return lot.gain_per_share * (cfg.stcg_rate if lot.bucket == "ST" else cfg.ltcg_rate)


def pick_order(
    lots: Sequence[PricedLot], shares: Sequence[int], method: str, cfg: TaxConfig
) -> list[int]:
    """The sold lots, in the order the method actually reached for them.

    A ranking rule works through a ticker's lots in a fixed sequence, so listing
    them that way is part of showing the rule. `optimal` has no such sequence -
    it settles every quantity at once - so its lots stay in portfolio order.
    """
    chosen = [i for i, n in enumerate(shares) if n > 0]
    if method == "fifo":
        return sorted(chosen, key=lambda i: (lots[i].lot.ticker, lots[i].lot.buy_date))
    if method == "ltfo":
        return sorted(
            chosen,
            key=lambda i: (
                lots[i].lot.ticker,
                statutory_per_share(lots[i], cfg),
                lots[i].lot.buy_date,
            ),
        )
    return chosen


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _rule_clause(lot: PricedLot, method: str, cfg: TaxConfig, pick: int) -> str:
    """Why this method reached for this lot, at this point in its sequence.

    `optimal` has no per-lot rule to quote: it settles every quantity jointly
    across the portfolio, so what stands in its place is the split itself.
    """
    ticker = lot.lot.ticker
    if method == "fifo":
        return f"{_ordinal(pick)} pick for {ticker}: the oldest lot still available."
    if method == "ltfo":
        return (
            f"{_ordinal(pick)} pick for {ticker}: the lowest statutory tax per share "
            f"of the lots still available, at Rs {statutory_per_share(lot, cfg):,.2f}."
        )
    return ""


def boundary_note(cfg: TaxConfig) -> str:
    """The reasons a quantity can be a boundary, for a footnote or a tooltip.

    Written as alternatives because it is one of them at a time, not all four -
    which of them applies depends on where the rest of the portfolio has left
    the exemption and the losses.
    """
    return (
        "Moving off this number tends to waste one of the reliefs: either part "
        f"of the Rs {cfg.ltcg_exemption:,.0f} exemption goes unused, or long-term "
        f"gain is pushed above it, or short-term gain is realised at "
        f"{cfg.stcg_rate:.0%} while long-term was still free, or a loss is spent "
        "where it cancels less than it could elsewhere."
    )


def _boundary_clause(n: int) -> str:
    """Why the split stops here rather than one share either side.

    Qualitative on purpose. The exact cost of a move depends on which other lot
    absorbs the share, and with three lots in a ticker that differs by partner
    and by direction, which buries the only point worth making.
    """
    return f"{n} is the boundary for this lot."


def _fill_clause(
    lot: PricedLot, n: int, required: int, before: int, pick: int, sequential: bool
) -> str:
    """How much of the ticker's requirement this lot covers, and what is left.

    The brief turns on exactly this: a lot is emptied, the quantity still needed
    spills into the next one, and the shares beyond that stay put. A rule works
    through its lots in sequence, so after the first pick it counts against what
    was still outstanding rather than against the original total. The solver
    settles every quantity at once and has no sequence to count along.
    """
    ticker, quantity = lot.lot.ticker, lot.lot.quantity
    left, after = quantity - n, before - n
    usage = (
        "taking the whole lot"
        if n == quantity
        else f"{n} of its {quantity}, leaving {left} untouched"
    )

    if sequential and pick > 1 and after == 0:
        return (
            f"Covers the remaining {n}, taking the whole lot."
            if n == quantity
            else f"Covers the remaining {n} from its {quantity}, leaving {left} untouched."
        )
    if sequential and pick > 1:
        head = f"Supplies {n} of the {before} still outstanding"
    elif n == required:
        head = f"Supplies all {required} shares {ticker} must give up"
    else:
        head = f"Supplies {n} of the {required} shares {ticker} must give up"

    spill = f"; {after} still to find from the next lot" if sequential and after else ""
    return f"{head}, {usage}{spill}."


def lot_sales(
    lots: Sequence[PricedLot],
    shares: Sequence[int],
    cfg: TaxConfig,
    sale_date: date,
    method: str = "optimal",
) -> list[LotSale]:
    """One record per lot the plan sells, with its reasoning attached."""
    net_st, net_lt = aggregates(lots, shares)
    required: dict[str, int] = {}
    for lot, n in zip(lots, shares):
        required[lot.lot.ticker] = required.get(lot.lot.ticker, 0) + n

    out: list[LotSale] = []
    picks: dict[str, int] = {}
    outstanding: dict[str, int] = {}
    sequential = method in ("fifo", "ltfo")

    for i in pick_order(lots, shares, method, cfg):
        lot = lots[i]
        n = shares[i]
        ticker = lot.lot.ticker
        picks[ticker] = picks.get(ticker, 0) + 1
        before = outstanding.get(ticker, required[ticker])
        outstanding[ticker] = before - n
        rate = effective_rate(net_st, net_lt, lot.bucket, cfg)
        realized = lot.gain_per_share * n
        left = lot.lot.quantity - n

        parts = [
            _rule_clause(lot, method, cfg, picks[ticker]),
            _fill_clause(lot, n, required[ticker], before, picks[ticker], sequential),
        ]
        if method == "optimal" and 0 < n < lot.lot.quantity:
            parts.append(_boundary_clause(n))
        text = " ".join(p for p in parts if p)

        out.append(
            LotSale(
                lot_id=lot.lot.lot_id,
                ticker=lot.lot.ticker,
                buy_date=lot.lot.buy_date,
                holding=holding_label(lot.lot.buy_date, sale_date),
                classification=lot.classification,
                shares_sold=n,
                lot_quantity=lot.lot.quantity,
                remaining_shares=left,
                price=lot.price,
                cost_basis_per_share=lot.lot.buy_price,
                gain_per_share=lot.gain_per_share,
                realized_gain=realized,
                marginal_tax_rate=rate,
                reason=text,
            )
        )
    return out


def setoff_summary(tax: TaxBreakdown, cfg: TaxConfig) -> str:
    """One line describing how losses and the exemption were applied."""
    parts = []
    if tax.ltcl_used_against_ltcg:
        parts.append(
            f"Rs {tax.ltcl_used_against_ltcg:,.0f} of long-term loss cancelled "
            "long-term gain"
        )
    if tax.stcl_used_against_stcg:
        parts.append(
            f"Rs {tax.stcl_used_against_stcg:,.0f} of short-term loss cancelled "
            f"short-term gain at {cfg.stcg_rate:.1%}"
        )
    if tax.stcl_used_against_ltcg:
        parts.append(
            f"Rs {tax.stcl_used_against_ltcg:,.0f} of surplus short-term loss "
            "carried over to the long-term side"
        )
    setoff = ("Set-off: " + "; ".join(parts) + ". ") if parts else ""
    return (
        f"{setoff}Exemption used: Rs {tax.exemption_used:,.0f} of "
        f"Rs {cfg.ltcg_exemption:,.0f}. Taxable: Rs {tax.taxable_stcg:,.0f} "
        f"short-term at {cfg.stcg_rate:.1%} and Rs {tax.taxable_ltcg:,.0f} "
        f"long-term at {cfg.ltcg_rate:.1%}."
    )
