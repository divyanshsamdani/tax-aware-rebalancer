"""The sentence attached to each lot in a plan.

FIFO and LTFO work through a ticker one lot at a time, so their reasoning names
the place in that sequence and where the remainder spills. The solver settles
every quantity at once and has no sequence to name, so it prices the alternative
instead: what one share either side of the split would have cost.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from .models import Alternative, LotSale, PricedLot, SwapLine, TaxBreakdown, TaxConfig
from .tax import aggregates, breakdown, effective_rate, holding_label


def statutory_per_share(lot: PricedLot, cfg: TaxConfig) -> float:
    """Gain per share at the lot's own statutory rate: what LTFO ranks on."""
    return lot.gain_per_share * (cfg.stcg_rate if lot.bucket == "ST" else cfg.ltcg_rate)


def pick_order(
    lots: Sequence[PricedLot], shares: Sequence[int], method: str, cfg: TaxConfig
) -> list[int]:
    """The sold lots, in the order the method reached for them. `optimal` has no
    such order, so its lots stay in portfolio order."""
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
    ticker = lot.lot.ticker
    if method == "fifo":
        return f"{_ordinal(pick)} pick for {ticker}: the oldest lot still available."
    if method == "ltfo":
        return (
            f"{_ordinal(pick)} pick for {ticker}: the lowest statutory tax per share "
            f"of the lots still available, at Rs {statutory_per_share(lot, cfg):,.2f}."
        )
    return ""


# Used in place of the ledger where it will not reconcile. Written as
# alternatives because it is one of them at a time, and which one depends on
# where the rest of the portfolio has left the exemption and the losses.
UNCERTAIN_NOTE = (
    "Which relief is at stake here depends on the rest of the portfolio. It is "
    "usually one of these: part of the Rs 1,25,000 exemption goes unused, or "
    "long-term gain is pushed above it, or short-term gain is realised at 20% "
    "while long-term was still free, or a loss is spent where it cancels less "
    "than it could elsewhere."
)


def _ledger(before: TaxBreakdown, after: TaxBreakdown, cfg: TaxConfig) -> list[SwapLine]:
    """Break the difference between two set-off calculations into priced terms.

    The tax is `stcg_rate * taxable_stcg + ltcg_rate * taxable_ltcg`, and each
    taxable figure is a gain less the reliefs applied to it, so these six terms
    are the definition of the difference rather than an attribution of it. They
    add up to the exact cost.
    """
    s, l = cfg.stcg_rate, cfg.ltcg_rate
    terms = [
        ("short-term gain", after.stcg - before.stcg, s, 1),
        ("short-term loss cancelling short-term gain",
         after.stcl_used_against_stcg - before.stcl_used_against_stcg, s, -1),
        ("long-term gain", after.ltcg - before.ltcg, l, 1),
        ("long-term loss cancelling long-term gain",
         after.ltcl_used_against_ltcg - before.ltcl_used_against_ltcg, l, -1),
        ("short-term loss spilling to the long-term side",
         after.stcl_used_against_ltcg - before.stcl_used_against_ltcg, l, -1),
        (f"of the Rs {cfg.ltcg_exemption:,.0f} exemption used",
         after.exemption_used - before.exemption_used, l, -1),
    ]
    return [
        SwapLine(
            change=f"Rs {abs(delta):,.0f} {'more' if delta > 0 else 'less'} {label}",
            rate=rate,
            tax_effect=sign * rate * delta,
        )
        for label, delta, rate, sign in terms
        if abs(delta) > 1e-9
    ]


def cheapest_alternative(
    lots: Sequence[PricedLot], shares: Sequence[int], i: int, cfg: TaxConfig
) -> Alternative | None:
    """The cheapest single-share change to this lot's quantity, and what it costs.

    The ticker's total is fixed, so a share can only move to or from another lot
    of the same ticker. Where no sibling can take it the quantity is forced
    arithmetic rather than a decision, and where some swap is free the tax is
    flat and this quantity is one of several equally cheap answers. Neither is a
    trade-off, so neither is reported.
    """
    ticker = lots[i].lot.ticker
    base = breakdown(lots, shares, cfg)
    best: tuple[float, int, str, TaxBreakdown] | None = None

    for j, sibling in enumerate(lots):
        if j == i or sibling.lot.ticker != ticker:
            continue
        for frm, to, sells_one in ((i, j, "fewer"), (j, i, "more")):
            if shares[frm] < 1 or shares[to] >= lots[to].lot.quantity:
                continue
            moved = list(shares)
            moved[frm] -= 1
            moved[to] += 1
            after = breakdown(lots, moved, cfg)
            cost = after.total_tax - base.total_tax
            if best is None or cost < best[0]:
                best = (cost, j, sells_one, after)

    if best is None or best[0] <= 1e-9:
        return None

    cost, j, sells_one, after = best
    lines = _ledger(base, after, cfg)
    reconciles = abs(sum(line.tax_effect for line in lines) - cost) < 0.005
    return Alternative(
        lot_id=lots[j].lot.lot_id,
        sells_one=sells_one,
        extra_tax=cost,
        lines=lines if reconciles else [],
        note="" if reconciles else UNCERTAIN_NOTE,
    )


def _fill_clause(
    lot: PricedLot, n: int, required: int, before: int, pick: int, sequential: bool
) -> str:
    """How much of the ticker's requirement this lot covers, and what is left.

    The brief turns on exactly this: a lot is emptied, the quantity still needed
    spills into the next one, and the shares beyond that stay put. A sequential
    rule counts against what was still outstanding after its first pick.
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

        parts = [
            _rule_clause(lot, method, cfg, picks[ticker]),
            _fill_clause(lot, n, required[ticker], before, picks[ticker], sequential),
        ]
        alternative = None
        if method == "optimal" and 0 < n < lot.lot.quantity:
            alternative = cheapest_alternative(lots, shares, i, cfg)
        if alternative is not None:
            parts.append(
                f"Moving one share off {n} costs at least "
                f"Rs {alternative.extra_tax:,.2f} more."
            )

        out.append(
            LotSale(
                lot_id=lot.lot.lot_id,
                ticker=ticker,
                buy_date=lot.lot.buy_date,
                holding=holding_label(lot.lot.buy_date, sale_date),
                classification=lot.classification,
                shares_sold=n,
                lot_quantity=lot.lot.quantity,
                remaining_shares=lot.lot.quantity - n,
                price=lot.price,
                cost_basis_per_share=lot.lot.buy_price,
                gain_per_share=lot.gain_per_share,
                realized_gain=lot.gain_per_share * n,
                marginal_tax_rate=effective_rate(net_st, net_lt, lot.bucket, cfg),
                reason=" ".join(p for p in parts if p),
                alternative=alternative,
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
