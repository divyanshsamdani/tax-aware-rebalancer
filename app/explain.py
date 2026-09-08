"""Why each lot was chosen.

A sentence template restating what the solver did would read exactly as
confident if the solver were wrong. So instead, for every lot in the plan this
module takes the other lots of the same ticker, actually moves the shares
across, re-runs the tax calculation and reports the rupee difference. The sign
of that difference is itself a check: at a genuine optimum, no swap comes back
cheaper. It is affordable because tax depends only on the two realised-gain
totals, so pricing a swap is a handful of multiplications, not another solve.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from .models import Alternative, LotSale, PricedLot, TaxBreakdown, TaxConfig
from .tax import aggregates, effective_rate, holding_label, tax_on

_BUCKET_NAME = {"ST": "short-term", "LT": "long-term"}
_MAX_ALTERNATIVES = 3


def swap_tax_delta(
    lots: Sequence[PricedLot],
    shares: Sequence[int],
    source: int,
    target: int,
    n: int,
    cfg: TaxConfig,
) -> float:
    """Tax change from moving n shares between two lots of the same ticker, which
    leaves the plan valid."""
    trial = list(shares)
    trial[source] -= n
    trial[target] += n
    return tax_on(*aggregates(lots, trial), cfg) - tax_on(*aggregates(lots, shares), cfg)


def _alternatives(
    lots: Sequence[PricedLot], shares: Sequence[int], i: int, cfg: TaxConfig
) -> list[Alternative]:
    out: list[Alternative] = []
    ticker = lots[i].lot.ticker
    for j, other in enumerate(lots):
        if j == i or other.lot.ticker != ticker:
            continue
        room = other.lot.quantity - shares[j]
        n = min(shares[i], room)
        if n <= 0:
            continue
        delta = swap_tax_delta(lots, shares, i, j, n, cfg)
        if delta > 0:
            verdict = f"would cost Rs {delta:,.2f} more in tax"
        elif delta < 0:
            verdict = f"would save Rs {-delta:,.2f}, so this plan is not optimal"
        else:
            verdict = "costs exactly the same"
        out.append(
            Alternative(
                lot_id=other.lot.lot_id,
                buy_date=other.lot.buy_date,
                classification=other.classification,
                gain_per_share=other.gain_per_share,
                shares=n,
                tax_delta=delta,
                note=(
                    f"moving {n} share{'' if n == 1 else 's'} to lot "
                    f"{other.lot.lot_id} ({other.classification}) {verdict}"
                ),
            )
        )
    out.sort(key=lambda a: a.tax_delta)
    return out[:_MAX_ALTERNATIVES]


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

    `optimal` has no per-lot rule to quote: it chooses every quantity jointly
    across the portfolio, so its justification is the priced swap that follows.
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
    """One record per lot the plan sells, with its reasoning attached.

    The prose carries only what the structured fields cannot: why this method
    reached for this lot, what the next rupee actually costs as against the
    statutory rate, and the priced cost of having chosen differently.
    """
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
        alternatives = _alternatives(lots, shares, i, cfg)

        parts = [
            _rule_clause(lot, method, cfg, picks[ticker]),
            _fill_clause(lot, n, required[ticker], before, picks[ticker], sequential),
        ]
        # Only a plan that weighed alternatives should be judged against one. A
        # ranking rule never considered the swap, so "this plan is not optimal"
        # answers a question it never asked; the plan comparison says it better.
        if alternatives and method == "optimal":
            best = alternatives[0].note
            parts.append(best[0].upper() + best[1:] + ".")
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
                alternatives=alternatives,
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


def optimality_note(sales: Sequence[LotSale], certified: bool = True) -> str:
    """What the table of alternatives actually proves.

    Surviving every single-lot swap is a local check. For the solver's plan it
    corroborates a guarantee that already exists; for a ranking rule it proves
    nothing at all, because the cheaper plan may be several shares away in two
    lots at once - which is exactly how LTFO loses on `loss_priority`.
    """
    swaps = [a for s in sales for a in s.alternatives]
    if not swaps:
        return "No substitution was possible: the plan is forced."
    cheapest = min(a.tax_delta for a in swaps)
    if cheapest < -1e-6:
        return (
            f"A single swap would save Rs {-cheapest:,.2f}, so this plan is not "
            "optimal."
        )
    if certified:
        return (
            f"Checked {len(swaps)} alternative lot substitution(s); every one costs "
            "the same or more, which corroborates the solver's guarantee."
        )
    return (
        f"Checked {len(swaps)} alternative lot substitution(s) and none is cheaper, "
        "but that is not a proof: a cheaper plan can lie several shares away across "
        "two lots at once. Only the solver's plan carries a guarantee."
    )
