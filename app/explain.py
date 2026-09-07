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
            verdict = f"would save Rs {-delta:,.2f} - this plan is not optimal"
        else:
            verdict = "costs exactly the same in tax"
        out.append(
            Alternative(
                lot_id=other.lot.lot_id,
                buy_date=other.lot.buy_date,
                classification=other.classification,
                gain_per_share=other.gain_per_share,
                shares=n,
                tax_delta=delta,
                note=(
                    f"selling {n} share(s) from lot {other.lot.lot_id} instead "
                    f"(bought {other.lot.buy_date}, {other.classification}, "
                    f"{'loss' if other.gain_per_share < 0 else 'gain'} of Rs "
                    f"{abs(other.gain_per_share):,.2f}/share) {verdict}"
                ),
            )
        )
    out.sort(key=lambda a: a.tax_delta)
    return out[:_MAX_ALTERNATIVES]


def _rate_sentence(bucket: str, rate: float, is_loss: bool, cfg: TaxConfig) -> str:
    """The marginal rate, worded as a cost for a gain and a saving for a loss."""
    side = _BUCKET_NAME[bucket]
    zero = abs(rate) < 1e-9

    if is_loss:
        if zero and bucket == "LT":
            why = (
                f"the Rs {cfg.ltcg_exemption:,.0f} exemption already covers the "
                "long-term gain, so this loss saves nothing this year"
            )
        elif zero:
            why = "there is no realised gain left for it to cancel"
        elif bucket == "LT":
            why = "it cancels long-term gain that sits above the exemption"
        elif abs(rate - cfg.ltcg_rate) < 1e-9:
            why = (
                "no short-term gain is left, so it carries over to the long-term "
                "side where the rate is lower"
            )
        else:
            why = "it cancels short-term gain, which is taxed at the higher rate"
        return f"Each rupee of this loss saves {rate:.2%}, because {why}."

    if zero and bucket == "LT":
        why = f"the Rs {cfg.ltcg_exemption:,.0f} long-term exemption is not yet used up"
    elif zero:
        why = "realised losses still absorb it completely"
    elif bucket == "LT":
        why = "the exemption has been used up"
    elif abs(rate - cfg.ltcg_rate) < 1e-9:
        why = (
            "short-term losses absorb it and the surplus carries over to the "
            "long-term side, where the rate is lower"
        )
    else:
        why = "there is no realised loss left to absorb it"
    return f"One more rupee of {side} gain would cost {rate:.2%}, because {why}."


def lot_sales(
    lots: Sequence[PricedLot],
    shares: Sequence[int],
    cfg: TaxConfig,
    sale_date: date,
) -> list[LotSale]:
    """One record per lot the plan sells, with its reasoning attached."""
    net_st, net_lt = aggregates(lots, shares)
    out: list[LotSale] = []

    for i, lot in enumerate(lots):
        n = shares[i]
        if n <= 0:
            continue
        rate = effective_rate(net_st, net_lt, lot.bucket, cfg)
        realized = lot.gain_per_share * n
        left = lot.lot.quantity - n
        alternatives = _alternatives(lots, shares, i, cfg)

        is_loss = lot.gain_per_share < 0
        movement = (
            f"a loss of Rs {abs(lot.gain_per_share):,.2f}/share, "
            f"Rs {abs(realized):,.2f} in all"
            if is_loss
            else f"a gain of Rs {lot.gain_per_share:,.2f}/share, "
            f"Rs {realized:,.2f} in all"
        )
        text = (
            f"{lot.lot.ticker}: sell {n} of {lot.lot.quantity} shares from the lot "
            f"bought {lot.lot.buy_date} - "
            f"{holding_label(lot.lot.buy_date, sale_date)}, "
            f"{_BUCKET_NAME[lot.bucket]}, {lot.classification}. "
            f"Cost Rs {lot.lot.buy_price:,.2f}/share against Rs {lot.price:,.2f} "
            f"today, so {movement}. "
            f"{_rate_sentence(lot.bucket, rate, is_loss, cfg)}"
        )
        if alternatives:
            best = alternatives[0].note
            text += " " + best[0].upper() + best[1:] + "."
        elif sum(1 for o in lots if o.lot.ticker == lot.lot.ticker) == 1:
            text += " This is the only lot held in this ticker, so the choice was forced."
        else:
            text += (
                " Every other lot of this ticker is already fully committed in this "
                "plan, so there are no spare shares to swap for these."
            )
        if left > 0:
            text += (
                f" The remaining {left} shares stay untouched and keep their "
                f"original buy date of {lot.lot.buy_date}."
            )

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


def optimality_note(sales: Sequence[LotSale]) -> str:
    """What the table of alternatives actually proves."""
    swaps = [a for s in sales for a in s.alternatives]
    if not swaps:
        return "No substitution was possible: the plan is forced."
    cheapest = min(a.tax_delta for a in swaps)
    if cheapest < -1e-6:
        return (
            f"Warning: a single swap would save Rs {-cheapest:,.2f}, so this plan "
            "is not optimal."
        )
    return (
        f"Checked {len(swaps)} alternative lot substitution(s); every one costs the "
        "same or more, which is what makes this plan the cheapest available."
    )
