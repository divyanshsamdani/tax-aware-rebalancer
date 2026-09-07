"""Orchestration. No tax rule is implemented in this file.

    portfolio.validate_targets    are the target weights a legal allocation?
    portfolio.positions           lots -> quantity and value per ticker
    portfolio.sell_requirements   weights -> shares to sell per ticker
    tax.price_lot                 lot -> short-term or long-term, gain per share
    optimizer.solve               which lots supply those shares
    tax.breakdown                 the set-off calculation on the result
    explain.lot_sales             why those lots, priced not narrated
    portfolio.buy_plan            proceeds -> shares to buy back

`sale_date` is passed in rather than read from the clock, because every lot's
classification depends on it and the tests would otherwise fail as lots aged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping, Optional, Sequence

from . import explain, optimizer, portfolio, tax
from .models import Lot, Plan, PricedLot, SellRequirement, Summary, TaxConfig, Trade, WeightRow
from .optimizer import Method
from .portfolio import Position


@dataclass(frozen=True)
class _Setup:
    """Method-independent work, done once so the FIFO comparison need not repeat it."""

    positions: dict[str, Position]
    total_value: float
    requirements: list[SellRequirement]
    priced: list[PricedLot]


class Engine:
    def __init__(
        self,
        lots: Sequence[Lot],
        prices: Mapping[str, float],
        targets: Mapping[str, float],
        cfg: Optional[TaxConfig] = None,
        sale_date: Optional[date] = None,
    ) -> None:
        self.lots = list(lots)
        self.prices = dict(prices)
        self.targets = dict(targets)
        self.cfg = cfg or TaxConfig()
        self.sale_date = sale_date or date.today()
        self._cached: Optional[_Setup] = None

    def _setup(self) -> _Setup:
        if self._cached is not None:
            return self._cached

        if not self.lots:
            raise ValueError("no lots supplied: nothing to rebalance")

        future = sorted(l.lot_id for l in self.lots if l.buy_date > self.sale_date)
        if future:
            raise ValueError(
                f"lot(s) {', '.join(future[:5])} were bought after the sale date "
                f"{self.sale_date}"
            )

        portfolio.validate_targets(self.targets)
        positions = portfolio.positions(self.lots, self.prices)

        unknown = sorted(set(self.targets) - set(positions))
        if unknown:
            raise ValueError(
                f"target weight given for {', '.join(unknown)} but no lots are held "
                "in it. This engine rebalances existing holdings; it does not open "
                "new positions."
            )

        total_value = sum(p.value for p in positions.values())
        if total_value <= 0:
            raise ValueError("portfolio has no value to rebalance")

        requirements = portfolio.sell_requirements(positions, self.targets, total_value)

        # Only lots of tickers that must be sold: an unconstrained lot would be a
        # free variable, and the solver would sell losses nobody asked for.
        selling = {r.ticker for r in requirements}
        priced = [
            tax.price_lot(lot, self.prices[lot.ticker], self.sale_date, self.cfg)
            for lot in self.lots
            if lot.ticker in selling
        ]

        self._cached = _Setup(positions, total_value, requirements, priced)
        return self._cached

    def run(self, method: Method = "exact", compare: bool = True) -> Plan:
        s = self._setup()
        allocation = optimizer.solve(s.priced, s.requirements, self.cfg, method)

        sales = explain.lot_sales(s.priced, allocation.shares, self.cfg, self.sale_date)
        breakdown = tax.breakdown(s.priced, allocation.shares, self.cfg)

        sold: dict[str, int] = {}
        for sale in sales:
            sold[sale.ticker] = sold.get(sale.ticker, 0) + sale.shares_sold
        proceeds = sum(n * self.prices[t] for t, n in sold.items())

        buys = portfolio.buy_plan(
            s.positions, self.targets, sold, s.total_value, proceeds
        )
        spent = sum(n * self.prices[t] for t, n in buys.items())

        trades = [
            Trade(t, "SELL", n, self.prices[t], n * self.prices[t])
            for t, n in sorted(sold.items())
        ] + [
            Trade(t, "BUY", n, self.prices[t], n * self.prices[t])
            for t, n in sorted(buys.items())
        ]

        summary = Summary(
            total_tax=breakdown.total_tax,
            shares_sold=sum(s.shares_sold for s in sales),
            shares_bought=sum(buys.values()),
            gross_proceeds=proceeds,
            cash_left_over=proceeds - spent,
        )

        plan = Plan(
            method=allocation.method,
            certified_optimal=allocation.certified_optimal,
            summary=summary,
            trades=trades,
            lot_sales=sales,
            tax=breakdown,
            weights=self._weights(sold, buys, proceeds - spent),
        )

        if compare:
            fifo = optimizer.solve(s.priced, s.requirements, self.cfg, "fifo").tax(
                s.priced, self.cfg
            )
            summary.tax_if_fifo = fifo
            summary.saving_vs_fifo = fifo - breakdown.total_tax

        plan.reasoning = self._reasoning(plan, breakdown)
        return plan

    def _weights(
        self, sold: Mapping[str, int], buys: Mapping[str, int], cash: float
    ) -> list[WeightRow]:
        """Weights before and after, against the same total. Undeployed proceeds stay
        in the denominator as cash, so the "after" column is honest about rounding."""
        s = self._setup()
        before = portfolio.weights(s.positions)
        after_positions = {
            t: Position(t, p.quantity - sold.get(t, 0) + buys.get(t, 0), p.price)
            for t, p in s.positions.items()
        }
        after = portfolio.weights(after_positions, cash)
        return [
            WeightRow(
                ticker=t,
                target_pct=self.targets.get(t, 0.0) * 100.0,
                before_pct=before.get(t, 0.0) * 100.0,
                after_pct=after.get(t, 0.0) * 100.0,
            )
            for t in sorted(set(before) | set(self.targets))
        ]

    def _reasoning(self, plan: Plan, breakdown) -> list[str]:
        if not plan.trades:
            return [
                "Every ticker is already at its target weight to within one whole "
                "share, so there are no trades, no realised gains and no tax.",
                self._comparison_line(plan),
            ]

        lines = [
            f"Sold {plan.summary.shares_sold} share(s) from {len(plan.lot_sales)} "
            f"lot(s) for Rs {plan.summary.gross_proceeds:,.2f} and bought "
            f"{plan.summary.shares_bought} share(s) back. "
            f"Tax on this plan: Rs {breakdown.total_tax:,.2f}.",
            self._comparison_line(plan),
            explain.setoff_summary(breakdown, self.cfg),
            explain.optimality_note(plan.lot_sales),
        ]
        if plan.summary.cash_left_over > 0.005:
            lines.append(
                f"Trades are whole shares only, so Rs "
                f"{plan.summary.cash_left_over:,.2f} of the proceeds could not be "
                "reinvested. That is why the closing weights sit a fraction under "
                "target."
            )
        if plan.method != "exact":
            lines.append(
                f"Lot selection used the '{plan.method}' baseline rather than the "
                "tax-minimising solver, so this plan is not claimed to be cheapest."
            )
        return [line for line in lines if line]

    def _comparison_line(self, plan: Plan) -> str:
        """The FIFO comparison in words. A bare saving of zero reads as "picking lots
        achieved nothing" when it means FIFO was optimal here and was proved so."""
        fifo = plan.summary.tax_if_fifo
        if fifo is None:
            return ""
        if not plan.trades:
            return "No sale is required, so every lot-selection method costs nothing."
        mine = plan.summary.total_tax
        if fifo - mine > 0.005:
            return (
                f"Choosing lots to minimise tax costs Rs {mine:,.2f} against "
                f"Rs {fifo:,.2f} selling oldest-first, a saving of "
                f"Rs {fifo - mine:,.2f}."
            )
        return (
            f"Selling oldest-first reaches the same figure here, Rs {mine:,.2f}. "
            "That is a result the engine proved, not one it assumed."
        )
