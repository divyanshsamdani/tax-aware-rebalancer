"""End to end. The three cases the brief asks for come first."""

from __future__ import annotations

from datetime import date

import pytest

from app import ingest
from app.engine import Engine
from app.models import Lot

SALE = date(2026, 9, 6)
SCENARIOS = ["edge_case", "all_ltcg", "at_target", "loss_offset",
             "exemption_split", "exemption_vs_loss", "loss_priority"]


def run(scenario: str, method: str = "optimal"):
    lots, prices, targets = ingest.load_scenario(f"samples/{scenario}")
    return Engine(lots, prices, targets, sale_date=SALE).run(method)


def sold(plan) -> dict[str, int]:
    return {s.lot_id: s.shares_sold for s in plan.lot_sales}


# (a) the required partial-lot edge case
def test_edge_case_splits_the_sale_across_two_lots():
    """ACME is 60 long-term shares and 40 short-term. 75 must go, so the whole
    long-term lot goes and 15 spill into the short-term lot."""
    plan = run("edge_case")
    assert sold(plan) == {"L1": 60, "L2": 15}
    l1, l2 = plan.lot_sales
    assert (l1.classification, l1.remaining_shares) == ("LTCG", 0)
    assert (l2.classification, l2.remaining_shares) == ("STCG", 25)
    # the 25 left behind keep their own buy date for a future sale
    assert l2.buy_date == date(2026, 4, 1)


def test_edge_case_blends_the_tax_across_both_lots():
    """Rs 12,000 of long-term gain inside the exemption costs nothing; Rs 750 of
    short-term gain costs 20%."""
    plan = run("edge_case")
    assert plan.tax.ltcg == pytest.approx(12_000.0)
    assert plan.tax.stcg == pytest.approx(750.0)
    assert plan.tax.exemption_used == pytest.approx(12_000.0)
    assert plan.tax.taxable_ltcg == pytest.approx(0.0)
    assert plan.tax.total_tax == pytest.approx(150.0)


def test_edge_case_hits_the_target_weights():
    plan = run("edge_case")
    weights = {w.ticker: w for w in plan.weights}
    assert weights["ACME"].before_pct == pytest.approx(80.0)
    assert weights["ACME"].after_pct == pytest.approx(20.0)
    assert weights["NOVA"].after_pct == pytest.approx(80.0)
    assert plan.summary.cash_left_over == pytest.approx(0.0)


# (b) a straightforward all-long-term rebalance
def test_all_ltcg_takes_the_lot_with_the_smaller_gain():
    """525 shares of the high-cost lot realise Rs 52,500, inside the exemption."""
    plan = run("all_ltcg")
    assert sold(plan) == {"INFRA-2": 525}
    assert all(s.classification == "LTCG" for s in plan.lot_sales)
    assert plan.tax.total_tax == pytest.approx(0.0)
    assert plan.summary.tax_if_fifo == pytest.approx(43_437.50)
    assert plan.summary.saving_vs_fifo == pytest.approx(43_437.50)


# (c) nothing to do
def test_a_portfolio_at_target_produces_no_trades_and_no_tax():
    plan = run("at_target")
    assert plan.trades == []
    assert plan.lot_sales == []
    assert plan.summary.total_tax == pytest.approx(0.0)
    for w in plan.weights:
        assert w.after_pct == pytest.approx(w.target_pct)
    assert any("already at its target weight" in line for line in plan.reasoning)


# losses, and the asymmetry between the two kinds
def test_the_short_term_loss_lot_beats_the_older_long_term_one():
    """Both LOSSCO lots lose Rs 400 a share, but the long-term one could only
    cancel long-term gain, of which there is none."""
    plan = run("loss_offset")
    assert sold(plan) == {"GAINCO-1": 100, "LOSSCO-ST": 200}
    assert plan.tax.stcl_used_against_stcg == pytest.approx(80_000.0)
    assert plan.tax.total_tax == pytest.approx(4_000.0)
    assert plan.summary.tax_if_fifo == pytest.approx(20_000.0)


def test_a_long_term_loss_is_spent_only_down_to_the_exemption_line():
    """GAINCO leaves Rs 1,80,000 of long-term gain, Rs 55,000 of it above the
    exemption, so the long-term loss stops paying after 55 shares."""
    plan = run("exemption_vs_loss")
    assert sold(plan)["H1"] == 55 and sold(plan)["H2"] == 45
    assert plan.tax.ltcg - plan.tax.ltcl_used_against_ltcg == 125_000.0
    assert plan.summary.total_tax == 8_400.0


def test_the_priority_between_two_losses_flips_part_way_through():
    """The short-term loss saves 20% only while there is short-term gain to
    cancel. After 43 shares the long-term loss is worth more per share."""
    plan = run("loss_priority")
    assert sold(plan)["H2"] == 43 and sold(plan)["H1"] == 57
    assert plan.tax.taxable_stcg == 0.0
    assert plan.summary.total_tax == 14_700.0


@pytest.mark.parametrize("scenario", ["exemption_vs_loss", "loss_priority", "exemption_split"])
def test_neither_shortcut_finds_the_split(scenario):
    best, ltfo, fifo = (run(scenario, m).summary.total_tax for m in ("optimal", "ltfo", "fifo"))
    assert best < ltfo <= fifo


# rebalancing mechanics
def test_the_fifo_baseline_reports_itself_as_not_optimal():
    plan = run("all_ltcg", method="fifo")
    assert plan.certified_optimal is False
    assert plan.tax.total_tax == pytest.approx(43_437.50)
    assert any("not claimed to be cheapest" in line for line in plan.reasoning)


def test_a_held_ticker_missing_from_the_targets_is_sold_off():
    lots = [
        Lot("A", "KEEP", date(2023, 1, 1), 100, 100.0),
        Lot("B", "DROP", date(2023, 1, 1), 100, 100.0),
    ]
    plan = Engine(lots, {"KEEP": 100.0, "DROP": 100.0}, {"KEEP": 1.0}, sale_date=SALE).run()
    assert sold(plan) == {"B": 100}
    assert {w.ticker: w.after_pct for w in plan.weights}["DROP"] == pytest.approx(0.0)


def test_a_ticker_is_never_sold_and_bought_back_on_the_same_day():
    lots = [
        Lot("A", "AAA", date(2023, 1, 1), 100, 90.0),
        Lot("B", "BBB", date(2023, 1, 1), 33, 90.0),
        Lot("C", "CCC", date(2023, 1, 1), 20, 90.0),
    ]
    plan = Engine(
        lots,
        {"AAA": 101.0, "BBB": 97.0, "CCC": 103.0},
        {"AAA": 0.34, "BBB": 0.33, "CCC": 0.33},
        sale_date=SALE,
    ).run()
    sells = {t.ticker for t in plan.trades if t.action == "SELL"}
    buys = {t.ticker for t in plan.trades if t.action == "BUY"}
    assert not (sells & buys)
    assert plan.summary.cash_left_over >= -1e-9  # the buy leg pays for itself


def test_trades_are_whole_shares_with_the_sells_first():
    plan = run("edge_case")
    actions = [t.action for t in plan.trades]
    assert actions == sorted(actions, reverse=True)
    assert all(isinstance(t.shares, int) and t.shares > 0 for t in plan.trades)


def test_the_trade_date_changes_the_classification_and_so_the_tax():
    """The same portfolio a month later, once the newer lot has aged past twelve
    months. Nothing else changes; the bill goes from Rs 1,000 to nil."""
    lots = [
        Lot("OLD", "T", date(2024, 1, 1), 50, 100.0),
        Lot("NEW", "T", date(2025, 9, 20), 50, 100.0),
        Lot("KEEP", "U", date(2020, 1, 1), 100, 100.0),
    ]
    prices, targets = {"T": 200.0, "U": 200.0}, {"U": 1.0}
    before = Engine(lots, prices, targets, sale_date=date(2026, 9, 6)).run()
    after = Engine(lots, prices, targets, sale_date=date(2026, 10, 6)).run()

    assert {s.classification for s in before.lot_sales} == {"LTCG", "STCG"}
    assert {s.classification for s in after.lot_sales} == {"LTCG"}
    assert before.summary.total_tax == pytest.approx(1_000.0)
    assert after.summary.total_tax == pytest.approx(0.0)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_bundled_scenario_loads_and_solves(scenario):
    assert run(scenario).certified_optimal is True


# validation
@pytest.mark.parametrize(
    "prices,targets,message",
    [
        ({"X": 100.0}, {"X": 0.9}, "add up to"),
        ({}, {"X": 1.0}, "no current price"),
        ({"X": 100.0}, {"X": 0.5, "Y": 0.5}, "no lots are held"),
    ],
)
def test_bad_input_is_rejected_with_a_reason(prices, targets, message):
    lots = [Lot("A", "X", date(2023, 1, 1), 10, 100.0)]
    with pytest.raises(ValueError, match=message):
        Engine(lots, prices, targets, sale_date=SALE).run()


def test_a_lot_bought_after_the_trade_date_is_rejected():
    """A year typed as 2027 instead of 2017 would otherwise look like an
    ordinary short-term holding."""
    lots = [Lot("A", "X", date(2027, 1, 1), 10, 100.0)]
    with pytest.raises(ValueError, match="bought after the sale date"):
        Engine(lots, {"X": 100.0}, {"X": 1.0}, sale_date=SALE).run()


# the reasoning belongs to the plan it is attached to
def test_each_method_explains_itself_in_its_own_terms():
    fifo, ltfo, best = run("edge_case", "fifo"), run("edge_case", "ltfo"), run("edge_case")

    fifo_l1 = next(s for s in fifo.lot_sales if s.lot_id == "L1")
    assert fifo_l1.reason.startswith("1st pick for ACME: the oldest lot still available")

    ltfo_l2 = next(s for s in ltfo.lot_sales if s.lot_id == "L2")
    assert ltfo_l2.reason.startswith("1st pick for ACME: the lowest statutory tax per share")
    assert "at Rs 10.00" in ltfo_l2.reason

    # a ranking rule lists its lots in the order it reached for them
    assert [s.lot_id for s in ltfo.lot_sales] == ["L2", "L1"]
    assert [s.lot_id for s in fifo.lot_sales] == ["L1", "L2"]
    # the solver has no such order, so it has no pick to name
    assert "pick for ACME" not in next(s for s in best.lot_sales if s.lot_id == "L1").reason


def test_the_reasoning_shows_the_fill_spilling_into_the_next_lot():
    """The mechanic the brief turns on: a lot is emptied, the shortfall spills
    into the next lot, and the shares beyond that stay put."""
    plan = run("edge_case", "fifo")
    by_id = {s.lot_id: s for s in plan.lot_sales}
    assert (
        "Supplies 60 of the 75 shares ACME must give up, taking the whole lot; "
        "15 still to find from the next lot." in by_id["L1"].reason
    )
    assert "Covers the remaining 15 from its 40, leaving 25 untouched." in by_id["L2"].reason


def test_a_partial_lot_prices_the_alternative_rather_than_asserting_one():
    """On loss_priority, moving a share off 57 really does cost Rs 5. On
    all_ltcg the swap is free, so that lot claims no trade-off at all."""
    h1 = next(s for s in run("loss_priority").lot_sales if s.lot_id == "H1")
    assert "Moving one share off 57 costs at least Rs 5.00 more." in h1.reason
    assert h1.alternative.lot_id == "H2" and h1.alternative.extra_tax == 5.0

    infra = next(s for s in run("all_ltcg").lot_sales if s.lot_id == "INFRA-2")
    assert 0 < infra.shares_sold < infra.lot_quantity
    assert infra.alternative is None
    assert "Moving one share" not in infra.reason


def test_the_ledger_names_the_reliefs_that_move_and_adds_up():
    """The share carries Rs 800 of short-term loss, but only Rs 400 of it was
    cancelling short-term gain at 20% - the rest had already spilled to the
    long-term side at 12.5%. That split is why the answer is 57 and not 58."""
    h1 = next(s for s in run("loss_priority").lot_sales if s.lot_id == "H1")
    lines = {line.change: line for line in h1.alternative.lines}
    assert lines["Rs 400 less short-term loss cancelling short-term gain"].tax_effect == 80.0
    assert lines["Rs 400 less short-term loss spilling to the long-term side"].tax_effect == 50.0
    assert lines["Rs 1,000 more long-term loss cancelling long-term gain"].tax_effect == -125.0
    assert sum(line.tax_effect for line in h1.alternative.lines) == 5.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_ledger_reconciles_with_the_cost_it_explains(scenario):
    """The six terms are the definition of the difference between two set-off
    calculations, so they add up or the ledger is wrong."""
    for sale in run(scenario).lot_sales:
        if sale.alternative is None:
            continue
        assert sale.alternative.lines and not sale.alternative.note
        assert sum(line.tax_effect for line in sale.alternative.lines) == pytest.approx(
            sale.alternative.extra_tax, abs=0.005
        )


def test_only_the_solver_prices_the_alternative():
    """A ranking rule never weighed the share either side, so it makes no such
    claim."""
    sales = run("loss_priority", "fifo").lot_sales
    assert all(s.alternative is None for s in sales)
    assert all("Moving one share" not in s.reason for s in sales)
