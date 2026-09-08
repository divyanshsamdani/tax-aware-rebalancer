"""End-to-end tests over the whole engine.

The three the assignment asks for are first: the partial-lot edge case, a plain
all-long-term rebalance, and a portfolio already at target.
"""

from __future__ import annotations

from datetime import date

import pytest

from app import ingest
from app.engine import Engine
from app.models import Lot

SALE = date(2026, 9, 6)


def run(scenario: str, method: str = "optimal"):
    lots, prices, targets = ingest.load_scenario(f"samples/{scenario}")
    engine = Engine(lots, prices, targets, sale_date=SALE)
    return engine, engine.run(method)


def sold(plan) -> dict[str, int]:
    return {s.lot_id: s.shares_sold for s in plan.lot_sales}


# Required case (a): the partial-lot edge case
def test_edge_case_sells_the_whole_long_term_lot_and_part_of_the_short_term_one():
    """ACME is 60 long-term shares and 40 short-term ones. The rebalance needs
    75 gone, so the whole long-term lot goes and 15 shares spill into the
    short-term lot."""
    _, plan = run("edge_case")
    assert sold(plan) == {"L1": 60, "L2": 15}

    l1, l2 = plan.lot_sales
    assert (l1.classification, l1.remaining_shares) == ("LTCG", 0)
    assert (l2.classification, l2.remaining_shares) == ("STCG", 25)


def test_edge_case_leaves_the_remainder_with_its_original_buy_date():
    """The 25 shares left in the short-term lot must keep 2026-04-01, so a
    future sale classifies them correctly."""
    _, plan = run("edge_case")
    remainder = next(s for s in plan.lot_sales if s.lot_id == "L2")
    assert remainder.remaining_shares == 25
    assert remainder.buy_date == date(2026, 4, 1)
    assert remainder.lot_quantity - remainder.shares_sold == 25


def test_edge_case_blends_the_tax_across_both_lots():
    """Rs 12,000 of long-term gain inside the exemption costs nothing; Rs 750 of
    short-term gain costs 20%. Total Rs 150."""
    _, plan = run("edge_case")
    assert plan.tax.ltcg == pytest.approx(12_000.0)
    assert plan.tax.stcg == pytest.approx(750.0)
    assert plan.tax.exemption_used == pytest.approx(12_000.0)
    assert plan.tax.taxable_ltcg == pytest.approx(0.0)
    assert plan.tax.total_tax == pytest.approx(150.0)


def test_edge_case_prices_its_alternative_rather_than_asserting_it():
    """The reasoning for the long-term lot must contain a number that came from
    re-running the tax calculation on the swap, not a template."""
    _, plan = run("edge_case")
    l1 = next(s for s in plan.lot_sales if s.lot_id == "L1")
    alt = l1.alternatives[0]
    assert alt.lot_id == "L2"
    assert alt.tax_delta == pytest.approx(250.0)
    assert "Rs 250.00 more in tax" in alt.note


def test_edge_case_hits_the_target_weights():
    _, plan = run("edge_case")
    weights = {w.ticker: w for w in plan.weights}
    assert weights["ACME"].before_pct == pytest.approx(80.0)
    assert weights["ACME"].after_pct == pytest.approx(20.0)
    assert weights["NOVA"].after_pct == pytest.approx(80.0)
    assert plan.summary.cash_left_over == pytest.approx(0.0)


# Required case (b): a straightforward all-long-term rebalance
def test_all_ltcg_rebalance_takes_the_lot_with_the_smaller_gain():
    """Both INFRA lots are long-term. Selling 525 shares of the high-cost lot
    realises Rs 52,500, which fits inside the exemption and costs nothing."""
    _, plan = run("all_ltcg")
    assert sold(plan) == {"INFRA-2": 525}
    assert all(s.classification == "LTCG" for s in plan.lot_sales)
    assert plan.tax.ltcg == pytest.approx(52_500.0)
    assert plan.tax.total_tax == pytest.approx(0.0)


def test_all_ltcg_saves_the_entire_bill_against_selling_oldest_first():
    _, plan = run("all_ltcg")
    assert plan.summary.tax_if_fifo == pytest.approx(43_437.50)
    assert plan.summary.saving_vs_fifo == pytest.approx(43_437.50)


def test_the_fifo_baseline_is_reported_as_not_optimal():
    _, plan = run("all_ltcg", method="fifo")
    assert plan.method == "fifo"
    assert plan.certified_optimal is False
    assert plan.tax.total_tax == pytest.approx(43_437.50)
    assert any("not claimed to be cheapest" in line for line in plan.reasoning)


# Required case (c): nothing to do
def test_a_portfolio_already_at_target_produces_no_trades_and_no_tax():
    _, plan = run("at_target")
    assert plan.trades == []
    assert plan.lot_sales == []
    assert plan.summary.total_tax == pytest.approx(0.0)
    assert plan.summary.shares_sold == 0
    assert plan.summary.shares_bought == 0
    for w in plan.weights:
        assert w.after_pct == pytest.approx(w.target_pct)
    assert any("already at its target weight" in line for line in plan.reasoning)


# Losses, and the asymmetry between the two kinds
def test_the_short_term_loss_lot_is_preferred_over_the_older_long_term_one():
    """Both LOSSCO lots lose Rs 400 a share. The short-term one cancels
    short-term gain at 20%; the long-term one could only cancel long-term gain,
    and there is none, so it would save nothing at all."""
    _, plan = run("loss_offset")
    assert sold(plan) == {"GAINCO-1": 100, "LOSSCO-ST": 200}
    assert plan.tax.stcl_used_against_stcg == pytest.approx(80_000.0)
    assert plan.tax.total_tax == pytest.approx(4_000.0)


def test_selling_oldest_first_would_waste_the_loss_entirely():
    _, plan = run("loss_offset")
    assert plan.summary.tax_if_fifo == pytest.approx(20_000.0)
    assert plan.summary.saving_vs_fifo == pytest.approx(16_000.0)


# Rebalancing mechanics
def test_a_ticker_held_but_missing_from_the_targets_is_sold_off_entirely():
    lots = [
        Lot("A", "KEEP", date(2023, 1, 1), 100, 100.0),
        Lot("B", "DROP", date(2023, 1, 1), 100, 100.0),
    ]
    engine = Engine(lots, {"KEEP": 100.0, "DROP": 100.0}, {"KEEP": 1.0}, sale_date=SALE)
    plan = engine.run()
    assert sold(plan) == {"B": 100}
    assert {w.ticker: w.after_pct for w in plan.weights}["DROP"] == pytest.approx(0.0)


def test_a_ticker_is_never_sold_and_bought_back_on_the_same_day():
    lots = [
        Lot("A", "AAA", date(2023, 1, 1), 100, 90.0),
        Lot("B", "BBB", date(2023, 1, 1), 33, 90.0),
        Lot("C", "CCC", date(2023, 1, 1), 20, 90.0),
    ]
    engine = Engine(
        lots,
        {"AAA": 101.0, "BBB": 97.0, "CCC": 103.0},
        {"AAA": 0.34, "BBB": 0.33, "CCC": 0.33},
        sale_date=SALE,
    )
    plan = engine.run()
    sells = {t.ticker for t in plan.trades if t.action == "SELL"}
    buys = {t.ticker for t in plan.trades if t.action == "BUY"}
    assert not (sells & buys)


def test_the_buy_leg_never_costs_more_than_the_sales_raise():
    lots = [
        Lot("A", "AAA", date(2023, 1, 1), 77, 90.0),
        Lot("B", "BBB", date(2023, 1, 1), 41, 90.0),
        Lot("C", "CCC", date(2023, 1, 1), 13, 90.0),
    ]
    engine = Engine(
        lots,
        {"AAA": 137.0, "BBB": 211.0, "CCC": 89.0},
        {"AAA": 0.2, "BBB": 0.3, "CCC": 0.5},
        sale_date=SALE,
    )
    plan = engine.run()
    assert plan.summary.cash_left_over >= -1e-9


def test_trades_are_whole_shares_with_the_sells_listed_first():
    _, plan = run("edge_case")
    actions = [t.action for t in plan.trades]
    assert actions == sorted(actions, reverse=True)  # SELL before BUY
    assert all(isinstance(t.shares, int) and t.shares > 0 for t in plan.trades)


def test_the_trade_date_changes_the_classification_and_so_the_tax():
    """The same portfolio, a month later, once the newer lot has aged past
    twelve months. Nothing else changes; the bill goes from Rs 1,000 to nil."""
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


# Input validation
def test_target_weights_must_add_up_to_one_hundred():
    lots = [Lot("A", "X", date(2023, 1, 1), 10, 100.0)]
    with pytest.raises(ValueError, match="add up to"):
        Engine(lots, {"X": 100.0}, {"X": 0.9}, sale_date=SALE).run()


def test_a_held_ticker_with_no_price_is_rejected():
    lots = [Lot("A", "X", date(2023, 1, 1), 10, 100.0)]
    with pytest.raises(ValueError, match="no current price"):
        Engine(lots, {}, {"X": 1.0}, sale_date=SALE).run()


def test_a_target_for_a_ticker_that_is_not_held_is_rejected():
    lots = [Lot("A", "X", date(2023, 1, 1), 10, 100.0)]
    with pytest.raises(ValueError, match="no lots are held"):
        Engine(lots, {"X": 100.0}, {"X": 0.5, "Y": 0.5}, sale_date=SALE).run()


def test_a_lot_bought_after_the_trade_date_is_rejected():
    """A year typed as 2027 instead of 2017 would otherwise price and classify
    as a perfectly ordinary short-term holding."""
    lots = [Lot("A", "X", date(2027, 1, 1), 10, 100.0)]
    with pytest.raises(ValueError, match="bought after the sale date"):
        Engine(lots, {"X": 100.0}, {"X": 1.0}, sale_date=SALE).run()


# CSV reading
LOTS_CSV = "ticker,buy_date,quantity,buy_price\nacme,2023-02-10,60,800\nACME,01/03/2024,40,950\n"


def test_lot_ids_are_generated_per_ticker_in_file_order():
    lots = ingest.parse_lots(LOTS_CSV)
    assert [l.lot_id for l in lots] == ["ACME-1", "ACME-2"]
    assert [l.ticker for l in lots] == ["ACME", "ACME"]


def test_both_date_formats_are_accepted():
    lots = ingest.parse_lots(LOTS_CSV)
    assert lots[0].buy_date == date(2023, 2, 10)
    assert lots[1].buy_date == date(2024, 3, 1)


def test_targets_are_converted_from_percent_to_fractions():
    targets = ingest.parse_targets("ticker,target_weight_pct\nACME,20\nNOVA,80\n")
    assert targets == {"ACME": 0.2, "NOVA": 0.8}


def test_a_bad_row_names_the_file_the_row_and_the_value():
    bad = "ticker,buy_date,quantity,buy_price\nACME,2023-02-10,sixty,800\n"
    with pytest.raises(ValueError, match="lots.csv row 2: quantity='sixty' is not a number"):
        ingest.parse_lots(bad)


def test_a_fractional_share_count_is_rejected_rather_than_rounded():
    bad = "ticker,buy_date,quantity,buy_price\nACME,2023-02-10,60.5,800\n"
    with pytest.raises(ValueError, match="whole number of shares"):
        ingest.parse_lots(bad)


def test_a_missing_column_is_named():
    with pytest.raises(ValueError, match="missing column\\(s\\) buy_price"):
        ingest.parse_lots("ticker,buy_date,quantity\nACME,2023-02-10,60\n")


def test_a_repeated_lot_id_is_rejected():
    bad = (
        "lot_id,ticker,buy_date,quantity,buy_price\n"
        "L1,ACME,2023-02-10,60,800\nL1,ACME,2024-02-10,40,900\n"
    )
    with pytest.raises(ValueError, match="repeats the id on row 2"):
        ingest.parse_lots(bad)


def test_a_repeated_price_row_is_rejected():
    with pytest.raises(ValueError, match="already priced"):
        ingest.parse_prices("ticker,current_price\nACME,100\nACME,200\n")


def test_a_file_with_a_header_but_no_data_is_rejected():
    with pytest.raises(ValueError, match="no data rows"):
        ingest.parse_prices("ticker,current_price\n")


def test_the_excel_byte_order_mark_does_not_hide_the_first_column():
    raw = "\ufeffticker,current_price\nACME,100\n".encode("utf-8")
    assert ingest.parse_prices(ingest.decode(raw, "prices.csv")) == {"ACME": 100.0}


def test_every_bundled_scenario_loads_and_runs():
    for name in (
        "edge_case", "all_ltcg", "at_target", "loss_offset",
        "exemption_split", "exemption_vs_loss", "loss_priority",
    ):
        _, plan = run(name)
        assert plan.certified_optimal is True


# Splits that only a solver finds
def test_a_long_term_loss_is_spent_only_down_to_the_exemption_line():
    """GAINCO leaves Rs 1,80,000 of long-term gain, so the first Rs 55,000 of
    long-term loss is worth 12.5% and every rupee after it is worth nothing:
    the exemption was always going to cover that gain. The plan stops the
    long-term lot at exactly 55 shares and takes the rest short-term."""
    _, plan = run("exemption_vs_loss")
    assert sold(plan)["H1"] == 55 and sold(plan)["H2"] == 45
    # Net long-term gain lands exactly on the exemption, with nothing taxable.
    assert plan.tax.ltcg - plan.tax.ltcl_used_against_ltcg == 125_000.0
    assert plan.tax.taxable_ltcg == 0.0
    assert plan.summary.total_tax == 8_400.0


def test_neither_shortcut_finds_the_exemption_line():
    _, best = run("exemption_vs_loss")
    _, ltfo = run("exemption_vs_loss", "ltfo")
    _, fifo = run("exemption_vs_loss", "fifo")
    assert best.summary.total_tax < ltfo.summary.total_tax < fifo.summary.total_tax
    # LTFO picks the right lot to start from and then cannot tell when to stop.
    assert sold(ltfo)["H1"] == 100


def test_the_short_term_loss_is_worth_more_only_while_it_has_gain_to_cancel():
    """The short-term loss saves 20% a rupee against the Rs 34,000 of short-term
    gain, beating the long-term loss's 12.5%. Once that gain is used up the
    surplus only carries over at 12.5%, and the bigger long-term loss is worth
    more per share, so the plan switches."""
    _, plan = run("loss_priority")
    assert sold(plan)["H2"] == 43 and sold(plan)["H1"] == 57
    assert plan.tax.taxable_stcg == 0.0
    assert plan.summary.total_tax == 14_700.0


def test_the_priority_between_the_two_losses_flips_part_way_through():
    _, best = run("loss_priority")
    _, ltfo = run("loss_priority", "ltfo")
    _, fifo = run("loss_priority", "fifo")
    assert best.summary.total_tax < ltfo.summary.total_tax < fifo.summary.total_tax
    # LTFO ranks the short-term loss first, correctly, and then takes all of it.
    assert sold(ltfo)["H2"] == 100
    # Both plans sell the same 100 shares; only the split differs.
    assert sum(sold(best).values()) == sum(sold(ltfo).values())


# The reasoning has to belong to the plan it is attached to
def test_each_method_explains_itself_in_its_own_terms():
    _, best = run("edge_case")
    _, fifo = run("edge_case", "fifo")
    _, ltfo = run("edge_case", "ltfo")

    fifo_l1 = next(s for s in fifo.lot_sales if s.lot_id == "L1")
    assert fifo_l1.reason.startswith("1st pick for ACME: the oldest lot still available")

    ltfo_l2 = next(s for s in ltfo.lot_sales if s.lot_id == "L2")
    assert ltfo_l2.reason.startswith(
        "1st pick for ACME: the lowest statutory tax per share"
    )
    assert "at Rs 10.00" in ltfo_l2.reason

    # Each rule works through a ticker in a fixed sequence, so the lots are listed
    # in the order it reached for them.
    assert [s.lot_id for s in ltfo.lot_sales] == ["L2", "L1"]
    assert [s.lot_id for s in fifo.lot_sales] == ["L1", "L2"]

    # The solver has no per-lot rule to quote: it settles every quantity at once,
    # so its justification is the priced swap rather than a ranking.
    best_l1 = next(s for s in best.lot_sales if s.lot_id == "L1")
    assert "pick for ACME" not in best_l1.reason
    assert "Rs 250.00 more in tax" in best_l1.reason


def test_the_reasoning_shows_the_fill_spilling_from_one_lot_into_the_next():
    """The case the brief turns on: a lot is emptied, the quantity still needed
    spills into the next one, and the shares beyond that stay put."""
    _, plan = run("edge_case", "fifo")
    by_id = {s.lot_id: s for s in plan.lot_sales}
    assert (
        "Supplies 60 of the 75 shares ACME must give up, taking the whole lot; "
        "15 still to find from the next lot." in by_id["L1"].reason
    )
    assert "Covers the remaining 15 from its 40, leaving 25 untouched." in by_id["L2"].reason
    # A later pick counts against what was still outstanding, not the original total.
    assert "of the 75" not in by_id["L2"].reason


def test_a_ranking_rule_is_not_judged_against_a_swap_it_never_weighed():
    """FIFO takes lots in date order. Telling it a swap would have been cheaper
    answers a question it never asked; the plan comparison says it better."""
    _, fifo = run("loss_priority", "fifo")
    assert all("not optimal" not in s.reason for s in fifo.lot_sales)
    assert all("would save" not in s.reason for s in fifo.lot_sales)
    # The priced swaps are still computed and still returned on the record.
    assert any(s.alternatives for s in fifo.lot_sales)


def test_the_solver_states_its_share_without_claiming_a_sequence():
    """It settles every quantity at once, so there is no "next lot" to spill to."""
    _, plan = run("edge_case")
    l1 = next(s for s in plan.lot_sales if s.lot_id == "L1")
    assert "Supplies 60 of the 75 shares ACME must give up" in l1.reason
    assert "still to find" not in l1.reason


def test_the_reasoning_does_not_repeat_what_the_structured_fields_already_say():
    """Buy date, holding period, cost and price are all fields on the record and
    all on screen beside it; the prose is for what they cannot say."""
    _, plan = run("edge_case")
    l1 = next(s for s in plan.lot_sales if s.lot_id == "L1")
    for restated in ("2023-02-10", "42 months held", "800.00/share", "60 of 60"):
        assert restated not in l1.reason


def test_a_ranking_rule_never_claims_the_swap_check_proves_it_optimal():
    """LTFO survives every single-lot swap on loss_priority and is still Rs 1,425
    off, because the cheaper plan is a partial split across two lots."""
    _, ltfo = run("loss_priority", "ltfo")
    _, best = run("loss_priority")
    assert all(a.tax_delta >= 0 for s in ltfo.lot_sales for a in s.alternatives)
    assert ltfo.summary.total_tax > best.summary.total_tax
    note = next(line for line in ltfo.reasoning if "substitution" in line)
    assert "not a proof" in note
    assert "cheapest available" not in note

    certified = next(line for line in best.reasoning if "substitution" in line)
    assert "corroborates the solver's guarantee" in certified
