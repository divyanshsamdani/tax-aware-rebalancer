"""The Streamlit front end's data path.

Only the parts that do not touch Streamlit: importing `ui` must be free of side
effects, and the tables it renders must be built from the engine's own output.
"""

from datetime import date

import ui
from app.api import DEMO_SALE_DATE, SAMPLES
from app.models import TaxConfig

EDGE = date(2026, 9, 6)
CFG = TaxConfig()


def read(name: str) -> bytes:
    return (SAMPLES / "edge_case" / name).read_bytes()


def plan(scenario: str = "edge_case", method: str = "optimal"):
    return ui.scenario_engine(scenario, EDGE).run(method, compare=False)


def test_every_bundled_scenario_is_reachable_from_the_picker():
    for name in ui.SCENARIOS:
        assert (SAMPLES / name).is_dir()


def test_all_three_plans_are_offered_and_reach_a_real_solver():
    assert [key for _, key in ui.PLANS] == ["optimal", "fifo", "ltfo"]
    for _, key in ui.PLANS:
        assert plan("loss_offset", key).method == key


def test_scenario_path_reproduces_the_required_edge_case():
    p = plan()
    assert {s.lot_id: s.shares_sold for s in p.lot_sales} == {"L1": 60, "L2": 15}
    assert p.summary.total_tax == 150.0


def test_upload_path_matches_the_scenario_path():
    uploaded = ui.uploaded_engine(
        read("lots.csv"), read("prices.csv"), read("targets.csv"), EDGE
    ).run("optimal", compare=False)
    assert uploaded.summary.total_tax == plan().summary.total_tax


def test_the_required_shares_are_the_same_whichever_lots_supply_them():
    """The sell requirement comes from price and target alone, so the three plans
    must agree on it even when they disagree about which lots to use."""
    rows = [ui.requirement_rows(plan("loss_offset", key)) for _, key in ui.PLANS]
    assert rows[0] == rows[1] == rows[2]
    assert any(r["Shares to trade"] < 0 for r in rows[0])


def test_holdings_table_shows_every_lot_with_its_own_buy_date():
    engine = ui.scenario_engine("edge_case", EDGE)
    rows = ui.holdings_rows(engine)
    assert [r["Lot"] for r in rows] == ["L1", "L2", "N1"]
    assert [r["Buy date"] for r in rows[:2]] == ["2023-02-10", "2026-04-01"]
    assert rows[0]["Market value"] == 60 * 1000


def test_the_requirement_table_carries_buys_and_sells_in_one_signed_column():
    rows = {r["Ticker"]: r["Shares to trade"] for r in ui.requirement_rows(plan())}
    assert rows == {"ACME": -75, "NOVA": 300}


def test_trades_table_is_one_row_per_ticker():
    rows = ui.trade_rows(plan())
    assert [(r["Action"], r["Ticker"], r["Shares"]) for r in rows] == [
        ("SELL", "ACME", 75),
        ("BUY", "NOVA", 300),
    ]


def test_the_sell_leg_is_broken_out_lot_by_lot_with_signed_gains():
    rows = ui.sell_lot_rows(plan(), CFG)
    assert [r["Lot"] for r in rows] == ["L1", "L2"]
    assert sum(r["Shares to sell"] for r in rows) == 75
    assert rows[0]["Gain per share"] == 200.0 and rows[1]["Left"] == 25
    assert (rows[0]["Term"], rows[1]["Term"]) == ("🟢 LT", "🟠 ST")


def test_the_sell_plan_drops_what_the_holdings_table_already_carries():
    assert set(ui.sell_lot_rows(plan(), CFG)[0]).isdisjoint(
        {"Buy date", "Held", "Buy price"}
    )


def test_tax_per_share_is_the_statutory_rate_and_is_what_ltfo_sorts_on():
    """It is also the wrong number, which is the point: on edge_case the
    long-term lot looks dearer per share and is in fact free."""
    assert ui.tax_per_share(200.0, "LT", CFG) == "25.00"
    assert ui.tax_per_share(50.0, "ST", CFG) == "10.00"
    # LTFO sorts on exactly this, reaches for the cheaper-looking lot, and loses.
    assert plan("edge_case", "ltfo").summary.total_tax > plan().summary.total_tax


def test_a_short_term_loss_shows_both_rates_it_could_be_realised_at():
    """20% against short-term gain, 12.5% on whatever spills to the long-term
    side once that gain runs out. Which one applies is settled by the rest of
    the plan, which is precisely what a per-lot ranking cannot see."""
    assert ui.tax_per_share(-800.0, "ST", CFG) == "-160.00 / -100.00"


def test_nothing_else_is_ambiguous_enough_to_need_two_figures():
    for gain, bucket in [(-1000.0, "LT"), (900.0, "LT"), (340.0, "ST")]:
        assert "/" not in ui.tax_per_share(gain, bucket, CFG)


def test_the_loss_priority_table_carries_the_three_numbers_that_decide_it():
    """160 against short-term gain, 125 for the long-term loss, 100 for
    short-term loss that has spilled over. Sorted, that is the whole plan."""
    rows = {r["Lot"]: r for r in ui.sell_lot_rows(plan("loss_priority"), CFG)}
    assert rows["H2"]["Tax per share"] == "-160.00 / -100.00"
    assert rows["H1"]["Tax per share"] == "-125.00"


def test_net_gains_are_the_two_numbers_the_tax_depends_on():
    net_st, net_lt = ui.net_gains(plan())
    assert (net_st, net_lt) == (750.0, 12_000.0)


def test_net_gains_go_negative_when_losses_outweigh_gains():
    """loss_priority realises slightly more short-term loss than gain, and a net
    long-term loss on the long-term side of the harvest."""
    net_st, net_lt = ui.net_gains(plan("loss_priority"))
    assert net_st < 0 < net_lt
    assert ui.signed_rupees(net_st).startswith("-₹")


def test_a_loss_lot_carries_a_negative_gain_per_share():
    rows = ui.sell_lot_rows(plan("loss_offset"), CFG)
    assert any(r["Gain per share"] < 0 for r in rows)


def test_holdings_table_marks_long_and_short_term_lots():
    rows = ui.holdings_rows(ui.scenario_engine("edge_case", EDGE))
    assert rows[0]["Holding"] == "🟢 Long-term"
    assert rows[1]["Holding"] == "🟠 Short-term"


def test_only_the_plan_that_carries_a_guarantee_is_green():
    """The colour marks the guarantee, not the outcome: FIFO ties the optimum on
    edge_case and still stays red, because it cannot tell you it has tied."""
    assert ui.PLAN_SHADE == {"optimal": "green", "fifo": "red", "ltfo": "red"}
    assert plan("edge_case", "fifo").summary.total_tax == 150.0
    assert not plan("edge_case", "fifo").certified_optimal


def test_each_method_gets_its_own_explanation():
    notes = ui.METHOD_NOTE
    assert set(notes) == {key for _, key in ui.PLANS}
    assert len(set(notes.values())) == 3, "the three notes must say different things"
    assert notes["fifo"].startswith("FIFO") and notes["ltfo"].startswith("LTFO")


def test_the_three_way_scenario_separates_all_three_methods():
    """optimal < ltfo < fifo, and the cheapest answer splits a lot part-way."""
    taxes = {key: plan("exemption_split", key).summary.total_tax for _, key in ui.PLANS}
    assert taxes["optimal"] < taxes["ltfo"] < taxes["fifo"]
    best = plan("exemption_split", "optimal")
    partial = [s for s in best.lot_sales if 0 < s.shares_sold < s.lot_quantity]
    assert partial, "the point of this scenario is that the answer is not a whole lot"


def test_nothing_in_the_rendered_text_claims_the_plan_has_been_executed():
    """This is a planner, not a trade blotter."""
    p = plan()
    text = " ".join(p.reasoning + [s.reason for s in p.lot_sales]).lower()
    for word in ("sold ", "bought back", "were sold"):
        assert word not in text, f"{word!r} reads as though the plan already ran"


def test_bad_upload_surfaces_the_engine_message_not_a_crash():
    try:
        ui.uploaded_engine(
            b"ticker,quantity\nACME,10\n", read("prices.csv"), read("targets.csv"), EDGE
        )
    except ValueError as exc:
        assert "missing column" in str(exc)
    else:
        raise AssertionError("expected a ValueError naming the missing columns")


def test_prettify_only_swaps_the_currency_prefix():
    assert ui.prettify("a gain of Rs 12,000.00 in all") == "a gain of ₹12,000.00 in all"


def test_demo_date_is_shared_with_the_api_rather_than_copied():
    assert ui.DEMO_SALE_DATE is DEMO_SALE_DATE
