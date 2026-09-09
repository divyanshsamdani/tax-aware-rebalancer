"""The front end's data path. Only the parts that do not touch Streamlit:
importing `ui` must be free of side effects, and every table it renders must be
built from the engine's own output."""

from datetime import date

import ui
from app.api import SAMPLES
from app.models import TaxConfig

EDGE = date(2026, 9, 6)
CFG = TaxConfig()


def plan(scenario: str = "edge_case", method: str = "optimal"):
    return ui.scenario_engine(scenario, EDGE).run(method, compare=False)


def test_the_picker_the_api_and_the_samples_directory_all_agree():
    from typing import get_args
    from app.api import Scenario

    on_disk = {p.name for p in SAMPLES.iterdir() if p.is_dir()}
    assert set(ui.SCENARIOS) == on_disk
    assert set(get_args(Scenario)) == on_disk


def test_all_three_plans_are_offered_and_reach_a_real_solver():
    assert [key for _, key in ui.PLANS] == ["optimal", "fifo", "ltfo"]
    for _, key in ui.PLANS:
        assert plan("loss_offset", key).method == key


def test_both_input_paths_reproduce_the_edge_case():
    def read(name):
        return (SAMPLES / "edge_case" / name).read_bytes()

    p = plan()
    assert {s.lot_id: s.shares_sold for s in p.lot_sales} == {"L1": 60, "L2": 15}
    assert p.summary.total_tax == 150.0
    uploaded = ui.uploaded_engine(
        read("lots.csv"), read("prices.csv"), read("targets.csv"), EDGE
    ).run("optimal", compare=False)
    assert uploaded.summary.total_tax == p.summary.total_tax


def test_the_tables_carry_what_the_screen_needs():
    holdings = ui.holdings_rows(ui.scenario_engine("edge_case", EDGE))
    assert [r["Lot"] for r in holdings] == ["L1", "L2", "N1"]
    assert [r["Holding period"] for r in holdings[:2]] == ["42 months", "5 months"]
    assert [r["Class"] for r in holdings] == ["LTCG", "STCG", "LTCG"]

    # buys and sells share one signed column
    assert {r["Ticker"]: r["Shares to trade"] for r in ui.requirement_rows(plan())} == {
        "ACME": -75,
        "NOVA": 300,
    }
    assert [(r["Action"], r["Ticker"], r["Shares"]) for r in ui.trade_rows(plan())] == [
        ("SELL", "ACME", 75),
        ("BUY", "NOVA", 300),
    ]

    sells = ui.sell_lot_rows(plan(), CFG)
    assert [r["Lot"] for r in sells] == ["L1", "L2"]
    assert sells[0]["Gain per share"] == 200.0 and sells[1]["Left"] == 25
    # the sell table drops what the holdings table already carries
    assert set(sells[0]).isdisjoint({"Buy date", "Held", "Buy price"})


def test_the_required_shares_are_the_same_whichever_lots_supply_them():
    rows = [ui.requirement_rows(plan("loss_offset", key)) for _, key in ui.PLANS]
    assert rows[0] == rows[1] == rows[2]
    assert any(r["Shares to trade"] < 0 for r in rows[0])


def test_tax_per_share_is_the_statutory_rate_that_ltfo_sorts_on():
    """It is also the wrong number, which is the point: on edge_case the
    long-term lot looks dearer per share and is in fact free."""
    assert ui.tax_per_share(200.0, "LT", CFG) == "25.00"
    assert ui.tax_per_share(50.0, "ST", CFG) == "10.00"
    assert plan("edge_case", "ltfo").summary.total_tax > plan().summary.total_tax


def test_a_short_term_loss_shows_both_rates_it_could_be_realised_at():
    """20% against short-term gain, 12.5% on whatever spills to the long-term
    side once that gain runs out."""
    assert ui.tax_per_share(-800.0, "ST", CFG) == "-160.00 / -100.00"
    for gain, bucket in [(-1000.0, "LT"), (900.0, "LT"), (340.0, "ST")]:
        assert "/" not in ui.tax_per_share(gain, bucket, CFG)


def test_net_gains_are_the_two_numbers_the_tax_depends_on():
    assert ui.net_gains(plan()) == (750.0, 12_000.0)
    net_st, net_lt = ui.net_gains(plan("loss_priority"))
    assert net_st < 0 < net_lt
    assert ui.signed_rupees(net_st).startswith("-₹")


def test_the_three_way_scenario_separates_all_three_methods():
    taxes = {key: plan("exemption_split", key).summary.total_tax for _, key in ui.PLANS}
    assert taxes["optimal"] < taxes["ltfo"] < taxes["fifo"]
    best = plan("exemption_split")
    assert [s for s in best.lot_sales if 0 < s.shares_sold < s.lot_quantity]


def test_a_ranking_rule_lists_its_lots_in_the_order_it_picked_them():
    assert [s.lot_id for s in plan("edge_case", "ltfo").lot_sales] == ["L2", "L1"]
    assert [s.lot_id for s in plan("edge_case", "fifo").lot_sales] == ["L1", "L2"]


def test_bad_upload_surfaces_the_engine_message_not_a_crash():
    def read(name):
        return (SAMPLES / "edge_case" / name).read_bytes()

    try:
        ui.uploaded_engine(
            b"ticker,quantity\nACME,10\n", read("prices.csv"), read("targets.csv"), EDGE
        )
    except ValueError as exc:
        assert "missing column" in str(exc)
    else:
        raise AssertionError("expected a ValueError naming the missing columns")


def test_the_tooltip_shows_the_ledger_with_signs_and_a_net():
    h1 = next(s for s in plan("loss_priority").lot_sales if s.lot_id == "H1")
    text = ui.alternative_help(h1.alternative)
    assert "one share more here and one fewer from lot `H2`" in text
    assert "**+₹80.00** — ₹400 less short-term loss cancelling short-term gain at 20.0%" in text
    assert "**-₹125.00**" in text
    assert text.endswith("**Net +₹5.00**")


def test_the_tooltip_passes_the_qualitative_note_through_untouched():
    """The fallback for an input whose ledger will not reconcile. It must not be
    dressed up as a sum."""
    from app.models import Alternative

    note = ui.alternative_help(Alternative("H2", "more", 5.0, [], "Rs 1 or Rs 2, who knows"))
    assert note == "₹1 or ₹2, who knows"


def test_prettify_only_swaps_the_currency_prefix():
    assert ui.prettify("a gain of Rs 12,000.00 in all") == "a gain of ₹12,000.00 in all"
