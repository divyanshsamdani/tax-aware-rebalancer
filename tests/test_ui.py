"""The Streamlit front end's data path.

Only the parts that do not touch Streamlit: importing `ui` must be free of side
effects, and the two input paths must produce the same plan the API returns.
"""

from datetime import date

import ui
from app.api import DEMO_SALE_DATE, SAMPLES

EDGE = date(2026, 9, 6)


def read(name: str) -> bytes:
    return (SAMPLES / "edge_case" / name).read_bytes()


def test_every_bundled_scenario_is_reachable_from_the_picker():
    for name in ui.SCENARIOS:
        assert (SAMPLES / name).is_dir()


def test_scenario_path_reproduces_the_required_edge_case():
    plan = ui.scenario_engine("edge_case", EDGE).run("exact")
    assert {s.lot_id: s.shares_sold for s in plan.lot_sales} == {"L1": 60, "L2": 15}
    assert plan.summary.total_tax == 150.0


def test_upload_path_matches_the_scenario_path():
    uploaded = ui.uploaded_engine(
        read("lots.csv"), read("prices.csv"), read("targets.csv"), EDGE
    ).run("exact")
    bundled = ui.scenario_engine("edge_case", EDGE).run("exact")
    assert uploaded.summary.total_tax == bundled.summary.total_tax
    assert [s.shares_sold for s in uploaded.lot_sales] == [
        s.shares_sold for s in bundled.lot_sales
    ]


def test_the_method_selector_reaches_both_solvers():
    engine = ui.scenario_engine("loss_offset", EDGE)
    exact = engine.run(ui.METHODS["Tax-minimising"])
    fifo = engine.run(ui.METHODS["Oldest first (FIFO)"])
    assert exact.certified_optimal and not fifo.certified_optimal
    assert exact.summary.total_tax < fifo.summary.total_tax


def test_bad_upload_surfaces_the_engine_message_not_a_crash():
    try:
        ui.uploaded_engine(b"ticker,quantity\nACME,10\n", read("prices.csv"), read("targets.csv"), EDGE)
    except ValueError as exc:
        assert "missing column" in str(exc)
    else:
        raise AssertionError("expected a ValueError naming the missing columns")


def test_prettify_only_swaps_the_currency_prefix():
    assert ui.prettify("a gain of Rs 12,000.00 in all") == "a gain of ₹12,000.00 in all"


def test_demo_date_is_shared_with_the_api_rather_than_copied():
    assert ui.DEMO_SALE_DATE is DEMO_SALE_DATE
