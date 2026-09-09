"""The HTTP surface: the demo routes, both input paths, and the error contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)
SAMPLES = Path("samples")
SCENARIOS = ["edge_case", "all_ltcg", "at_target", "loss_offset",
             "exemption_split", "exemption_vs_loss", "loss_priority"]

JSON_BODY = {
    "lots": [
        {"lot_id": "L1", "ticker": "ACME", "buy_date": "2023-02-10",
         "quantity": 60, "buy_price": 800},
        {"lot_id": "L2", "ticker": "ACME", "buy_date": "2026-04-01",
         "quantity": 40, "buy_price": 950},
        {"lot_id": "N1", "ticker": "NOVA", "buy_date": "2022-08-15",
         "quantity": 100, "buy_price": 200},
    ],
    "prices": {"ACME": 1000, "NOVA": 250},
    "targets_pct": {"ACME": 20, "NOVA": 80},
    "sale_date": "2026-09-06",
}


def upload(scenario: str = "edge_case", **form):
    files = {
        name + "_file": (f"{name}.csv", (SAMPLES / scenario / f"{name}.csv").read_bytes(), "text/csv")
        for name in ("lots", "prices", "targets")
    }
    return client.post(
        "/rebalance/upload", files=files, data={"sale_date": "2026-09-06", **form}
    )


def test_health_and_openapi():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/openapi.json").status_code == 200


def test_the_edge_case_demo_returns_the_whole_plan():
    body = client.get("/demo/edge_case").json()

    assert body["as_of"] == "2026-09-06"
    assert body["certified_optimal"] is True
    assert body["summary"]["total_tax"] == 150.0

    assert [(s["lot_id"], s["shares_sold"], s["remaining_shares"]) for s in body["lot_sales"]] == [
        ("L1", 60, 0),
        ("L2", 15, 25),
    ]
    assert [(t["action"], t["ticker"], t["shares"]) for t in body["trades"]] == [
        ("SELL", "ACME", 75),
        ("BUY", "NOVA", 300),
    ]
    assert {w["ticker"]: w["after_pct"] for w in body["weights"]} == {"ACME": 20.0, "NOVA": 80.0}

    first = body["lot_sales"][0]
    assert (first["holding"], first["classification"]) == ("42 months held", "LTCG")
    assert "Supplies 60 of the 75 shares ACME must give up" in first["reason"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_demo_scenario_responds(scenario):
    assert client.get(f"/demo/{scenario}").status_code == 200


def test_an_unknown_scenario_is_rejected():
    assert client.get("/demo/nonsense").status_code == 422


def test_the_three_input_paths_agree():
    assert upload().json() == client.get("/demo/edge_case").json()
    assert client.post("/rebalance", json=JSON_BODY).json() == upload().json()


def test_upload_without_a_trade_date_still_runs():
    assert upload(sale_date="").status_code == 200


def test_upload_can_ask_for_the_fifo_baseline():
    body = upload("all_ltcg", method="fifo").json()
    assert body["method"] == "fifo"
    assert body["certified_optimal"] is False
    assert body["summary"]["total_tax"] == 43437.5


def test_the_fifo_comparison_can_be_switched_off():
    body = upload(compare_with_fifo="false").json()
    assert body["summary"]["tax_if_fifo"] is None
    assert body["summary"]["saving_vs_fifo"] is None


def test_a_plan_that_loses_to_fifo_says_so():
    """LTFO costs Rs 400 on the edge case against FIFO's Rs 150. The comparison
    must not report that as a tie or a saving."""
    body = client.post("/rebalance", json=dict(JSON_BODY, method="ltfo")).json()
    assert body["summary"]["saving_vs_fifo"] == -250.0
    assert any("Rs 250.00 worse" in line for line in body["reasoning"])


def test_lot_ids_are_generated_when_the_json_omits_them():
    body = dict(JSON_BODY, lots=[{k: v for k, v in l.items() if k != "lot_id"}
                                 for l in JSON_BODY["lots"]])
    ids = [s["lot_id"] for s in client.post("/rebalance", json=body).json()["lot_sales"]]
    assert ids == ["ACME-1", "ACME-2"]


@pytest.mark.parametrize(
    "override,message",
    [
        ({"targets_pct": {"ACME": 20, "NOVA": 70}}, "add up to 90"),
        ({"prices": {"ACME": 1000}}, "no current price supplied for: NOVA"),
    ],
)
def test_bad_input_comes_back_as_422_with_the_reason(override, message):
    response = client.post("/rebalance", json=dict(JSON_BODY, **override))
    assert response.status_code == 422
    assert message in response.json()["detail"]


def test_a_malformed_csv_row_is_reported_with_its_row_number():
    files = {
        "lots_file": ("lots.csv", b"ticker,buy_date,quantity,buy_price\nACME,2023-02-10,x,800\n", "text/csv"),
        "prices_file": ("prices.csv", (SAMPLES / "edge_case" / "prices.csv").read_bytes(), "text/csv"),
        "targets_file": ("targets.csv", (SAMPLES / "edge_case" / "targets.csv").read_bytes(), "text/csv"),
    }
    response = client.post("/rebalance/upload", files=files, data={"sale_date": "2026-09-06"})
    assert response.status_code == 422
    assert "row 2: quantity='x' is not a number" in response.json()["detail"]


def test_a_negative_quantity_is_caught_by_the_schema():
    body = dict(JSON_BODY, lots=[dict(JSON_BODY["lots"][0], quantity=-5)])
    assert client.post("/rebalance", json=body).status_code == 422
