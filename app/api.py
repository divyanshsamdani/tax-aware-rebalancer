"""The HTTP layer: parse input, call `Engine`, return the plan. The response
shapes are the engine's own types, so there is no second data model to keep in
step. Start at GET /demo/edge_case, or /docs.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import ingest
from .engine import Engine
from .models import Lot, LotSale, Plan, Summary, TaxBreakdown, Trade, WeightRow
from .optimizer import Method

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
Scenario = Literal[
    "edge_case", "all_ltcg", "at_target", "loss_offset", "exemption_split",
    "exemption_vs_loss", "loss_priority",
]

# Pinned: the scenarios turn on particular lots being short-term, which a real
# "today" would quietly undo once those lots aged past twelve months.
DEMO_SALE_DATE = date(2026, 9, 6)

app = FastAPI(
    title="Tax-Aware Rebalancing Engine",
    version="1.0.0",
    description=(
        "Rebalances an Indian listed-equity portfolio to its target weights and "
        "picks which specific lots to sell so that capital gains tax is as small "
        "as possible. Short-term gains 20%, long-term 12.5% above a Rs 1,25,000 "
        "yearly exemption. Start at GET /demo/edge_case."
    ),
)


@app.exception_handler(ValueError)
async def _value_error(_, exc: ValueError) -> JSONResponse:
    """Bad input comes back as a 422 with the engine's own message, not a 500."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _round(value):
    """Trim floating-point dust before the plan goes out as JSON."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


class LotIn(BaseModel):
    ticker: str
    buy_date: date
    quantity: int = Field(gt=0)
    buy_price: float = Field(gt=0)
    lot_id: Optional[str] = Field(
        default=None, description="Generated as {TICKER}-{n} in list order if omitted"
    )


class RebalanceRequest(BaseModel):
    lots: list[LotIn]
    prices: dict[str, Annotated[float, Field(gt=0)]] = Field(
        description="Ticker -> current market price"
    )
    targets_pct: dict[str, Annotated[float, Field(ge=0)]] = Field(
        description="Ticker -> target weight in percent. Must add up to 100."
    )
    sale_date: Optional[date] = Field(
        default=None, description="Trade date. Defaults to today."
    )
    method: Method = "optimal"
    compare_with_fifo: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "method": "optimal",
            }
        }
    }


class PlanResponse(BaseModel):
    """The engine's own types, with the trade date added."""

    as_of: date
    method: str
    certified_optimal: bool
    summary: Summary
    reasoning: list[str]
    trades: list[Trade]
    lot_sales: list[LotSale]
    tax: TaxBreakdown
    weights: list[WeightRow]


def _respond(engine: Engine, plan: Plan) -> dict:
    return _round({"as_of": engine.sale_date, **asdict(plan)})


def _to_lots(rows: list[LotIn]) -> list[Lot]:
    lots: list[Lot] = []
    counter: dict[str, int] = {}
    seen: set[str] = set()
    for row in rows:
        ticker = row.ticker.strip().upper()
        counter[ticker] = counter.get(ticker, 0) + 1
        lot_id = row.lot_id or ingest.default_lot_id(ticker, counter[ticker])
        if lot_id in seen:
            raise ValueError(f"duplicate lot_id: {lot_id}")
        seen.add(lot_id)
        lots.append(
            Lot(
                lot_id=lot_id,
                ticker=ticker,
                buy_date=row.buy_date,
                quantity=row.quantity,
                buy_price=row.buy_price,
            )
        )
    return lots


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/demo/{scenario}", response_model=PlanResponse, tags=["demo"])
def demo(scenario: Scenario) -> dict:
    """Run one of the bundled scenarios. No input needed. `edge_case` is the one
    the brief requires: 60 long-term ACME shares and 40 short-term, 75 must go,
    so the long-term lot goes whole and 15 spill into the short-term one."""
    lots, prices, targets = ingest.load_scenario(SAMPLES / scenario)
    engine = Engine(lots, prices, targets, sale_date=DEMO_SALE_DATE)
    return _respond(engine, engine.run("optimal"))


@app.post("/rebalance/upload", response_model=PlanResponse, tags=["rebalance"])
async def rebalance_upload(
    lots_file: UploadFile = File(..., description="ticker,buy_date,quantity,buy_price"),
    prices_file: UploadFile = File(..., description="ticker,current_price"),
    targets_file: UploadFile = File(..., description="ticker,target_weight_pct"),
    sale_date: str = Form("", description="YYYY-MM-DD or DD/MM/YYYY. Blank means today."),
    method: Method = Form("optimal"),
    compare_with_fifo: bool = Form(True),
) -> dict:
    """Upload the three CSVs, as in samples/edge_case/{lots,prices,targets}.csv."""
    engine = Engine(
        lots=ingest.parse_lots(
            ingest.decode(await lots_file.read(), "lots"),
            lots_file.filename or "lots.csv",
        ),
        prices=ingest.parse_prices(
            ingest.decode(await prices_file.read(), "prices"),
            prices_file.filename or "prices.csv",
        ),
        targets=ingest.parse_targets(
            ingest.decode(await targets_file.read(), "targets"),
            targets_file.filename or "targets.csv",
        ),
        sale_date=ingest.parse_date(sale_date, "sale_date") if sale_date.strip() else None,
    )
    return _respond(engine, engine.run(method, compare=compare_with_fifo))


@app.post("/rebalance", response_model=PlanResponse, tags=["rebalance"])
def rebalance(body: RebalanceRequest) -> dict:
    """The same thing as a JSON body. Target weights are percentages adding to 100."""
    engine = Engine(
        lots=_to_lots(body.lots),
        prices={t.strip().upper(): p for t, p in body.prices.items()},
        targets={t.strip().upper(): w / 100.0 for t, w in body.targets_pct.items()},
        sale_date=body.sale_date,
    )
    return _respond(engine, engine.run(body.method, compare=body.compare_with_fifo))
