"""Streamlit front end.

Thin by design: it collects input, calls `Engine`, and renders the plan. Every
number and every sentence on screen comes from the engine - the UI computes
nothing of its own, so there is no second implementation to disagree with the
API.

    streamlit run ui.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import streamlit as st

from app import ingest
from app.api import DEMO_SALE_DATE, SAMPLES
from app.engine import Engine
from app.models import Plan

SCENARIOS = {
    "edge_case": "The required edge case: 75 shares must go, which empties the "
    "long-term lot and forces a partial sale into the short-term one.",
    "loss_offset": "A short-term loss sheltering a gain elsewhere. Oldest-first "
    "takes the long-term loss lot and wastes it.",
    "all_ltcg": "A plain long-term rebalance that fits inside the exemption.",
    "at_target": "Already at target, so there is nothing to do.",
}

METHODS = {"Tax-minimising": "exact", "Oldest first (FIFO)": "fifo"}

REPO_SAMPLES = (
    "https://github.com/divyanshsamdani/tax-aware-rebalancer/tree/main/samples"
)


def scenario_engine(name: str, sale_date: date) -> Engine:
    lots, prices, targets = ingest.load_scenario(SAMPLES / name)
    return Engine(lots, prices, targets, sale_date=sale_date)


def uploaded_engine(lots: bytes, prices: bytes, targets: bytes, sale_date: date) -> Engine:
    return Engine(
        lots=ingest.parse_lots(ingest.decode(lots, "lots"), "lots.csv"),
        prices=ingest.parse_prices(ingest.decode(prices, "prices"), "prices.csv"),
        targets=ingest.parse_targets(ingest.decode(targets, "targets"), "targets.csv"),
        sale_date=sale_date,
    )


def rupees(x: float) -> str:
    return f"₹{x:,.2f}"


def prettify(text: str) -> str:
    """The engine writes 'Rs' to keep its output plain ASCII; a screen can do better."""
    return text.replace("Rs ", "₹")


def badge(classification: str) -> str:
    colour = "green" if classification.startswith("LT") else "orange"
    return f":{colour}-background[**{classification}**]"


def collect_input():
    """The sidebar. Returns (engine, method, compare) with engine None until the
    input is complete."""
    st.sidebar.title("🧾 Rebalancer")
    st.sidebar.caption(
        "Indian listed equity. Short-term 20%, long-term 12.5% above a "
        "₹1,25,000 yearly exemption."
    )
    st.sidebar.divider()

    source = st.sidebar.radio("Portfolio", ["Bundled scenario", "Upload CSVs"])
    engine = None
    error = None

    if source == "Bundled scenario":
        name = st.sidebar.selectbox("Scenario", list(SCENARIOS))
        st.sidebar.caption(SCENARIOS[name])
        st.sidebar.caption(f"[See this scenario's input CSVs ↗]({REPO_SAMPLES}/{name})")
        sale_date = st.sidebar.date_input("Trade date", value=DEMO_SALE_DATE)
        st.sidebar.caption(
            "Pinned to 2026-09-06: the scenarios turn on particular lots still "
            "being short-term."
        )
        try:
            engine = scenario_engine(name, sale_date)
        except ValueError as exc:
            error = str(exc)
    else:
        st.sidebar.caption(
            f"Need a set to try? [Download the sample CSVs ↗]({REPO_SAMPLES}/edge_case)"
        )
        lots = st.sidebar.file_uploader("lots.csv", type="csv")
        prices = st.sidebar.file_uploader("prices.csv", type="csv")
        targets = st.sidebar.file_uploader("targets.csv", type="csv")
        sale_date = st.sidebar.date_input("Trade date", value=date.today())
        if lots and prices and targets:
            try:
                engine = uploaded_engine(
                    lots.getvalue(), prices.getvalue(), targets.getvalue(), sale_date
                )
            except ValueError as exc:
                error = str(exc)

    st.sidebar.divider()
    method = METHODS[st.sidebar.radio("Lot selection", list(METHODS))]
    compare = st.sidebar.checkbox("Compare against FIFO", value=True)

    return engine, method, compare, error


def render(plan: Plan, sale_date: date) -> None:
    st.title("Tax-aware rebalancing plan")
    st.caption(f"Trade date {sale_date} · lot selection: {plan.method}")

    cols = st.columns(4)
    cols[0].metric("Total tax", rupees(plan.summary.total_tax))
    if plan.summary.tax_if_fifo is None:
        cols[1].metric("If sold oldest-first", "not compared")
        cols[2].metric("Saving", "—")
    else:
        cols[1].metric("If sold oldest-first", rupees(plan.summary.tax_if_fifo))
        cols[2].metric("Saving", rupees(plan.summary.saving_vs_fifo))
    cols[3].metric(
        "Shares sold / bought",
        f"{plan.summary.shares_sold} / {plan.summary.shares_bought}",
    )

    if plan.certified_optimal:
        st.success(
            "Proved optimal — the solver certified that no cheaper legal plan exists.",
            icon="✅",
        )
    else:
        st.warning(
            "Oldest-first baseline. This plan is not claimed to be the cheapest.",
            icon="⚠️",
        )

    st.subheader("What the engine did")
    for line in plan.reasoning:
        st.markdown(f"- {prettify(line)}")

    st.subheader("Trades")
    if plan.trades:
        st.dataframe(
            [
                {
                    "Action": t.action,
                    "Ticker": t.ticker,
                    "Shares": t.shares,
                    "Price": t.price,
                    "Value": t.value,
                }
                for t in plan.trades
            ],
            hide_index=True,
        )
    else:
        st.info("No trades — every ticker is already at its target weight.")

    if plan.lot_sales:
        st.subheader("Which lots were sold, and why")
        for sale in plan.lot_sales:
            with st.container(border=True):
                st.markdown(
                    f"**{sale.ticker} · lot `{sale.lot_id}`** &nbsp; "
                    f"{badge(sale.classification)} &nbsp; bought {sale.buy_date} "
                    f"· {sale.holding}"
                )
                row = st.columns(4)
                row[0].metric("Sold", f"{sale.shares_sold} of {sale.lot_quantity}")
                row[1].metric(
                    "Cost → price",
                    f"₹{sale.cost_basis_per_share:,.0f} → ₹{sale.price:,.0f}",
                )
                row[2].metric("Realised gain", rupees(sale.realized_gain))
                row[3].metric("Next rupee costs", f"{sale.marginal_tax_rate:.2%}")

                st.markdown(prettify(sale.reason))

                if sale.remaining_shares:
                    st.info(
                        f"{sale.remaining_shares} shares stay in this lot, keeping "
                        f"their original buy date of {sale.buy_date}.",
                        icon="🔒",
                    )
                if sale.alternatives:
                    with st.expander(
                        f"What selling a different lot would have cost "
                        f"({len(sale.alternatives)} priced)"
                    ):
                        st.dataframe(
                            [
                                {
                                    "Lot": a.lot_id,
                                    "Bought": str(a.buy_date),
                                    "Class": a.classification,
                                    "Shares": a.shares,
                                    "Tax difference": a.tax_delta,
                                }
                                for a in sale.alternatives
                            ],
                            hide_index=True,
                        )
                        st.caption(
                            "Each figure is the tax recomputed on that swap, not a "
                            "label. Nothing negative means nothing cheaper exists."
                        )

    st.subheader("Weights")
    st.dataframe(
        [
            {
                "Ticker": w.ticker,
                "Target %": round(w.target_pct, 2),
                "Before %": round(w.before_pct, 2),
                "After %": round(w.after_pct, 2),
            }
            for w in plan.weights
        ],
        hide_index=True,
    )

    with st.expander("The full set-off calculation"):
        st.dataframe(
            [{"Figure": k.replace("_", " "), "Amount": v} for k, v in asdict(plan.tax).items()],
            hide_index=True,
        )

    with st.expander("Raw API response"):
        st.json(json.loads(json.dumps(asdict(plan), default=str)))


def main() -> None:
    st.set_page_config(
        page_title="Tax-Aware Rebalancing Engine", page_icon="🧾", layout="wide"
    )
    engine, method, compare, error = collect_input()

    if error:
        st.error(error, icon="🚫")
        return
    if engine is None:
        st.title("Tax-aware rebalancing engine")
        st.markdown(
            "Pick a **bundled scenario** in the sidebar, or upload your own "
            "`lots.csv`, `prices.csv` and `targets.csv`. Samples to try are in "
            "`samples/edge_case/`."
        )
        return

    try:
        plan = engine.run(method, compare=compare)
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return

    render(plan, engine.sale_date)


if __name__ == "__main__":
    main()
