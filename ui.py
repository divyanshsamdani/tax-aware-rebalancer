"""Streamlit front end.

Thin by design: it collects input, calls `Engine` once per method, and renders
what comes back. It computes nothing of its own, so there is no second
implementation that could drift away from the API.

    streamlit run ui.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date

import streamlit as st

from app import ingest
from app import tax as taxrules
from app.api import DEMO_SALE_DATE, SAMPLES
from app.engine import Engine
from app.models import Plan, TaxConfig

SCENARIOS = {
    "edge_case": "The required edge case: 75 shares must go, which empties the "
    "long-term lot and forces a partial sale into the short-term one.",
    "exemption_split": "All three methods disagree. The cheapest answer stops "
    "part-way through a lot, exactly where the exemption runs out.",
    "exemption_vs_loss": "A long-term loss is worth nothing once the exemption "
    "already covers the gain. The best plan spends it down to that line exactly, "
    "then switches to the short-term loss.",
    "loss_priority": "The short-term loss saves more per share — but only until "
    "the short-term gain it offsets runs out. The best plan switches part-way.",
    "loss_offset": "A short-term loss sheltering a gain elsewhere.",
    "all_ltcg": "A plain long-term rebalance that fits inside the exemption.",
    "at_target": "Already at target, so there is nothing to do.",
}

PLANS = [
    ("Optimal", "optimal"),
    ("FIFO (oldest first)", "fifo"),
    ("LTFO (least tax first out)", "ltfo"),
]

SHORT_NAME = {"optimal": "Optimal", "fifo": "FIFO", "ltfo": "LTFO"}

# Green marks the plan that carries a guarantee, not the one that happens to win
# on this input. FIFO and LTFO stay red even where they tie, because neither can
# tell you it has tied.
PLAN_SHADE = {"optimal": "green", "fifo": "red", "ltfo": "red"}

SHADES = {
    "green": ("#16a34a", "rgba(22, 163, 74, 0.08)", "rgba(22, 163, 74, 0.22)"),
    "red": ("#dc2626", "rgba(220, 38, 38, 0.08)", "rgba(220, 38, 38, 0.22)"),
}


METHOD_NOTE = {
    "optimal": "Solved as a mixed-integer linear program over every lot at once.",
    "fifo": "FIFO makes no tax decision at all — the order is fixed by the buy "
    "dates. Whatever it costs here is a coincidence.",
    "ltfo": "LTFO does rank on tax, but it charges every lot its statutory rate. "
    "The rate that actually applies depends on how much exemption is left and "
    "which losses are absorbing gains elsewhere, so it ranks on the wrong number.",
}


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


def signed_rupees(x: float) -> str:
    return f"-₹{abs(x):,.2f}" if x < 0 else f"₹{x:,.2f}"


def net_gains(plan: Plan) -> tuple[float, float]:
    """(net short-term, net long-term) realised gain, signed. These two numbers
    are the entire tax base: the whole calculation is a function of just them."""
    return (plan.tax.stcg - plan.tax.stcl, plan.tax.ltcg - plan.tax.ltcl)


def prettify(text: str) -> str:
    """The engine writes 'Rs' to keep its output plain ASCII; a screen can do better."""
    return text.replace("Rs ", "₹")


def marker(bucket: str) -> str:
    """Long-term green, short-term orange, everywhere a lot is listed."""
    return "🟢" if bucket == "LT" else "🟠"


def holding_cell(bucket: str) -> str:
    return f"{marker(bucket)} {'Long-term' if bucket == 'LT' else 'Short-term'}"


def term_cell(bucket: str) -> str:
    """The dense form, for tables that are already carrying a lot of numbers."""
    return f"{marker(bucket)} {bucket}"


def holdings_rows(engine: Engine) -> list[dict]:
    rows = []
    for lot in engine.lots:
        price = engine.prices[lot.ticker]
        priced = taxrules.price_lot(lot, price, engine.sale_date, engine.cfg)
        rows.append(
            {
                "Lot": lot.lot_id,
                "Ticker": lot.ticker,
                "Buy date": str(lot.buy_date),
                "Holding": holding_cell(priced.bucket),
                "Held": taxrules.holding_label(lot.buy_date, engine.sale_date),
                "Quantity": lot.quantity,
                "Buy price": lot.buy_price,
                "Current price": price,
                "Gain / share": priced.gain_per_share,
                "Market value": lot.quantity * price,
            }
        )
    return rows


def requirement_rows(plan: Plan) -> list[dict]:
    """Current weight against target, and the shares that implies per ticker.

    One signed column rather than two: a sale is negative, a purchase positive,
    so a ticker's whole instruction sits on its own row.
    """
    trades = {t.ticker: t for t in plan.trades}
    rows = []
    for w in plan.weights:
        trade = trades.get(w.ticker)
        shares = 0
        if trade is not None:
            shares = -trade.shares if trade.action == "SELL" else trade.shares
        rows.append(
            {
                "Ticker": w.ticker,
                "Current %": round(w.before_pct, 2),
                "Target %": round(w.target_pct, 2),
                "Shares to trade": shares,
            }
        )
    return rows


def trade_rows(plan: Plan) -> list[dict]:
    """One row per ticker. Which lots supply the shares is the next table down."""
    return [
        {
            "Action": t.action,
            "Ticker": t.ticker,
            "Shares": t.shares,
            "Price": t.price,
            "Value": t.value,
        }
        for t in plan.trades
    ]


def tax_per_share(gain: float, bucket: str, cfg: TaxConfig) -> str:
    """What one share of this lot does to the bill, at statutory rates.

    A short-term loss has two answers, so it gets both: it saves 20% against
    short-term gain, and only 12.5% on whatever spills to the long-term side
    once that gain runs out. Which one applies is decided by the rest of the
    plan, which is exactly what a per-lot ranking cannot see.
    """
    if bucket == "ST" and gain < 0:
        return f"{gain * cfg.stcg_rate:,.2f} / {gain * cfg.ltcg_rate:,.2f}"
    rate = cfg.stcg_rate if bucket == "ST" else cfg.ltcg_rate
    return f"{gain * rate:,.2f}"


def sell_lot_rows(plan: Plan, cfg: TaxConfig) -> list[dict]:
    """The sell leg, lot by lot. Gains are signed: a loss is negative."""
    rows = []
    for s in plan.lot_sales:
        bucket = "LT" if s.classification.startswith("LT") else "ST"
        rows.append(
            {
                "Lot": s.lot_id,
                "Ticker": s.ticker,
                "Term": term_cell(bucket),
                "Shares to sell": s.shares_sold,
                "Of lot": s.lot_quantity,
                "Left": s.remaining_shares,
                "Gain per share": s.gain_per_share,
                "Tax per share": tax_per_share(s.gain_per_share, bucket, cfg),
                "Gain to realise": s.realized_gain,
            }
        )
    return rows


def inject_card_css(shades: dict[str, str], selected: str) -> None:
    rules = []
    for key, shade in shades.items():
        border, light, strong = SHADES[shade]
        rules.append(
            f".st-key-plan-{key} {{"
            f" background: {strong if key == selected else light};"
            f" border: 1px solid {border};"
            f" border-radius: 0.6rem;"
            f" padding: 0.9rem 1rem 0.6rem 1rem; }}"
        )
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def collect_input():
    """The sidebar. Returns (engine, error); engine is None until input is complete."""
    st.sidebar.title("🧾 Rebalancing planner")
    st.sidebar.caption(
        "Indian listed equity. Short-term 20%, long-term 12.5% above a "
        "₹1,25,000 yearly exemption."
    )
    st.sidebar.divider()

    source = st.sidebar.radio("Portfolio", ["Bundled scenario", "Upload CSVs"])

    if source == "Bundled scenario":
        name = st.sidebar.selectbox(
            "Scenario",
            list(SCENARIOS),
            index=None,
            placeholder="Choose a scenario…",
        )
        if name is None:
            return None, None
        st.sidebar.caption(SCENARIOS[name])
        st.sidebar.caption(f"[See this scenario's input CSVs ↗]({REPO_SAMPLES}/{name})")
        sale_date = st.sidebar.date_input("Trade date", value=DEMO_SALE_DATE)
        st.sidebar.caption(
            "Pinned to 2026-09-06: the scenarios turn on particular lots still "
            "being short-term."
        )
        try:
            return scenario_engine(name, sale_date), None
        except ValueError as exc:
            return None, str(exc)

    st.sidebar.caption(
        f"Need a set to try? [Download the sample CSVs ↗]({REPO_SAMPLES}/edge_case)"
    )
    lots = st.sidebar.file_uploader("lots.csv", type="csv")
    prices = st.sidebar.file_uploader("prices.csv", type="csv")
    targets = st.sidebar.file_uploader("targets.csv", type="csv")
    sale_date = st.sidebar.date_input("Trade date", value=date.today())
    if not (lots and prices and targets):
        return None, None
    try:
        return uploaded_engine(
            lots.getvalue(), prices.getvalue(), targets.getvalue(), sale_date
        ), None
    except ValueError as exc:
        return None, str(exc)


def render_welcome() -> None:
    st.markdown(
        "Rebalancing to target weights is arithmetic. The hard part is that the "
        "tax depends on **which specific lots** you sell, because every lot has "
        "its own holding period and cost basis. This planner chooses the lots, "
        "and proves the choice is the cheapest available."
    )
    st.markdown(
        "**Choose a scenario in the sidebar**, or upload your own `lots.csv`, "
        "`prices.csv` and `targets.csv`."
    )
    left, right = st.columns(2)
    left.markdown(
        "**Rates applied**  \n"
        "🟠 Short-term, held 12 months or less — **20%**, no exemption  \n"
        "🟢 Long-term, held more than 12 months — **12.5%** above **₹1,25,000** a year"
    )
    right.markdown(
        "**Three plans, side by side**  \n"
        "Optimal — a linear program over every lot at once  \n"
        "FIFO (oldest first) — what a demat account does by default  \n"
        "LTFO (least tax first out) — the obvious ranking rule"
    )
    st.caption(
        "Rates from the Income Tax Department, set by the Finance (No. 2) Act 2024 "
        "for transfers on or after 23 July 2024."
    )


def render_input(engine: Engine, plan: Plan) -> None:
    st.subheader("The portfolio")
    st.caption(
        "Every lot, with its own buy date and cost basis. 🟢 long-term, "
        "🟠 short-term, as at the trade date."
    )
    st.dataframe(holdings_rows(engine), hide_index=True)

    st.subheader("What the rebalance requires")
    st.caption(
        "What each ticker has to trade to reach its target weight — negative to "
        "sell, positive to buy. This is fixed by price and target alone, so all "
        "three plans below trade exactly this much; they differ only in which "
        "lots supply the shares."
    )
    st.dataframe(requirement_rows(plan), hide_index=True)


def render_cards(plans: dict[str, Plan]) -> str:
    st.subheader("Tax on each plan")
    taxes = {key: plans[key].summary.total_tax for _, key in PLANS}
    selected = st.session_state.setdefault("plan", "optimal")
    inject_card_css(PLAN_SHADE, selected)

    best = min(taxes.values())
    for col, (label, key) in zip(st.columns(3), PLANS):
        with col, st.container(key=f"plan-{key}"):
            st.markdown(f"**{label}**")
            st.markdown(f"## {rupees(taxes[key])}")
            extra = taxes[key] - best
            st.caption("cheapest" if extra <= 0.005 else f"{rupees(extra)} more")
            if st.button(
                "Showing details" if key == selected else "View details",
                key=f"btn-{key}",
                width="stretch",
                type="primary" if key == selected else "secondary",
            ):
                st.session_state["plan"] = key
                st.rerun()
    return st.session_state["plan"]


def render_plan(plan: Plan, cfg: TaxConfig) -> None:
    if plan.certified_optimal:
        st.info(METHOD_NOTE[plan.method], icon="🧮")
    else:
        st.warning(METHOD_NOTE[plan.method], icon="⚠️")

    st.markdown("#### Ticker-level trade plan")
    if plan.trades:
        st.dataframe(trade_rows(plan), hide_index=True)
    else:
        st.info("No trades — every ticker is already at its target weight.")

    if plan.lot_sales:
        st.markdown("#### Lot-level sell plan")
        st.caption(
            "🟢 LT long-term, 🟠 ST short-term. Gains are signed, so a loss is "
            "negative and a negative tax is a saving. **Tax per share** uses "
            "statutory rates — the number a ranking rule sorts on, not what the "
            "lot actually costs. A short-term loss carries two: what it saves "
            "against short-term gain, and the 12.5% it drops to once that gain "
            "runs out."
        )
        st.dataframe(sell_lot_rows(plan, cfg), hide_index=True)

        net_st, net_lt = net_gains(plan)
        totals = st.columns(2)
        totals[0].metric("Net short-term gain", signed_rupees(net_st))
        totals[1].metric("Net long-term gain", signed_rupees(net_lt))
        st.caption(
            "What this plan realises on each side, after losses cancel gains within "
            "that side. The tax is a function of just these two numbers — which is "
            "what makes the choice of lots a linear program rather than a search."
        )

        st.markdown("#### Why these lots")
        for sale in plan.lot_sales:
            with st.container(border=True):
                st.markdown(
                    f"**{sale.ticker} · lot `{sale.lot_id}`** &nbsp; "
                    f"{marker('LT' if sale.classification.startswith('LT') else 'ST')} "
                    f"{sale.classification} &nbsp;·&nbsp; bought {sale.buy_date} "
                    f"·  {sale.holding}"
                )
                st.markdown(prettify(sale.reason))
                if sale.remaining_shares:
                    st.info(
                        f"{sale.remaining_shares} shares stay in this lot, keeping "
                        f"their original buy date of {sale.buy_date}.",
                        icon="🔒",
                    )
                if sale.alternatives:
                    with st.expander(
                        f"What a different lot would cost ({len(sale.alternatives)} priced)"
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

    st.markdown("#### Weights after this plan")
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
            [
                {"Figure": k.replace("_", " "), "Amount": v}
                for k, v in asdict(plan.tax).items()
            ],
            hide_index=True,
        )

    with st.expander("Raw API response"):
        st.json(json.loads(json.dumps(asdict(plan), default=str)))


def main() -> None:
    st.set_page_config(
        page_title="Tax-Aware Rebalancing Planner", page_icon="🧾", layout="wide"
    )
    engine, error = collect_input()

    st.title("Tax-aware rebalancing planner")
    st.caption(
        "Rebalances to target weights and chooses which specific lots to sell so "
        "the capital gains tax on the plan is as small as it can be."
    )

    if error:
        st.error(error, icon="🚫")
        return
    if engine is None:
        render_welcome()
        return

    try:
        plans = {key: engine.run(key, compare=False) for _, key in PLANS}
    except ValueError as exc:
        st.error(str(exc), icon="🚫")
        return

    render_input(engine, plans["optimal"])
    st.divider()
    selected = render_cards(plans)
    st.divider()
    render_plan(plans[selected], engine.cfg)


if __name__ == "__main__":
    main()
