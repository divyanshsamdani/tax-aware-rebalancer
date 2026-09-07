"""The tax rules themselves: holding periods, set-off order, and the identity
that the rest of the engine is built on."""

from __future__ import annotations

import random
from datetime import date

import pytest

from app.models import Lot, TaxConfig
from app.tax import breakdown, classify, effective_rate, holding_label, price_lot, tax_on

CFG = TaxConfig()
SALE = date(2026, 9, 6)


def priced(buy: date, buy_price: float, price: float, qty: int = 100):
    return price_lot(Lot("X", "T", buy, qty, buy_price), price, SALE, CFG)


# Holding period
def test_twelve_months_exactly_is_still_short_term():
    """Long-term needs MORE than 12 months, so the anniversary itself is not it."""
    assert classify(date(2025, 9, 6), SALE, CFG) == "ST"
    assert classify(date(2025, 9, 5), SALE, CFG) == "LT"


def test_month_end_is_clamped_not_overflowed():
    """31-Aug plus twelve months is 31-Aug, and 29-Feb behaves across a leap year."""
    assert classify(date(2025, 8, 31), date(2026, 8, 31), CFG) == "ST"
    assert classify(date(2025, 8, 31), date(2026, 9, 1), CFG) == "LT"
    assert classify(date(2024, 2, 29), date(2025, 2, 28), CFG) == "ST"
    assert classify(date(2024, 2, 29), date(2025, 3, 1), CFG) == "LT"


def test_holding_label_reads_in_months():
    assert holding_label(date(2023, 2, 10), SALE) == "42 months held"
    assert holding_label(date(2026, 9, 1), SALE) == "5 days held"


# The four-formula identity
def test_four_formulas_match_the_step_by_step_calculation():
    """tax_on() is what the optimiser minimises; breakdown() is what the report
    shows. If they ever disagree, one of them is lying."""
    rng = random.Random(0)
    for _ in range(400):
        st_gain = rng.uniform(0, 300_000)
        st_loss = rng.uniform(0, 300_000)
        lt_gain = rng.uniform(0, 400_000)
        lt_loss = rng.uniform(0, 400_000)
        lots = [
            priced(date(2026, 5, 1), 100.0, 100.0 + st_gain / 100),
            priced(date(2026, 5, 1), 100.0 + st_loss / 100, 100.0),
            priced(date(2020, 5, 1), 100.0, 100.0 + lt_gain / 100),
            priced(date(2020, 5, 1), 100.0 + lt_loss / 100, 100.0),
        ]
        shares = [100, 100, 100, 100]
        net_st = st_gain - st_loss
        net_lt = lt_gain - lt_loss
        assert breakdown(lots, shares, CFG).total_tax == pytest.approx(
            tax_on(net_st, net_lt, CFG), abs=1e-6
        )


# Set-off order
def test_long_term_loss_cannot_shelter_short_term_gain():
    """The asymmetry that makes lot choice matter: a long-term loss is useless
    against a short-term gain, so it does not reduce the bill at all."""
    with_loss = tax_on(100_000.0, -100_000.0, CFG)
    without = tax_on(100_000.0, 0.0, CFG)
    assert with_loss == pytest.approx(without) == pytest.approx(20_000.0)


def test_short_term_loss_is_spent_on_short_term_gain_first():
    """It saves 20% there and only 12.5% against long-term gain, so the order is
    forced rather than chosen."""
    # Rs 50,000 of short-term loss, against Rs 50,000 short-term gain and
    # Rs 300,000 long-term gain.
    lots = [
        priced(date(2026, 5, 1), 100.0, 600.0),  # +500/sh short-term gain
        priced(date(2026, 5, 1), 600.0, 100.0),  # -500/sh short-term loss
        priced(date(2020, 5, 1), 100.0, 3100.0),  # +3000/sh long-term gain
    ]
    b = breakdown(lots, [100, 100, 100], CFG)
    assert b.stcl_used_against_stcg == pytest.approx(50_000.0)
    assert b.stcl_used_against_ltcg == pytest.approx(0.0)
    assert b.taxable_stcg == pytest.approx(0.0)
    assert b.taxable_ltcg == pytest.approx(175_000.0)


def test_surplus_short_term_loss_spills_into_the_long_term_side():
    lots = [
        priced(date(2026, 5, 1), 600.0, 100.0),  # -500/sh, Rs 50,000 short-term loss
        priced(date(2020, 5, 1), 100.0, 3100.0),  # +3000/sh long-term gain
    ]
    b = breakdown(lots, [100, 100], CFG)
    assert b.stcl_used_against_stcg == pytest.approx(0.0)
    assert b.stcl_used_against_ltcg == pytest.approx(50_000.0)
    assert b.taxable_ltcg == pytest.approx(300_000.0 - 50_000.0 - 125_000.0)


def test_exemption_covers_long_term_gain_up_to_the_threshold():
    assert tax_on(0.0, 125_000.0, CFG) == pytest.approx(0.0)
    assert tax_on(0.0, 125_001.0, CFG) == pytest.approx(0.125)


# Effective rate
def test_effective_rate_is_not_the_statutory_rate():
    """A long-term gain under the exemption costs nothing, and a short-term gain
    with losses to absorb it costs nothing either. A rule that prices lots at
    20% and 12.5% is using the wrong multiplier."""
    assert effective_rate(0.0, 50_000.0, "LT", CFG) == pytest.approx(0.0)
    assert effective_rate(0.0, 200_000.0, "LT", CFG) == pytest.approx(0.125)
    assert effective_rate(-50_000.0, 0.0, "ST", CFG) == pytest.approx(0.0)
    assert effective_rate(50_000.0, 0.0, "ST", CFG) == pytest.approx(0.20)
