"""Holding periods, set-off order, and the identity the optimiser relies on."""

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


def test_twelve_months_exactly_is_short_term():
    assert classify(date(2025, 9, 6), SALE, CFG) == "ST"
    assert classify(date(2025, 9, 5), SALE, CFG) == "LT"


def test_month_ends_and_leap_days():
    assert classify(date(2025, 8, 31), date(2026, 8, 31), CFG) == "ST"
    assert classify(date(2025, 8, 31), date(2026, 9, 1), CFG) == "LT"
    assert classify(date(2024, 2, 29), date(2025, 2, 28), CFG) == "ST"
    assert classify(date(2024, 2, 29), date(2025, 3, 1), CFG) == "LT"


def test_holding_label():
    assert holding_label(date(2023, 2, 10), SALE) == "42 months held"
    assert holding_label(date(2026, 9, 1), SALE) == "5 days held"


def test_four_formulas_match_the_step_by_step_calculation():
    """tax_on() is what the optimiser minimises, breakdown() is what the report
    shows. If they disagree, one of them is lying."""
    rng = random.Random(0)
    for _ in range(400):
        st_gain, st_loss = rng.uniform(0, 300_000), rng.uniform(0, 300_000)
        lt_gain, lt_loss = rng.uniform(0, 400_000), rng.uniform(0, 400_000)
        lots = [
            priced(date(2026, 5, 1), 100.0, 100.0 + st_gain / 100),
            priced(date(2026, 5, 1), 100.0 + st_loss / 100, 100.0),
            priced(date(2020, 5, 1), 100.0, 100.0 + lt_gain / 100),
            priced(date(2020, 5, 1), 100.0 + lt_loss / 100, 100.0),
        ]
        assert breakdown(lots, [100] * 4, CFG).total_tax == pytest.approx(
            tax_on(st_gain - st_loss, lt_gain - lt_loss, CFG), abs=1e-6
        )


def test_long_term_loss_cannot_shelter_short_term_gain():
    assert tax_on(100_000.0, -100_000.0, CFG) == pytest.approx(20_000.0)
    assert tax_on(100_000.0, 0.0, CFG) == pytest.approx(20_000.0)


def test_short_term_loss_goes_to_short_term_gain_first():
    """It saves 20% there and only 12.5% against long-term gain."""
    lots = [
        priced(date(2026, 5, 1), 100.0, 600.0),  # +500/sh short-term
        priced(date(2026, 5, 1), 600.0, 100.0),  # -500/sh short-term
        priced(date(2020, 5, 1), 100.0, 3100.0),  # +3000/sh long-term
    ]
    b = breakdown(lots, [100, 100, 100], CFG)
    assert b.stcl_used_against_stcg == pytest.approx(50_000.0)
    assert b.stcl_used_against_ltcg == pytest.approx(0.0)
    assert b.taxable_stcg == pytest.approx(0.0)
    assert b.taxable_ltcg == pytest.approx(175_000.0)


def test_surplus_short_term_loss_spills_to_the_long_term_side():
    lots = [
        priced(date(2026, 5, 1), 600.0, 100.0),  # -500/sh short-term
        priced(date(2020, 5, 1), 100.0, 3100.0),  # +3000/sh long-term
    ]
    b = breakdown(lots, [100, 100], CFG)
    assert b.stcl_used_against_ltcg == pytest.approx(50_000.0)
    assert b.taxable_ltcg == pytest.approx(300_000.0 - 50_000.0 - 125_000.0)


def test_exemption_covers_gain_up_to_the_threshold():
    assert tax_on(0.0, 125_000.0, CFG) == pytest.approx(0.0)
    assert tax_on(0.0, 125_001.0, CFG) == pytest.approx(0.125)


def test_effective_rate_is_not_the_statutory_rate():
    """Long-term gain under the exemption is free, and so is short-term gain
    with losses left to absorb it."""
    assert effective_rate(0.0, 50_000.0, "LT", CFG) == pytest.approx(0.0)
    assert effective_rate(0.0, 200_000.0, "LT", CFG) == pytest.approx(0.125)
    assert effective_rate(-50_000.0, 0.0, "ST", CFG) == pytest.approx(0.0)
    assert effective_rate(50_000.0, 0.0, "ST", CFG) == pytest.approx(0.20)
