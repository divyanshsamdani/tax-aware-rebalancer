"""Lot selection: the solver against exhaustive search, and the simpler rules
it replaces getting the wrong answer."""

from __future__ import annotations

import random
from datetime import date

import pytest

from app.models import Lot, SellRequirement, TaxConfig
from app.optimizer import Allocation, solve_fifo, solve_ltfo, solve_optimal
from app.tax import price_lot
from tests.bruteforce import solve_bruteforce

CFG = TaxConfig()
SALE = date(2026, 9, 6)


def priced(lots, prices):
    return [price_lot(l, prices[l.ticker], SALE, CFG) for l in lots]


def test_the_split_lands_where_the_exemption_runs_out():
    """AAA is forced into Rs 10,000 of long-term gain. BBB must give up 100
    shares from a lot gaining Rs 1,200/share (long-term) or Rs 300/share
    (short-term). Filling entirely from the long-term lot overshoots the
    exemption, so the answer stops at 96 and takes 4 short-term - which no rule
    that picks a lot and fills from it can produce."""
    lots = [
        Lot("A1", "AAA", date(2023, 1, 1), 100, 900.0),
        Lot("B1", "BBB", date(2023, 1, 1), 100, 800.0),
        Lot("B2", "BBB", date(2026, 5, 1), 100, 1700.0),
    ]
    p = priced(lots, {"AAA": 1000.0, "BBB": 2000.0})
    reqs = [SellRequirement("AAA", 100), SellRequirement("BBB", 100)]

    best = solve_optimal(p, reqs, CFG)
    assert best.shares == (100, 96, 4)
    assert best.tax(p, CFG) == pytest.approx(265.0)
    assert Allocation((100, 100, 0), "fifo", False).tax(p, CFG) == pytest.approx(625.0)
    assert Allocation((100, 0, 100), "fifo", False).tax(p, CFG) == pytest.approx(6_000.0)


def test_the_exemption_couples_tickers_that_look_independent():
    """Same lots, same required sale, different answer - purely because another
    ticker used up the exemption."""
    bbb = [
        Lot("B1", "BBB", date(2023, 1, 1), 100, 800.0),
        Lot("B2", "BBB", date(2026, 5, 1), 100, 1700.0),
    ]
    alone = priced(bbb, {"BBB": 2000.0})
    assert solve_optimal(alone, [SellRequirement("BBB", 100)], CFG).shares == (100, 0)

    with_neighbour = priced(
        [Lot("A1", "AAA", date(2023, 1, 1), 100, 100.0)] + bbb,
        {"AAA": 1500.0, "BBB": 2000.0},
    )
    reqs = [SellRequirement("AAA", 100), SellRequirement("BBB", 100)]
    assert solve_optimal(with_neighbour, reqs, CFG).shares != (100, 100, 0)


def test_one_ticker_can_split_across_a_short_and_a_long_loss():
    """Per rupee the short-term loss is worth more; per share the long-term loss
    is bigger. The crossover is where the short-term gain elsewhere runs out."""
    lots = [
        Lot("A", "X", date(2026, 3, 1), 100, 1200.0),  # -200/sh, short-term
        Lot("B", "X", date(2024, 3, 1), 100, 1300.0),  # -300/sh, long-term
        Lot("C", "Y", date(2026, 3, 1), 100, 940.0),  # +60/sh, short-term
        Lot("D", "Z", date(2023, 3, 1), 100, 1000.0),  # +2000/sh, long-term
    ]
    p = priced(lots, {"X": 1000.0, "Y": 1000.0, "Z": 3000.0})
    reqs = [SellRequirement(t, 100) for t in ("X", "Y", "Z")]

    assert solve_optimal(p, reqs, CFG).shares == (30, 70, 100, 100)
    assert solve_optimal(p, reqs, CFG).tax(p, CFG) == pytest.approx(6_750.0)
    assert solve_fifo(p, reqs, CFG).tax(p, CFG) == pytest.approx(6_825.0)
    assert solve_bruteforce(p, reqs, CFG).tax(p, CFG) == pytest.approx(6_750.0)


def _random_case(rng):
    lots, prices = [], {}
    for t in range(rng.randint(1, 3)):
        ticker = f"T{t}"
        prices[ticker] = float(rng.randint(50, 3000))
        for k in range(rng.randint(1, 3)):
            lots.append(
                Lot(
                    f"{ticker}-{k}",
                    ticker,
                    date(rng.choice([2021, 2023, 2026]), rng.randint(1, 12), 1),
                    rng.randint(1, 12),
                    float(rng.randint(50, 3000)),
                )
            )
    p = priced(lots, prices)
    reqs = [
        SellRequirement(t, rng.randint(1, sum(l.lot.quantity for l in p if l.lot.ticker == t)))
        for t in prices
    ]
    return p, reqs


@pytest.mark.parametrize("seed", range(60))
def test_solver_matches_exhaustive_search(seed):
    p, reqs = _random_case(random.Random(seed))
    assert solve_optimal(p, reqs, CFG).tax(p, CFG) == pytest.approx(
        solve_bruteforce(p, reqs, CFG).tax(p, CFG), abs=1e-6
    )


@pytest.mark.parametrize("seed", range(60))
def test_solver_beats_fifo_and_sells_exactly_what_was_asked(seed):
    p, reqs = _random_case(random.Random(1000 + seed))
    best = solve_optimal(p, reqs, CFG)
    assert best.tax(p, CFG) <= solve_fifo(p, reqs, CFG).tax(p, CFG) + 1e-9
    for r in reqs:
        sold = sum(n for l, n in zip(p, best.shares) if l.lot.ticker == r.ticker)
        assert sold == r.shares
    for lot, n in zip(p, best.shares):
        assert 0 <= n <= lot.lot.quantity


def test_selling_more_than_is_held_is_rejected():
    p = priced([Lot("A", "X", date(2023, 1, 1), 10, 100.0)], {"X": 200.0})
    with pytest.raises(ValueError, match="only 10 are held"):
        solve_optimal(p, [SellRequirement("X", 11)], CFG)


def test_two_requirements_for_one_ticker_are_rejected():
    p = priced([Lot("A", "X", date(2023, 1, 1), 10, 100.0)], {"X": 200.0})
    with pytest.raises(ValueError, match="duplicate"):
        solve_optimal(p, [SellRequirement("X", 1), SellRequirement("X", 2)], CFG)


def test_the_tie_break_pass_never_buys_a_worse_tax_outcome():
    lots = [
        Lot("A", "X", date(2023, 1, 1), 100, 100.0),  # huge long-term gain
        Lot("B", "X", date(2026, 5, 1), 100, 2990.0),  # tiny short-term gain
    ]
    p = priced(lots, {"X": 3000.0})
    reqs = [SellRequirement("X", 100)]
    assert solve_optimal(p, reqs, CFG).tax(p, CFG) == pytest.approx(
        solve_bruteforce(p, reqs, CFG).tax(p, CFG), abs=1e-6
    )


def test_ltfo_takes_the_cheapest_lot_per_share_first():
    """Long-term gain of Rs 1,000/share ranks at 125; short-term of Rs 100/share
    ranks at 20, so the short-term lot goes first despite the higher rate."""
    lots = [
        Lot("L1", "A", date(2023, 1, 1), 100, 1_000.0),
        Lot("L2", "A", date(2026, 5, 1), 100, 1_900.0),
    ]
    p = priced(lots, {"A": 2_000.0})
    got = solve_ltfo(p, [SellRequirement("A", 100)], CFG)
    assert got.shares == (0, 100)
    assert not got.certified_optimal


def test_ltfo_respects_quantity_and_lot_capacity():
    lots = [
        Lot("L1", "A", date(2023, 1, 1), 30, 1_990.0),
        Lot("L2", "A", date(2026, 5, 1), 30, 1_995.0),
    ]
    p = priced(lots, {"A": 2_000.0})
    got = solve_ltfo(p, [SellRequirement("A", 45)], CFG)
    assert sum(got.shares) == 45
    assert all(n <= lot.lot.quantity for n, lot in zip(got.shares, p))
