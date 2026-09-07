"""Lot selection.

Two things are checked here: that the solver's answer matches an exhaustive
search, and that the simpler rules it replaces really do get the wrong answer.
"""

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


# Why a solver and not a ranking rule
def test_the_answer_stops_where_the_exemption_runs_out_not_at_a_lot_boundary():
    """AAA is forced into Rs 10,000 of long-term gain. BBB must give up 100
    shares, from either a lot gaining Rs 1,200/share (long-term) or one gaining
    Rs 300/share (short-term). Filling entirely from the long-term lot
    overshoots the exemption, so the best answer stops at the boundary and takes
    the last 4 shares short-term.

    No rule that picks a lot and fills from it can produce 96 + 4. That is the
    whole argument for solving this rather than sorting it.
    """
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

    # Both "obvious" pure answers cost more.
    assert Allocation((100, 100, 0), "fifo", False).tax(p, CFG) == pytest.approx(625.0)
    assert Allocation((100, 0, 100), "fifo", False).tax(p, CFG) == pytest.approx(6_000.0)
    assert solve_bruteforce(p, reqs, CFG).tax(p, CFG) == pytest.approx(265.0)


def test_the_exemption_couples_tickers_that_look_independent():
    """The exemption is one pool shared across the whole portfolio, so what is
    cheapest for one ticker depends on what every other ticker is doing. Here
    the same ticker, with the same lots and the same required sale, gets a
    different answer purely because another ticker used up the exemption."""
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


def test_the_best_plan_can_split_one_ticker_across_a_short_and_a_long_loss():
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

    best = solve_optimal(p, reqs, CFG)
    assert best.shares == (30, 70, 100, 100)
    assert best.tax(p, CFG) == pytest.approx(6_750.0)
    assert solve_fifo(p, reqs, CFG).tax(p, CFG) == pytest.approx(6_825.0)
    assert solve_bruteforce(p, reqs, CFG).tax(p, CFG) == pytest.approx(6_750.0)


# Agreement with an exhaustive search
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
    reqs = []
    for ticker in prices:
        held = sum(l.lot.quantity for l in p if l.lot.ticker == ticker)
        reqs.append(SellRequirement(ticker, rng.randint(1, held)))
    return p, reqs


@pytest.mark.parametrize("seed", range(120))
def test_solver_matches_exhaustive_search_on_random_portfolios(seed):
    rng = random.Random(seed)
    p, reqs = _random_case(rng)
    assert solve_optimal(p, reqs, CFG).tax(p, CFG) == pytest.approx(
        solve_bruteforce(p, reqs, CFG).tax(p, CFG), abs=1e-6
    )


@pytest.mark.parametrize("seed", range(120))
def test_solver_is_never_worse_than_fifo_and_always_sells_the_right_amount(seed):
    rng = random.Random(1000 + seed)
    p, reqs = _random_case(rng)
    best = solve_optimal(p, reqs, CFG)
    assert best.tax(p, CFG) <= solve_fifo(p, reqs, CFG).tax(p, CFG) + 1e-9
    for r in reqs:
        sold = sum(n for l, n in zip(p, best.shares) if l.lot.ticker == r.ticker)
        assert sold == r.shares
    for lot, n in zip(p, best.shares):
        assert 0 <= n <= lot.lot.quantity


# Guard rails
def test_asking_for_more_shares_than_are_held_is_rejected():
    p = priced([Lot("A", "X", date(2023, 1, 1), 10, 100.0)], {"X": 200.0})
    with pytest.raises(ValueError, match="only 10 are held"):
        solve_optimal(p, [SellRequirement("X", 11)], CFG)


def test_two_requirements_for_one_ticker_are_rejected():
    p = priced([Lot("A", "X", date(2023, 1, 1), 10, 100.0)], {"X": 200.0})
    with pytest.raises(ValueError, match="duplicate"):
        solve_optimal(p, [SellRequirement("X", 1), SellRequirement("X", 2)], CFG)


def test_tie_break_never_buys_a_worse_tax_outcome():
    """The second pass settles ties. It must never trade real tax for its own
    preference, which is why it runs pinned to the first pass's answer."""
    lots = [
        Lot("A", "X", date(2023, 1, 1), 100, 100.0),  # huge long-term gain
        Lot("B", "X", date(2026, 5, 1), 100, 2990.0),  # tiny short-term gain
    ]
    p = priced(lots, {"X": 3000.0})
    reqs = [SellRequirement("X", 100)]
    best = solve_optimal(p, reqs, CFG)
    assert best.tax(p, CFG) == pytest.approx(
        solve_bruteforce(p, reqs, CFG).tax(p, CFG), abs=1e-6
    )


# Least tax first out
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


def test_ltfo_reaches_for_a_loss_lot_first_because_its_ranking_goes_negative():
    lots = [
        Lot("L1", "A", date(2023, 1, 1), 50, 1_200.0),
        Lot("L2", "A", date(2026, 5, 1), 50, 2_200.0),
    ]
    p = priced(lots, {"A": 2_000.0})
    assert solve_ltfo(p, [SellRequirement("A", 50)], CFG).shares == (0, 50)


def test_ltfo_never_beats_the_solver_on_the_counterexample():
    """It prices each ticker in isolation, so it cannot see that the exemption is
    one pool shared across the portfolio."""
    lots = [
        Lot("A1", "AAA", date(2023, 1, 1), 100, 900.0),
        Lot("B1", "BBB", date(2023, 1, 1), 100, 800.0),
        Lot("B2", "BBB", date(2026, 5, 1), 100, 1700.0),
    ]
    p = priced(lots, {"AAA": 1000.0, "BBB": 2000.0})
    reqs = [SellRequirement("AAA", 100), SellRequirement("BBB", 100)]
    assert solve_optimal(p, reqs, CFG).tax(p, CFG) < solve_ltfo(p, reqs, CFG).tax(p, CFG)


def test_ltfo_respects_the_required_quantity_and_lot_capacity():
    lots = [
        Lot("L1", "A", date(2023, 1, 1), 30, 1_990.0),
        Lot("L2", "A", date(2026, 5, 1), 30, 1_995.0),
    ]
    p = priced(lots, {"A": 2_000.0})
    got = solve_ltfo(p, [SellRequirement("A", 45)], CFG)
    assert sum(got.shares) == 45
    assert all(n <= lot.lot.quantity for n, lot in zip(got.shares, p))


def test_ltfo_agrees_with_the_solver_whenever_only_one_lot_exists():
    lots = [Lot("L1", "A", date(2023, 1, 1), 40, 1_000.0)]
    p = priced(lots, {"A": 2_000.0})
    reqs = [SellRequirement("A", 25)]
    assert solve_ltfo(p, reqs, CFG).shares == solve_optimal(p, reqs, CFG).shares
