"""CSV parsing. A bad row must stop the run naming the file, the row and the
value, because a silently dropped lot changes the tax answer."""

from __future__ import annotations

from datetime import date

import pytest

from app import ingest

LOTS_CSV = (
    "ticker,buy_date,quantity,buy_price\n"
    "acme,2023-02-10,60,800\n"
    "ACME,01/03/2024,40,950\n"
)


def test_lot_ids_are_generated_per_ticker_in_file_order():
    lots = ingest.parse_lots(LOTS_CSV)
    assert [l.lot_id for l in lots] == ["ACME-1", "ACME-2"]
    assert [l.ticker for l in lots] == ["ACME", "ACME"]


def test_both_date_formats_are_accepted():
    lots = ingest.parse_lots(LOTS_CSV)
    assert [l.buy_date for l in lots] == [date(2023, 2, 10), date(2024, 3, 1)]


def test_targets_are_converted_from_percent_to_fractions():
    targets = ingest.parse_targets("ticker,target_weight_pct\nACME,20\nNOVA,80\n")
    assert targets == {"ACME": 0.2, "NOVA": 0.8}


def test_the_excel_byte_order_mark_does_not_hide_the_first_column():
    raw = "﻿ticker,current_price\nACME,100\n".encode("utf-8")
    assert ingest.parse_prices(ingest.decode(raw, "prices.csv")) == {"ACME": 100.0}


@pytest.mark.parametrize(
    "text,message",
    [
        ("ticker,buy_date,quantity,buy_price\nACME,2023-02-10,sixty,800\n",
         "lots.csv row 2: quantity='sixty' is not a number"),
        ("ticker,buy_date,quantity,buy_price\nACME,2023-02-10,60.5,800\n",
         "whole number of shares"),
        ("ticker,buy_date,quantity\nACME,2023-02-10,60\n",
         r"missing column\(s\) buy_price"),
        ("lot_id,ticker,buy_date,quantity,buy_price\n"
         "L1,ACME,2023-02-10,60,800\nL1,ACME,2024-02-10,40,900\n",
         "repeats the id on row 2"),
    ],
)
def test_a_bad_lot_row_names_the_file_the_row_and_the_value(text, message):
    with pytest.raises(ValueError, match=message):
        ingest.parse_lots(text)


@pytest.mark.parametrize(
    "text,message",
    [
        ("ticker,current_price\nACME,100\nACME,200\n", "already priced"),
        ("ticker,current_price\n", "no data rows"),
    ],
)
def test_a_bad_price_file_is_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        ingest.parse_prices(text)
