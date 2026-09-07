"""Reading the three input CSVs: lots, prices and target weights.

One rule throughout: never guess. A bad quantity, an unreadable date or a
missing column stops the run naming the file, the row and the offending value,
because a silently dropped lot changes the tax answer and nothing downstream
could tell that it happened.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .models import Lot

# ISO first because it is unambiguous; the second is what Indian brokers export.
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y")

LOT_COLUMNS = ("ticker", "buy_date", "quantity", "buy_price")
PRICE_COLUMNS = ("ticker", "current_price")
TARGET_COLUMNS = ("ticker", "target_weight_pct")


def _rows(text: str, source: str, required: Sequence[str]) -> Iterator[tuple[int, dict]]:
    """Yield (line number, row) and fail loudly on a bad or missing header."""
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError(f"{source}: file is empty, expected a header row")

    header = {(f or "").strip().lower() for f in reader.fieldnames}
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(
            f"{source}: missing column(s) {', '.join(missing)}. "
            f"Found: {', '.join(sorted(header)) or '(none)'}"
        )

    found_any = False
    for row in reader:
        clean = {
            k.strip().lower(): (v.strip() if isinstance(v, str) else v)
            for k, v in row.items()
            if k is not None
        }
        if not any(clean.get(c) for c in required):
            continue  # blank line, common at the end of an exported sheet
        found_any = True
        yield reader.line_num, clean

    if not found_any:
        raise ValueError(f"{source}: header found but no data rows")


def _fail(source: str, line: int, column: str, value, why: str) -> ValueError:
    shown = "(empty)" if value in (None, "") else repr(value)
    return ValueError(f"{source} row {line}: {column}={shown} {why}")


def _text(source: str, line: int, row: Mapping[str, str], column: str) -> str:
    value = row.get(column)
    if not value:
        raise _fail(source, line, column, value, "is required")
    return value


def _ticker(source: str, line: int, row: Mapping[str, str]) -> str:
    return _text(source, line, row, "ticker").upper()


def parse_date(value: str, label: str = "date") -> date:
    """The only place a date string becomes a date, shared with the API form fields
    so the accepted formats cannot drift apart."""
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"{label}={value!r} is not a date; use YYYY-MM-DD or DD/MM/YYYY")


def _date(source: str, line: int, row: Mapping[str, str], column: str) -> date:
    value = _text(source, line, row, column)
    try:
        return parse_date(value, column)
    except ValueError:
        raise _fail(
            source, line, column, value, "is not a date; use YYYY-MM-DD or DD/MM/YYYY"
        ) from None


def _positive(source: str, line: int, row: Mapping[str, str], column: str) -> float:
    value = _text(source, line, row, column)
    try:
        number = float(value)
    except ValueError:
        raise _fail(source, line, column, value, "is not a number") from None
    if number <= 0:
        raise _fail(source, line, column, value, "must be greater than zero")
    return number


def _whole(source: str, line: int, row: Mapping[str, str], column: str) -> int:
    """Accepts 60 and 60.0, rejects 60.5: there are no fractional shares."""
    number = _positive(source, line, row, column)
    if abs(number - round(number)) > 1e-9:
        raise _fail(
            source, line, column, row.get(column), "must be a whole number of shares"
        )
    return int(round(number))


def default_lot_id(ticker: str, n: int) -> str:
    """A missing id is filled in from file order, not randomly: it is the audit trail."""
    return f"{ticker}-{n}"


def parse_lots(text: str, source: str = "lots.csv") -> list[Lot]:
    """Columns: ticker, buy_date, quantity, buy_price, and an optional lot_id."""
    lots: list[Lot] = []
    counter: dict[str, int] = {}
    seen: dict[str, int] = {}

    for line, row in _rows(text, source, LOT_COLUMNS):
        ticker = _ticker(source, line, row)
        counter[ticker] = counter.get(ticker, 0) + 1
        lot_id = row.get("lot_id") or default_lot_id(ticker, counter[ticker])
        if lot_id in seen:
            raise _fail(
                source, line, "lot_id", lot_id, f"repeats the id on row {seen[lot_id]}"
            )
        seen[lot_id] = line
        lots.append(
            Lot(
                lot_id=lot_id,
                ticker=ticker,
                buy_date=_date(source, line, row, "buy_date"),
                quantity=_whole(source, line, row, "quantity"),
                buy_price=_positive(source, line, row, "buy_price"),
            )
        )
    return lots


def parse_prices(text: str, source: str = "prices.csv") -> dict[str, float]:
    """Columns: ticker, current_price."""
    prices: dict[str, float] = {}
    seen: dict[str, int] = {}
    for line, row in _rows(text, source, PRICE_COLUMNS):
        ticker = _ticker(source, line, row)
        if ticker in seen:
            raise _fail(
                source, line, "ticker", ticker, f"already priced on row {seen[ticker]}"
            )
        seen[ticker] = line
        prices[ticker] = _positive(source, line, row, "current_price")
    return prices


def parse_targets(text: str, source: str = "targets.csv") -> dict[str, float]:
    """Columns: ticker, target_weight_pct. Percent in, fractions out, converted here
    once so nothing downstream has to know which convention it holds."""
    targets: dict[str, float] = {}
    seen: dict[str, int] = {}
    for line, row in _rows(text, source, TARGET_COLUMNS):
        ticker = _ticker(source, line, row)
        if ticker in seen:
            raise _fail(
                source, line, "ticker", ticker, f"already targeted on row {seen[ticker]}"
            )
        seen[ticker] = line
        value = _text(source, line, row, "target_weight_pct")
        try:
            pct = float(value)
        except ValueError:
            raise _fail(source, line, "target_weight_pct", value, "is not a number") from None
        if pct < 0:
            raise _fail(source, line, "target_weight_pct", value, "must not be negative")
        targets[ticker] = pct / 100.0
    return targets


def decode(raw: bytes, source: str) -> str:
    """utf-8-sig strips the byte-order mark Excel writes, which would otherwise make
    the first column name read as missing."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: not valid UTF-8 text ({exc})") from None


def load_scenario(
    directory: str | Path,
) -> tuple[list[Lot], dict[str, float], dict[str, float]]:
    """Read a samples/<name>/ set: lots.csv, prices.csv, targets.csv."""
    d = Path(directory)
    files = {}
    for name in ("lots.csv", "prices.csv", "targets.csv"):
        path = d / name
        if not path.is_file():
            raise ValueError(f"{path}: not found")
        files[name] = path.read_text(encoding="utf-8-sig")
    return (
        parse_lots(files["lots.csv"], f"{d.name}/lots.csv"),
        parse_prices(files["prices.csv"], f"{d.name}/prices.csv"),
        parse_targets(files["targets.csv"], f"{d.name}/targets.csv"),
    )
