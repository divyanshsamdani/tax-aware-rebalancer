# Tax-Aware Rebalancing Engine

Rebalances an Indian listed-equity portfolio to its target weights and picks
**which specific lots to sell** so that the capital gains tax on the plan is as
small as it can be. Lot selection is solved as a linear program with whole-number
variables, so the answer comes back with a proof that no cheaper plan exists.

## Running it

**Python 3.11 or newer.** Install with either tool:

```bash
uv sync                                              # with uv

python3 -m venv .venv && source .venv/bin/activate   # without uv
pip install -r requirements.txt
```

Then pick a way in. Nothing is installed as a package and there is no build step.

```bash
uv run streamlit run ui.py     # the UI, on http://localhost:8501
uv run uvicorn app.api:app     # the API: /docs for Swagger, /demo/edge_case for the brief's case
uv run pytest                  # the tests
```

Drop the `uv run` prefix if you are inside an activated venv.

---

## Tax rates and sources

| | rate | exemption | provision |
|---|---|---|---|
| Short-term — held 12 months or less | **20%** | none | section 111A |
| Long-term — held more than 12 months | **12.5%**, no indexation | **₹1,25,000** a year | section 112A |

Listed equity on which STT is paid. Source: the Income Tax Department,
<https://incometaxindia.gov.in/>, which hosts the Act and the Finance Acts. The
**Finance (No. 2) Act 2024** set these for transfers on or after **23 July 2024**
— short-term 15% → 20%, long-term 10% → 12.5%, exemption ₹1,00,000 → ₹1,25,000.

> The brief quotes a ₹1 lakh exemption, which is the pre-23-July-2024 figure.
> This engine uses ₹1,25,000. Every rate lives in `TaxConfig` in `app/models.py`.

---

## How lot selection works, and why it minimises tax

### The tax depends on two numbers, not on lots

Write **A** for the net short-term realised gain and **B** for the net long-term,
both signed so losses are negative. Let `s = 20%`, `l = 12.5%`, `E = ₹1,25,000`.

The order in which losses are used up is fixed by law, not chosen. A long-term
loss can only be set off against long-term gain. A short-term loss saves 20%
against short-term gain but at most 12.5% against long-term gain, so it is spent
on short-term gain first and anything left over spills across. Substitute that
order and the whole calculation becomes:

```
A ≥ 0 :   tax = s·A + l·max(0, B − E)
A < 0 :   tax =       l·max(0, A + B − E)
```

which is the single expression

```
tax(A, B) = max( 0 ,  s·A ,  l·(A+B−E) ,  s·A + l·(B−E) )
```

Four straight lines. `test_tax.py` checks this identity against the step-by-step
calculation in `tax.breakdown()` on 400 random inputs.

### So lot selection is a linear program

Let `xᵢ` be the number of shares sold out of lot `i`. Each lot adds
`gain_per_share × xᵢ` to either A or B, so both are linear in x, and so is each
of the four lines:

```
minimise    t
subject to  t ≥ aₖ·A(x) + bₖ·B(x) + cₖ      for each of the four lines
            Σ xᵢ over ticker T's lots  =  shares T must give up
            0 ≤ xᵢ ≤ lotᵢ.quantity,  xᵢ a whole number
```

`t` stands in for "the tax": you cannot write `max(...)` in a linear objective,
so add one extra variable, force it above all four lines and minimise it.
`scipy.optimize.milp` hands this to HiGHS, which returns a proved global optimum
in milliseconds.

### Why no ranking rule can match it

**The ₹1,25,000 exemption is one pool shared across the whole portfolio.** The
real cost of a rupee of long-term gain is 0% until the pool runs out and 12.5%
after, and where that happens depends on what every other ticker is doing. So
you cannot decide ticker by ticker, and the best answer can land **part-way
through a lot** — which is never where a lot boundary is.

Here is the counterexample, automated in
`test_optimizer.py::test_the_split_lands_where_the_exemption_runs_out`. AAA is
forced into ₹10,000 of long-term gain. BBB must give up 100 shares, from either a
lot gaining ₹1,200/share (long-term) or one gaining ₹300/share (short-term):

| plan | tax |
|---|---|
| all 100 from the long-term lot | ₹625 |
| all 100 from the short-term lot | ₹6,000 |
| **96 long-term + 4 short-term** | **₹265** |

No rule that picks a lot and fills from it produces 96 + 4. Whole numbers matter
for the same reason: the unrestricted answer lands between two shares, so
allowing fractions would return a plan you cannot trade.

---

## The required edge case

One ticker held in two lots, where the rebalance needs only part of the position:
enough that one lot can go entirely as LTCG, but the required quantity forces a
partial sale into a second, STCG-eligible lot.

**Input** — `samples/edge_case/`, trade date 2026-09-06:

```csv
# lots.csv
lot_id,ticker,buy_date,quantity,buy_price
L1,ACME,2023-02-10,60,800
L2,ACME,2026-04-01,40,950
N1,NOVA,2022-08-15,100,200

# prices.csv            # targets.csv
ticker,current_price    ticker,target_weight_pct
ACME,1000               ACME,20
NOVA,250                NOVA,80
```

The portfolio is ACME 100 × ₹1,000 + NOVA 100 × ₹250 = **₹1,25,000**. ACME sits
at 80% against a 20% target, so ₹75,000 has to go: **75 shares**.

**Output** — `GET /demo/edge_case`:

| lot | bought | held | class | sold | left | gain | tax |
|---|---|---|---|---|---|---|---|
| L1 | 2023-02-10 | 42 months | LTCG | **60** of 60 | 0 | ₹12,000 | ₹0 — inside the exemption |
| L2 | 2026-04-01 | 5 months | STCG | **15** of 40 | **25** | ₹750 | ₹150 at 20% |

Trades: `SELL 75 ACME @ ₹1,000` → `BUY 300 NOVA @ ₹250`. **Total tax ₹150.**
Weights land on 20% / 80% exactly. The remaining 25 shares of L2 keep their
2026-04-01 buy date, so a future sale classifies them correctly.

The reasoning the engine emits for the two lots:

> **L1** — Supplies 60 of the 75 shares ACME must give up, taking the whole lot.
>
> **L2** — Supplies 15 of the 75 shares ACME must give up, 15 of its 40, leaving
> 25 untouched. Moving one share off 15 costs at least ₹10.00 more.

Buy date, holding period, cost basis and share counts are structured fields on
the same record, so the prose does not restate them. It carries the fill, and —
for a partly-sold lot — what one share either side of the split would have cost.

That last figure is computed, not asserted. Where the quantity is forced, or
where some equally cheap alternative exists, the sentence says nothing at all:
on `all_ltcg` the plan sells 525 of a 1,000-share lot and claims no trade-off,
because shifting a share to the sibling lot is free.

Where it does speak, the `alternative` field on the record — the ℹ️ beside it in
the UI — breaks the figure down. The tax is
`20% × taxable_stcg + 12.5% × taxable_ltcg`, and each taxable figure is a gain
less the reliefs applied to it, so differencing the two set-off calculations
gives terms that are the definition of the cost rather than a guess at it. On
`loss_priority`, why the answer is 57 shares and not 58:

| change | rate | tax |
|---|---|---|
| ₹400 less short-term loss cancelling short-term gain | 20% | **+₹80** |
| ₹1,000 more long-term loss cancelling long-term gain | 12.5% | **−₹125** |
| ₹400 less short-term loss spilling to the long-term side | 12.5% | **+₹50** |
| **net** | | **+₹5** |

The share carries ₹800 of short-term loss, but only ₹400 of it was still
cancelling short-term gain at 20% — the rest had already spilled to the
long-term side, where it is worth 12.5% like everything else. That split is the
whole reason the boundary falls where it does, and it is not a property of the
lot: it is the point where the rest of the portfolio changes what the lot is
worth. Across 800 random portfolios the ledger reconciled with the cost it
explains every time; `Alternative.note` carries the qualitative reasons instead
if it ever does not.

Under `fifo` and `ltfo` each lot additionally names its place in the sequence and
where the remainder goes — *"1st pick for ACME: the oldest lot still available.
Supplies 60 of the 75 shares ACME must give up, taking the whole lot; 15 still to
find from the next lot."* — because those rules really do work through a ticker
one lot at a time.

Automated in `test_engine.py`, first three tests.

---

## Endpoints

| | path | |
|---|---|---|
| `GET` | `/demo/{scenario}` | Run a bundled scenario. No input. |
| `POST` | `/rebalance/upload` | Three CSVs. Works through Swagger's file picker. |
| `POST` | `/rebalance` | The same thing as JSON. |
| `GET` | `/health` | Liveness. |

```bash
curl -s localhost:8000/demo/edge_case | python3 -m json.tool

curl -s -X POST localhost:8000/rebalance/upload \
  -F lots_file=@samples/edge_case/lots.csv \
  -F prices_file=@samples/edge_case/prices.csv \
  -F targets_file=@samples/edge_case/targets.csv \
  -F sale_date=2026-09-06 | python3 -m json.tool
```

**Input format.** `lots.csv` — `ticker, buy_date, quantity, buy_price`, plus an
optional `lot_id` (generated as `{TICKER}-{n}` in file order if absent). Dates may
be `YYYY-MM-DD` or `DD/MM/YYYY`. `prices.csv` — `ticker, current_price`.
`targets.csv` — `ticker, target_weight_pct`, in **percent**, adding to 100.

Nothing is coerced. A bad row stops the run naming the file, the row number and
the offending value, and arrives as a **422** with the message intact.

**The response** carries the total tax, the plain-English reasoning, per-ticker
buy/sell instructions, a per-lot breakdown of the sells each with its own
reasoning and — for a partly-sold lot — the priced alternative, the full set-off
calculation, and target/before/after weights.

### The three lot-selection methods

`method` picks one; only the first claims to be cheapest.

| `method` | rule |
|---|---|
| `optimal` | the linear program above, solved over every lot at once |
| `fifo` | oldest lot first — what a demat account does by default |
| `ltfo` | least tax first out: within each ticker, the lot with the smallest `gain_per_share × statutory rate` goes first |

`ltfo` is the rule a reader is most likely to reach for instead of a solver, so
it is worth being able to price that instinct. It ranks each lot in isolation, so
it cannot see that the exemption is a single pool — and on the required edge case
that costs it ₹250. It sees a short-term lot gaining ₹50/share (₹10 of tax at
20%) against a long-term lot gaining ₹200/share (₹25 at 12.5%), so it empties the
short-term lot first, never noticing that the long-term gain was free because the
exemption had not been touched.

`compare_with_fifo` reports the FIFO figure alongside whichever plan was asked
for, in words as well as numbers — a saving of zero usually means FIFO happened
to be optimal and the solver proved it, which is not the same as lot selection
achieving nothing.

### Bundled scenarios

| scenario | shows | `optimal` | `ltfo` | `fifo` |
|---|---|---|---|---|
| `edge_case` | the required 60/40 partial-lot split | **₹150** | ₹400 | ₹150 |
| `exemption_split` | all three disagree; the best answer stops part-way through a lot | **₹80** | ₹1,600 | ₹5,625 |
| `exemption_vs_loss` | a long-term loss spent only down to the exemption line | **₹8,400** | ₹12,000 | ₹21,375 |
| `loss_priority` | which of two losses is worth more, and when that flips | **₹14,700** | ₹16,125 | ₹16,175 |
| `loss_offset` | a short-term loss sheltering a gain elsewhere | **₹4,000** | ₹4,000 | ₹20,000 |
| `all_ltcg` | a plain long-term rebalance | **₹0** | ₹0 | ₹43,437.50 |
| `at_target` | already at target, nothing to do | ₹0 | ₹0 | ₹0 |

The two `loss` scenarios are where the ranking rules break down most clearly.
Both pick the right lot to start from and have no way to know when to stop: a
long-term loss is worth 12.5% a rupee only while there is long-term gain above
the exemption to cancel, and a short-term loss is worth 20% only while there is
short-term gain left. The stopping point is not a property of the lot — it is the
point where the rest of the portfolio changes what that lot is worth.

---

## The UI

```bash
streamlit run ui.py
```

One file, and deliberately thin: it collects input, calls `Engine`, and renders
what comes back. It computes nothing of its own, so there is no second
implementation that could drift away from the API. Pick a bundled scenario or
upload three CSVs in the sidebar; the page shows the portfolio lot by lot, the
shares each ticker must trade, then the three plans side by side with their tax,
and opens whichever you click. It calls the engine directly rather than over
HTTP, so there is only one process to start.

---

## Tests

```bash
uv run pytest
```

| file | covers |
|---|---|
| `test_tax.py` | holding periods, the four-line identity against the step-by-step calculation, set-off order |
| `test_optimizer.py` | the solver against exhaustive search on 60 random portfolios, 60 more against FIFO, plus the counterexamples that rule out simpler rules |
| `test_engine.py` | the three cases the brief requires, the loss-and-exemption splits, rebalancing mechanics, validation, the per-lot reasoning and its ledger |
| `test_ingest.py` | CSV parsing and the error messages |
| `test_api.py` | every endpoint, both input paths, the error contract |
| `test_ui.py` | the front end's data path |

The three the brief asks for are the first three sections of `test_engine.py`:
the partial-lot edge case, a straightforward all-long-term rebalance, and a
portfolio already at target.

`tests/bruteforce.py` is an exhaustive reference solver used only by the tests.
It tries every split and keeps the cheapest — useless in production, but it
shares no logic with the linear program, so agreement between the two is
evidence rather than the same idea agreeing with itself.

---

## Assumptions

| | |
|---|---|
| **A1** | Ordinary shares bought and sold on an Indian stock exchange. These rates do not apply to debt funds, unlisted shares, gold, property or derivatives. |
| **A2** | Holding period counted in calendar months, **strictly more than 12**. The anniversary itself is short-term; a 365-day count is off by one across a leap year. |
| **A3** | Rates as above. Short-term gain gets no exemption. |
| **A4** | Set-off order is forced: long-term loss against long-term gain; short-term loss against short-term gain before long-term gain. |
| **A5** | The ₹1,25,000 exemption applies **after** loss set-off, not before. Arguable, but both readings subtract the same amounts from the same base, so this year's bill is unchanged; what the order changes is which relief is left over, which only matters for carry-forward — out of scope under A6. Isolated to one line in `tax.breakdown()`. |
| **A6** | Stateless. The full exemption is assumed available, nothing carries between runs, and unused losses are not tracked. A second rebalance in the same financial year would under-state the tax. |
| **A7** | Whole shares only, halves rounded away from zero. |
| **A8** | Target weights add to 100% ± 0.01%, and every targeted ticker must already be held. A ticker held but **absent** from the targets has a 0% target and is sold off. The portfolio is assumed fully invested, so the engine rebalances between existing holdings and cannot open a new position. |
| **A9** | One trade date, supplied or today. The engine chooses **what** to sell, not **when**. |
| **A10** | Cess and surcharge excluded. Cess is a flat multiplier on the final figure, so it cannot change which lots are best; surcharge depends on total income, which a single-portfolio calculation does not know. |

## Limits

- **Choosing specific lots is not generally available in India.** A demat account
  sells oldest-first. So read the output two ways: as the ceiling on what lot
  selection is worth, and as an executable plan where lots are already separable.
- **The engine is myopic about timing.** A lot at 350 days is taxed at 20% today
  and 12.5% in a fortnight, which often beats any lot choice available now.
- **Weights drift by a rupee or two.** Whole shares cannot absorb the proceeds
  exactly, so `all_ltcg` leaves ₹600 of ₹5,25,000 uninvested. The buy leg is
  capped by what the sells raise, so the plan always pays for itself, and buying
  realises nothing so this touches no tax.
- **Not modelled:** cess and surcharge, losses carried across financial years,
  exemption already used earlier in the year, a cash balance, opening new
  positions, corporate actions, and the price impact of the trades themselves.
- **Not tax advice.** A deterministic calculator over a documented set of
  assumptions.

---

## Layout

```
app/
  models.py      data types and the tax rates
  tax.py         holding period, the four-line tax function, the set-off calculation
  portfolio.py   weights -> shares to sell per ticker, and the buy leg
  optimizer.py   the linear program, plus the FIFO and LTFO baselines
  explain.py     the per-lot reasoning, and the ledger behind a partial split
  engine.py      orchestration; implements no tax rule of its own
  ingest.py      CSV parsing
  api.py         FastAPI routes
ui.py            Streamlit front end; renders the plan, computes nothing
samples/         seven scenarios, three CSVs each
tests/           test_tax, test_optimizer, test_engine, test_ingest, test_api, test_ui, bruteforce
requirements.txt pinned dependencies for the pip path; mirrors uv.lock
```
