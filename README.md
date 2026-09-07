# Tax-Aware Rebalancing Engine

Rebalances an Indian listed-equity portfolio to its target weights and picks
**which specific lots to sell** so that the capital gains tax on the plan is as
small as it can be. Lot selection is solved as a linear program with whole-number
variables, so the answer comes back proved optimal — not scored, not ranked, not
filled oldest-first.

**Python 3.11 or newer.** Install with either tool, then pick a way in.

```bash
uv sync                                     # with uv

python3 -m venv .venv && source .venv/bin/activate   # without uv
pip install -r requirements.txt
```

**The UI** — one command, nothing else to start:

```bash
uv run streamlit run ui.py        # or, in the activated venv: streamlit run ui.py
```

Opens on <http://localhost:8501>. `edge_case` is selected in the sidebar by
default — that is the scenario the brief requires. Or upload your own CSVs.

**The API**:

```bash
uv run uvicorn app.api:app        # or: uvicorn app.api:app
```

- Swagger: <http://127.0.0.1:8000/docs>
- **The required edge case, no input needed:** <http://127.0.0.1:8000/demo/edge_case>

**Tests**: `uv run pytest`, or just `pytest` inside the activated venv.

Nothing is installed as a package and there is no build step — both paths only
install dependencies, and everything is imported from the repo root.

---

## Tax rates and sources

| | rate | exemption | provision |
|---|---|---|---|
| Short-term — held 12 months or less | **20%** | none | section 111A |
| Long-term — held more than 12 months | **12.5%**, no indexation | **₹1,25,000** a year | section 112A |

Listed equity on which STT is paid. Source: the Income Tax Department,
<https://incometaxindia.gov.in/>, which hosts the Act and the Finance Acts.

The **Finance (No. 2) Act 2024** set these for transfers on or after
**23 July 2024** — short-term 15% → 20%, long-term 10% → 12.5%, exemption
₹1,00,000 → ₹1,25,000. They have not changed since.

> The brief quotes a ₹1 lakh exemption, which is the pre-23-July-2024 figure.
> This engine uses ₹1,25,000. Every rate lives in one place: `TaxConfig` in
> `app/models.py`.

---

## How lot selection works, and why it minimises tax

### Step 1 — the tax depends on two numbers, not on lots

Write **A** for the net short-term realised gain and **B** for the net
long-term, both signed so losses are negative. Let `s = 20%`, `l = 12.5%` and
`E = ₹1,25,000`.

The order in which losses are used up is fixed by law, not chosen:

- A **long-term loss** can only be set off against long-term gain. It has no
  other use, so it goes there.
- A **short-term loss** saves 20% against short-term gain but at most 12.5%
  against long-term gain, so it is always spent on short-term gain first.
  Anything left over spills across to the long-term side.

Substitute that fixed order and the whole calculation becomes:

```
A ≥ 0 :   tax = s·A + l·max(0, B − E)
A < 0 :   tax =       l·max(0, A + B − E)
```

which is the single expression

```
tax(A, B) = max( 0 ,  s·A ,  l·(A+B−E) ,  s·A + l·(B−E) )
```

**Four straight lines.** `tests/test_tax.py` checks this identity against the
step-by-step calculation in `tax.breakdown()` on 400 random inputs.

### Step 2 — so lot selection is a linear program

Let `xᵢ` be the number of shares sold out of lot `i`. Each lot adds
`gain_per_share × xᵢ` to either A or B, so both are linear in x, and so is each
of the four lines. That makes the problem:

```
minimise    t
subject to  t ≥ aₖ·A(x) + bₖ·B(x) + cₖ      for each of the four lines
            Σ xᵢ over ticker T's lots  =  shares T must give up
            0 ≤ xᵢ ≤ lotᵢ.quantity,  xᵢ a whole number
```

`t` stands in for "the tax". You cannot write `max(...)` in a linear objective,
so instead you add one extra variable, force it above all four lines, and
minimise it — minimising pushes `t` down until it rests on whichever line is
currently largest, which is exactly the tax.

`scipy.optimize.milp` hands this to the HiGHS solver, which returns a proved
global optimum in milliseconds.

### Step 3 — why no ranking rule can match it

**The ₹1,25,000 exemption is one pool shared across the whole portfolio.** So
the real cost of a rupee of long-term gain is 0% until the pool runs out and
12.5% after — which depends on what *every other ticker* is doing. Two things
follow:

- You cannot decide ticker by ticker. Selling the cheap lot in one name changes
  the right answer in another.
- The best answer can land **part-way through a lot**, exactly where the pool is
  exhausted, which is never a lot boundary.

Here is the counterexample, automated in
`tests/test_optimizer.py::test_the_answer_stops_where_the_exemption_runs_out_not_at_a_lot_boundary`.
AAA is forced into ₹10,000 of long-term gain. BBB must give up 100 shares, from
either a lot gaining ₹1,200/share (long-term) or one gaining ₹300/share
(short-term):

| plan | tax |
|---|---|
| all 100 from the long-term lot | ₹625 |
| all 100 from the short-term lot | ₹6,000 |
| **96 long-term + 4 short-term** | **₹265** |

No rule that picks a lot and fills from it produces 96 + 4. Whole numbers matter
for the same reason: the unrestricted answer lands between two shares, so
allowing fractions would return a plan you cannot trade.

A second counterexample in the same file shows the answer splitting one ticker
30/70 across a short-term and a long-term loss lot, because per rupee the
short-term loss is worth more while per share the long-term loss is bigger.

### Step 4 — how the engine explains itself

The brief asks the engine to show its lot-selection reasoning. The easy way is a
sentence template that restates what the solver did — but that text is not
derived from anything, so if the solver were wrong it would read exactly as
confident.

Instead, for every lot in the plan, `app/explain.py` takes the other lots of the
same ticker, moves the shares across, **re-runs the tax calculation**, and
reports the rupee difference. Every "why" sentence therefore contains a
recomputed number, and the sign of that number is itself a check: at a genuine
optimum, no swap can come back cheaper. This is affordable because of Step 1 —
pricing a swap is a few multiplications, not another solve.

---

## The required edge case

One ticker held in two lots, where the rebalance needs only part of the
position: enough that one lot can go entirely as LTCG, but the required quantity
forces a partial sale into a second, STCG-eligible lot.

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
Post-rebalance weights land on 20% / 80% exactly. The remaining 25 shares of L2
are untouched and keep their 2026-04-01 buy date, so a future sale classifies
them correctly.

The reasoning the engine emits for L1:

> ACME: sold 60 of 60 shares from the lot bought 2023-02-10 — 42 months held,
> long-term, LTCG. Cost ₹800.00/share against ₹1,000.00 today, so a gain of
> ₹200.00/share, ₹12,000.00 in all. One more rupee of long-term gain would cost
> 0.00%, because the ₹1,25,000 long-term exemption is not yet used up.
> **Selling 25 share(s) from lot L2 instead (bought 2026-04-01, STCG, gain of
> ₹50.00/share) would have cost ₹250.00 more in tax.**

That ₹250 is the tax function re-evaluated on the swap, not a label.

Automated as
`tests/test_engine.py::test_edge_case_sells_the_whole_long_term_lot_and_part_of_the_short_term_one`
and the three tests after it.

---

## The UI

```bash
streamlit run ui.py
```

One file, `ui.py`, and deliberately thin: it collects input, calls `Engine`, and
renders what comes back. It computes nothing of its own, so there is no second
implementation that could drift away from the API.

- **Portfolio** — a bundled scenario, or upload `lots.csv`, `prices.csv` and
  `targets.csv`. Bad input surfaces the engine's own message, naming the file,
  the row and the offending value.
- **Lot selection** — tax-minimising or oldest-first, plus a toggle for the FIFO
  comparison. Switching to oldest-first replaces the "proved optimal" badge with
  a warning, because that plan carries no such claim.
- Every sold lot gets a card: how much of it went, the gain, what the next rupee
  of gain costs, the engine's own sentence, and an expander pricing what each
  other lot would have cost instead. Lots left partly intact say so, with the
  buy date that is being preserved.

The UI calls the engine directly rather than over HTTP, so there is only one
process to start. The API is the programmatic way in.

---

## Endpoints

| | path | |
|---|---|---|
| `GET` | `/demo/{scenario}` | Run a bundled scenario. No input. |
| `POST` | `/rebalance/upload` | Three CSVs. Works through Swagger's file picker. |
| `POST` | `/rebalance` | The same thing as JSON, for programmatic use. |
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
optional `lot_id` (generated as `{TICKER}-{n}` in file order if absent). Dates
may be `YYYY-MM-DD` or `DD/MM/YYYY`. `prices.csv` — `ticker, current_price`.
`targets.csv` — `ticker, target_weight_pct`, in **percent**, adding to 100.

Nothing is coerced. A bad row stops the run naming the file, the row number and
the offending value, and every such error arrives as a **422** with the message
intact — never a bare 500.

**The response** carries: a summary with the total tax; the plain-English
reasoning; per-ticker buy/sell instructions; a per-lot breakdown of the sells,
each with its own reasoning and priced alternatives; the full set-off
calculation; and the target, before and after weights.

### The FIFO comparison (the optional toggle)

`compare_with_fifo` is on by default. It runs the same sell requirements through
oldest-lot-first as well, and reports both figures plus the saving. Selling
oldest-first is the right baseline because it is what actually happens by
default for shares held in a demat account, so the gap is what picking lots is
worth.

The comparison is also stated in words, because a bare saving of zero reads as
"picking lots achieved nothing" when it usually means FIFO happened to be
optimal here and the engine proved it.

`method=fifo` produces the plan that way instead; it comes back with
`certified_optimal: false`.

### Bundled scenarios

| scenario | shows | tax | oldest-first |
|---|---|---|---|
| `edge_case` | the required 60/40 partial-lot split | ₹150 | ₹150 |
| `all_ltcg` | a plain long-term rebalance | ₹0 | ₹43,437.50 |
| `at_target` | already at target, nothing to do | ₹0 | ₹0 |
| `loss_offset` | a short-term loss sheltering a gain elsewhere | ₹4,000 | ₹20,000 |

`loss_offset` is the clearest demonstration of the asymmetry. Both LOSSCO lots
lose exactly ₹400 a share, but the short-term one cancels short-term gain at
20%, while the long-term one could only cancel long-term gain — of which there
is none, so it would save nothing at all. Selling oldest-first takes the
long-term lot and wastes the loss entirely.

---

## Tests

```bash
uv run pytest     # or: pytest
```

| file | covers |
|---|---|
| `test_tax.py` | holding-period rules, the four-line identity against the step-by-step calculation, set-off order |
| `test_optimizer.py` | the solver against an exhaustive search on 240 random portfolios, plus the counterexamples that rule out simpler rules |
| `test_engine.py` | the three cases the brief requires, rebalancing mechanics, validation, CSV parsing |
| `test_api.py` | every endpoint, both input paths, the error contract |
| `test_ui.py` | the front end's data path: both input sources, the method selector, bad uploads |

The three the brief asks for are the first three sections of `test_engine.py`:
the partial-lot edge case, a straightforward all-long-term rebalance, and a
portfolio already at target.

`tests/bruteforce.py` is an exhaustive reference solver used only by the tests.
It tries every possible split and keeps the cheapest — useless in production,
but it shares no logic with the linear program, so agreement between the two is
real evidence rather than the same idea agreeing with itself.

---

## Assumptions

| | |
|---|---|
| **A1** | Ordinary shares bought and sold on an Indian stock exchange. These rates do not apply to debt funds, unlisted shares, gold, property or derivatives, which are taxed under different rules. |
| **A2** | Holding period counted in calendar months, **strictly more than 12**. The anniversary itself is short-term; a 365-day count is off by one across a leap year. |
| **A3** | Rates as above. Short-term gain gets no exemption. |
| **A4** | Set-off order is forced: long-term loss against long-term gain; short-term loss against short-term gain before long-term gain. |
| **A5** | The ₹1,25,000 exemption applies **after** loss set-off, not before. Genuinely arguable and material; isolated to one line in `tax.breakdown()`. |
| **A6** | Stateless. The full exemption is assumed available, nothing carries between runs, and unused losses are not tracked. A second rebalance in the same financial year would under-state the tax. |
| **A7** | Whole shares only, halves rounded away from zero. |
| **A8** | Target weights add to 100% ± 0.01%, and every targeted ticker must already be held. A ticker held but **absent** from the targets has a 0% target and is sold off. The portfolio is assumed fully invested — there is no idle cash to deploy — and the engine only rebalances between tickers already held, so you cannot set a target for a stock you do not own. |
| **A9** | One trade date, supplied or today. The engine chooses **what** to sell, not **when**. |
| **A10** | Cess and surcharge are excluded. Cess is a flat multiplier on the final figure, so it cannot change which lots are best; surcharge depends on total income, which a single-portfolio calculation does not know. |

---

## Limits

- **Choosing specific lots is not generally available in India.** Shares in a
  demat account are sold oldest-first by default. So read the output two ways:
  as the **ceiling** on what lot selection is worth — the number you would need
  to justify holding lots in separate accounts — and as an executable plan where
  they already are.
- **The engine is myopic about timing.** A lot at 350 days is taxed at 20% today
  and 12.5% in a fortnight, which often beats any lot choice available now.
- **Weights drift by a rupee or two.** Whole shares cannot absorb the proceeds
  exactly, so `all_ltcg` leaves ₹600 of ₹5,25,000 uninvested. The buy leg is
  capped by what the sells actually raise, so the plan always pays for itself,
  and buying realises nothing so this touches no tax.
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
  optimizer.py   the linear program, plus the oldest-first baseline
  explain.py     per-lot reasoning, priced by re-running the tax calculation
  engine.py      orchestration; implements no tax rule of its own
  ingest.py      CSV parsing
  api.py         FastAPI routes
ui.py            Streamlit front end; renders the plan, computes nothing
samples/         four scenarios, three CSVs each
tests/           test_tax, test_optimizer, test_engine, test_api, test_ui, bruteforce
requirements.txt pinned dependencies for the pip path; mirrors uv.lock
```
