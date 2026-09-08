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
sentence template that restates what came out — but that text is not derived
from anything, so if the answer were wrong it would read exactly as confident.

For the two ranking rules the reasoning is straightforward, because they really
do work lot by lot: each names its place in the sequence, the rule that put it
there, how much of the ticker's requirement it covers and how many shares are
left for the next lot. That last part is the mechanic the brief turns on.

The optimal plan is a different problem, and worth being honest about. The
mathematics does not work lot by lot at all — it settles every quantity across
the portfolio at once, so there is no sequence inside it to narrate. Narrating
one would describe an implementation detail rather than a reason; on these
scenarios HiGHS returns the answer without branching even once.

So instead of retracing how the split was reached, the engine **checks the
split**. Every lot that is only partly sold reports the tax with one more share
taken from it and one fewer, each against the cheapest partner lot of the same
ticker. On `loss_priority` the long-term loss lot stops at 57 because one more
share costs ₹5.00 and one fewer costs ₹25.00 — both directions dearer, both
figures recomputed rather than asserted, and the two unequal because the
stopping point sits on a kink rather than in the middle of a slope. Where a
move turns out to be free the engine says so instead, since that means several
plans tie rather than that a boundary was found.

### Why there is no order to report

The obvious objection is that the answer must be *describable* as an order even
if it was not found that way. It is not, and this is worth showing rather than
asserting. Rank every lot by gain per share at the rate that genuinely applies
at the optimum — not the statutory rate, the real one — and fill greedily. If
the answer were an ordering, this would reproduce it:

| scenario | optimal | ranked at the true rates |
|---|---|---|
| `edge_case` | ₹150 | ₹150 |
| `loss_offset` | ₹4,000 | ₹4,000 |
| `exemption_split` | **₹80** | ₹5,625 |
| `exemption_vs_loss` | **₹8,400** | ₹12,000 |
| `loss_priority` | **₹14,700** | ₹16,175 |
| `all_ltcg` | **₹0** | ₹43,437.50 |

Four of six, and the reason is the one that runs through this whole document: a
lot's worth changes as you take more of it. The first 95 shares of a lot can be
free and the 96th cost 12.5%, and no per-lot score — at any rates, statutory or
real — can hold that. There is no ordering to find, which is why the quantities
are solved rather than sorted, and why the optimal plan is always at least as
cheap as any rule that sorts them.

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

The reasoning the engine emits for L1, under the optimal plan:

> Supplies 60 of the 75 shares ACME must give up, taking the whole lot.

and for L2, which is the lot the split lands in:

> Supplies 15 of the 75 shares ACME must give up, 15 of its 40, leaving 25
> untouched.

The buy date, holding period, cost basis and share counts are structured fields
on the same record and sit on screen beside it, so the prose does not restate
them. It carries the fill and nothing else — how much of the requirement this
lot covers, how much of the lot that uses, and what is left behind.

Under `fifo` and `ltfo` the same two lots additionally name their place in the
sequence and where the remainder goes — *"1st pick for ACME: the oldest lot
still available. Supplies 60 of the 75 shares ACME must give up, taking the
whole lot; 15 still to find from the next lot."* — because those rules really
do work through a ticker one lot at a time.

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
- It opens on nothing but the rates and the choice of input. Pick a scenario or
  upload three CSVs and the page fills in.
- **The portfolio comes first**: every lot with its own buy date and cost basis,
  classified LTCG, LTCL, STCG or STCL as at the trade date. Then the shares that
  must leave each ticker — fixed by price and target alone, so it is the same
  whichever lots supply them.
- **The three plans are the navigation.** Their tax sits side by side in three
  cards and clicking one opens it. Only `optimal` is green: the colour marks the
  plan that carries a guarantee, not the one that happens to win on this input.
  FIFO ties the optimum on `edge_case` and stays red, because it cannot tell you
  it has tied.
- Inside a plan: the trades one row per ticker, then the sell leg broken out lot
  by lot with signed gains, then the engine's own sentence for each lot. Lots
  left partly intact say so, naming the buy date being preserved.

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
each with its own reasoning; the full set-off
calculation; and the target, before and after weights.

### The three lot-selection methods

`method` picks one. Only the first claims to be cheapest; the other two come
back with `certified_optimal: false`.

| `method` | rule |
|---|---|
| `optimal` | the linear program above, solved over every lot at once |
| `fifo` | oldest lot first — what a demat account does by default |
| `ltfo` | least tax first out: within each ticker, the lot with the smallest `gain_per_share × statutory rate` goes first |

`ltfo` is there because it is the rule a reader is most likely to reach for
instead of a solver, and it is worth being able to price that instinct. It
ranks each lot in isolation, so it cannot see that the ₹1,25,000 exemption is a
single pool shared across the portfolio — and on the required edge case that
costs it ₹250:

| | `optimal` | `ltfo` | `fifo` |
|---|---|---|---|
| `edge_case` | **₹150** | ₹400 | ₹150 |
| `exemption_split` | **₹80** | ₹1,600 | ₹5,625 |
| `exemption_vs_loss` | **₹8,400** | ₹12,000 | ₹21,375 |
| `loss_priority` | **₹14,700** | ₹16,125 | ₹16,175 |
| `loss_offset` | **₹4,000** | ₹4,000 | ₹20,000 |
| `all_ltcg` | **₹0** | ₹0 | ₹43,437.50 |
| `at_target` | ₹0 | ₹0 | ₹0 |

Neither shortcut is safe. `fifo` is beaten badly on three of the five, which is
unsurprising — it makes no tax decision at all, so whatever it costs is a
coincidence. `ltfo` is the interesting one: it *does* rank on tax, and it still
loses on the very edge case the brief specifies. It sees a short-term lot
gaining ₹50/share (₹10 of tax at 20%) against a long-term lot gaining
₹200/share (₹25 at 12.5%), so it empties the short-term lot first — never
noticing the long-term gain was free, because the exemption had not been
touched. Charging each lot its statutory rate is the whole mistake: the rate
that actually applies depends on the rest of the portfolio.

`compare_with_fifo` additionally reports the FIFO figure and the saving
alongside whichever plan was asked for. It is stated in words as well as
numbers, because a bare saving of zero reads as "picking lots achieved nothing"
when it usually means FIFO happened to be optimal and the engine proved it.

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

#### Where the splits come from

The tax function has two kinks, and each one is a place where the right answer
stops part-way through a lot. Both scenarios below sell **100 shares of
HARVEST**; the only question is which lots supply them.

**`exemption_vs_loss` — a loss spent on gain the exemption already covers is
wasted.** GAINCO is sold off entirely, leaving ₹1,80,000 of long-term gain,
₹55,000 of it above the exemption. HARVEST offers a long-term loss of
₹1,000/share and a short-term loss of ₹400/share.

Per share the long-term loss saves ₹125 (12.5% of ₹1,000) against the
short-term loss's ₹80 (20% of ₹400), so it goes first — and `ltfo` ranks it
first for exactly that reason. But it is only worth ₹125 a share **while there
is long-term gain above the exemption to cancel**. That runs out after 55
shares, and every share after that saves nothing at all. The plan takes 55 from
the long-term lot, landing net long-term gain on ₹1,25,000 exactly, and the
remaining 45 short-term. `ltfo` takes all 100 and throws 45 shares of loss away.

**`loss_priority` — the more valuable loss stops being the more valuable one.**
GAINCO leaves ₹34,000 of short-term gain and ₹3,00,000 of long-term. HARVEST
offers a short-term loss of ₹800/share and a long-term loss of ₹1,000/share.

The short-term loss saves ₹160 a share (20% of ₹800) against ₹125 for the
long-term one, so it goes first, and again `ltfo` gets that right. But it only
saves 20% while there is short-term gain left to cancel. After 43 shares that
gain is gone, the surplus carries to the long-term side at 12.5%, and ₹800 at
12.5% (₹100) is now worth less than the long-term lot's ₹1,000 at 12.5%
(₹125). So the priority flips and the plan takes the last 57 shares long-term.

Both times the ranking rule picks the right lot to start from and has no way to
know when to stop. The stopping point is not a property of the lot — it is the
point where the rest of the portfolio changes what the lot is worth.

`exemption_split` is the one that separates all three. AAA is sold off entirely,
putting ₹10,000 of long-term gain on the books and leaving ₹1,15,000 of
exemption. BBB must give up 100 shares from three lots: ₹1,600/share long-term,
₹1,200/share long-term, and ₹80/share short-term. `fifo` takes the oldest lot
and realises ₹1,60,000 of gain. `ltfo` ranks the short-term lot cheapest and
takes all of it. The best answer takes **95 shares from the ₹1,200 lot** —
filling the exemption to ₹1,24,000 of the ₹1,25,000 available — and the last
5 short-term, for ₹80.

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
| `test_optimizer.py` | the solver against an exhaustive search on 240 random portfolios, plus the counterexamples that rule out simpler rules, and the FIFO and LTFO baselines |
| `test_engine.py` | the three cases the brief requires, the two loss-and-exemption splits, rebalancing mechanics, validation, CSV parsing |
| `test_api.py` | every endpoint, both input paths, the error contract |
| `test_ui.py` | the front end's data path: both input sources, all three plans, the coalesced trades table, bad uploads |

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
| **A5** | The ₹1,25,000 exemption applies **after** loss set-off, not before. Arguable, but it does not change this year's bill: both readings subtract the same amounts from the same base, and on 200,000 random portfolios the two orderings differ by ₹0.00. What the order changes is which resource is left over, which matters only for carry-forward — out of scope under A6. Isolated to one line in `tax.breakdown()`. |
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
