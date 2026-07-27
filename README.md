# Problem 2 — LLM-as-Judge Evaluation Pipeline

A structured-verdict judging pipeline: feed it a test suite (input + model output,
optionally a reference answer), get back rubric-scored verdicts, suite-level
aggregates, and — for comparing two configs — a declared winner with a position-bias
sanity check baked in.

## Architecture

```
test_suite.json (15 cases, tagged incl. adversarial probes)
        │
        ▼
judge/schemas.py  (TestCase, PointwiseVerdict, PairwiseVerdict — typed contracts)
judge/prompts.py  (rubric + calibration anchors + pointwise/pairwise prompt builders)
judge/judge.py    (calls Claude, robust JSON parsing w/ 1 corrective retry, audit log)
judge/bias.py     (position-swap check, verbosity-padding probe, sycophancy probe)
judge/aggregate.py(suite report: pass rate, mean scores; A/B winner declaration)
        │
        ▼
pipeline.py  score    -> results/report.json
pipeline.py  compare  -> results/ab_report.json  (pairwise + position-bias flip rate)
validation/validate.py -> results/judge_validation.json (agreement, test-retest, adversarial)
```

## Setup

```bash
cd problem2_judge
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY; JUDGE_MODEL defaults to claude-opus-4-8
```

## Step-by-step: run it

**1. Offline logic tests first** (no API key needed — JSON parsing, aggregation,
Cohen's kappa, Pearson correlation, position-bias remapping logic, all pure functions):

```bash
python judge/test_offline.py
```

**2. Score the test suite (pointwise, rubric-based):**

```bash
python pipeline.py score --suite test_suite.json --out results/report.json
```

Each of the 15 cases gets scored 1-5 on its relevant rubric criteria
(correctness, faithfulness, completeness, instruction_following, tone, safety),
with a grounded rationale per criterion. Output includes a suite summary (pass
rate, mean scores per criterion, parse-failure rate, total judge tokens spent).

**3. Compare two configs head-to-head (pairwise + position-bias check):**

```bash
python pipeline.py compare --suite test_suite.json \
  --config-a outputs/prompt_v1_outputs.json \
  --config-b outputs/prompt_v2_outputs.json \
  --out results/ab_report.json
```

The sample `outputs/` files simulate a "prompt v1 vs v2" comparison on 6 of the
15 cases (v2 outputs are visibly more complete/grounded). Every case is judged in
**both physical orders** and the winner is only counted if both orders agree — see
Bias Handling below.

**4. Validate the judge itself** (agreement with human labels, test-retest
consistency, adversarial probes):

```bash
python -m validation.validate --out results/judge_validation.json
```

## Judging modes: pointwise vs pairwise — when each fits

- **Pointwise** (`judge_pointwise`, used by `pipeline.py score`): scores one output
  in isolation against an absolute rubric. Use this for regression gates ("did this
  response clear our safety/quality bar"), monitoring production output quality over
  time, or any case where there's no second output to compare against.
- **Pairwise** (`judge_pairwise`, used by `pipeline.py compare`): compares two
  outputs for the same input and picks a winner. Use this for A/B decisions — "is
  prompt v2 actually better than v1?" — because two independent pointwise scores
  from separate calls don't calibrate against each other reliably (this is the
  "score clustering" bias below), while a direct head-to-head comparison forces a
  sharper, better-calibrated relative judgment.
- **Reference-based vs reference-free**: both modes accept an optional
  `expected_output` in a test case. When present, the judge grades against it
  directly (higher precision, needs gold answers). When absent, it falls back to
  rubric + world-knowledge grading (more scalable, necessary for open-ended
  generation, but noisier) — see `judge/prompts.py` for exactly how this is worded
  into the system prompt.

## Bias handling

| Bias | Mitigation | Where |
|---|---|---|
| **Position (A/B order)** | Every pairwise comparison is run in **both physical orders**, remapped back to stable labels, and only counted as a real winner if both orders agree; disagreement is recorded as a flip and the case counts as a tie rather than a coin-flip winner. `pipeline.py compare` reports an aggregate flip rate. | `judge/bias.py::run_position_bias_check` |
| **Verbosity / length** | (a) Prompt-level: explicit instruction not to reward length, with a grounding requirement per rationale. (b) Code-level probe: pads a correct terse answer with content-free filler and checks the score doesn't rise. | `judge/prompts.py` (instructions) + `judge/bias.py::run_verbosity_probe` |
| **Self-enhancement** (judge favors its own model family's style) | `JUDGE_MODEL` is configured independently from `GENERATOR_MODEL` in `.env`. This repo defaults to a different Claude tier (Opus judging Sonnet-generated output) as a partial mitigation — the strongest version of this mitigation is a genuinely different vendor (e.g. judge with GPT-4o or Gemini while generating with Claude), which is a one-line swap in `judge/judge.py`'s client construction; not done here since the repo only has Anthropic API access. |
| **Sycophancy / fluency bias** | Prompt-level instruction to penalize confident-but-wrong content. Code-level probe: `c05_confidently_wrong_fluent` in the test suite is a fluently-worded, confidently-stated wrong answer (wrong Apollo 11 astronaut); the probe checks the judge scored correctness/faithfulness low despite the fluent tone. | `judge/prompts.py` + `judge/bias.py::run_sycophancy_probe` |
| **Score clustering** (everything scored 3-4) | Few-shot calibration anchors in the system prompt spell out concretely what a 1, 3, and 5 look like, so the judge has a fixed reference scale instead of free-associating a number. | `judge/prompts.py::CALIBRATION_ANCHORS` |

## Judge validation (`validation/validate.py`)

1. **Agreement with human labels**: all 15 cases have hand-written gold labels in
   `validation/gold_labels.json` (independent human review against the same
   rubric). The script reports raw agreement rate, **Cohen's kappa** (chance-
   corrected agreement on pass/fail, implemented from scratch — see
   `validation/validate.py::cohens_kappa`, no sklearn dependency), and Pearson
   correlation between judge and human numeric scores.
2. **Test-retest consistency**: every case is judged twice independently; the
   script reports how often the pass/fail verdict flips between runs and the mean
   absolute score difference. High flip rate = the judge is noisy, not just biased.
3. **Adversarial probe set**: reuses the verbosity and sycophancy probes above as
   a batch over all suitably-tagged cases (`adversarial_terse_correct`,
   `adversarial_confidently_wrong`), reporting how many probes "fooled" the judge.

## Results

*(Populate by running the commands above with a real `ANTHROPIC_API_KEY` — this
repo ships the harness and pre-verified logic, not fabricated scores, since actual
verdicts depend on the live judge model. All four `results/*.json` files are
written by the scripts above.)*

What's already verified without an API key (see `judge/test_offline.py` and the
inline mock-based tests run during development — 15/15 offline logic tests pass,
plus targeted mock tests of the retry-on-malformed-JSON path, the position-bias
order-remapping logic against both a simulated biased and unbiased judge, and both
adversarial probes against simulated good/bad judges):
- JSON extraction handles clean JSON, markdown-fenced JSON, and JSON-with-prose-
  wrapper; raises cleanly on true garbage (triggering the one-shot corrective retry).
- `aggregate_pointwise_suite` correctly separates parse failures from valid
  verdicts and computes pass rate only over valid ones.
- `declare_ab_winner` respects the tie margin (a 1-point win out of 20 isn't a
  declared winner).
- Position-bias detection correctly distinguishes a judge that's biased toward
  physical order A from one that's just correctly, consistently preferring the
  better content regardless of which slot it's in.

## Discussion: when NOT to trust this judge as ground truth

- **It's still one model's opinion.** Even with calibration anchors and bias
  probes, a single LLM judge (or even judge-model-family) can share correlated
  blind spots with the generator, especially on domains requiring real expertise
  (legal, medical, niche technical correctness) where "sounds right" and "is right"
  diverge in ways a fellow LLM won't reliably catch. Cohen's kappa against human
  labels tells you how far this specific judge has drifted from a human reviewer
  on *this* test suite — it doesn't generalize to domains not represented in
  `test_suite.json`.
- **Small suite, small gold set.** 15 cases (all 15 also human-labeled here) is
  enough to prove the pipeline and catch gross miscalibration, not enough to
  trust a kappa estimate with real statistical confidence — treat it as a smoke
  test, and grow the gold-labeled set before using this to gate real deploys.
- **The position-bias check doubles judge cost** for every pairwise comparison
  (two calls per case). For a large-scale A/B this is a real, deliberate
  trade-off documented in `results/ab_report.json`'s cost fields — worth it for
  a launch decision, possibly overkill for iterating on prompts hourly.
