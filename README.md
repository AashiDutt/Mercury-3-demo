# Mercury 3: measuring a diffusion LLM through its API

Code and logged results for a tutorial on [Mercury 3](https://docs.inceptionlabs.ai),
Inception's diffusion large language model.

A diffusion LLM does not generate left to right. This repository measures what that
actually means from a client, using one fixed task — a customer-support escalation
case — and changing one model control at a time.

Everything here is reproducible with an API key. No GPU.

## What the runs showed

Four findings, all from the logged data in this repository.

**Mercury rewrites text it has already shown you.** In one 12-state diffusion run,
**0 of 11** state transitions were a pure append. Every transition diverged from the
one before it within the first **14 characters** of a 2,636-character answer, six of
them after a single character. The final state replaced 2,622 of its 2,636
characters. An autoregressive model cannot do this — a committed token stays
committed.

**`diffusing=True` changes the contract, not the clock.** Standard streaming chunks
**append**; diffusion chunks **replace**, each one carrying the whole current answer.
Diffusion was faster at the median (2.560 s against 3.375 s) but p = 0.31, and an
earlier scenario produced the opposite ordering. Treat the latency as undetermined
and the semantics as the real difference.

**Reasoning tokens share `max_completion_tokens`.** At `reasoning_effort="high"` with
a 1,600-token cap, all **1,539 of 1,539** completion tokens went to reasoning. The
API returned `content: None` and `finish_reason: "length"` — no error. Inception's
docs state the budget but not that reasoning competes for it.

**Higher effort buys hidden work, not a longer answer.** Median latency rose 2.77×
(2.936 s → 8.146 s) and median reasoning rose from 828 to 2,292 tokens, while the
visible answer did not reliably change (468 vs 412 tokens, p = 0.15, ranges
overlapping). One `low` trial reported **2** reasoning tokens and still answered
completely in 1.60 s.

## Requirements

- Python 3.9+
- `pip install -r requirements.txt` (one package, `inceptionai`)
- An Inception API key, exported as `INCEPTION_API_KEY`

New accounts get 100 million free tokens. The 12 calls here that report usage
consumed 9,617 input and 23,014 output tokens — under one cent at list price.

## Quick start

`mercury3_demo.ipynb` is self-contained: it embeds the case, installs the SDK, and
asks for your key with `getpass`. Open it in Colab or Jupyter and run every cell.
Three API calls, about ten seconds.

It covers the two most visual findings — the diffusion replay with a
longest-common-prefix proof, and the token-budget trap.

## Reproducing every measurement

```bash
export INCEPTION_API_KEY="your_key_here"
./run_support_experiments.sh
```

23 API calls, two to three minutes. The script runs the harness five times and
writes only to `*_support.json` filenames, so a re-run cannot clobber a previous
scenario's data.

To run one configuration by hand:

```bash
python3 mercury3_incident_demo.py --mode diffusion --repeat 5 --effort low \
  --incident case_support.json --results results_diffusion_support.json
```

`--mode` takes `dry-run`, `probe`, `compare`, `standard` or `diffusion`. `dry-run`
prints the assembled prompt and makes no API call, which is the cheapest way to
check a new case file.

## Files

| File | What it is |
| --- | --- |
| `mercury3_demo.ipynb` | Self-contained notebook. Needs nothing else here. |
| `mercury3_incident_demo.py` | The measurement harness. Standard library only, plus the SDK. |
| `case_support.json` | The task: a support case carrying its own system and task prompts. |
| `run_support_experiments.sh` | All 23 calls, in order. |
| `requirements.txt` | `inceptionai` |

The `results_*.json` and `diffusion_states_support.json` files are **outputs** of a
run, committed so the findings above can be checked without spending tokens.
`diffusion_states_support.json` holds the full 12-state trajectory and is the only
way to verify the rewrite claim yourself.

## About the case file

`case_support.json` carries its own `system_prompt`, `task_prompt` and
`packet_label`. The harness strips those three keys before serializing the rest as
evidence, so one harness can drive any scenario — pass a different file to
`--incident`.

The case is synthetic and deliberately unresolvable. Two payment entries 94 seconds
apart share a card but differ in status and authorization code; the fulfillment log
shows an orphan reservation two seconds after the second payment, later released;
the customer's history contains a precedent that contradicts the stated policy; and
the authorization expiry date appears nowhere in the evidence, only in
`available_agent_actions` as something the agent could request. A good answer has to
name what it cannot conclude.

Keep all of it. The trimmed versions that appear in some drafts drop
`available_agent_actions`, which is the only reason the model knows the expiry date
is obtainable.

## Reading the data

Each record in `results_compare_support.json` carries `usage`, with
`completion_tokens_details.reasoning_tokens` nested inside it — the top-level
`completion_tokens` includes the reasoning. Two fields are undocumented and survive
only because the SDK's `BaseModel` allows extra attributes:

- `diffusion_meta.diffusion_progress` — not a progress ramp. It is `0` for every
  state and `1` on the last one.
- `reasoning_details` — the reasoning trace, returned **encrypted**
  (`"type": "reasoning.encrypted"`). You are billed for those tokens and can count
  them, but you cannot read them.

## Limitations

One machine, one network, one synthetic case, one day: 2026-11-02, `mercury-3` at
`temperature=0.75`. Five trials per configuration, with medians reported because the
distributions overlap. Every comparison carries an exact two-sided Mann-Whitney
p-value; with five trials per side the smallest achievable p is **0.008**, so
p = 0.15 means "these runs cannot tell the two apart", not "the difference is small".

The low and high requests were grouped rather than interleaved, so provider load
could affect the latency comparison. Prompt-cache hits varied between requests.

These are observations about API behaviour, not a benchmark of the model.
