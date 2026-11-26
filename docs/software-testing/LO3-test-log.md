# LO3 — Test log (lo-agent)

LO 3: "Apply a wide variety of testing techniques and compute test coverage and
yield according to a variety of criteria."

Log for the testing run on lo-agent for LO3, planned testing is in
[`LO2-test-plan.md`](LO2-test-plan.md) and the results will be in
[`LO3-evidence/`](LO3-evidence/)

## Summary

| Technique | What ran | Adequacy criterion | Result |
|---|---|---|---|
| Combinatorial testing: the composability DAGs | 12 DAGs: every combination of 2 system prompts × 2 tools × 3 ways to stop. Each ran through the terminal interface (TUI), over the real session server, on the mock model. | The LO2 plan's rule (2.2), on the mock: every DAG in the pool passes the break list | 12 of 12 pass. All 12 combinations, 16 pairs and 7 options are covered, counted from the run tags. Every run replays identically. |
| Planted faults, to check that the DAG suite finds real faults | 5 one-line faults in lo, one at a time | Each fault makes at least one DAG fail | 5 of 5 caught |
| Structural coverage | The whole suite (600 tests) under coverage.py | Statement and branch coverage | 60.3% of statements and 54.3% of branches. By area, from 10% of statements (`jlens`, whose tests were skipped) to 98% (`events`). |
| Mutation testing (fault-based) | mutmut 3.8.1 on `agent/goal.py` (1004 mutants) | Killed ÷ total | 77.1% → 91.4% after 19 new tests (94.1% without the 28 equivalent mutants) |
| Unit and integration tests (existing, plus the new ones) | `uv run --group dev pytest` | All pass | 600 passed, 8 skipped, 9 deselected |

Yield: the course materials don't define it, so we use it to mean the share of the faults in
the code that the tests find. Real faults can't be counted, so we estimate it the way the
course's fault-based testing slides judge a test suite: plant known faults, then count how many
the tests catch. For `agent/goal.py` that's the mutation score; the 77.1% from the first run is
the estimate for the suite as it was. The 91.4% after the new tests isn't an independent
estimate, because those tests were written after seeing which mutants survived. For the DAG
suite it's the 5 planted faults, all of which were caught.

## Runs

All runs were on 23 Nov 2026, with our existing python environment and lo 0.2.24 at commit
912b371 plus this work.

### 1. The composability DAGs, on the mock

`uv run --group dev pytest tests/test_dag_composability.py`. Setting
`LO_DAG_REPORT=<file>` also writes the results page,
[`LO3-evidence/dag-results.md`](LO3-evidence/dag-results.md).

Each DAG tests one lo feature per step:

1. The system prompt: context engineering with and without AGENTS.md
2. A tool call: memory or session_search
3. A stop: The model calls eos, the step budget is used up, or the doom-loop guardrail

After the full round we should see the session come back to a ready state.

- Driver: The journey driver which simulates a user in the TUI
- Stub: MockLlamaCpp, scripted
- Oracle: The plan's break list checked against the event log, the requests received, and the
  screen
  - crashes: exceptions that create end events
  - swallowed errors: stop reason should appear on screen and zero tool calls can fail
  - incorrect outputs: tools need to run the expected number of times and return the correct
    results; the result must be inserted into conversation history and go back to the model;
    the round cannot stop in an unexpected way after the expected number of model calls

Each run is tagged with DAG and options

The coverage of the pool is counted using those tags

Result: 12 of 12 passed, 100% of the DAGs, pairs, and options covered. Each DAG takes about 9 s;
the file takes about 100 s.

The oracle itself is tested with 5 synthetic runs

### 2. Planted faults: does the DAG suite find real faults?

`uv run --group dev python tests/run_dag_faults.py`. Results:
[`LO3-evidence/dag-fault-injection.md`](LO3-evidence/dag-fault-injection.md).

Each of these is a planted fault that breaks one item of the break list. Each was caught
successfully by the DAGs

| Fault | DAGs that failed |
|---|---|
| F1: The chat doesn't show what went wrong on a failed turn | the 4 step-budget DAGs. The guardrail DAGs still showed their reason on the guardrail's own line, so for them nothing was hidden. |
| F2: AGENTS.md never reaches the system prompt | the 6 AGENTS.md DAGs |
| F3: a turn that uses up its step budget logs no end event | the 4 step-budget DAGs |
| F4: a tool's result isn't sent back to the model | all 12, after the oracle fix in issue 6 below. Before the fix, only the 6 session_search DAGs. |
| F5: the doom-loop guardrail never stops a run | the 4 guardrail DAGs |

### 3. Branch coverage of the whole suite

`uv run --group dev coverage run -m pytest`, then `uv run --group dev coverage
report`. Results: [`LO3-evidence/coverage.md`](LO3-evidence/coverage.md).

Result: 7969 of 13215 statements (60.3%) and 2215 of 4078 branches (54.3%)
ran. The figure coverage.py reports, which combines the two, is 58.9%.

- **Best covered:** `events` (98.4% of statements), `skills` (91.9%),
  `server` (91.7%) and `agent` (89.6%).
- **Least covered:** `jlens` (10.1%) and `native` (18.0%). Their tests were
  skipped in this run, because the optional packages they need (numpy, gguf,
  torch, transformers) aren't in the dev group. Next are `tui` (48.5%) and
  `cli` (55.5%).
- **The modules this work tested most:**
  - `agent/goal.py`: 98.4% of statements, 94.4% of branches;
  - `events/export.py`: 100% of both;
  - `events/log.py`: 100% of statements, 94.1% of branches.

Code run in subprocesses isn't measured, and the pseudo-terminal and
live-model tests are deselected by default.

### 4. Mutation testing of `agent/goal.py`

`uv run --group dev --with mutmut==3.8.1 mutmut run --max-children 4`. Results:
[`LO3-evidence/mutation-goal.md`](LO3-evidence/mutation-goal.md).

The first run was 77.1% successful on the mutations. The ones that got through showed 13 groups
of gaps in the tests. No code changes needed. Examples:

- a failed round at the cap could have run past the cap;
- a chat turn between pause and resume could have hidden the resume;
- the judge could have been shown no conversation at all.

19 tests were added over four more runs, and the score rose to 918 of 1004
(91.4%). Of the 86 survivors:
- 56 change only wording sent to the model, which the mock ignores;
- 28 are equivalent: they can't change behaviour;
- 2 are untested exact limits.

## Issues found, and what happened to them

| # | Found | How | Issue | What happened |
|---|---|---|---|---|
| 1 | 19 Nov | Designing goal mode's step budget | The 95-step budget counted model calls across the whole conversation. Once a chat passed 95 calls, every later message failed at once with "max_steps exceeded". | Fixed in a765657: the budget counts from the start of each turn. Tests are in `tests/test_agent.py`. |
| 2 | 19 Nov | A script that interrupts a follow-up turn | Interrupting a follow-up turn logged no end event, because the check accepted any end event in the run and the first turn's counted. The run stayed "running", so the TUI refused every later message. | Fixed in be2fec8. Tests are in `tests/test_server.py`. |
| 3 | 19 Nov | A script that follows a follow-up turn's events from the server | A follower using `replay=1&once=1`, which is how the TUI follows a turn, stops at the first turn's end event and misses the follow-up turn. At the server level, a request to approve a tool is never delivered, and the tool is denied after the 120 s wait. The effect in the TUI, no live tokens in follow-ups, was worked out from the code rather than observed. | Open. Planned fix: an `after=<seq>` option on the events route. |
| 4 | 19 Nov | A script that cancels a running command | Cancelling `HostSandbox.exec` doesn't stop the command it started. From the code, this affects a shell command when its turn is interrupted, and a goal's verify command when the goal is stopped. | Open. |
| 5 | 23 Nov | Mutation testing of `agent/goal.py` | Gaps in the tests (13 groups), not bugs in the code. See run 4. | Fixed: 19 tests added. The score afterwards, 91.4%, isn't an independent estimate (see the yield note). |
| 6 | 23 Nov | Planting fault F4 under the DAG suite | The oracle's check that a tool's result goes back to the model looked for the result's text anywhere in the next request. A memory call returns "ok", and that text also appears elsewhere in the request, so a lost result went unnoticed in the 6 memory DAGs. | Fixed: the oracle now requires a tool message, paired with the call, that carries the result. A planted-break test covers this case, and F4 is now caught by all 12 DAGs. |
| 7 | 23 Nov | Timing the DAG runs | Each DAG took about 21 s. The driver types one key at a time (about 0.1 s a key), and the prompts were long. | Fixed in the tests: shorter prompts, about 9 s a DAG. |
| 8 | 23 Nov | Reading the mutation survivors | `Goal.set_seq` and `Round.start_seq` are written but never read. | Open. |
| 9 | 23 Nov | Branch coverage (run 3) | The new full export's lines for goal rounds, the judge, the verify command, goal status, context compaction, a guardrail's fatal reason and the nesting limit never ran in any test. | Fixed: 2 tests added. `events/export.py` is now at 100% of statements and branches. |

## Planned in the LO2 test plan, not run yet

- The composability DAGs tested with a real model. The plan needs every DAG to pass both on the
  mock and the real model, and we've only run the mock
- The comparison with the other harnesses
- Replay of recorded sessions against a real model on real hardware. We've only run the mock
  replay
- Longer runs, the source of the different orders. They don't yet have a stopping rule.
- The pseudo-terminal journeys and the live-model journeys
- Running the tests in a CI pipeline. There's no pipeline yet.
