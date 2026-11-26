# Composability DAG results

Run 2026-11-23 18:21 UTC · lo 0.2.24 · Python 3.13.11 · model: MockLlamaCpp (scripted per DAG)

Command: `LO_DAG_REPORT=<this file> pytest tests/test_dag_composability.py`

12 of 12 DAGs passed. Each DAG is one round (system prompt → tool call → stop) followed by one more message, which checks that the session came back to ready. Replay re-sends every logged model call and compares.

| DAG | system prompt | tool | stop | model calls | tool runs | round ended | follow-up answered | replay | result |
|---|---|---|---|---|---|---|---|---|---|
| `agents-md.memory.finish` | agents-md | memory | finish | 2 | 1 | completed | yes | 3/3 | pass |
| `agents-md.memory.step-budget` | agents-md | memory | step-budget | 8 | 8 | failed: max_steps exceeded | yes | 9/9 | pass |
| `agents-md.memory.guardrail` | agents-md | memory | guardrail | 6 | 4 | failed: guardrails: doom loop: memory called 6 times with identical arguments | yes | 7/7 | pass |
| `agents-md.session_search.finish` | agents-md | session_search | finish | 2 | 1 | completed | yes | 3/3 | pass |
| `agents-md.session_search.step-budget` | agents-md | session_search | step-budget | 8 | 8 | failed: max_steps exceeded | yes | 9/9 | pass |
| `agents-md.session_search.guardrail` | agents-md | session_search | guardrail | 6 | 4 | failed: guardrails: doom loop: session_search called 6 times with identical arguments | yes | 7/7 | pass |
| `no-agents-md.memory.finish` | no-agents-md | memory | finish | 2 | 1 | completed | yes | 3/3 | pass |
| `no-agents-md.memory.step-budget` | no-agents-md | memory | step-budget | 8 | 8 | failed: max_steps exceeded | yes | 9/9 | pass |
| `no-agents-md.memory.guardrail` | no-agents-md | memory | guardrail | 6 | 4 | failed: guardrails: doom loop: memory called 6 times with identical arguments | yes | 7/7 | pass |
| `no-agents-md.session_search.finish` | no-agents-md | session_search | finish | 2 | 1 | completed | yes | 3/3 | pass |
| `no-agents-md.session_search.step-budget` | no-agents-md | session_search | step-budget | 8 | 8 | failed: max_steps exceeded | yes | 9/9 | pass |
| `no-agents-md.session_search.guardrail` | no-agents-md | session_search | guardrail | 6 | 4 | failed: guardrails: doom loop: session_search called 6 times with identical arguments | yes | 7/7 | pass |

## Coverage, counted from the runs' RUN_TAG events

| criterion | covered | total | % |
|---|---|---|---|
| dags (3-wise) | 12 | 12 | 100% |
| pairs (2-wise) | 16 | 16 | 100% |
| options (1-wise) | 7 | 7 | 100% |
