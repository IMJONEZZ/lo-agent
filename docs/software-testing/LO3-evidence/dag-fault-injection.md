# Planted faults: does the DAG suite find real faults?

This page checks whether the composability DAGs fail when lo breaks. Each fault is a one-line
change to lo's source that breaks one item of the LO2 test plan at one step of the round.

`uv run --group dev python tests/run_dag_faults.py` copies `src/` to a temp folder and creates the
fault. The copy then makes sure the tests import the broken code, and the script then reads which
DAGs failed from pytest's report. This way the repo stays intact and doesn't break.

The run was on 23 Nov 2026, with our existing environment and lo 0.2.24 plus this work.

## Results

| Fault | What it breaks | DAGs that failed | Which |
|---|---|---|---|
| F1-error-not-shown | the chat no longer shows a failed turn's error (an error the user doesn't see) | 4 of 12 | `agents-md.memory.step-budget`: AssertionError: on screen: step 'round agents-md.memory.step-budget': marker 'max_steps exceeded' not seen within 60.0s<br>`agents-md.session_search.step-budget`: AssertionError: on screen: step 'round agents-md.session_search.step-budget': marker 'max_steps exceeded' not seen within 60.0s<br>`no-agents-md.memory.step-budget`: AssertionError: on screen: step 'round no-agents-md.memory.step-budget': marker 'max_steps exceeded' not seen within 60.0s<br>`no-agents-md.session_search.step-budget`: AssertionError: on screen: step 'round no-agents-md.session_search.step-budget': marker 'max_steps exceeded' not seen within 60.0s |
| F2-agents-md-dropped | AGENTS.md never reaches the system prompt (an incorrect output at step 1) | 6 of 12 | `agents-md.memory.finish`: AssertionError: incorrect output: AGENTS.md rule in the system prompt = False<br>`agents-md.memory.step-budget`: AssertionError: incorrect output: AGENTS.md rule in the system prompt = False<br>`agents-md.memory.guardrail`: AssertionError: incorrect output: AGENTS.md rule in the system prompt = False<br>`agents-md.session_search.finish`: AssertionError: incorrect output: AGENTS.md rule in the system prompt = False<br>`agents-md.session_search.step-budget`: AssertionError: incorrect output: AGENTS.md rule in the system prompt = False<br>`agents-md.session_search.guardrail`: AssertionError: incorrect output: AGENTS.md rule in the system prompt = False |
| F3-no-end-event-on-budget | a turn that uses up its step budget logs no end event (not returning to ready) | 4 of 12 | `agents-md.memory.step-budget`: AssertionError: on screen: step 'round agents-md.memory.step-budget': marker 'max_steps exceeded' not seen within 60.0s<br>`agents-md.session_search.step-budget`: AssertionError: on screen: step 'round agents-md.session_search.step-budget': marker 'max_steps exceeded' not seen within 60.0s<br>`no-agents-md.memory.step-budget`: AssertionError: on screen: step 'round no-agents-md.memory.step-budget': marker 'max_steps exceeded' not seen within 60.0s<br>`no-agents-md.session_search.step-budget`: AssertionError: on screen: step 'round no-agents-md.session_search.step-budget': marker 'max_steps exceeded' not seen within 60.0s |
| F4-tool-result-not-sent-back | a tool's result isn't sent back to the model (step 2 → step 3 broken) | 12 of 12 | `agents-md.memory.finish`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`agents-md.memory.step-budget`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`agents-md.memory.guardrail`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`agents-md.session_search.finish`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`agents-md.session_search.step-budget`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`agents-md.session_search.guardrail`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`no-agents-md.memory.finish`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`no-agents-md.memory.step-budget`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`no-agents-md.memory.guardrail`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`no-agents-md.session_search.finish`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`no-agents-md.session_search.step-budget`: AssertionError: the tool's result was not sent back to the model, paired with its call<br>`no-agents-md.session_search.guardrail`: AssertionError: the tool's result was not sent back to the model, paired with its call |
| F5-doom-loop-cap-off | the doom-loop guardrail never stops a run (step 3's guardrail stop broken) | 4 of 12 | `agents-md.memory.guardrail`: AssertionError: on screen: step 'round agents-md.memory.guardrail': marker 'doom loop' not seen within 60.0s<br>`agents-md.session_search.guardrail`: AssertionError: on screen: step 'round agents-md.session_search.guardrail': marker 'doom loop' not seen within 60.0s<br>`no-agents-md.memory.guardrail`: AssertionError: on screen: step 'round no-agents-md.memory.guardrail': marker 'doom loop' not seen within 60.0s<br>`no-agents-md.session_search.guardrail`: AssertionError: on screen: step 'round no-agents-md.session_search.guardrail': marker 'doom loop' not seen within 60.0s |

## Reading the results

Every fault was caught successfully by the DAGs. F2 failed only the AGENTS.md DAGs, F5 only the
guardrail DAGs, and F4 all DAGs because every DAG calls a tool. F1 and F3 both failed only the
step-budget DAGs, for different reasons. F3 failed because the round's error never reached the
screen and the session didn't come back to ready. F1 the guardrail DAGs still passed, because a
guardrail stop also shows its reason (e.g. "doom loop: ..."), so hiding the error line hid nothing
from the user, they still saw the guardrail stop reason. A finish is successful and doesn't have an
error.

## A problem we discovered with the oracle

The first run of F4 only caught the session_search DAGs, and the memory DAGs passed with the
fault. The memory tool returns ok, which also appears elsewhere in the request, so the oracle
passed even though the memory tool's result never went back to the model.

Now the oracle requires a tool message with the call's id and
`test_the_oracle_catches_a_planted_break` has a case to test against: When the result's expected
text appears elsewhere in the request, but not as the result. With that fix, F4 fails all of the
DAGs.
