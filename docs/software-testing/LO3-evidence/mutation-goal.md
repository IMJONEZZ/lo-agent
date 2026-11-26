# Mutation testing: `agent/goal.py`

This page measures our canaries in the coal mine (mutations) in `src/local_harness/agent/goal.py`
that are caught. mutmut makes one small change at a time, such as `>=` to `>`, `continue` to
`break`, or a value to `None` and then runs the tests that cover the changed function. The
mutation is caught if a test fails and gets through if every test still passes, and the mutation
score is the number caught.

`goal.py` has the decision logic for the goal skill, which decides what happens after each round
and when a goal stops. Its tests are fast so all 1004 mutations run in under a minute.

The runs used mutmut 3.8.1 on 23 Nov 2026 with the settings in the `[tool.mutmut]` section of our
pyproject.toml. These commands reproduce the final result:

```
uv run --group dev --with mutmut==3.8.1 mutmut run --max-children 4
uv run --group dev --with mutmut==3.8.1 mutmut results
```

## Results

| Run | Tests added before the run | Mutants | Killed | Survived | Timed out | Score |
|---|---|---|---|---|---|---|
| 1 (baseline) | none | 1004 | 774 | 230 | 0 | 77.1% |
| 2 | 9 tests for the decision-logic gaps (items 1–5 and 11 below, and smaller checks) | 1004 | 856 | 148 | 0 | 85.3% |
| 3 | 2 tests and 2 more checks: reading a goal back from the log, the goal tool's schema, the judge's payload | 1004 | 873 | 131 | 0 | 87.0% |
| 4 | 5 tests for what the judge is shown (items 6–10) | 1004 | 905 | 99 | 0 | 90.1% |
| 5 | 3 tests: the judge's message roles, the verify feedback's limits, a guided goal's fields in the prompts | 1004 | 918 | 86 | 0 | 91.4% |

The 77.1% from the first run is the estimate of how good the tests were. The later scores aren't
independent estimates, because each round of new tests was written after seeing which mutations
had been caught. 28 of the final mutations that got through are equivalent, meaning no test will
be able to catch them because they can't change behavior. Leaving them out, the final score is 918
of 976 or 94.1%

Tests in `tests/test_goal.py` increased from 29 to 48, and the 19 new tests are in its last 2
sections. No change to `goal.py` was needed, because every problem was in the tests, not in the
code.

## The gaps the baseline run found

Each item here is a mutation that got through, the behaviour it shows that was untested, and the
test that we added for that behaviour.

1. A failed last round at the cap. `r.number >= goal.cap` → `>` survived in the branch for failed
   rounds, along with 6 other mutants of the same stop, so a goal whose final allowed round failed
   would have been given another round past its cap. That bears on the LO2 requirement that a goal
   ends eventually. Test: `test_a_failed_round_at_the_cap_stops_the_goal`.
2. A chat turn between pause and resume. `continue` → `break` survived in `goal_state`, which then
   stopped reading the log at the first event outside a round, so the goal would have ignored its
   own resume. Test: `test_a_chat_turn_between_pause_and_resume_doesnt_hide_the_resume`.
3. A "true" from the judge followed by a failing verify command. This path never ran. Test:
   `test_a_true_from_the_judge_still_needs_the_verify_command_to_pass`.
4. Stall detection. Comparing each round with the one two back, instead of the one before,
   survived, and would stall a goal whose rounds alternate. Test:
   `test_stalling_compares_each_round_with_the_one_before`.
5. The cap's lower bound, a boundary value. `cap < 1` → `cap <= 1` survived, which would refuse a
   cap of 1. Test: `test_goal_set_payload_accepts_a_cap_of_one_and_explains_refusals`, which also
   checks each refusal's message, since the HTTP route returns it to the client.
6. The judge shown no conversation. `transcript = None` survived in `judge_request`. Test:
   `test_the_judge_request_carries_the_transcript`.
7. The judge reading an older model call. Searching back two events at a time survived. Test:
   `test_the_judge_reads_the_latest_model_call_wherever_it_sits`.
8. Tool results shown twice. Starting one event too early survived. Test:
   `test_tool_results_from_before_the_last_call_appear_once`.
9. The cuts for long tool results (2,000 characters) and long transcripts (60,000). Neither ever
   ran. Test: `test_the_judge_transcript_cuts_long_tool_results_and_then_old_text`.
10. Message content given as a list of parts, such as text plus an image. This never ran. Test:
    `test_the_judge_reads_messages_whose_content_is_a_list_of_parts`.
11. The goal tool's two replies. The replies for "complete" and "blocked" could be swapped (`==` →
    `!=`) unnoticed. Test: `test_the_goal_tools_reply_says_what_happens_next`, which also checks
    the schema the model is shown and a claim with no summary.
12. A guided goal's mode and fields, which were never read back from the log or checked in the
    prompts. Tests: `test_a_guided_goals_mode_and_fields_are_read_back_from_the_log` and
    `test_a_guided_goals_fields_reach_the_prompts`.

## The 86 that still survive

| Kind | Count | Why they survive |
|---|---|---|
| Wording sent to the model | 56 | Changes to prompt text, the judge's question, the goal tool's description and replies, and transcript labels. The mock model ignores wording, so only runs on a real model could show whether one of these changes matters. |
| Equivalent, or unreachable in practice | 28 | They can't change behaviour. 5 change `Goal.set_seq` or `Round.start_seq`, which are stored but never read. 7 change the verify feedback's fallback to the goal's command, which is never used because the log always records the command. 3 only matter when the text to parse is empty, or change the case of a regex that ignores case. 4 only matter for event layouts lo never writes, such as a judged round with no model call. 9 change defaults for fields that every event lo writes already carries, or swap one empty value for another. |
| Untested exact limits | 2 | The transcript cap's default (60,000 → 60,001) and the token cap for a verdict answer (400 → 401). |

Every mutation that got through, with its diff and its kind, is listed in
[`mutation-goal-survivors.md`](mutation-goal-survivors.md). Reading the survivors also showed that
a couple of our starting and setting sequences are written but not read or executed anywhere.
