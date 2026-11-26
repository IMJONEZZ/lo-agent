# Surviving mutants: `agent/goal.py` (final run)

The 86 mutants that survived run 5 of [`mutation-goal.md`](mutation-goal.md): mutmut 3.8.1, 23 Nov 2026. Each heading gives the function, mutmut's mutant number and the kind:

- wording: a change to text sent to the model, which the mock model ignores;
- equivalent: a change that can't alter behaviour, or only for event layouts lo never writes;
- limit: an untested exact limit.

In each diff, `-` is the original line and `+` is the mutant.

### Goal.from_set #1 [equivalent]
```diff
-def from_set(cls, p: dict[str, Any], seq: int = 0) -> "Goal":
+def from_set(cls, p: dict[str, Any], seq: int = 1) -> "Goal":
```

### Goal.from_set #7 [equivalent]
```diff
-               fields=dict(p.get("fields") or {}), set_seq=seq,
+               fields=dict(p.get("fields") or {}), set_seq=None,
```

### goal_state #2 [equivalent]
```diff
-    current: Round | None = None  # the round that events are attributed to
+    current: Round | None = ""  # the round that events are attributed to
```

### goal_state #7 [equivalent]
```diff
-            goal = Goal.from_set(p, ev.seq)
+            goal = Goal.from_set(p, None)
```

### goal_state #9 [equivalent]
```diff
-            goal = Goal.from_set(p, ev.seq)
+            goal = Goal.from_set(p, )
```

### goal_state #10 [equivalent]
```diff
-            current = None
+            current = ""
```

### goal_state #22 [equivalent]
```diff
-            goal.reason = p.get("reason") or ""
+            goal.reason = p.get("reason") or "XXXX"
```

### goal_state #30 [equivalent]
```diff
-                            start_seq=ev.seq)
+                            start_seq=None)
```

### goal_state #41 [equivalent]
```diff
-            current = Round(number=int(p["round"]), source=p.get("source") or "goal",
+            current = Round(number=int(p["round"]), source=p.get("source") or "XXgoalXX",
```

### goal_state #42 [equivalent]
```diff
-            current = Round(number=int(p["round"]), source=p.get("source") or "goal",
+            current = Round(number=int(p["round"]), source=p.get("source") or "GOAL",
```

### goal_state #59 [equivalent]
```diff
-                current.work.append(f"{p.get('name')}:{p.get('arguments') or ''}")
+                current.work.append(f"{p.get('name')}:{p.get('arguments') or 'XXXX'}")
```

### goal_state #72 [equivalent]
```diff
-                current.error = p.get("error") or ""
+                current.error = p.get("error") or "XXXX"
```

### decide #182 [equivalent]
```diff
-    return _not_yet(goal, r, "")
+    return _not_yet(goal, r, None)
```

### _verify_reason #6 [wording]
```diff
-    out = (v.get("output") or "").strip()
+    out = (v.get("output") or "XXXX").strip()
```

### _verify_reason #15 [equivalent]
```diff
-    head = f"`{v.get('command') or goal.verify}` exited {v.get('exit_code')}"
+    head = f"`{v.get('command') and goal.verify}` exited {v.get('exit_code')}"
```

### _verify_reason #16 [equivalent]
```diff
-    head = f"`{v.get('command') or goal.verify}` exited {v.get('exit_code')}"
+    head = f"`{v.get(None) or goal.verify}` exited {v.get('exit_code')}"
```

### _verify_reason #17 [equivalent]
```diff
-    head = f"`{v.get('command') or goal.verify}` exited {v.get('exit_code')}"
+    head = f"`{v.get('XXcommandXX') or goal.verify}` exited {v.get('exit_code')}"
```

### _verify_reason #18 [equivalent]
```diff
-    head = f"`{v.get('command') or goal.verify}` exited {v.get('exit_code')}"
+    head = f"`{v.get('COMMAND') or goal.verify}` exited {v.get('exit_code')}"
```

### _verify_reason #23 [wording]
```diff
-    return f"{head}:\n{tail}" if tail else head
+    return f"{head}:\n{tail}" if (tail) or True else head
```

### _after_verify #20 [equivalent]
```diff
-        return _stop(MET, _verify_reason(goal, r.verify), r)
+        return _stop(MET, _verify_reason(None, r.verify), r)
```

### _after_verify #32 [equivalent]
```diff
-        return _not_yet(goal, r, _verify_reason(goal, r.verify))
+        return _not_yet(goal, r, _verify_reason(None, r.verify))
```

### _after_verify #45 [equivalent]
```diff
-    return _stop(PAUSED, "the verify command is broken: " + _verify_reason(goal, r.verify), r)
+    return _stop(PAUSED, "the verify command is broken: " + _verify_reason(None, r.verify), r)
```

### _fields_block #2 [wording]
```diff
-        return ""
+        return "XXXX"
```

### _fields_block #7 [wording]
```diff
-    return "\n" + "\n".join(f"{k}: {v}" for k, v in goal.fields.items()) + "\n"
+    return "\n" + "XX\nXX".join(f"{k}: {v}" for k, v in goal.fields.items()) + "\n"
```

### kickoff_prompt #1 [wording]
```diff
-    verify = (f"\nThe goal counts as met when `{goal.verify}` exits with code 0.\n"
-              if goal.verify else "")
+    verify = None
```

### kickoff_prompt #2 [wording]
```diff
-              if goal.verify else "")
+              if (goal.verify) and False else "")
```

### kickoff_prompt #3 [wording]
```diff
-              if goal.verify else "")
+              if (goal.verify) or True else "")
```

### kickoff_prompt #4 [wording]
```diff
-              if goal.verify else "")
+              if goal.verify else "XXXX")
```

### kickoff_prompt #6 [wording]
```diff
-        "Work toward this goal until it is met. You work in rounds: after each of "
+        "XXWork toward this goal until it is met. You work in rounds: after each of XX"
```

### kickoff_prompt #7 [wording]
```diff
-        "Work toward this goal until it is met. You work in rounds: after each of "
+        "work toward this goal until it is met. you work in rounds: after each of "
```

### kickoff_prompt #8 [wording]
```diff
-        "Work toward this goal until it is met. You work in rounds: after each of "
+        "WORK TOWARD THIS GOAL UNTIL IT IS MET. YOU WORK IN ROUNDS: AFTER EACH OF "
```

### kickoff_prompt #9 [wording]
```diff
-        "your replies, the harness checks whether the goal is met, and if it isn't, "
+        "XXyour replies, the harness checks whether the goal is met, and if it isn't, XX"
```

### kickoff_prompt #10 [wording]
```diff
-        "your replies, the harness checks whether the goal is met, and if it isn't, "
+        "YOUR REPLIES, THE HARNESS CHECKS WHETHER THE GOAL IS MET, AND IF IT ISN'T, "
```

### kickoff_prompt #11 [wording]
```diff
-        "budget of model calls.\n\n"
+        "XXbudget of model calls.\n\nXX"
```

### kickoff_prompt #12 [wording]
```diff
-        "budget of model calls.\n\n"
+        "BUDGET OF MODEL CALLS.\n\n"
```

### kickoff_prompt #13 [wording]
```diff
-        "status \"complete\" and a summary of the evidence. If the goal can't be met "
+        "XXstatus \"complete\" and a summary of the evidence. If the goal can't be met XX"
```

### kickoff_prompt #14 [wording]
```diff
-        "status \"complete\" and a summary of the evidence. If the goal can't be met "
+        "status \"complete\" and a summary of the evidence. if the goal can't be met "
```

### kickoff_prompt #16 [wording]
```diff
-        "(it needs access you don't have, a decision from the user, or it contradicts "
+        "XX(it needs access you don't have, a decision from the user, or it contradicts XX"
```

### kickoff_prompt #17 [wording]
```diff
-        "(it needs access you don't have, a decision from the user, or it contradicts "
+        "(IT NEEDS ACCESS YOU DON'T HAVE, A DECISION FROM THE USER, OR IT CONTRADICTS "
```

### kickoff_prompt #18 [wording]
```diff
-        "itself), call it with status \"blocked\" and the reason."
+        "XXitself), call it with status \"blocked\" and the reason.XX"
```

### continue_prompt #4 [wording]
```diff
-    feedback = f"Feedback from the check:\n{reason}\n\n" if reason else ""
+    feedback = f"Feedback from the check:\n{reason}\n\n" if reason else "XXXX"
```

### continue_prompt #5 [wording]
```diff
-        "or \"blocked\" if it can't be."
+        "XXor \"blocked\" if it can't be.XX"
```

### continue_prompt #6 [wording]
```diff
-        "or \"blocked\" if it can't be."
+        "OR \"BLOCKED\" IF IT CAN'T BE."
```

### parse_check #10 [equivalent]
```diff
-    m = re.match(r"^\W*(true|false)\b", (text or "").strip().lower())
+    m = re.match(r"^\W*(true|false)\b", (text or "XXXX").strip().lower())
```

### parse_verdict #9 [equivalent]
```diff
-    m = re.match(r"^\W*(met|not yet met|impossible)\b\s*[-:–—]?\s*(.*)",
+    m = re.match(r"^\W*(MET|NOT YET MET|IMPOSSIBLE)\b\s*[-:–—]?\s*(.*)",
```

### parse_verdict #11 [equivalent]
```diff
-                 (text or "").strip(), flags=re.IGNORECASE | re.DOTALL)
+                 (text or "XXXX").strip(), flags=re.IGNORECASE | re.DOTALL)
```

### judge_transcript #1 [limit]
```diff
-def judge_transcript(events: Iterable, max_chars: int = 60_000,
+def judge_transcript(events: Iterable, max_chars: int = 60001,
```

### judge_transcript #8 [equivalent]
```diff
-                 if events[i].type == MODEL_CALL), None)
+                 if events[i].type == MODEL_CALL), )
```

### judge_transcript #17 [equivalent]
```diff
-    last = next((i for i in range(len(events) - 1, -1, -1)
+    last = next((i for i in range(len(events) - 1, +1, -1)
```

### judge_transcript #18 [equivalent]
```diff
-    last = next((i for i in range(len(events) - 1, -1, -1)
+    last = next((i for i in range(len(events) - 1, -2, -1)
```

### judge_transcript #23 [equivalent]
```diff
-        return ""
+        return "XXXX"
```

### judge_transcript #86 [wording]
```diff
-            lines.append(f"[tool result: {m.get('name') or 'tool'}]\n{text}")
+            lines.append(f"[tool result: {m.get('name') or 'XXtoolXX'}]\n{text}")
```

### judge_transcript #87 [wording]
```diff
-            lines.append(f"[tool result: {m.get('name') or 'tool'}]\n{text}")
+            lines.append(f"[tool result: {m.get('name') or 'TOOL'}]\n{text}")
```

### judge_transcript #102 [wording]
```diff
-            calls = [f"[called {tc['function']['name']}({tc['function'].get('arguments') or ''})]"
+            calls = [f"[called {tc['function']['name']}({tc['function'].get('arguments') or 'XXXX'})]"
```

### judge_transcript #112 [wording]
```diff
-            lines.append("[assistant]\n" + "\n".join(x for x in [text, *calls] if x))
+            lines.append("[assistant]\n" + "XX\nXX".join(x for x in [text, *calls] if x))
```

### judge_request #9 [wording]
```diff
-        claim = f"The agent says the goal is complete: {claim_summary}\n\n" if claim_summary else ""
+        claim = f"The agent says the goal is complete: {claim_summary}\n\n" if claim_summary else "XXXX"
```

### judge_request #10 [wording]
```diff
-        question = (
-            "Is the goal met? Answer on one line, in one of these forms:\n"
-            "met - <the evidence>\n"
-            "not yet met - <what is still missing>\n"
-            "impossible - <why it can't be met>"
-        )
+        question = None
```

### judge_request #11 [wording]
```diff
-            "Is the goal met? Answer on one line, in one of these forms:\n"
+            "XXIs the goal met? Answer on one line, in one of these forms:\nXX"
```

### judge_request #12 [wording]
```diff
-            "Is the goal met? Answer on one line, in one of these forms:\n"
+            "is the goal met? answer on one line, in one of these forms:\n"
```

### judge_request #13 [wording]
```diff
-            "Is the goal met? Answer on one line, in one of these forms:\n"
+            "IS THE GOAL MET? ANSWER ON ONE LINE, IN ONE OF THESE FORMS:\n"
```

### judge_request #14 [wording]
```diff
-            "met - <the evidence>\n"
+            "XXmet - <the evidence>\nXX"
```

### judge_request #15 [wording]
```diff
-            "met - <the evidence>\n"
+            "MET - <THE EVIDENCE>\n"
```

### judge_request #16 [wording]
```diff
-            "not yet met - <what is still missing>\n"
+            "XXnot yet met - <what is still missing>\nXX"
```

### judge_request #17 [wording]
```diff
-            "not yet met - <what is still missing>\n"
+            "NOT YET MET - <WHAT IS STILL MISSING>\n"
```

### judge_request #18 [wording]
```diff
-            "impossible - <why it can't be met>"
+            "XXimpossible - <why it can't be met>XX"
```

### judge_request #19 [wording]
```diff
-            "impossible - <why it can't be met>"
+            "IMPOSSIBLE - <WHY IT CAN'T BE MET>"
```

### judge_request #21 [limit]
```diff
-        grammar, max_tokens = verdict_grammar(), 400
+        grammar, max_tokens = verdict_grammar(), 401
```

### judge_request #22 [wording]
```diff
-        claim = ""
+        claim = None
```

### judge_request #23 [wording]
```diff
-        claim = ""
+        claim = "XXXX"
```

### goal_tool #5 [equivalent]
```diff
-        status = (status or "").strip().lower()
+        status = (status or "XXXX").strip().lower()
```

### goal_tool #33 [wording]
```diff
-                    "Finish your reply now.")
+                    "XXFinish your reply now.XX")
```

### goal_tool #34 [wording]
```diff
-                    "Finish your reply now.")
+                    "finish your reply now.")
```

### goal_tool #35 [wording]
```diff
-                    "Finish your reply now.")
+                    "FINISH YOUR REPLY NOW.")
```

### goal_tool #39 [wording]
```diff
-                "Finish your reply now.")
+                "XXFinish your reply now.XX")
```

### goal_tool #40 [wording]
```diff
-                "Finish your reply now.")
+                "finish your reply now.")
```

### goal_tool #41 [wording]
```diff
-                "Finish your reply now.")
+                "FINISH YOUR REPLY NOW.")
```

### goal_tool #50 [wording]
```diff
-            "Report on the active goal. Call it with status \"complete\" when you "
+            "XXReport on the active goal. Call it with status \"complete\" when you XX"
```

### goal_tool #51 [wording]
```diff
-            "Report on the active goal. Call it with status \"complete\" when you "
+            "report on the active goal. call it with status \"complete\" when you "
```

### goal_tool #53 [wording]
```diff
-            "believe the goal is met, with the evidence as the summary; the goal is "
+            "XXbelieve the goal is met, with the evidence as the summary; the goal is XX"
```

### goal_tool #54 [wording]
```diff
-            "believe the goal is met, with the evidence as the summary; the goal is "
+            "BELIEVE THE GOAL IS MET, WITH THE EVIDENCE AS THE SUMMARY; THE GOAL IS "
```

### goal_tool #55 [wording]
```diff
-            "checked after your reply. Call it with status \"blocked\" when the goal "
+            "XXchecked after your reply. Call it with status \"blocked\" when the goal XX"
```

### goal_tool #56 [wording]
```diff
-            "checked after your reply. Call it with status \"blocked\" when the goal "
+            "checked after your reply. call it with status \"blocked\" when the goal "
```

### goal_tool #58 [wording]
```diff
-            "can't be met without the user, with what is blocking it as the summary."),
+            "XXcan't be met without the user, with what is blocking it as the summary.XX"),
```

### goal_tool #59 [wording]
```diff
-            "can't be met without the user, with what is blocking it as the summary."),
+            "CAN'T BE MET WITHOUT THE USER, WITH WHAT IS BLOCKING IT AS THE SUMMARY."),
```

### goal_tool #86 [wording]
```diff
-                            "description": "the evidence (complete) or the blocker (blocked)"},
+                            "description": "XXthe evidence (complete) or the blocker (blocked)XX"},
```

### goal_tool #87 [wording]
```diff
-                            "description": "the evidence (complete) or the blocker (blocked)"},
+                            "description": "THE EVIDENCE (COMPLETE) OR THE BLOCKER (BLOCKED)"},
```
