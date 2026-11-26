# Branch coverage: the whole test suite

This page records which statements, and which branches (each way out of an
`if`, a loop or a `try`), the test suite ran. It was measured with coverage.py
with `branch = true` over the `local_harness` package, using the
`[tool.coverage]` settings in `pyproject.toml`:

```
uv run --group dev coverage run -m pytest
uv run --group dev coverage report
```

Suite result in that run: 600 passed, 8 skipped, 9 deselected (23 Nov 2026, lo 0.2.24 at 912b371 plus this work, Python 3.13.11). The pseudo-terminal and live-model tests
are deselected by default (`-m 'not pty and not live'`), and code run in
subprocesses isn't measured.

## Totals

| Measure | Covered | Total | % |
|---|---|---|---|
| Statements | 7969 | 13215 | 60.3% |
| Branches | 2215 | 4078 | 54.3% |
| Combined (coverage.py's figure) | | | 58.9% |

## By area of the package

| Area | Statements | Statement % | Branches | Branch % |
|---|---|---|---|---|
| `tui` | 1754/3613 | 48.5% | 517/1146 | 45.1% |
| `agent` | 1803/2013 | 89.6% | 651/762 | 85.4% |
| `jlens` | 176/1735 | 10.1% | 37/540 | 6.9% |
| `cli` | 632/1138 | 55.5% | 100/284 | 35.2% |
| `inference` | 481/564 | 85.3% | 113/164 | 68.9% |
| `server` | 487/531 | 91.7% | 125/156 | 80.1% |
| `(top level)` | 201/397 | 50.6% | 33/84 | 39.3% |
| `skills` | 365/397 | 91.9% | 149/178 | 83.7% |
| `events` | 381/387 | 98.4% | 107/112 | 95.5% |
| `logits` | 276/382 | 72.3% | 68/114 | 59.6% |
| `native` | 55/305 | 18.0% | 2/60 | 3.3% |
| `guardrails` | 260/295 | 88.1% | 82/98 | 83.7% |
| `proxy` | 252/285 | 88.4% | 77/102 | 75.5% |
| `sim` | 127/219 | 58.0% | 16/46 | 34.8% |
| `tree` | 161/193 | 83.4% | 25/36 | 69.4% |
| `signals` | 129/191 | 67.5% | 29/50 | 58.0% |
| `background` | 131/181 | 72.4% | 28/58 | 48.3% |
| `optimize` | 105/159 | 66.0% | 27/44 | 61.4% |
| `integrations` | 115/146 | 78.8% | 21/32 | 65.6% |
| `structured` | 78/84 | 92.9% | 8/12 | 66.7% |

## The modules this LO3 work added to or tested

| Module | Statement % | Branch % | Missing lines |
|---|---|---|---|
| `agent/goal.py` | 98.4% | 94.4% | 149, 150, 151, 152, 480 |
| `server/goals.py` | 90.2% | 76.7% | 70, 77, 79, 92, 121, 133, 157, 158, 159 |
| `server/sessions.py` | 92.2% | 81.0% | 107, 160, 161, 165, 166, 167, 174, 205, 239, 240, 281, 321 … |
| `events/export.py` | 100.0% | 100.0% | none |
| `events/log.py` | 100.0% | 94.1% | none |
| `events/replay.py` | 90.9% | 75.0% | 52, 53, 58, 59, 60 |
| `guardrails/loops.py` | 96.0% | 87.5% | 40 |
| `agent/loop.py` | 92.0% | 86.9% | 200, 201, 205, 206, 311, 386, 403, 416, 426, 427, 428, 444 … |

## Full report (coverage report)

```
Name                                                Stmts   Miss Branch BrPart  Cover
-------------------------------------------------------------------------------------
src/local_harness/__init__.py                           1      0      0      0 100.0%
src/local_harness/agent/__init__.py                     3      0      0      0 100.0%
src/local_harness/agent/codemode.py                   157     40     42      3  75.4%
src/local_harness/agent/commands.py                    71      6     16      0  93.1%
src/local_harness/agent/compaction.py                 111     20     50      8  80.1%
src/local_harness/agent/frontmatter.py                115     11     58      8  86.7%
src/local_harness/agent/goal.py                       321      5    144      4  97.2%
src/local_harness/agent/loop.py                       426     34    176     19  90.5%
src/local_harness/agent/memory.py                      42      2      6      2  91.7%
src/local_harness/agent/notebook.py                   121     10     42      7  89.6%
src/local_harness/agent/permissions.py                 44      0     24      1  98.5%
src/local_harness/agent/presets.py                    111      9     28      4  90.6%
src/local_harness/agent/structured_memory.py           17      0      2      0 100.0%
src/local_harness/agent/tools.py                      474     73    174     27  83.6%
src/local_harness/background/__init__.py                7      0      0      0 100.0%
src/local_harness/background/auto_skills.py            33      2     12      3  88.9%
src/local_harness/background/consolidate.py            20      0      6      1  96.2%
src/local_harness/background/cycle.py                  15      0      0      0 100.0%
src/local_harness/background/induce.py                 54     46     24      0  10.3%
src/local_harness/background/propose.py                28      2     10      2  89.5%
src/local_harness/background/reflect.py                24      0      6      0 100.0%
src/local_harness/bench.py                             90     90     22      0   0.0%
src/local_harness/cli/main.py                        1092    460    268     24  53.8%
src/local_harness/cli/progress.py                      46     46     16      0   0.0%
src/local_harness/debuglog.py                          64      8      6      0  88.6%
src/local_harness/events/__init__.py                    3      0      0      0 100.0%
src/local_harness/events/bus.py                        49      1     12      1  96.7%
src/local_harness/events/export.py                    100      0     58      0 100.0%
src/local_harness/events/log.py                       180      0     34      2  99.1%
src/local_harness/events/replay.py                     55      5      8      0  88.9%
src/local_harness/guardrails/__init__.py                7      0      0      0 100.0%
src/local_harness/guardrails/concept_watch.py          19     19      4      0   0.0%
src/local_harness/guardrails/errors.py                 21      0      0      0 100.0%
src/local_harness/guardrails/guardrails.py             57      4     22      2  89.9%
src/local_harness/guardrails/loops.py                  25      1      8      1  93.9%
src/local_harness/guardrails/nudges.py                 26      1      4      1  93.3%
src/local_harness/guardrails/rescue.py                 65      8     36      4  88.1%
src/local_harness/guardrails/steps.py                  35      0      8      0 100.0%
src/local_harness/guardrails/validator.py              40      2     16      2  92.9%
src/local_harness/inference/__init__.py                 4      0      0      0 100.0%
src/local_harness/inference/adapters/__init__.py        6      0      0      0 100.0%
src/local_harness/inference/adapters/base.py           32      3      0      0  90.6%
src/local_harness/inference/adapters/generic.py         9      0      0      0 100.0%
src/local_harness/inference/adapters/llamacpp.py       22      3      2      1  83.3%
src/local_harness/inference/adapters/vllm.py           10      1      0      0  90.0%
src/local_harness/inference/capabilities.py           173     20     52      8  85.8%
src/local_harness/inference/client.py                 138     27     52     16  74.2%
src/local_harness/inference/lora.py                    46      9     24      3  82.9%
src/local_harness/inference/types.py                  124     20     34      1  79.1%
src/local_harness/integrations/load.py                 30      6     10      2  75.0%
src/local_harness/integrations/mcp.py                  69     12     12      1  81.5%
src/local_harness/integrations/utcp.py                 47     13     10      2  70.2%
src/local_harness/jlens/__init__.py                    22     14      2      1  37.5%
src/local_harness/jlens/cli.py                        293    222     92      4  23.1%
src/local_harness/jlens/exports.py                    104    104     38      0   0.0%
src/local_harness/jlens/fit_torch.py                   77     77     26      0   0.0%
src/local_harness/jlens/fitting.py                    117    117     42      0   0.0%
src/local_harness/jlens/interventions.py               73     73     20      0   0.0%
src/local_harness/jlens/lens.py                       103    103     32      0   0.0%
src/local_harness/jlens/manager.py                    272    175     96      6  31.2%
src/local_harness/jlens/model_reader.py                83     83     22      0   0.0%
src/local_harness/jlens/native_client.py              141    141     36      0   0.0%
src/local_harness/jlens/readout.py                    131    131     30      0   0.0%
src/local_harness/jlens/service.py                    302    302     98      0   0.0%
src/local_harness/jlens/topology.py                    17     17      6      0   0.0%
src/local_harness/logits/__init__.py                    5      0      0      0 100.0%
src/local_harness/logits/antislop.py                   68      3     28      4  92.7%
src/local_harness/logits/bias.py                       84     46     20      0  36.5%
src/local_harness/logits/budget.py                     46      6     12      6  79.3%
src/local_harness/logits/grammar_stage.py              30      2     14      2  90.9%
src/local_harness/logits/guidance.py                   45     45     10      0   0.0%
src/local_harness/logits/pipeline.py                   48      1     12      2  95.0%
src/local_harness/logits/samplers.py                   30      2     14      2  90.9%
src/local_harness/logits/slop.py                       26      1      4      0  96.7%
src/local_harness/native/__init__.py                    2      0      0      0 100.0%
src/local_harness/native/backend.py                    87     67     22      0  18.3%
src/local_harness/native/lora.py                       38     38      8      0   0.0%
src/local_harness/native/skill_exec.py                 37      4      4      2  85.4%
src/local_harness/native/steering.py                  141    141     26      0   0.0%
src/local_harness/optimize/__init__.py                  2      0      0      0 100.0%
src/local_harness/optimize/bootstrap.py                73     15     20      3  74.2%
src/local_harness/optimize/dspy_opt.py                  7      7      0      0   0.0%
src/local_harness/optimize/prompt_opt.py               47      2     18      2  93.8%
src/local_harness/optimize/weight_opt.py               30     30      6      0   0.0%
src/local_harness/proxy/__init__.py                     4      0      0      0 100.0%
src/local_harness/proxy/anthropic.py                   95     14     50     11  81.4%
src/local_harness/proxy/app.py                         36      3      4      0  92.5%
src/local_harness/proxy/config.py                      23      0      4      1  96.3%
src/local_harness/proxy/engine.py                     127     16     44      9  84.2%
src/local_harness/sandbox.py                          138     67     24      1  49.4%
src/local_harness/server/__init__.py                    2      0      0      0 100.0%
src/local_harness/server/app.py                       107      8     20      3  91.3%
src/local_harness/server/coordinator.py                24      4      6      2  80.0%
src/local_harness/server/goals.py                      92      9     30      7  86.9%
src/local_harness/server/sessions.py                  295     23    100     15  89.4%
src/local_harness/server/webui.py                      11      0      0      0 100.0%
src/local_harness/signals/__init__.py                   3      0      0      0 100.0%
src/local_harness/signals/jspace.py                    43     43     10      0   0.0%
src/local_harness/signals/metrics.py                   44     12     16      1  68.3%
src/local_harness/signals/policies.py                  31      1      8      1  94.9%
src/local_harness/signals/semantic_entropy.py          70      6     16      3  89.5%
src/local_harness/sim/__init__.py                       2      0      0      0 100.0%
src/local_harness/sim/pty_driver.py                   111     80     24      0  23.0%
src/local_harness/sim/scenario.py                      59     11     22      4  79.0%
src/local_harness/sim/scenarios.py                     47      1      0      0  97.9%
src/local_harness/skills/__init__.py                    4      0      0      0 100.0%
src/local_harness/skills/exec.py                       61     12     18      4  72.2%
src/local_harness/skills/ir.py                        219      9    112      9  94.6%
src/local_harness/skills/skill.py                     113     11     48      8  87.0%
src/local_harness/structured/__init__.py                2      0      0      0 100.0%
src/local_harness/structured/builder.py                82      6     12      4  89.4%
src/local_harness/tree/__init__.py                      4      0      0      0 100.0%
src/local_harness/tree/search/beam.py                  26      4     16      4  81.0%
src/local_harness/tree/search/best_of_n.py             64     18     10      1  68.9%
src/local_harness/tree/search/plan_search.py           11      0      2      0 100.0%
src/local_harness/tree/search/self_consistency.py      34      6      6      2  80.0%
src/local_harness/tree/state.py                        54      4      2      0  92.9%
src/local_harness/tui/__init__.py                       2      0      0      0 100.0%
src/local_harness/tui/advantages.py                   394    394    106      0   0.0%
src/local_harness/tui/app.py                         2102    876    658    105  56.1%
src/local_harness/tui/lens_render.py                  196    196     78      0   0.0%
src/local_harness/tui/lens_screen.py                  348    348     66      0   0.0%
src/local_harness/tui/render.py                       571     45    238     34  89.2%
src/local_harness/tuned_replay.py                     104     31     32      6  66.9%
-------------------------------------------------------------------------------------
TOTAL                                               13215   5246   4078    429  58.9%

3 empty files skipped.
```
