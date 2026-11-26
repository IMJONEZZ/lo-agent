# LO2 — Test Planning Document (lo-agent)

LO 2: "Design and implement comprehensive test plans with instrumented code."
Built around requirements chosen from the LO1 requirements document
([`LO1-requirements.md`](LO1-requirements.md)).

## Chosen requirements (priority order)

1. **Composability** — it never breaks when people combine known-to-work systems in a new
   order: they should all work regardless, unless specifically marked as not working within a
   particular context or order. *(primary — the plan is built around this)*
2. **Comparative effectiveness** — local/self-hosted promises more ability to manipulate model
   behaviour and context than a normal harness — that must ACTUALLY work better than standard
   non-local harnesses.
3. **Determinism & byte-level replay** — lo's determinism and byte-level replay functionalities
   must work.

## Priority and pre-requisites (composability)

### Why it is important

Composability is the closest to usability. I need to be able to guarantee users a packaged
product that they can be confident building on top of, and won't need to debug later. If it
fails, the user suffers — pretty badly, bad enough to stop using (churn).

### Quality level

Highest level of quality/testing is what this requirement deserves (given limited resources,
this is where the depth goes).

### How it could be tested

The main thing: create as many valid DAGs as possible with the components in different orders,
with known and machine-checkable outputs after the workflow — essentially synthetic data
generation for these DAGs, maybe even with simulated user personas (i.e. Tau² infinite DAG). We
can also use this for regression testing; I think it's the only valid way of testing
agentic systems effectively.

Given that we're doing t-wise testing, we're treating 3 as the minimum number of features for
that DAG. That way, our Tau² infinite DAG testing won't actually have an infinite number of
tests needed to run, but also we're testing a reasonable number to conclude that we're covered.

#### What counts as a break

Crashes, errors that are swallowed and the user doesn't see them, incorrect outputs, promises
unfulfilled (we said this would be byte-level identical and it isn't), and the session not
returning to ready are all examples of breaks.

#### Test levels

For an agentic system, the smallest meaningful unit isn't a single function or single tool call,
the smallest meaningful unit is a 3-turn round, Comprehension, solve, report. "Hey lo, will you
check what time it is?"

- "The user is asking for the current time. I should check their machine's location and clock."
- toolcall(check_current_time)
- It is now 11:21am in Atlanta Georgia.

So the features of each step:

- Like all general-purpose harnesses, lo compiles context, memories, AGENTS.md rules into a
  system prompt that's used. The first step is the system prompt feature.
- Also like all general-purpose harnesses, lo parses tool calls and actions and pairs valid ones
  with handlers that execute the tool. The second line shows the tool calling feature.
- Lastly, like all other general purpose harnesses, lo must give models the ability to stop
  generating and return to a ready state (even with a /goal active, it must end eventually).

So we have unit tests for our individual pieces like regular software, then slightly longer unit
tests that are for our agents, then integration tests for the full DAG after the unit tests pass.

#### Components and orders

**Pool:** Let's keep the pool small so it all stays buildable before December 4

**Known-to-work:** Unit tests passing is a reasonable way of calling something known-to-work
here.

**Orders:** I think that our differing orders will come from our longer runs and we can use our
user action example in LO1 as an invalid order case.

## Priority and pre-requisites (comparative effectiveness)

### Why it is important

Requirement 2 is similar to 3, as a differentiator from other harnesses, but it's referring to
the other pieces of the harness, such as memory management.

### Quality level

I would put the quality level of this requirement one peg below composability. Reasonably high,
but unlike requirements 1 and 3, this one is validation, so the result is a measurement rather than
a binary pass or fail. The success criteria is that, with the same model, same configs, same chat
templates, etc, does lo perform better than the baseline harnesses across repeats, both in terms
of average performance and also in terms of deterministic output. It gets less effort than
composability, and because each comparison needs runs in every harness it'll get a fixed number
of repeats instead of going for every combination.

### How it could be tested

Is lo-agent better at returning the most-relevant memory when semantic memory is invoked? Is
lo-agent the best at creating summaries and rules for itself during episodic memory sessions?
Same with procedural memory. Beyond that, does lo-agent's logit manipulation beat the
schema-matching of other harnesses when it comes down to it and we repeat the same
requirements/queries 1000x? We're essentially quanitfying what lo-agent actually gives you for
using it in a research or enterprise context.

The baselines are the ones in LO1: Claude Code, Codex, pi, and opencode, running the same model
on the same generated DAG tasks.

## Priority and pre-requisites (determinism & byte-level replay)

### Why it is important

Requirement 3 is important because it's one of the biggest differentiators between this harness
and others. If requirement 3 doesn't exist, there is no reason to use lo-agent over pi or
opencode or claude code even.

### Quality level

This has the least effort required of it, but the highest quality level of the 3. Any mismatch
between replays is a fail, it's literally just exact match. It needs little effort because the
check runs on its own and the tests already exist: test_replay_is_bit_identical and
test_replay_detects_divergence in tests/test_agent.py, test_seeded_determinism in
tests/test_native.py. The claim only covers the model/config tiers I've shown to replay
deterministically: untested GPUs (Intel a770, for example), batching, or a custom setup
(multi-node GPU offloading to system RAM + SSH for example) isn't covered (LO1 limitation 6).

### How it could be tested

The testing for this is by far the easiest of all of our tests, because it's an exact match
test. Do generations A, B, and C match when generated with our deterministic replay settings
on? If yes, pass, if no, diagnose and fail, then fix.

## Scaffolding and instrumentation

### Instrumentation

Most of the instrumentation was already in lo before this plan. The event log is append-only and
records every model call (the full request, the response, the seed and the timing), every tool
call with its arguments and result, every guardrail decision, and how each turn ended. Replay
compares against it. What the plan still needed was two things, and both are now built:

- **The full nested conversation history, in a readable format:** `lo export --full`
  (`src/local_harness/events/export.py`). It adds what the normal export left out: reasoning
  blocks, tool results in full instead of cut at 2,000 characters, guardrail nudges, how each
  turn ended (including the error), goal rounds, and every sub-agent run, nested under the run
  that spawned it.
- **A tag on each run:** a new `run_tag` event (`EventLog.tag_run`). It records which DAG a run
  was and the option it used at each step, so DAG coverage is counted from the event log instead
  of from the test code. The DAG runner tags every run it makes.

### Scaffolding

**Driver:** The existing journey drivers will be added on to to run the DAGs

**Stub:** The existing MockLlamaCpp will stub for us

**Oracle:** We already have an Oracle, but we'll add to it based on the break list.

**Where isolation can't happen:** ML observability is different from other system
observability in that it isn't "is the server running" — it's "is the server running AND is
the model producing good output?" — which requires isolation to NOT be occurring.

We can isolate for unit tests, but I think I actually have those adequately covered.

## Process and risk

### Lifecycle

I have been using DevOps CI/CD up to this point. Lo is released in versions as it runs locally
on users' machines, and it keeps changing after each release. DevOps keeps testing after
deployment, which feels ideal for this project and class. Can't do the others for various
reasons, from the fact that lo relies on unit tests already to the release schedule happening
more than once. The only weakness I see from DevOps for this project is that improvement is
normally tied to data from operation, and I'm not collecting any telemetry from users running
locally on their own devices. I'm dealing with it slightly through collecting user feedback and
issues on the github project for now, along with dogfooding, but it's not ideal right now.
There's no CI in the repo yet, this is one of the things I'm planning on adding in later as part
of this class.

This is our test ordering:

- On build: DAG Generator - None of the DAG tests can run until we have the actual tests ready
  to run
- On push: Our CI tests once I've put CI in - Unit tests, the journey tests, and the DAG runs
  (mocked for composability), then the replay tests (also mocked for determinism and replay).
  None of these need a real model, so they aren't computationally expensive and can run on every
  push
- Before Release: The end to end live tests, and a sample of the DAG tests on a real model,
  because a real model will show whether the output is good. `lo replay` on recorded sessions to
  test determinism against a real model and system, and the comparison against the other
  harnesses because they need a real model and a run in every harness, which is too expensive
  for every push.
- During operation: user feedback, GH issues, and dogfooding. I discover new things and so do my
  users pretty often that then get incorporated into new tests for the next release. Breaks
  become new DAGs, replay mismatches become new warnings for broken determinism in specific
  circumstances, and more information for the users to figure out if the product is a match for
  them

### Risks

Here are the risks as I see them:

Personnel: I'm the only developer, and sort of the only tester. There are 2 Fortune 10s that
have onboarded my product, but they are giving me feedback over email instead of in the repo so
there isnt' a good record of why every change is made and I have a potential conflict between
the two roles. To mitigate this, the on-push tests run in CI once it's in, so I can't skip any
of them, and user feedback gives me a very strong outside view of the product.

Technology: The on-push DAG runs are mocked, which definitely in no way resembles the behavior
of a real model running on a real inference engine, so a DAG could pass on the mock and fail for
real. That said, the real model sample happens before each release, so even if it's failing we
can fix before a release happens.

Schedule: none of the DAG tests can run until the DAG generator is built, and this class is
freaking flying now after the strikes. So I'm building the smallest version first and will grow
from there after the class concludes.

Development: Until CI is done, nothing stops me from pushing without running the tests. So CI is
a known requirement that I need to build to mitigate this.

Test execution: the tests before release all need a real model, and the comparison needs a run
in every harness, and it's currently all running on my own hardware. I need to minimize parts
that require the full system to be executed, and likely also run the harnesses in parallel to
minimize the amount of time the hardware is maxed out.

Requirements: The Generated DAGS and the failures I hear about from users may not match exactly
how people really use lo. with no in-app telemetry I only hear from users who choose to report
(selection bias), and also I only hear their opinions of what went wrong and how. I don't know
how to fully mitigate this, but every reported break is becoming a new DAG at this point, and
using the simulated user personas should expand my visibility as well.

Contingency plan:

- If the DAG generator isn't ready in time, I write the combinations by hand as journeys
- If a real model test fails because of the model rather than lo, the event log shows what lo
  sent and what came back. I record it and don't count it as a harness break.

## Evaluation of the plan (criterion 2.2)

Omissions:

- Long-horizon use isn't covered at this stage. Because we're doing only 3-wise DAGs at this
  point and working towards combinatorial work, tasks that run over many chats for months at a
  time compacted several times aren't going to be tested.
- Longer runs have no stopping rule yet, so I don't know precisely when there have been enough
  of them

Vulnerabilities:

- The replay tests and all of the other tests that run in CI are mocked, so the only real model
  determinism check is before release and only on the setups I'm testing it on.
- The comparison I'm making omits all frontier models due to them not exposing weights or weight
  spaces, meaning that the models we test may suit some harnesses better than others.

How we ensure adequate testing:

- Composability is adequately tested when every combination in the pool of DAGs passes both on
  the mocked version (testing the system) and on the real model (testing the harness). An
  example would be: 2 system prompts (with and without our AGENTS.md) x 3 ways to stop (normal
  finish, step budget used up, guardrail stop) x 2 tools (memory, session_search) = 12 DAGs.
- Determinism and replay is adequately tested when every recorded session replays identically on
  the setups I support.
- The comparison is adequately tested when every harness has done the same fixed number of
  repeats on the same tasks.
- Overall, I would say this plan is strong for the parts that can be isolated, and weaker for
  real model output quality, but should still be representative of real model performance
  despite the sampling, including for long-horizon use (which will be covered later).

## Evaluation of the instrumentation and scaffolding (criterion 2.4)

Instrumentation:

- Most of the instrumentation already exists. We already have an event log, it's append-only,
  and it records every model call and every tool call. It's also what we compare replays
  against, so the determinism checks run right against it.
- `lo export` command turns a run into a readable markdown transcript, and `lo doctor` checks
  the server, model, event log, and sandbox at the start of every test session when you run it.

How we can improve it:

- The readable transcript isn't the full nested history I want yet. `lo export` leaves out the
  reasoning blocks and truncates tool call results and doesn't follow subagent runs at all.
- Errors that are swallowed so the user doesn't see them and the session not returning to ready
  are both errors that the event log currently doesn't show. Checks for both can be added, and
  I'm planning on doing it.
- Nothing records which DAG a run was or which option it used at each step, so I can't count
  coverage from the event log. Each DAG run needs a tag.
- Replay compares each reply's text, reasoning, and tool calls, not the whole raw response, but
  this is easily _easily_ fixed.
- The event log only covers lo, none of the other harnesses that we're testing against, so their
  traces need to be captured outside lo in a comparable format.

What does the scaffolding look like?

- The driver, the stub, and the oracle already exist. The journey drivers run lo's terminal
  interface the same way a user does. The inference engine mocks work for the model, and the
  journey checks are the oracle. The DAG generator is the main piece I still need to build.
- MockLlamaCpp imitates a real llama.cpp server, but doesn't load a real model. It also can't
  judge whether an output is good, so output quality is only checked on the real model runs.
- test_seeded_determinism uses a tiny random-weight GPT-2 model, so it checks a real forward
  pass without needing a running model, but it's skipped if pytorch and transformers aren't
  installed.
- Overall, the instrumentation and scaffolding are strong for lo's logic and weak where
  isolation can't happen
