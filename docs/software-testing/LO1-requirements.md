# LO1 — Requirements & Testing Strategies (lo-agent)

`lo` — a self-hosted agent harness for local LLMs: CLI/TUI, OpenAI-compatible proxy mode,
grammar-constrained skills, token-level logit pipeline, tree search, FTS5 memory, guardrails;
installable via uv/brew; existing test suite.

## Stakeholders

1. Me — creator & maintainer (primary stakeholder)
2. Users who have ideas and want updates (direct user community)
3. Companies interested in open-source local harnesses; interested in it performing well
4. Regular people who interact with systems built on top of lo-agent (indirect end users — the
   "other drivers on the road": they don't know lo is underneath, but experience its
   behaviour). They want that software to behave normally and reliably.

## Requirements by type, with test approaches

### Safety

- **lo must NEVER steal user information and send it to an undisclosed server.** *(all users)*
  - *Test (verification):* observe network calls as well as any data writes, to verify where
    the data goes and stays. To fight optimistic inaccuracy, we will inspect every place in
    the code that can send data out and check that each one goes to a destination the user
    approves.
- **It never breaks when people combine known-to-work systems in a new order: they should all
  work regardless, unless specifically marked as not working within a particular context or
  order.** *(me)*
  - *Test (verification):* run all functionality in every possible order. There is a
    limitation because of the number of features that lo-agent contains, but we have an
    advantage that keeps this from being n! if n is the number of features: ordering.
    Essentially, I say in every possible order, but _possible_ covers problems like, for
    example, the user starting a chat before connecting to a model. So the practice would be
    t-wise testing where t is 3 features, constituting the smallest DAG for each of our runs.
- **It never lies to them about what it's doing (its account of its own behaviour is
  truthful).** *(direct users; companies)*
  - *Test (verification):* harder to test, but unit tests on logging are the answer. The logs
    must contain the actual independent reassurance from the system. We are already working
    with byte-level replays of sessions, we can also make sure the logs repeat what's happening
    (for example, tool calling) at least at the sandbox level.

### Security

- **lo must NEVER run under another name or without user approval.** *(all users)*
  - *Test (verification):* a violation would look like the harness allowing the model either
    (a) out of the sandbox, into folders it doesn't have access to, or (b) to run with
    permissions it is not granted, i.e. root. Check by keeping track of the usergroup the
    model is running under and what permissions it has at any point.

### Correctness

- **It works — the claims I make about it are true.** *(me)*
  - *Test (verification):* establish success criteria for each claim I make, then run lo
    against those criteria and check it passes (scripted acceptance checks, runnable in CI).
- **The software works properly.** *(direct users; companies)*
  - *Test (validation):* ask users for feedback — because lo runs locally on their machines,
    I'm not collecting crash logs from their computers, so user-reported experience is the
    signal.
- **lo's determinism and byte-level replay functionalities must work.** *(all users)*
  - *Test (verification):* this means we have actually engineered a scenario with absolute
    determinism within the models we're working with, which is difficult; partially solved by
    showcasing the tiers of ability (which model/config tiers achieve deterministic replay).

### Liveness

- **It tells them if it's not working properly (failures are surfaced, not silent).**
  *(direct users; companies)*
  - *Test (verification):* test this myself: unit-test every possible break point I can find,
    asserting that the logs show (a) before the break: the user was warned, and (b) after the
    break: the user was shown WHY it happened, and (c) both before and after the break: the
    user is shown that the system is ready. This is already taken care of in the lo-agent, so
    it wouldn't be an addition, but it's worth mentioning.

### Resource use

- **lo must keep token cost under control through its weight-level mechanisms: (a) when the
  shape of the desired output is known, logit manipulation bakes the bias into the system
  engineering and eliminates the need for reasoning blocks beforehand; (b) code mode keeps the
  reasoning and toolcalling blocks flat on their token usage.** *(all users)*
  - *Test (verification):* very testable against all the other popular harnesses: opencode,
    pi, claude code, codex, cursor, etc. (comparative token-count benchmarks).

### Usability ("-ility")

- **The quickstart makes onboarding easy.** *(me)*
  - *Test (validation):* test it with OTHER users, not me — I built the onboarding so I'm
    biased; first-time-user usability trials are the real test.
- **Onboarding is easy and intuitive.** *(direct users)*
  - *Test (validation):* two ways: (a) time how long onboarding takes; (b) ask users for their
    feedback.
- **Change management for adding lo into their organisation is easy (adoption/deployment
  friction is low).** *(companies)*
  - *Test (validation):* can't adequately test beyond nailing the onboarding and then getting
    some company to be users (real-world pilot).

### Reliability / predictability ("-ility")

- **Systems running on lo do not behave unpredictably when they use or interact with them.**
  *(indirect end users)*
  - *Test (validation):* dogfood it: create a golden dataset of both long- and short-horizon
    tasks and complete the Tau² infinite DAG testing on them.
- **They do not feel cheated or discriminated against because of the way lo does model
  interactions.** *(indirect end users)*
  - *Test (validation):* impossible to support ALL new models and inference engines that come
    out, but not hard to support and test the median plus 1–2 standard deviations from it
    (i.e. a representative slice of the model/engine space, not exhaustive coverage).

### Comparative effectiveness

- **Local/self-hosted promises more ability to manipulate model behaviour and context than a
  normal harness — that must ACTUALLY work better than standard non-local harnesses.**
  *(all users)*
  - *Test (validation):* test against Claude Code, Codex, and Antigravity (named baseline
    harnesses). The measure would be the same models running the same generated DAG tasks in
    those harnesses, in this case currently Qwen3.8-27B via a command like
    `UNSLOTH_STUDIO_URL=http://<host-address>:8888 unsloth start claude --model unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL --reasoning-effort max`
    for claude code or
    `UNSLOTH_STUDIO_URL=http://<host-address>:8888 unsloth start codex --model unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_XL --reasoning-effort max`.
    Not really a problem to control. So essentially everything is constant and controlled
    except for the harness itself, which is what we're testing.

### Economics / legal

- **They do not have to pay extra because the system they're using uses lo under the hood.**
  *(indirect end users)*
  - *Test (validation):* easy on my end: check what the license says and how it is executed
    against real customers.
- **Licensing: usable for commercial purposes without crazy hoops or payment.** *(companies)*
  - *Test (verification):* known to be solved; testing = showing the license.

## Levels of requirements (system → sub-system → unit)

| Sub-system | Inherited system requirement | Likely unit tests | Integration tests needed |
|---|---|---|---|
| Sandbox & permissions | "NEVER run under another name or without user approval" | unit-test the sandbox: process user/group, permission sets, path-escape attempts | A tool call to the user's inference server must have the same restrictions as from the CLI |
| Guardrails | "NEVER steal user information and send it to an undisclosed server" | unit-test guardrail rules: destination allow-lists, payload scanning | guardrails come before the model, during inference (j-space, logit manipulation), and after the model |
| Skills & grammar pipeline | "never breaks when systems are combined in a new order" | unit-test each grammar-constrained skill against its grammar | skills × skills: composition of features |
| Logit pipeline | token cost kept under control | unit-test logit bias application against reference logits | biases survive the full request path |
| Replay / determinism | determinism & byte-level replay must work | unit-test event-log serialisation round-trips | replaying a recorded session reproduces byte-identical behaviour |
| FTS5 memory | "systems running on lo do not behave unpredictably" | unit-test index/update/query behaviour at boundary sizes | memory ↔ session lifecycle: memories survive restart and are scoped correctly per project |

Take a look at these unit tests that existed before the class: `test_permissions.py`,
`test_sandbox.py`, `test_guardrails.py`, `test_structured.py`

## Limitations of these testing approaches

1. We'll never be able to FULLY verify usability, regardless of how much or how well we do it,
   because that's so dependent on who is actually using it.
2. Unit tests are characteristically difficult on LLM-based products because we're not testing
   "is the server up" — we're also testing "AND is the model giving good answers?"
3. User feedback as a signal for "works properly": silent churn — unhappy local users often
   just leave without reporting, so you hear from a biased subset; and failures you weren't
   there for can't be reproduced (no crash logs by design).
4. Composability, an interaction that's a long-horizon task that runs over multiple chats for
   months on end and is compacted multiple times isn't something I'm able to test using the
   3-wise or DAG testing.
5. Comparative effectiveness: we're controlling for harness, but the model I've chosen may
   inherently behave better with some harnesses over others.
6. Determinism tiers: the hardware is a much more real problem here than with most. If you're
   running with an untested GPU, with batching for requests, or with a custom setup I won't be
   able to guarantee determinism unfortunately due to floating point arithmetic and other
   low-level hardware and firmware quirks.
7. If you are using an unusual model architecture or (increasingly more common) a vibe-coded
   inference engine, lo-agent will be untested

## Approaches that are not appropriate

- Crash telemetry doesn't work for lo-agent because that would send user information to an
  undiscosed server.
- Exhaustive ordering doesn't work for composability both because many of the orders don't
  make any sense and also because the number would be n! tests
- I can't test my own onboarding because of my bias having worked on this extensively
- Unit tests by themselves don't show usability unfortunately
- Timings are really unhelpful here. We don't control the inference engine, which is the main
  amount of time anything here takes. We are testing the harness, not the user's hardware.
