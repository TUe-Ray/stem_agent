# demo_001 Analysis Report

## Executive Summary

`Baseline score` and `evolved score` appear identical because this run saturated the current evaluator almost immediately, and no mutation produced a score increase large enough to satisfy the promotion threshold.

The most important detail is that the run was not perfectly flat in raw floating-point terms. Some candidate genomes reached `0.48294` versus the baseline `0.48288`, but the report rounds both to `0.4829`, and Guardian requires an improvement of at least `0.03` to promote a mutation. In practice, this means the search space produced only tiny cosmetic or cost-related differences, not meaningful fitness gains.

## What Happened

### 1. The seed genome already maxed out the visible requirements

All train and validation cases hit:

- `requirement_coverage = 1.0`
- `format_validity = 1.0`
- `scenario_success = 1.0`
- `workflow_completion = 0.45`

That left almost no headroom except:

- reducing cost a tiny amount
- adding rewarded review/gate/tool traces
- avoiding complexity penalties

### 2. The evaluator is much coarser than the scenario suggests

The scenario says the success criteria are:

- requirement coverage
- usefulness
- format validity
- cost efficiency

But this run did not use scenario-specific weighted criteria. Instead, it used the evaluator's built-in heuristic score composed of:

- requirement coverage
- format validity
- artifact presence
- self-review usage
- workflow completion
- quality gate usage
- generated tool usage
- cost penalty
- complexity penalty

This matters because the score did not directly judge whether an answer was more insightful, specific, or useful for the user's request once the basic markdown pattern had been satisfied.

### 3. Early mutations targeted levers that could not help

The first two mutations only changed retry behavior. That had no practical upside because every case already completed successfully with no failures. If nothing fails, `revise_on_failure` never activates, so the evaluator sees no benefit.

### 4. Later mutations changed the genome shape, but not the rewarded traces

Whole-genome replacement proposals did produce different answers, and some of them were slightly cheaper. But they still did not create the specific signals the evaluator rewards, such as:

- `review_against_requirements`
- `revise_final_output`
- passed `quality_gate` traces
- generated tool usage
- environment artifacts

So most of the score remained unchanged.

### 5. Complexity penalties discouraged richer harnesses

The strongest example is generation 6. That candidate added an extra role and incurred a `complexity_penalty = 0.12`, which lowered the score to `0.47688`. In other words, richer structure was penalized before it created compensating evaluator-visible gains.

### 6. Promotion threshold was too high for this score landscape

The scenario uses:

- `min_delta = 0.03`

Observed improvements were around:

- `0.00006`

So even the best neutral candidates had no chance of promotion. The system effectively required a jump that was about 500x larger than the improvements this evaluator can currently express on this toy task.

### 7. The run directory shows signs of reuse

`convergence_log.jsonl` contains two generation sequences for the same run id. That does not explain the flat score by itself, but it does make the artifacts harder to read and can blur analysis if you compare logs naively.

## Root Causes

### Primary causes

- The task is too easy for the current evaluator.
- The evaluator saturates after basic structure requirements are met.
- The promotion threshold is far larger than the score granularity.

### Secondary causes

- Mutations targeted retry behavior even though no cases failed.
- Mutations rarely created evaluator-visible review or quality-gate behavior.
- Extra structure is penalized faster than it is rewarded on this scenario.

## Evidence

### Final reported state

- Baseline: `0.48288`
- Best visible candidate after rounding: still reported as `0.4829`
- Final frozen score: `0.48288`
- Promoted mutations: `0`
- Stop reason: max generations reached

### Score behavior

- Baseline cases: all scored `0.48288`
- Some later candidates: all scored `0.48294`
- Generation 6 mutation candidate: `0.47688` due to complexity penalty

### Why the tiny improvement happened

The only measurable lift in the better neutral candidates came from a lower `cost_penalty`:

- `cost_penalty = 0.004` for the baseline style outputs
- `cost_penalty = 0.002` for a cheaper candidate

That difference changes the total score only minimally, because cost is weighted weakly.

## How To Improve

### Highest priority: fix the scoring signal

1. Wire scenario-level criteria into actual scoring.
2. Add a real usefulness or specificity judge instead of relying on markdown-pattern heuristics.
3. Make the evaluator sensitive to input-specific quality, not just section headings.

Without this, evolution will keep optimizing for surface structure rather than better answers.

### Second priority: lower the promotion threshold

For this task family, try:

- `min_delta = 0.001` as a starting point
- or `min_delta = 0.0005` if you expect only subtle gains

With the current `0.03`, nearly any realistic improvement is invisible to promotion logic.

### Third priority: make the task harder to saturate

Expand the scenario so that generic advice stops scoring perfectly.

Good directions:

- add more train and validation cases
- include requests with constraints, tradeoffs, and priorities
- include adversarial vague prompts where generic checklists are not enough
- add hidden evaluation cases that punish boilerplate responses

### Fourth priority: reward the kinds of structure you want evolution to discover

If you want richer harnesses, the evaluator should reward them when they help.

Examples:

- explicit review steps
- revision passes
- acceptance checks
- artifact generation
- tool usage when justified

Right now those pathways exist in the score, but this scenario and mutation path barely activate them.

### Fifth priority: align mutation strategy with failure modes

Instead of mutating retry policy first, prefer mutations that address the actual bottleneck:

- introduce review steps with evaluator-visible trace ids
- tighten output structure around case-specific usefulness
- create lightweight quality gates before adding more roles

### Sixth priority: keep run artifacts clean

Use a fresh `run-id` for each experiment when comparing runs. Reusing a run directory makes convergence and lineage logs harder to trust during postmortems.

## Suggested Next Experiment

If the goal is to make `toy_structured_answer` actually evolve, the best next test is:

1. lower `min_delta` to `0.001`
2. add scenario scoring that measures usefulness/specificity directly
3. add 3 to 5 harder validation cases
4. rerun with a fresh run id

That combination should tell you whether the engine can improve on a meaningful signal instead of bouncing around a saturated heuristic.
