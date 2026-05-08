# StemOS

StemOS is a stem agent framework that differentiates into a task-specific operating harness.

It is not a universal agent. It is a universal differentiation mechanism. Each user-provided scenario produces a different specialized operating harness.

The user gives a scenario. Nucleus commands evolution. Genome mutates. Guardian protects. Harness materializes. The best organism freezes. The frozen harness executes future tasks.

## Architecture

```mermaid
flowchart LR
  A["Scenario signals"] --> B["Nucleus diagnosis"]
  B --> C["Genome mutation"]
  C --> D["Harness materialization"]
  D --> E["Evaluation"]
  E --> F["Guardian selection"]
  F --> C
  F --> G["Frozen specialized harness"]
```

## Why It Is Not A Universal Agent

StemOS does not hard-code bug fixing, research, QA, or any other task domain. It reads a scenario package, evolves a genome, materializes a candidate harness, evaluates it, and freezes the best specialized harness for that scenario.

The genome is the developmental blueprint. It can encode roles, workflow, tools, memory, self-evaluation, quality gates, retry policy, and environment layout.

StemOS may evolve roles, but roles are not predefined subagents. They are phenotypic structures produced by genome mutations and retained only if Guardian fitness improves. In the demo, adding more roles is not automatically rewarded; harmful or redundant roles are rolled back.

## Immutable vs Mutable Boundary

Immutable Guardian kernel:

- `src/stemos/kernel/guardian.py`
- `src/stemos/kernel/evaluator.py`
- `src/stemos/kernel/sandbox.py`
- `src/stemos/kernel/versioning.py`
- `src/stemos/kernel/budget.py`
- `src/stemos/kernel/validators.py`
- hidden evaluation, rollback, lineage, mutation validation, and budget limits

Mutable genome:

- roles
- workflow
- tool specs
- generated tools after sandbox checks
- memory schema
- quality gates
- self-evaluation rubric
- retry policy
- stop rule
- environment layout and artifacts

Nucleus may evolve mutable self-evaluation. It must never modify immutable Guardian fitness.

## Nucleus/Guardian Information Flow

StemOS uses a layered information policy between Guardian and Nucleus. The design is intentionally not simply "visible" or "hidden": Nucleus receives enough aggregate signal to improve search, but never receives validation answers, per-case outcomes, or Guardian's internal scoring details.

The policy lives in `src/stemos/kernel/signal_policy.py` and is enforced before Nucleus prompts are sent to an LLM.

### Signal Layers

`Layer 0` is always-visible context that does not require Guardian evaluation:

- scenario task description and task class
- current genome structure
- mutation type
- generation number
- directional result such as `improved`, `degraded`, or `neutral`

`Layer 1` is aggregate evaluation signal. It is configurable per scenario and is enabled by default:

- `aggregate_score`
- `previous_aggregate_score`
- `score_delta`

`Layer 2` is hard-blocked at the kernel boundary and must never reach Nucleus:

- validation case inputs
- validation expected outputs or answer keys
- Guardian criterion scores
- per-case scores
- hidden evaluation scores
- hidden evaluation case details

The practical rule is: Nucleus may see the dashboard, never the answer key.

### NucleusSignal

Guardian exposes evaluation feedback to Nucleus through `NucleusSignal`, not raw evaluation results. A signal contains only:

- `generation`
- `mutation_type`
- `direction`
- optional Layer 1 aggregate score fields when enabled

It cannot carry validation cases, expected outputs, hidden eval scores, or per-case score breakdowns. Tests in `tests/test_signal_policy.py` assert that these fields do not exist and that prompt construction rejects Layer 2 strings.

### Prompt Boundary

Nucleus prompts are constructed from:

- scenario task description
- current genome structure summary
- recent mutation history as `NucleusSignal` objects
- configured signal policy

Nucleus prompts are checked with `SignalPolicy.assert_no_layer2_leak()` before LLM invocation. If prompt text contains Layer 2 markers such as `validation_case`, `expected_output`, `ground_truth`, `criterion_score`, `per_case`, or `hidden_eval`, StemOS raises a `ValueError`.

Guardian and Nucleus also use separate system prompts and separate LLM calls. Guardian receives harness outputs and fixed evaluation criteria. Nucleus receives structure and policy-filtered signals. They do not share message history.

### Signal Policy In Scenario YAML

Scenarios can configure Layer 1 exposure:

```yaml
signal_policy:
  layer_1_enabled: true
  expose_aggregate_score: true
  expose_score_delta: true
  expose_direction: true
```

For ablation experiments, you can disable aggregate score exposure without editing the scenario file:

```bash
stemos evolve scenarios/gsm8k_mini --run-id gsm8k_blind \
  --signal-policy-override layer_1_enabled=false

stemos evolve scenarios/gsm8k_mini --run-id gsm8k_signal
```

In the blind run, lineage `nucleus_signal.aggregate_score` is `null`. In the signal-guided run, promoted mutation signals include aggregate scores.

## Setup

StemOS requires Python 3.10 or newer. The default setup runs in offline deterministic
mode and does not require an OpenAI API key.

### 1. Create a virtual environment

```bash
git clone <repo-url>
cd stem_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install the package

For local development and tests:

```bash
pip install -e ".[dev]"
```

For OpenAI-backed runs, install the optional OpenAI adapter too:

```bash
pip install -e ".[dev,openai]"
```

### 3. Configure environment variables

Copy the example file:

```bash
cp .env.example .env
```

The app reads settings from environment variables. A `.env` file is a convenient
template, but it is not loaded automatically by this repo. To load it in your shell:

```bash
set -a
source .env
set +a
```

For offline deterministic mode, keep:

```bash
STEMOS_OFFLINE_MODE=true
OPENAI_API_KEY=
```

For OpenAI-backed runs with Chat Completions, set:

```bash
OPENAI_API_KEY=...
STEMOS_OFFLINE_MODE=false
STEMOS_MODEL=gpt-4.1-mini
STEMOS_OPENAI_ENDPOINT=chat_completions
```

`STEMOS_OPENAI_ENDPOINT=auto` first tries the Responses API, then falls back to Chat Completions if the key is missing `api.responses.write` scope.

### 4. Verify the install

```bash
python -m pytest
stemos --help
stemos evolve scenarios/toy_structured_answer --run-id smoke_001
stemos execute runs/smoke_001/frozen_genome.yaml --input "Help me plan a focused workday." --stream-trace
```

If the `stemos` command is not found, confirm the virtual environment is active:

```bash
source .venv/bin/activate
```

## Provide A Scenario

Create a new scenario package:

```bash
stemos init-scenario scenarios/my_scenario
```

A scenario contains:

- `scenario.yaml`
- `train_cases.jsonl`
- `validation_cases.jsonl`
- optional `hidden_cases.jsonl`

Validation cases are used for promotion when present, so evolution cannot promote a genome only because it overfits training cases.

Hidden cases are run silently on promoted mutations when present. Hidden scores are written to audit logs and visualizations, but they are never passed back to Nucleus and never used for promotion.

### GSM8K Mini Benchmark

StemOS includes a `gsm8k_mini` benchmark scenario for grade-school math reasoning. Initialize or refresh its cases with:

```bash
stemos init-benchmark gsm8k_mini --n-train 30 --n-val 20
```

Then run evolution:

```bash
stemos evolve scenarios/gsm8k_mini --run-id gsm8k_001
stemos compare runs/gsm8k_001
stemos visualize runs/gsm8k_001
```

The GSM8K evaluator uses structured criteria:

- exact match after extracting the final number
- visible arithmetic reasoning steps
- a conservative check for hallucinated numbers

The baseline genome is a minimal zero-shot chain-of-thought math solver: one `solver` role, one `solve` workflow step, no tools, no quality gates, and an empty self-evaluation rubric.

## Run Evolution

Smoke test:

```bash
stemos evolve scenarios/toy_structured_answer --run-id smoke_001
```

To watch agent outputs and Nucleus/Guardian decisions while evolution runs:

```bash
stemos evolve scenarios/toy_structured_answer --run-id smoke_001 --stream-training-transcript
```

Stronger demo:

```bash
stemos evolve scenarios/tiny_task_operator --run-id demo_001
```

Harder benchmark scenario:

```bash
stemos evolve scenarios/hard_scenario --run-id hard_001 --stream-training-transcript
stemos progress runs/hard_001
```

Optional Git provenance is off by default:

```bash
stemos evolve scenarios/toy_structured_answer --run-id demo_001 --git-branch --git-commit
```

This creates a local branch named `stemos/run/<run_id>` and commits only at safe evolution points such as baseline evaluation, promoted mutations, rejected or rolled-back mutations, generation completion, pause/resume, and final freeze. Git is provenance only; runtime recovery still uses checkpoint files under `runs/<run_id>/checkpoint/latest/`.

Push is never automatic unless explicitly requested:

```bash
stemos evolve scenarios/toy_structured_answer --run-id demo_001 --git-branch --git-commit --git-push
stemos push-run demo_001
```

Before committing or pushing, StemOS scans staged files for `.env`, `OPENAI_API_KEY`, common secret patterns, and large temporary files.

The run writes:

- `runs/<run_id>/config_snapshot.yaml`
- `runs/<run_id>/baseline_genome.yaml`
- `runs/<run_id>/baseline_genome_score.json`
- `runs/<run_id>/frozen_genome.yaml`
- `runs/<run_id>/lineage.jsonl`
- `runs/<run_id>/archive.jsonl`
- `runs/<run_id>/llm_calls.jsonl`
- `runs/<run_id>/rubric_audit.jsonl` when self-evaluation rubric changes
- `runs/<run_id>/hidden_eval_log.jsonl` when hidden cases exist
- `runs/<run_id>/report.md`
- per-generation genomes, mutation plans, traces, outputs, and evaluation results

`archive.jsonl` is an append-only Top-K genome archive. Each entry contains a `genome_id`, genome payload, score, generation, parent id, and timestamp. Resume checkpoints include an archive snapshot so pause/resume preserves search state.

`llm_calls.jsonl` records auditable Nucleus and Guardian calls with role, prompt hash, response hash, token estimate, and prompt text. This is used to verify that Guardian and Nucleus calls are separate and that Nucleus prompts do not contain Layer 2 content.

## Inspect Before And After

```bash
stemos inspect runs/demo_001
stemos compare runs/demo_001
stemos visualize runs/demo_001
stemos progress runs/demo_001
stemos aggregate runs/openai_001 runs/openai_002 runs/openai_003
stemos compare-ablations runs/gsm8k_blind runs/gsm8k_signal
```

`report.md` includes:

- baseline genome score
- evolved genome score
- promoted mutations
- rejected or rolled-back mutations
- frozen genome path
- before/after comparison
- lineage narrative explaining how the harness differentiated
- genome archive table
- Mermaid evolution timeline and before/after harness graph
- Guardian selection board
- hidden eval vs train eval panel when hidden scores exist
- OpenAI run metadata without API keys

`stemos progress` writes or refreshes `runs/<run_id>/visuals/training_progress.md`, which includes a promotion-score chart, generation score table, and mutation decision timeline. It can be used while a run is still in progress:

```bash
watch -n 2 'stemos progress runs/smoke_001'
```

`stemos visualize` writes reviewer-facing files under `runs/<run_id>/visuals/`, including `training_progress.md`, `evolution_timeline.md`, `organism_shape.md`, `harness_before_after.md`, `guardian_selection_board.md`, `output_comparison.md`, and `visual_report.md`.

`stemos compare-ablations` writes `runs/ablation_comparison.md`. It compares runs on:

- generations to reach score greater than 0.7
- final validation score
- final hidden eval score
- validation-hidden overfitting gap
- mutation type entropy
- stagnation events

The command works with completed or in-progress runs. Missing metrics are shown as `N/A`.

## Safe Stop Controls

Each run has a control file at `runs/<run_id>/control.json`.

```bash
stemos pause demo_001
stemos resume demo_001
stemos status demo_001
stemos freeze-now demo_001
stemos abort demo_001
```

The evolution loop checks `control.json` at safe points. Pause saves a checkpoint and exits cleanly. Resume validates the scenario hash and continues from `checkpoint/latest/state.json`. Freeze-now freezes the best verified genome, never an unverified candidate. Abort saves state and does not promote the current candidate.

## Execute A Frozen Harness

```bash
stemos execute runs/demo_001/frozen_genome.yaml --input "Help me plan a focused workday."
```

## Safety Limitations

This MVP has a deterministic local fallback and a narrow OpenAI adapter. Generated tools are statically checked for forbidden imports and must include tests before activation. The sandbox is intentionally conservative and should be hardened further before running untrusted code outside local demos.

Forbidden imports for generated tools include:

- `os`
- `subprocess`
- `socket`
- `requests`
- `httpx`
- `shutil`
- `pathlib`

## Example Run

```bash
export STEMOS_OFFLINE_MODE=false
export STEMOS_MODEL=gpt-4.1-mini
export STEMOS_OPENAI_ENDPOINT=chat_completions

stemos evolve scenarios/toy_structured_answer --run-id openai_002
stemos visualize runs/openai_002
stemos compare runs/openai_002
```

Observed OpenAI-backed result:

```text
Baseline score: 0.1049
Final score: 0.9727
Promoted mutations:
  - modify_self_evaluation
  - modify_environment
  - add_workflow_step review_against_requirements
  - add_workflow_step revise_final_output
  - add_quality_gate required_sections_gate
  - create_tool requirement_sections_checker
Rejected or rolled back mutations:
  - unsafe generated tool rejected
  - redundant role rolled back
```
