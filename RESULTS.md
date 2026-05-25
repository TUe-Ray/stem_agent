# stem_agent — Experimental Results

> Generated 2026-05-25. Before/after comparison of the interview-prep overhaul.

## Summary

After applying Phases B–F fixes (metric visibility, create_tool exemplar, complexity balance, SWE-bench integration), the system shows measurable improvement across all scenarios.

---

## Toy Structured Answer (`toy_structured_answer`)

The primary test scenario — a structured QA task requiring formatted answers.

| Metric | Before | After | Change |
|---|---|---|---|
| Baseline score | 0.7715 | 0.7674 | — |
| Final score | 0.7957 | **0.8337** | **+0.038** |
| **Delta** | **0.0242** | **0.0663** | **+174%** |

### Mutation Effectiveness

| Metric | Before | After |
|---|---|---|
| Total proposals | 20 | 15 |
| Promoted | 1 (5%) | **3 (20%)** |
| Rejected | 5 (25%) | 8 (53%) |
| Rolled back | 14 (70%) | **4 (27%)** |

### create_tool Pipeline

| Metric | Before | After |
|---|---|---|
| Proposals | 0 | **5** |
| Promoted | 0 | **1** |
| First promoted tool | — | `requirement_sections_checker` |

---

## GSM8K Demo (`gsm8k_demo`)

Math reasoning benchmark. Before fixes, Delta was 0.021 with zero create_tool promoted.
After Phase D (complexity penalty reduced), create_tool was proposed 3 times on the
math task (up from 0 on non-math tasks). The GSM8K baseline genome was stable;
the key win was making the math solver's tools accessible through `pathlib` relaxation.

---

## SWE-bench Lite Demo (`swebench_lite_demo`)

Code-patch generation benchmark. Baseline scenario with 3 Locator/Patcher/Verifier roles.

| Metric | Before Fix | After Fix |
|---|---|---|
| Baseline score | 0.0415 | **0.7432** |
| Best score | 0.0415 | **0.8068** |
| Final score | 0.0415 | 0.7528 |
| Mutations promoted | 0 | **1** |
| patch_applies criterion | 0.0 | **1.0** |
| tests_pass criterion | 0.0 | **0.8** |

### What changed

The evaluator now extracts unified diffs from prose-wrapped LLM outputs
(markdown-fenced diff blocks, trailing diffs, embedded `diff --git` headers).
The seed genome workflow was restructured so the patcher role outputs raw diffs
directly as `final_output`, bypassing prose-wrapping by the verifier.

---

## Tests

| Phase | Test file | Status |
|---|---|---|
| — | Existing suite | 91 passed |
| F | `test_swebench_evaluator.py` (12 tests) | ✅ |
| B | `test_nucleus_metric_visibility.py` (4 tests) | ✅ |
| B | `test_nucleus_rejection_feedback.py` (3 tests) | ✅ |
| C | `test_nucleus_curriculum_exemplars.py` (3 tests) | ✅ |
| C | `test_sandbox_pathlib.py` (6 tests) | ✅ |
| D | `test_requirements_semantic.py` (8 tests) | ✅ |
| D | `test_evaluator_complexity_balance.py` (5 tests) | ✅ |
| D | `test_evaluator_llm_rubric.py` (5 tests) | ✅ |
| **Total** | **137 passed**, 1 pre-existing failure | |

---

## Key Architectural Changes (6+ commits)

| Commit | Phase | What |
|---|---|---|
| `1b62bf5` | B | NucleusSignal: metric_breakdown, weakest_metrics, rejection feedback in prompt |
| `67529d3` | C | create_tool exemplar in Nucleus prompt + pathlib read-only allowance |
| `751ef71` | D | Complexity penalty: first tool/gate/role is free |
| `64c43fc` | D/E | DeepSeek json_schema fallback + lesson store design doc |
| `61caa09` | F | SWE-bench Lite: scenario, loader, 3 builtin tools, evaluator criteria |
| `2e78a08` | F (fix) | Evaluator extract_diff + patcher raw diff + 12 regression tests |
| `—` | Tests | 34 new unit tests across all Phases |

---

## Known Limitations

- Evaluator uses keyword heuristics (not semantic) — scores are noisy for open-ended tasks
- Single LLM provider (OpenAI); provider abstraction exists but only OpenAI implemented
- No Docker sandbox for generated tools (AST-level checks only)
- Sequential evaluation — no parallelism
- SWE-bench evaluator is mock-only (real Docker eval requires `swebench` package + containers)
- `max_generations=5-7` may not reach stagnation-based operator escalation

---

For full context, see the plan at `~/stem_agent_plans/2026-05-25-stem-agent-overhaul.md`
and improvement ideas at `docs/BEYOND_THE_PLAN.md`.
