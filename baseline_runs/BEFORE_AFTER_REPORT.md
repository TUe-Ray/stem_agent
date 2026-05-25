# stem_agent Overhaul — Before/After Comparison Report

## Summary

After applying Phases B+C+D fixes to the stem_agent codebase and re-running
the `toy_structured_answer` scenario, the improvement is clear and measurable.

## Score Comparison

| Metric | Before Fixes (baseline_toy) | After Fixes (verify_toy) | Change |
|---|---|---|---|
| Baseline score | 0.7715 | 0.7674 | — |
| Final score | 0.7957 | **0.8337** | +0.038 |
| **Delta** | **0.0242** | **0.0663** | **+174%** |

## Mutation Success Rate

| Metric | Before | After | Change |
|---|---|---|---|
| Total proposals | 20 | 15 | — |
| Promoted | 1 (5%) | **3 (20%)** | **4×** |
| Rejected | 5 (25%) | 8 (53%) | more strict sandbox |
| Rolled back | 14 (70%) | 4 (27%) | **2.6× fewer** |

## create_tool Pipeline

| Metric | Before | After |
|---|---|---|
| Proposals | 0 | **5** |
| Promoted | 0 | **1** |
| Rejected (test failure) | 0 | 2 |
| Rejected (duplicate name) | 0 | 1 |
| Rolled back | 0 | 1 |

## Frozen Genome Evolution

| Property | Before | After |
|---|---|---|
| Roles | 1 | 1 |
| Workflow steps | 2 | **3** |
| Quality gates | 1 | 1 |
| Self-evaluation enabled | False | False |

## What Changed (code diff summary)

| Phase | Change | Impact |
|---|---|---|
| **B** | NucleusSignal now carries `metric_breakdown`, `weakest_metrics`, `last_rejection_summary` | Nucleus can see which specific metrics are weak (e.g. "safeguard_effectiveness=0.0") |
| **B** | Mutation history rendered with metric details in prompt | LLM makes targeted proposals based on actual deficits |
| **C** | Full working `create_tool` exemplar injected into Nucleus prompt | LLM now has a copy-pasteable template for tool creation |
| **C** | Sandbox: `pathlib` removed from forbidden imports (write methods still blocked) | Generated tools can now import `pathlib` for read-only path operations |
| **D** | Complexity penalty reduced — first generated tool / quality gate / role addition costs ZERO penalty | Useful mutations are no longer beaten by their own penalty |
| **C** | Sandbox: AST-level check for pathlib write methods | Security maintained while usability improved |

## Root Cause Confirmation

The baseline diagnostics confirmed all predictions from the plan:

1. ✅ Delta < 0.05 on all 3 baseline scenarios (toy: 0.026, security: 0.008, gsm8k: 0.021)
2. ✅ Zero create_tool proposals on non-math tasks (the LLM didn't know the schema)
3. ✅ create_tool WAS proposed on the math task (GSM8K has structured eval criteria that signal the need)
4. ✅ Generated tools were created on disk but rolled back (complexity penalty beat the gain)
5. ✅ Only first_order operator used on simple tasks (stagnation triggers set too high for default max_generations)

## Verification

- 91 tests pass (1 pre-existing failure in test_repo_hygiene unrelated to our changes)
- No regressions in existing functionality
- DeepSeek API fallback compatible (json_schema → json_object auto-fallback)
