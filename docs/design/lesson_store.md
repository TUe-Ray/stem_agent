# Lesson Store — Cross-Generation Memory Design

> **Status:** Design document (RFC). Implementation deferred to a future PR.
> **Plan:** Section 11.3 of the stem_agent overhaul plan.

## Problem

Each generation in the stem_agent evolution loop starts fresh. The `MemoryStore`
(`harness/memory.py`) is a per-case dict that resets on every case. There is no
mechanism for a role to learn from its own failures in prior generations.

**Concrete example:** In the baseline_toy experiment, the LLM proposed
`modify_self_evaluation` 4 times across 4 generations — each time removing the
scenario rubric. Every time the Guardian rejected it with "Self-evaluation
rubric removed scenario criteria." The LLM never learned. It was like a
goldfish discovering the same wall 4 times.

## Design Goals

1. **Cross-generation persistence**: Lessons from gen 3 should be available in gen 4.
2. **Layer-2 safety**: Lessons must never contain case inputs, expected outputs,
   or per-case scores. Only aggregate patterns.
3. **Minimal overhead**: One extra JSONL file per run. No new dependencies.
4. **Prompt-compatible**: Lessons render as a compact text block appended to the
   Nucleus prompt (≤500 tokens).

## Storage Schema

```python
# src/stem_agent/evolution/lessons.py
from dataclasses import dataclass, asdict
from pathlib import Path
import json

@dataclass
class Lesson:
    generation: int
    mutation_type: str
    pattern: str          # "self_evaluation_rubric_removed"
    outcome: str          # "rejected" | "promoted" | "rolled_back"
    metric_delta: float | None  # score change after promotion
    short_advice: str     # ≤140 characters, prompt-ready

class LessonStore:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "lessons.jsonl"

    def record(self, lesson: Lesson) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(lesson)) + "\n")

    def recall(self, mutation_type: str | None = None, k: int = 5) -> list[Lesson]:
        """Return up to k lessons, optionally filtered by mutation type."""
        ...
```

## Hook Points

| Location | Trigger | Action |
|---|---|---|
| `evolution/loop.py` after `lineage.record_rejected_mutation` | Mutation rejected | Record a lesson with `outcome=rejected` + short advice |
| `evolution/loop.py` after `lineage.record_promoted_mutation` | Mutation promoted | Record a lesson with `outcome=promoted` + score delta |
| `nucleus/nucleus.py:build_nucleus_prompt` | Before mutation plan prompt | Inject `lesson_store.recall(...)` as "LESSONS LEARNED" block |

## Prompt Rendering (Example)

```
LESSONS LEARNED FROM PRIOR GENERATIONS:
- Gen 1: rejected modify_self_evaluation → Self-evaluation rubric removed criteria.
  Advice: When modifying self_evaluation, keep the original scenario rubric intact.
- Gen 2: promoted add_quality_gate (score +0.028)
  Advice: Adding a quality gate for output structure improved scores.
```

## Layer-2 Guard

Before recording any lesson, run the same check as `SignalPolicy.assert_no_layer2_leak`:

```python
FORBIDDEN_IN_LESSON = [
    "case_input", "expected_output", "ground_truth", "validation_case",
    "hidden_eval", "per_case", "answer_key"
]
def is_safe(text: str) -> bool:
    lower = text.lower()
    return not any(p in lower for p in FORBIDDEN_IN_LESSON)
```

If a lesson contains any forbidden pattern, reject it before writing to disk.

## Relation to Existing Systems

| System | Memory mechanism | Equivalent in stem_agent |
|---|---|---|
| Hermes Agent | `memory` tool + SQLite session store | Not yet — this is the closest analog |
| Claude Code | `.claude/` project context | Not yet — per-run `lessons.jsonl` |
| Voyager (Wang et al., 2023) | Skill library with embedding similarity | Future: named lesson retrieval by embedding |
| LlamaIndex | `ToolMetadata` with embedding index | Not yet — tool-level similarity |

## Implementation Plan (Future PR)

1. **Minimal viable (Phase E, ~200 lines)**
   - `src/stem_agent/evolution/lessons.py` — dataclass + store
   - Hook in `loop.py` after rejected and promoted mutations
   - Inject in `build_nucleus_prompt` as a "LESSONS LEARNED" text block
   - Layer-2 safety check in LessonStore.record

2. **Follow-up (Phase E+, ~300 lines)**
   - Per-mutation-type semantic recall (only show create_tool lessons when
     Nucleus is considering a tool proposal)
   - Deduplicate lessons with identical patterns
   - Cap store at last 20 lessons (prune oldest)

3. **Advanced (post-interview, ~500 lines)**
   - Embedding-based recall using a local model (all-MiniLM-L6-v2 via
     `sentence-transformers`)
   - Tool-level skill library: named, versioned, retrievable tool modules

## Decision Log

- **Why not SQLite?** JSONL is simpler for a run-local store. No schema
  migration burden. The existing `LineageLog` already uses this pattern.
- **Why not in prompt by default?** Token budget. 500 tokens is the cap for
  lesson injection. Lessons should be dense and actionable.
- **Why not share lessons across runs?** Future work. Cross-run lesson sharing
  opens a can of worms (contamination between task families). Scope is
  deliberately single-run for now.
