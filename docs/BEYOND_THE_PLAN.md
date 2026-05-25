# Beyond the Plan — Additional Improvement Ideas

> Generated 2026-05-25. These are NOT implemented in the current codebase.
> They are discussion items for the Jack Francis interview or future PRs.

## 1. Multi-Provider LLM Support (High Priority)

The codebase currently works with OpenAI and OpenRouter (via the chat_completions
path + json_schema→json_object fallback). Adding explicit provider support would be
a strong architectural improvement:

- `src/stem_agent/llm/providers/` with one file per provider
- Each provider implements `LLMProvider.call(prompt, schema, **kwargs)`
- Configuration via `STEM_AGENT_PROVIDER=openai|anthropic|google|deepseek|ollama`

The existing `ModelClient` already has the right abstraction — the change is mostly
extracting the OpenAI-specific code into a provider class and adding dispatch.

## 2. Parallel Evaluation (Medium Priority)

The evaluation loop in `evolution/loop.py:_run_and_evaluate` runs cases
sequentially. For scenarios with 100+ cases, this is the dominant cost.

- Use `concurrent.futures.ThreadPoolExecutor` around `HarnessRunner.run_case`
- The LLM calls are the bottleneck — ThreadPool is enough since the waiting is I/O
- With 4 parallel workers, a 10-case evaluation goes from 30s → 8s

## 3. Database-Backed Archive (Medium Priority)

The `GenomeArchive` currently uses JSONL files. For production:

- Replace with SQLite (single file, zero-config)
- Benefits: atomic writes, query by score/fitness_vector, no file corruption on crash
- Keep JSONL as export format for portability

## 4. Docker-Based Tool Sandbox (High Priority)

The current sandbox only does AST-level static checks. For real security:

- Each generated tool runs in a minimal Docker container
- Container gets read-only volume mounts, no network, CPU/memory limits
- Use `docker-py` to manage containers from Python

## 5. Web Dashboard (Medium Priority)

- FastAPI + htmx for a training monitor
- Show live generation progress, score charts, genome diffs
- Replay past runs from JSONL artifacts
- Prototype: 200 lines of FastAPI serving a single HTML page

## 6. CI/CD Pipeline (Quick Win)

- `.github/workflows/ci.yml`: pytest + mypy + ruff on PR
- Trivial to add (20 lines of YAML)
- Signals professional engineering practices

## 7. Configuration Management

- Replace manual `.env` with `pydantic-settings`
- Auto-detect provider from env vars
- Validate at startup and give clear error messages

## 8. Streaming Support

- Add `stream=True` option to `ModelClient.call`
- Forward streaming chunks through `RoleRunner` to the CLI
- Improves UX for `stem_agent execute` on the frozen harness

## 9. Tool Similarity Detection

Discussed in Plan Section 11.2:

- Tier 1: name + description string similarity (difflib)
- Tier 2: shared test corpus cross-validation
- Tier 3: embedding-based recall via sentence-transformers

## 10. Cross-Run Lesson Database

Discussed in Plan Section 11.3 + `docs/design/lesson_store.md`:

- Per-run `lessons.jsonl` is Phase E MVP
- Cross-run sharing: SQLite DB with run_id foreign key
- Embedding-based recall: all-MiniLM-L6-v2 for similarity search

## 11. Performance Profiling

- Add `STEM_AGENT_PROFILE=1` to log timing per component
- Identify bottlenecks beyond LLM latency
- Currently unknown: how much time goes to genome serialization vs evaluator vs sandbox

## 12. Documentation

- API docs via Sphinx/autodoc from existing type hints
- Architecture diagram (mermaid flowchart already in README)
- Contributor guide (setup, test, PR workflow)
- `EXPERIMENTS.md` with curated results and plots

## 13. Multi-Tenant Run Isolation

- Each run gets its own temp directory
- Clean up old runs with configurable TTL
- Prevent run directory collisions

## 14. Benchmark Comparison Table

Add to README: comparison of stem_agent vs LangGraph, AutoGen, CrewAI, DSPy
on dimensions that matter: evolves harness, guards safety, task-specific, hidden eval.

## 15. GSM8K Full Result

Re-run the full GSM8K benchmark (100 train, 100 validation) with the Phase B-D
fixes and publish the promotion curve. The existing baseline_gsm8k was a smoke demo
(5/5/10 split). A full run would be a credible benchmark result.

## 16. What I Would Redesign (Honest Reflection)

If starting over with the lessons learned:

1. **`evolution/loop.py` is 1910 lines** — split into `BaselinePhase`,
   `GenerationPhase`, `FreezePhase` classes with shared context dataclass.
2. **`evaluator.py` is 895 lines** — one-file-per-metric with a decorator-based
   registry would make adding new metrics trivial.
3. **`cli.py` is 723 lines** — per-command logic should live next to its evolution
   counterpart, not in the CLI file.
4. **The `ModelClient` abstraction is good** — keep it, but make provider plugins
   first-class from day 1.
5. **The Genome YAML format evolved gradually** — a formal schema evolution policy
   (with version bumps and migration functions) would prevent the current ad-hoc
   backward compatibility hacks.
