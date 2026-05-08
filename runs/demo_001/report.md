# stem_agent Evolution Report

stem_agent is a universal differentiation mechanism. This run grew a specialized operating harness from scenario signals.

## Before / After
- Baseline genome score: 0.1049
- Evolved genome score: 0.9727
- Frozen genome: `runs/demo_001/frozen_genome.yaml`
- Promotion split policy: weighted_train_validation_40_60

## Organism Shape

### Initial organism
- Roles: 1
- Workflow steps: 2
- Self-evaluation: disabled
- Quality gates: 0
- Generated tools: 0
- Environment artifacts: 0

### Final organism
- Roles: 1
- Workflow steps: 4
- Self-evaluation: enabled
- Quality gates: 1
- Generated tools: 1 accepted, 1 rejected
- Environment artifacts: 5

## Why This Is Evolution, Not Subagent Orchestration
stem_agent does not start with a hand-written set of PM/Engineer/QA agents. It starts with a minimal Founder genome. Every new role, workflow step, quality gate, tool, or workspace artifact must appear as a mutation. Guardian evaluates the mutated harness and only promotes changes that improve fitness.

## Promoted Mutations
- modify_self_evaluation on self_evaluation: Previous runs missed required output sections.
- modify_environment on environment: The harness needs persistent task artifacts to preserve acceptance criteria, draft, QA, decisions, and final output.
- add_workflow_step on workflow: The harness currently produces final output without explicit review.
- add_workflow_step on workflow: Review notes should affect the delivered output.
- add_quality_gate on quality_gates: The harness should not finish without checking required output structure.
- create_tool on tools.generated: Add a safe checker tool that can support evolved quality gates.

## Rejected Or Rolled Back Mutations
- mutation_rejected create_tool on tools.generated: Forbidden import: os
- mutation_rolled_back add_role on roles: fitness did not improve enough for promotion

## Frozen Harness Shape
- Roles: Founder
- Workflow steps: understand_task, solve_task, review_against_requirements, revise_final_output
- Environment artifacts: artifacts/acceptance_criteria.md, artifacts/draft_output.md, artifacts/qa_report.md, artifacts/decision_log.md, artifacts/final_output.md
- Self-evaluation enabled: True

## Differentiation Story
stem_agent differentiation story

Scenario signals were interpreted by Nucleus, then converted into structured genome mutations.
- Generation 0 evaluated at 0.1049: Missing requirement: Must include a short summary; Missing requirement: Must include concrete steps; Missing requirement: Must include final answer
- Nucleus proposed modify_self_evaluation on self_evaluation: Previous runs missed required output sections.
- Guardian promoted modify_self_evaluation on self_evaluation because score improved from 0.1049 to 0.5229.
- Nucleus proposed create_tool on tools.generated: Try to inspect workspace files for extra context.
- Guardian rejected create_tool on tools.generated: Forbidden import: os.
- Generation 1 evaluated at 0.5229: Candidate satisfied visible requirements with acceptable complexity.
- Nucleus proposed modify_environment on environment: The harness needs persistent task artifacts to preserve acceptance criteria, draft, QA, decisions, and final output.
- Guardian promoted modify_environment on environment because score improved from 0.5229 to 0.6354.
- Generation 2 evaluated at 0.6354: Candidate satisfied visible requirements with acceptable complexity.
- Nucleus proposed add_workflow_step on workflow: The harness currently produces final output without explicit review.
- Guardian promoted add_workflow_step on workflow because score improved from 0.6354 to 0.7063.
- Generation 3 evaluated at 0.7063: Candidate satisfied visible requirements with acceptable complexity.
- Nucleus proposed add_workflow_step on workflow: Review notes should affect the delivered output.
- Guardian promoted add_workflow_step on workflow because score improved from 0.7063 to 0.7843.
- Generation 4 evaluated at 0.7843: Candidate satisfied visible requirements with acceptable complexity.
- Nucleus proposed add_quality_gate on quality_gates: The harness should not finish without checking required output structure.
- Guardian promoted add_quality_gate on quality_gates because score improved from 0.7843 to 0.9002.
- Generation 5 evaluated at 0.9002: Candidate satisfied visible requirements with acceptable complexity.
- Nucleus proposed create_tool on tools.generated: Add a safe checker tool that can support evolved quality gates.
- Guardian promoted create_tool on tools.generated because score improved from 0.9002 to 0.9727.
- Generation 6 evaluated at 0.9727: Candidate satisfied visible requirements with acceptable complexity.
- Nucleus proposed add_role on roles: Add an explicit reviewer role after self-evaluation has stabilized.
- Guardian rolled back add_role on roles because fitness did not improve enough for promotion.
- Evolution stopped at generation 6: maximum generations reached
