# StemOS Evolution Report

StemOS is a universal differentiation mechanism. This run grew a specialized operating harness from scenario signals.

## Before / After
- Baseline genome score: 0.4098
- Evolved genome score: 0.9423
- Frozen genome: `runs/demo_001/frozen_genome.yaml`
- Promotion split policy: weighted_train_validation_35_65

## Promoted Mutations
- modify_self_evaluation on self_evaluation: Previous runs missed required output sections.
- modify_environment on environment: The harness needs persistent task artifacts to preserve acceptance criteria, draft, QA, decisions, and final output.

## Rejected Or Rolled Back Mutations
- mutation_rejected create_tool on tools.generated: Forbidden import: os
- mutation_rolled_back add_role on roles: fitness did not improve enough for promotion

## Frozen Harness Shape
- Roles: Founder
- Workflow steps: understand_task, solve_task
- Environment artifacts: artifacts/acceptance_criteria.md, artifacts/draft_output.md, artifacts/qa_report.md, artifacts/decision_log.md, artifacts/final_output.md
- Self-evaluation enabled: True

## Differentiation Story
StemOS differentiation story

Scenario signals were interpreted by Nucleus, then converted into structured genome mutations.
- Generation 0 evaluated at 0.4098: Missing requirement: Must include a short summary; Missing requirement: Must include acceptance criteria; Missing requirement: Must include concrete steps
- Guardian promoted modify_self_evaluation on self_evaluation because score improved from 0.4098 to 0.7498.
- Guardian rejected create_tool on tools.generated: Forbidden import: os.
- Generation 1 evaluated at 0.7498: Missing requirement: Must include acceptance criteria; Missing requirement: Must include QA report; Missing requirement: Must include decision log
- Guardian promoted modify_environment on environment because score improved from 0.7498 to 0.9423.
- Generation 2 evaluated at 0.9423: Candidate satisfied visible requirements with acceptable complexity.
- Guardian rolled back add_role on roles because fitness did not improve enough for promotion.
- Generation 3 evaluated at 0.9423: Candidate satisfied visible requirements with acceptable complexity.
- Evolution stopped at generation 3: validation score plateaued for 2 generations
