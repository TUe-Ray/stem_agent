#!/usr/bin/env bash
set -euo pipefail
cd /home/hermes/projects/stem_agent_dev
rm -rf src/stem_agent/*/__pycache__
export $(grep -E '^DEEPSEEK_' ~/.hermes/.env | xargs)
source .venv/bin/activate
set -a
source .env
set +a
unset OPENAI_API_KEY
export STEM_AGENT_MODEL=deepseek/deepseek-v4-pro
export STEM_AGENT_LANGFUSE_DISABLED=0
export STEM_AGENT_SINGLE_AGENT=0
PYTHONUNBUFFERED=1 stem_agent evolve scenarios/swebench_lite_demo --run-id swe_v31 2>&1 | tee /tmp/swe_v31.log
