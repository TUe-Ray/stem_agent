#!/bin/bash
# Run stem_agent with OpenAI gpt-4o
cd ~/projects/stem_agent_dev
source .venv/bin/activate
set -a
source .env
set +a
export STEM_AGENT_MODEL=gpt-4o
export STEM_AGENT_LANGFUSE_DISABLED=1
export STEM_AGENT_SINGLE_AGENT=1
# Clear stale bytecode to ensure code changes take effect
find src -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
exec stem_agent "$@"
