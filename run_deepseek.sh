#!/bin/bash
# Setup DeepSeek env and run stem_agent
# Workaround for hermes .env having unquoted values with spaces

export $(grep -E '^DEEPSEEK_' ~/.hermes/.env | xargs)

cd ~/projects/stem_agent_dev
source .venv/bin/activate
set -a
source .env
set +a

unset OPENAI_API_KEY
export STEM_AGENT_MODEL=deepseek/deepseek-v4-flash
export STEM_AGENT_LANGFUSE_DISABLED=1

exec stem_agent "$@"
