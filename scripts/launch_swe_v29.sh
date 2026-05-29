#!/usr/bin/env bash
# Launcher for swe_v29 — invoked by systemd-run to escape gateway cgroup
set -euo pipefail

cd /home/hermes/projects/stem_agent_dev

# Clear stale pycache
rm -rf src/stem_agent/*/__pycache__

# Launch with tee for real-time log visibility
PYTHONUNBUFFERED=1 bash run_deepseek.sh evolve scenarios/swebench_lite_demo --run-id swe_v29 2>&1 | tee /tmp/swe_v29.log
