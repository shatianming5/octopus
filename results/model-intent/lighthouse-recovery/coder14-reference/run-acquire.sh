#!/usr/bin/env bash
set -euo pipefail

root=/home/user/octopus-qwen-coder14-reference-v1-20260916
cd "$root"
exec /home/user/octopus-r2-20260913/venv/bin/python acquire.py 2>&1 | tee download.log
