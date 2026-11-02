#!/bin/bash
# make_arms.sh (node-138, 10-31 v45 sweep): the per-arm settings dirs dirname(RUN)/run-sw-ARM, made as
# scripts/alpha/search_ablate.sh makes them (RUN's config.yaml + its episode_settings.json with the arm's changes; an
# existing dir must hold the same settings), then a dry run of each: the settings the eval's process would get
# (octopus.rl.config.run_episode_settings under env -i, as search_eval_arm.sh runs it) and the limits the episode reads
# from them (octopus/runtime/tree.py's functions, copied; agent.py's DEFAULTS by setdefault).
set -euo pipefail
D=/mnt/ssd4/user/search-v9; C=$D/runtime/code-local; PY=$D/areal-venv/.venv/bin/python
RUN=/home/user/search-v9-run/areal-exp/logs/user/rq1-rl/search-v9-rest
ARMS=("linear:OCTOPUS_LINEAR=1" "w1:OCTOPUS_MAX_BRANCHES=1" "w2:OCTOPUS_MAX_BRANCHES=2" "w5:OCTOPUS_MAX_BRANCHES=5"
  "w20:OCTOPUS_MAX_BRANCHES=20,OCTOPUS_MAX_THREADS=21" "d1:OCTOPUS_FORK_DEPTH=1,OCTOPUS_MAX_BRANCHES=10"
  "d2:OCTOPUS_FORK_DEPTH=2,OCTOPUS_MAX_BRANCHES=10" "d3:OCTOPUS_FORK_DEPTH=3,OCTOPUS_MAX_BRANCHES=10")
for arm in "${ARMS[@]}"; do
  a=${arm%%:*}; c=${arm#*:}; T=$(dirname "$RUN")/run-sw-$a
  s=$(python3 - "$RUN/episode_settings.json" "$c" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
for k, v in (kv.split("=", 1) for kv in filter(None, sys.argv[2].split(","))):
    s.pop(k, None) if v == "" else s.update({k: v})
print(json.dumps(s, indent=1, sort_keys=True))
EOF
)
  if [ -d "$T" ]; then [ "$(cat "$T/episode_settings.json")" = "$s" ] || { echo "$T exists with other settings: refused"; exit 1; }
  else rm -rf "$T.tmp"; mkdir "$T.tmp"; cp "$RUN/config.yaml" "$T.tmp/"; echo "$s" > "$T.tmp/episode_settings.json"; mv "$T.tmp" "$T"; fi
done
for a in - linear w1 w2 w5 w20 d1 d2 d3; do
  T=$RUN; [ "$a" = - ] || T=$(dirname "$RUN")/run-sw-$a
  (cd $C && env -i HOME=$HOME PATH=/usr/bin:/bin PYTHONPATH=$D/areal-overlay:$C OCTOPUS_ROUTE_KV=0.7 OCTOPUS_ROUTE_GAP=60 \
    "$PY" -c '
import json, os, sys
from pathlib import Path
from octopus.rl.config import EPISODE_SETTINGS, run_episode_settings
from octopus.envs.search.defaults import DEFAULTS
import math  # tree.py: fork_depth/max_threads/max_branches, verbatim (tree.py imports harbor, not in this venv)
fd = lambda: int(os.environ.get("OCTOPUS_FORK_DEPTH", "").strip() or 2) or math.inf
mt = lambda: int(os.environ.get("OCTOPUS_MAX_THREADS", "16"))
mb = lambda: int(os.environ.get("OCTOPUS_MAX_BRANCHES", "").strip() or 0) or None
run = json.loads(Path(sys.argv[2], EPISODE_SETTINGS).read_text())
run_episode_settings(run, [])
for k, v in DEFAULTS.items(): os.environ.setdefault(k, v)
knobs = ("OCTOPUS_LINEAR", "OCTOPUS_MAX_BRANCHES", "OCTOPUS_FORK_DEPTH", "OCTOPUS_MAX_THREADS")
print(sys.argv[1], "linear", os.environ.get(knobs[0], "-"), "branches", mb(), "depth", fd(),
      "threads", mt(), "other", {k: v for k, v in run.items() if k not in knobs})
' "$a" "$T")
done
