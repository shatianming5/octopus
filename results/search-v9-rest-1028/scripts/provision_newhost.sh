#!/bin/bash
# Code and Python environments for newhost.sh, at newhost.env's paths (README "New host (10-29)"):
#   provision_newhost.sh bundle HOST [REF...]  (control machine) git bundle of origin/main and the REFs (e.g.
#                    origin/chunk-major-p3) -> HOST:ROOT/code's origin refs, only what HOST lacks; main fast-forwarded
#   provision_newhost.sh venvs                 (host) scripts/nvwulf/install_env.sh with the HF env bundle: trainer venv
#                    venvs/areal/.venv (AREAL_PY), SGLang venv venvs/sglang (SGLANG_PY), overlay CURSOR (rebuilt each run)
#   provision_newhost.sh check                 (host) scripts/check_env.sh on AREAL_PY, SGLANG_PY and CURSOR
# bundle and venvs refuse while a trainer, an SGLang server, newhost.sh's watcher, a search_dev_eval or the search
# service runs (FORCE=1: anyway): a fast-forward or a rebuilt venv / overlay under a live run changes the code it loads.
set -euo pipefail
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); C=$(cd "$K/../../.." && pwd)
source "$K/newhost.env"; V=$ROOT/venvs; CODE=$ROOT/code
JOBS='octopus\.rl\.train|sglang\.launch_server|newhost\.sh _watch'
JOBS+='|octopus\.eval\.search_dev_eval|octopus\.envs\.search\.service'
idle() { [ "${FORCE:-}" = 1 ] || ! pgrep -u "$(id -un)" -fa "$JOBS"; }
case ${1:-} in
bundle)
  H=${2:?HOST}; REFS="origin/main ${*:3}"
  ssh "$H" "$(declare -f idle); FORCE=${FORCE:-}; JOBS='$JOBS'; idle" || { echo "$H runs jobs (FORCE=1: anyway)"; exit 1; }
  git -C "$C" fetch -q origin
  have=$(ssh "$H" "cd $CODE 2>/dev/null && for r in $REFS; do git rev-parse -q --verify \$r^{commit}; done" | sed 's/^/^/' || true)
  [ "$(git -C "$C" rev-list --count $REFS $have)" != 0 ] || { echo "$H has $REFS"; exit 0; }
  B=$(git -C "$C" rev-parse --absolute-git-dir)/newhost.bundle; git -C "$C" bundle create -q "$B" $REFS $have
  ssh "$H" "mkdir -p $ROOT/tmp" && scp -q "$B" "$H:$ROOT/tmp/octopus.bundle" && rm "$B"
  ssh "$H" "set -e; [ -d $CODE/.git ] || { git init -q -b main $CODE; git -C $CODE remote add origin git@github.com:shatianming5/octopus.git; }
    cd $CODE; git fetch -q $ROOT/tmp/octopus.bundle '+refs/remotes/origin/*:refs/remotes/origin/*'
    if git rev-parse -q --verify HEAD > /dev/null; then git merge -q --ff-only origin/main; else git reset -q --hard origin/main; git branch -q -u origin/main; fi
    git log --oneline -1"
  ;;
venvs)
  idle || { echo "jobs running (FORCE=1: anyway)"; exit 1; }
  for f in areal-commit areal-freeze.txt sglang-freeze.txt sglang-patch/files.tar; do
    curl -fsSL -m 600 --create-dirs -o "$V/bundle/$f" "https://huggingface.co/datasets/Tommysha/octopus-train/resolve/code/env-bundle/$f"; done
  # tantivy 0.27.0: the version that built this host's openresearcher index (README "Search index")
  SGLANG_VENV=${SGLANG_PY%/bin/python} CURSOR=$CURSOR TANTIVY=0.27.0 \
    bash "$C/scripts/nvwulf/install_env.sh" "$V" "$V/bundle" "$C" "areal sglang"
  ;;
check) bash "$C/scripts/check_env.sh" "$V" "$C" "$CURSOR" "$AREAL_PY" "$SGLANG_PY" && echo "check passed" ;;
*) sed -n 2,9p "$0"; exit 1 ;;
esac
