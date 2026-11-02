#!/bin/bash
# The RQ1 RL environment of node-94 on Stony Brook NVwulf (H200, sm90), from PyPI / GitHub / HF directly: the AReaL
# trainer venv (Python 3.12, torch 2.9.1+cu129), the SGLang rollout venv (Python 3.10, sglang 0.5.19 + node-94's local
# patches + the Qwen3.5 GDN LoRA patch), Harbor 0.23.0 (Python 3.13), Qwen3.5-4B with its text-only checkpoint and the
# zero rank-64 all-linear adapter. Exact versions come from the env bundle (node-94 /home/user/octopus-sft/env-bundle
# plus vendor/harbor-freeze.txt). CODE is this repo's code (git archive), already unpacked.
#
# usage: install_env.sh ROOT BUNDLE_DIR CODE [STEPS]     (on the login node: ~50 min, mostly copying wheels onto Lustre)
# STEPS: some of "areal sglang harbor model" (unset: all; "": none, e.g. the overlay alone:
#   CURSOR=ROOT/cursor-SHA install_env.sh ROOT BUNDLE_DIR CODE ""). Optional settings:
#   SGLANG_VENV  the SGLang venv (default ROOT/sglang-venv)
#   CURSOR       rebuilt each run: CURSOR/areal/areal (AReaL with the patches) and CURSOR/sglang/sglang (the SGLang
#                venv's, with the cursor and mount patches), the overlay put first on PYTHONPATH (docs/cursor-protocol.md
#                "Deployment"); deploy.sh install passes ROOT/cursor-SHA
#   TANTIVY      the search service's tantivy (default 0.26.2): it must be the version that built the BM25 index it serves,
#                0.26.2 and 0.27.0 stem differently ("international": intern / internat)
set -euo pipefail
ROOT=$1; B=$2; CODE=$3; STEPS=" ${4-areal sglang harbor model} "
SGLANG_VENV=${SGLANG_VENV:-$ROOT/sglang-venv}; SPY=$SGLANG_VENV/bin/python; A=$(cat "$B/areal-commit")
AREAL_PATCHES="logprobs cursor route-spread cpu-batch train-rpc-timeout"  # octopus/rl/proxy.py needs cursor
export UV_CACHE_DIR=$ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$ROOT/python UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
TORCH=https://download.pytorch.org/whl
step() { echo "=== $(date +%T) $*"; }

step uv
[ -x "$ROOT/bin/uv" ] || { mkdir -p "$ROOT/bin" "$ROOT/cache" && curl -fsSL -m 600 -o "$ROOT/cache/uv-0.11.8.tar.gz" \
  https://github.com/astral-sh/uv/releases/download/0.11.8/uv-x86_64-unknown-linux-gnu.tar.gz \
  && echo "56dd1b66701ecb62fe896abb919444e4b83c5e8645cca953e6ddd496ff8a0feb  $ROOT/cache/uv-0.11.8.tar.gz" | sha256sum -c --quiet \
  && tar xzf "$ROOT/cache/uv-0.11.8.tar.gz" -C "$ROOT/bin" --strip-components=1; }
UV=$ROOT/bin/uv

areal() {  # DIR: AReaL at the bundle's commit with the patches, built in DIR.tmp, then moved to DIR (keeps DIR/.venv)
  [ -d "$1/.venv" ] || [ ! -d "$1.tmp/.venv" ] || { mkdir -p "$1"; mv "$1.tmp/.venv" "$1/"; }  # an interrupted run's venv
  rm -rf "$1.tmp"; mkdir -p "$1.tmp"
  curl -fsSL -m 600 "https://github.com/areal-project/AReaL/archive/$A.tar.gz" | tar xz -C "$1.tmp" --strip-components=1
  for p in $AREAL_PATCHES; do patch -s -d "$1.tmp" -p1 < "$CODE/patches/areal-$p.patch"; done
  echo "$STAMP" > "$1.tmp/.octopus-patches"
  [ ! -d "$1/.venv" ] || mv "$1/.venv" "$1.tmp/"
  rm -rf "$1"; mv "$1.tmp" "$1"
}
# the commit and the patches' contents: a source built with other patches is rebuilt (check_env.sh checks the current ones)
STAMP="$A $(cd "$CODE/patches" && for p in $AREAL_PATCHES; do cat "areal-$p.patch"; done | sha256sum)"

if [[ $STEPS == *" areal "* ]]; then
step areal source
[ "$(cat "$ROOT/areal/.octopus-patches" 2>/dev/null)" = "$STAMP" ] || areal "$ROOT/areal"

step trainer venv
"$UV" venv -q --allow-existing --python 3.12.13 "$ROOT/areal/.venv"
sed -e "s|^-e file:///mnt/disk1/user/areal|-e file://$ROOT/areal|" "$B/areal-freeze.txt" > "$ROOT/areal-req.txt"
echo "tantivy==${TANTIVY:-0.26.2}" >> "$ROOT/areal-req.txt"  # the search service's BM25 (octopus/envs/search/service.py)
"$UV" pip install -q --python "$ROOT/areal/.venv/bin/python" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH/cu129" -r "$ROOT/areal-req.txt"
fi

if [[ $STEPS == *" sglang "* ]]; then
step sglang venv
"$UV" venv -q --allow-existing --python 3.10 "$SGLANG_VENV"
sed -e "s|^torch @ file://.*|torch==2.8.0+cu128|" "$B/sglang-freeze.txt" > "$ROOT/sglang-req.txt"
"$UV" pip install -q --python "$SPY" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH/cu128" -r "$ROOT/sglang-req.txt"
SP=$SGLANG_VENV/lib/python3.10/site-packages
tar xf "$B/sglang-patch/files.tar" -C "$SP"  # node-94's local patches
for p in lora_targets lora_abort_release top_k; do "$SPY" "$CODE/patches/apply_sglang_$p.py" "$SP/sglang"; done
# Triton 3.4.0 bug: tl.extra.cuda.gdc_* still take _builder, the compiler passes _semantic (hit on sm90+, PDL)
sed -i 's/_builder/_semantic/g' "$SP/triton/language/extra/cuda/gdc.py"
fi

if [ -n "${CURSOR:-}" ]; then
step "cursor overlay $CURSOR"
rm -rf "$CURSOR.tmp"; mkdir -p "$CURSOR.tmp/areal" "$CURSOR.tmp/sglang"
areal "$CURSOR.tmp/areal/src"; mv "$CURSOR.tmp/areal/src/areal" "$CURSOR.tmp/areal/"; rm -rf "$CURSOR.tmp/areal/src"  # its package only
cp -r "$("$SPY" -c 'import os, sglang; print(os.path.dirname(sglang.__file__))')" "$CURSOR.tmp/sglang/"
for p in cursor mount; do "$SPY" "$CODE/patches/apply_sglang_$p.py" "$CURSOR.tmp/sglang/sglang"; done
rm -rf "$CURSOR"; mv "$CURSOR.tmp" "$CURSOR"
fi

if [[ $STEPS == *" harbor "* ]]; then
step harbor venv
"$UV" venv -q --allow-existing --python 3.13 "$ROOT/harbor-venv"
"$UV" pip install -q --python "$ROOT/harbor-venv/bin/python" --no-deps -r "$B/harbor-freeze.txt"
fi

[[ $STEPS == *" model "* ]] || { step install done; exit 0; }
step model
M=$ROOT/models
[ -f "$M/Qwen3.5-4B/config.json" ] || "$ROOT/areal/.venv/bin/python" -c \
  "import sys; from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-4B', local_dir=sys.argv[1])" "$M/Qwen3.5-4B"
cd "$CODE"
[ -f "$M/Qwen3.5-4B-text/config.json" ] || \
  "$ROOT/areal/.venv/bin/python" -m octopus.rl.text_checkpoint --src "$M/Qwen3.5-4B" --out "$M/Qwen3.5-4B-text"
[ -f "$M/adapter-v0-4b-text-r64/adapter_config.json" ] || CUDA_VISIBLE_DEVICES="" OCTOPUS_SGLANG_PY="$SPY" \
  "$ROOT/areal/.venv/bin/python" -m octopus.rl.zero_adapter --model "$M/Qwen3.5-4B-text" --out "$M/adapter-v0-4b-text-r64" --rank 64 --alpha 64
step install done
