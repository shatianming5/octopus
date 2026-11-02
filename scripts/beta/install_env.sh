#!/bin/bash
# The Search RL stack on Cluster Beta (aarch64 Grace + GB200, sm_100), from Alpha's exact pins (ROOT_A/areal-req.txt,
# sglang-req.txt, the env bundle's harbor-freeze.txt; scripts/nvwulf/install_env.sh built those on x86) with the aarch64
# wheels: torch resolved for aarch64 from PyTorch's cu129 index (SGLang's torch
# 2.8.0 has no aarch64 cu128 wheel: +cu129, same version), sgl_kernel 0.3.5 +cu129 aarch64, flash-attn 2.8.3's aarch64
# prebuild, nvrtc for sgl_kernel; torchcodec (no aarch64 wheel, video only) is left out. Everything else --no-deps at
# Alpha's versions (the trainer's nvidia-* too: cudnn 9.19). Then
# node-94's SGLang patch files + lora_targets/lora_abort_release/top_k, Triton's gdc.py fix, tantivy, the cursor overlay
# ROOT/cursor-SHA (as deploy.sh install), ROOT/harbor/scripts/env.sh, ROOT/cuda-13.0 (nvcc for SGLang's JIT kernels). Models and search data are Alpha's (read only).
#   install_env.sh ROOT CODE      (ROOT = .../beta/octopus, CODE = ROOT/code-SHA; on the login node)
# Each venv and overlay is built once (its .octopus-installed): a rerun, e.g. for a new code dir's overlay, leaves them
# alone, since reinstalling rewrites .so files that running servers map (a judge died of SIGBUS that way, 10-23).
set -euo pipefail
ROOT=$1; CODE=$2; A=/projects/cluster/octopus/octopus; B=$A/harbor/datasets/env-bundle
export UV_CACHE_DIR=$ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$ROOT/python UV_HTTP_TIMEOUT=600 UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-managed  # ~/.local/bin (shared with Alpha) has x86 interpreters
UV=/projects/cluster/octopus/beta/envs/bin/uv; TORCH=https://download.pytorch.org/whl/cu129
step() { echo "=== $(date +%T) $*"; }
rest() { grep -vE "^(torch|torchvision|torchaudio|triton)==|^torchcodec==|^#" "$1"; }  # resolved per arch instead

step areal source
if [ ! -f "$ROOT/areal/pyproject.toml" ]; then
  mkdir -p "$ROOT/areal.tmp" && curl -fsSL -m 600 "https://github.com/areal-project/AReaL/archive/$(cat "$B/areal-commit").tar.gz" \
    | tar xz -C "$ROOT/areal.tmp" --strip-components=1
  for p in logprobs cursor route-spread cpu-batch train-rpc-timeout; do patch -d "$ROOT/areal.tmp" -p1 < "$CODE/patches/areal-$p.patch"; done
  mv "$ROOT/areal.tmp" "$ROOT/areal"
fi

step trainer venv
PY=$ROOT/areal/.venv/bin/python
if [ ! -f "$ROOT/areal/.venv/.octopus-installed" ]; then
"$UV" venv -q --allow-existing --python 3.12.13 "$ROOT/areal/.venv"
"$UV" pip install -q --python "$PY" --index-url "$TORCH" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  torch==2.9.1+cu129 torchvision==0.24.1 torchaudio==2.9.1 triton==3.5.1
rest "$A/areal-req.txt" | sed -e "s|^-e file://.*|-e file://$ROOT/areal|" -e "s|^flash-attn @ .*|flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-manylinux_2_34_aarch64.whl|" \
  -e "/^torch-memory-saver==/d" > "$ROOT/areal-req.txt"
"$UV" pip install -q --python "$PY" --no-deps -r "$ROOT/areal-req.txt" tantivy==0.26.2
touch "$ROOT/areal/.venv/.octopus-installed"; fi

step sglang venv
SPY=$ROOT/sglang-venv/bin/python; SP=$ROOT/sglang-venv/lib/python3.10/site-packages
if [ ! -f "$ROOT/sglang-venv/.octopus-installed" ]; then
"$UV" venv -q --allow-existing --python 3.10 "$ROOT/sglang-venv"
"$UV" pip install -q --python "$SPY" --index-url "$TORCH" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  torch==2.8.0+cu129 torchvision==0.23.0 torchaudio==2.8.0 triton==3.4.0
# torch 2.8.0's aarch64 wheel carries its CUDA 12.9 libraries in torch/lib (its nvidia-* deps are x86_64 only): Alpha's
# cu12.8 nvidia-* pins go, but not libnvrtc, which sgl_kernel needs (no RPATH; on x86 torch preloads pip's): torch's own
# nvrtc pin, preloaded RTLD_GLOBAL at interpreter start as torch does on x86
rest "$A/sglang-req.txt" | grep -vE "^nvidia-(cublas|cuda-cupti|cuda-nvrtc|cuda-runtime|cudnn|cufft|cufile|curand|cusolver|cusparse|cusparselt|nccl|nvjitlink|nvtx)-cu12==" | sed -e "s|^sgl-kernel @ .*|sgl-kernel @ https://github.com/sgl-project/whl/releases/download/v0.3.5/sgl_kernel-0.3.5+cu129-cp310-abi3-manylinux2014_aarch64.whl|" \
  > "$ROOT/sglang-req.txt"
"$UV" pip install -q --python "$SPY" --no-deps -r "$ROOT/sglang-req.txt" nvidia-cuda-nvrtc-cu12==12.9.86
# and libcudart.so.13 (RTLD_LOCAL: no symbol of it goes global), which nvcc 13 links SGLang's JIT kernels to: Alpha's
# nodes find it in /usr/local/cuda/lib64, here ROOT/cuda-13.0 (below)
echo "import ctypes; ctypes.CDLL('$SP/nvidia/cuda_nvrtc/lib/libnvrtc.so.12', ctypes.RTLD_GLOBAL); ctypes.CDLL('$ROOT/cuda-13.0/lib64/libcudart.so.13')" \
  > "$SP/octopus-nvrtc-preload.pth"
tar xf "$B/sglang-patch/files.tar" -C "$SP"  # node-94's local patches
for p in lora_targets lora_abort_release top_k; do "$SPY" "$CODE/patches/apply_sglang_$p.py" "$SP/sglang"; done
sed -i 's/_builder/_semantic/g' "$SP/triton/language/extra/cuda/gdc.py"  # Triton 3.4.0 (install_env.sh)
touch "$ROOT/sglang-venv/.octopus-installed"; fi

step harbor venv
if [ ! -f "$ROOT/harbor-venv/.octopus-installed" ]; then
"$UV" venv -q --allow-existing --python 3.13 "$ROOT/harbor-venv"
"$UV" pip install -q --python "$ROOT/harbor-venv/bin/python" --no-deps -r "$B/harbor-freeze.txt"
mkdir -p "$ROOT/harbor/bin" "$ROOT/harbor/scripts"; ln -sfn "$ROOT/harbor-venv/bin/harbor" "$ROOT/harbor/bin/harbor"
cat > "$ROOT/harbor/scripts/env.sh" <<EOF
# sourced by every search episode (octopus/envs/search/rl.py); no containers on Beta
export N=$ROOT/harbor OCTO_ENV=$ROOT OCTOPUS_HARBOR=$ROOT/harbor PATH="$ROOT/harbor/bin:\$PATH"
EOF
touch "$ROOT/harbor-venv/.octopus-installed"; fi

step "CUDA 13.0 toolkit $ROOT/cuda-13.0"  # nvcc for SGLang 0.5.19's JIT kernels: Alpha's nodes compile them with the
# same nvcc (13.0.88, /usr/local/cuda); Beta nodes have no toolkit, our container image has it
IMG=/projects/cluster/octopus/beta/containers/cuda13.0.3-devel-ubuntu24.04.sqsh
if [ ! -x "$ROOT/cuda-13.0/bin/nvcc" ]; then
  rm -rf "$ROOT/cuda-x.tmp"; unsquashfs -q -n -d "$ROOT/cuda-x.tmp" "$IMG" usr/local/cuda-13.0/{bin,include,lib64,nvvm,targets} > /dev/null 2>&1
  mv "$ROOT/cuda-x.tmp/usr/local/cuda-13.0" "$ROOT/cuda-13.0" && rm -rf "$ROOT/cuda-x.tmp"
fi
"$ROOT/cuda-13.0/bin/nvcc" --version | tail -1

step "models, search data: Alpha's"
for d in models search-data; do [ -e "$ROOT/$d" ] || ln -s "$A/$d" "$ROOT/$d"; done

CURSOR=$ROOT/cursor-${CODE##*/code-}
step "cursor overlay $CURSOR"
if [ ! -f "$CURSOR/.octopus-installed" ]; then
rm -rf "$CURSOR"; mkdir -p "$CURSOR/areal" "$CURSOR/sglang"
curl -fsSL -m 600 "https://github.com/areal-project/AReaL/archive/$(cat "$B/areal-commit").tar.gz" \
  | tar xz -C "$CURSOR/areal" --strip-components=1 "AReaL-$(cat "$B/areal-commit")/areal"
(cd "$CURSOR/areal" && for p in logprobs cursor route-spread cpu-batch train-rpc-timeout; do patch -p1 -s < "$CODE/patches/areal-$p.patch" || exit 1; done)
cp -r "$SP/sglang" "$CURSOR/sglang/" && "$SPY" "$CODE/patches/apply_sglang_cursor.py" "$CURSOR/sglang/sglang" \
  && "$SPY" "$CODE/patches/apply_sglang_mount.py" "$CURSOR/sglang/sglang" && touch "$CURSOR/.octopus-installed"
fi
step versions
for v in "$PY:triton transformers flash-linear-attention fla-core flash-attn peft sglang sglang-kernel flashinfer-python tantivy" \
         "$SPY:sglang sgl-kernel flashinfer-python triton transformers"; do
  "${v%%:*}" -c 'import importlib.metadata as m, sys, torch; print(sys.executable, "torch", torch.__version__, "cuda", torch.version.cuda,
    *(f"{p} {m.version(p)}" for p in sys.argv[1:]))' ${v#*:}
done
step install done
