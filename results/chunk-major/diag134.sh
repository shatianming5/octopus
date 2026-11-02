#!/bin/bash
# The review's checks of the fp32 cm8 gap (results/chunk-major/README.md), one card: diag134.sh GPU
#  (a) 101k fp32, SEG 8k: cm8 with the kept tensors in pinned host memory against cm8 with them in GPU memory
#  (b) 41k fp32, SEG 4k, k1 / k1s2 / k1s8 noise floors and cm8 / cm4: TF32 as before, then IEEE fp32 throughout
#  (c) the GPU tests of the pinned slab's other user (tests/test_host_activations.py)
D=/mnt/disk4/user/chunk-major-p2
R=$D/code/results/chunk-major/run134.sh
G=$1
$R "$G" diag-a-101k-host-vs-gpu.log mb-101431.pt --fp32 --no-fsdp --seg 8 --variants cm8-fp32 cm8-fp32-gpu
$R "$G" diag-b-41k-tf32.log mb-41398.pt --fp32 --no-fsdp --seg 4 --variants k1 k1s2 k1s8 cm8-fp32 cm4-fp32
TRITON_F32_DEFAULT=ieee TRITON_CACHE_DIR=/mnt/disk4/user/tmp/chunk-major-p2/triton-ieee \
    $R "$G" diag-b-41k-ieee.log mb-41398.pt --fp32 --no-fsdp --ieee --seg 4 --variants k1 k1s2 k1s8 cm8-fp32 cm4-fp32
cd $D/code && CUDA_VISIBLE_DEVICES=$G PYTHONPATH=$D/areal-overlay:$D/code TMPDIR=/mnt/disk4/user/tmp/chunk-major-p2 \
    $D/.venv/bin/python -m pytest -q -p no:warnings tests/test_host_activations.py > $D/logs/diag-c-host-activations.log 2>&1
