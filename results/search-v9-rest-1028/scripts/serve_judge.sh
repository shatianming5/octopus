#!/bin/bash
# serve_judge.sh STATE MODEL GPUS: the Qwen3.5-27B-text judge (served as "judge", HOST:32500, HOST default 127.0.0.1;
# another host's address to serve other hosts, README "Judge on another host"), tp 2 on GPUS (e.g.
# 4,5), on GPUs shared with another user: launch only with BUDGET + 6 GiB free on each, at --mem-fraction-static
# BUDGET / (the smaller free memory now) (default BUDGET 30000 MiB). That is an estimate (serve_f2.sh's BUDGET note): SGLang
# applies the fraction to the free memory at its own distributed init, so memory another user frees in between makes the
# judge larger (10-29: 46.8 GB per GPU on a 30000 MiB budget). KV_TOKENS and MAMBA_SLOTS fix the pools instead
# (serve_f2.sh's note; recommended on shared GPUs; at 30000 MiB the weights' 25.2 GB leave almost no pool). share_watch.sh
# stops it when the other user needs the room.
# Flags: the 10-28 node-134 judge's, --disable-prefill-cuda-graph included. judge.pid/log in STATE; CU as in serve_f2.sh.
# On GPUs of our own (newhost.sh): MEM, the --mem-fraction-static to use instead of the BUDGET estimate; TP, the tp size
# (default 2; must equal the number of GPUS); SGLANG_PY, the SGLang venv's python.
S=$1; M=$2; G=$3; B=${BUDGET:-30000}; CU=${CU:-/usr/local/cuda-12.9}; PY=${SGLANG_PY:-/home/user/octo-env/sglang-venv/bin/python}
[ $(ls $M/model-*.safetensors $M/model.safetensors-*.safetensors 2>/dev/null | wc -l) -eq 11 ] || { echo "$M incomplete"; exit 1; }
free=$(nvidia-smi -i $G --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n 1)
[ -n "$free" ] && [ "$free" -ge $((B + 6144)) ] || { echo "GPUs $G: ${free:-unreadable} MiB free"; exit 1; }
mem=${MEM:-$(awk -v b=$B -v f=$free 'BEGIN {printf "%.3f", b / f}')}
export PATH=$CU/bin:$PATH CUDA_HOME=$CU SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1  # else PackageNotFoundError: sglang-kernel
cd $S
CUDA_VISIBLE_DEVICES=$G setsid "$PY" -m sglang.launch_server --model-path $M \
  --host ${HOST:-127.0.0.1} --port 32500 --tp-size ${TP:-2} --mem-fraction-static $mem --attention-backend triton --sampling-backend pytorch \
  --context-length 32768 --disable-prefill-cuda-graph --served-model-name judge \
  ${KV_TOKENS:+--max-total-tokens $KV_TOKENS} ${MAMBA_SLOTS:+--max-mamba-cache-size $MAMBA_SLOTS} > judge.log 2>&1 < /dev/null &
echo $! > judge.pid; echo "judge pid $! on GPUs $G, mem $mem (min free $free MiB)"
