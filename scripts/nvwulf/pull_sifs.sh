#!/bin/bash
# One read-only SIF per repo image of a Harbor dataset, for octopus.envs.swe.singularity, plus <repo>.root.tar: the
# image's /root (conda activation in .bashrc), which is mode 700 root and so unreadable inside an unprivileged container.
#   pull_sifs.sh DATASET_DIR SIF_DIR [parallel=3]
# Not on the login node: its 5 GB per-user memory cgroup OOM-kills mksquashfs. A CPU-only debug job (h200x4 refuses jobs
# without a GPU) builds in node RAM:
#   sbatch -p debug-h200x4 -c 32 --mem 200G -t 2:00:00 --wrap "SINGULARITY_TMPDIR=/dev/shm pull_sifs.sh DATASET SIF_DIR 8"
# Tasks name 64 images by their local transport tag (swebench/swesmith.x86_64.<repo>:transport-*); on Docker Hub they are
# jyangballin/swesmith.x86_64.<repo>:latest. The SIF is <SIF_DIR>/swesmith.x86_64.<repo>.sif either way. Existing files
# are kept; per image the log is SIF_DIR/logs/<repo>.log and failures are listed in SIF_DIR/failed.txt.
set -uo pipefail
D=$1; S=$2; P=${3:-3}
mkdir -p "$S/logs"; : > "$S/failed.txt"
pull() {  # hub image -> sif; build in a private temp dir, no layer cache (90 images would leave ~90 GB of blobs)
  local hub=$1 base t id; base=${hub#*/}; base=${base%%:*}
  t=$(mktemp -d "${SINGULARITY_TMPDIR:-/tmp}/pull-XXXXXX")
  if [ ! -f "$S/$base.sif" ]; then
    if SINGULARITY_TMPDIR=$t APPTAINER_TMPDIR=$t singularity pull --disable-cache "$S/$base.sif.part" "docker://$hub" > "$S/logs/$base.log" 2>&1; then
      mv "$S/$base.sif.part" "$S/$base.sif"
    else
      rm -rf "$S/$base.sif.part" "$t"; echo "$hub" >> "$S/failed.txt"; return
    fi
  fi
  if [ ! -f "$S/$base.root.tar" ]; then  # unsquashfs 4.4 has no -offset: dump the squashfs partition first
    id=$(singularity sif list "$S/$base.sif" | awk '/Squashfs/ {print $1; exit}')
    singularity sif dump "$id" "$S/$base.sif" > "$t/fs.sqfs" && unsquashfs -q -n -d "$t/fs" "$t/fs.sqfs" root > /dev/null \
      && tar -cf "$S/$base.root.tar.part" -C "$t/fs/root" . && mv "$S/$base.root.tar.part" "$S/$base.root.tar" \
      || echo "$hub (root.tar)" >> "$S/failed.txt"
  fi
  rm -rf "$t"
}
export -f pull; export S
grep -h '^docker_image' "$D"/*/task.toml | sed -E 's/.*"(.*)"/\1/' | sort -u \
  | sed -E 's#^swebench/(swesmith\.x86_64\.[^:]+):transport-.*#jyangballin/\1:latest#' \
  | xargs -P "$P" -I{} bash -c 'pull "$@"' _ {}
echo "$(ls "$S"/*.sif | wc -l) SIFs, $(ls "$S"/*.root.tar | wc -l) root tars, $(wc -l < "$S/failed.txt") failed"
