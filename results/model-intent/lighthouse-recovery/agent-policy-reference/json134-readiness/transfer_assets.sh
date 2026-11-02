#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/octopus-agent7-json134-readiness-v1-20260916
SRC_HOST=user@10.0.2.83
SRC_ACTOR=/home/user/octopus-swe-gym-agent7-reference-v1-20260916/model/snapshots/97054e70501d79c27499518af5ce613be6278bcb
SRC_TOKENIZER=/home/user/octopus-qwen-coder-reference-v1-20260915/model/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242
DST_ACTOR=/home/user/octopus-swe-gym-agent7-reference-v1-20260916/model/snapshots/97054e70501d79c27499518af5ce613be6278bcb
DST_TOKENIZER=/home/user/octopus-qwen-coder-reference-v1-20260915/model/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242
PYTHON=/home/user/octopus-r2-20260913/venv/bin/python
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_SECONDS=$(date +%s)
STATUS=failed
DETAIL="asset transfer ended before digest verification"

mkdir -p "$ROOT" "$DST_ACTOR" "$DST_TOKENIZER"
exec > >(tee "$ROOT/transfer.log") 2>&1

finish() {
  local rc=$?
  local finished elapsed
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  elapsed=$(($(date +%s) - START_SECONDS))
  if [[ ! -e "$ROOT/terminal.json" ]]; then
    "$PYTHON" - "$ROOT/terminal.json" "$STATUS" "$DETAIL" \
      "$STARTED_AT" "$finished" "$elapsed" "$rc" <<'PY'
import json
import sys
from pathlib import Path

path, status, detail, started, finished, elapsed, rc = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema": "octopus-agent7-json134-transfer-terminal-v1",
    "status": status,
    "detail": detail,
    "started_at": started,
    "finished_at": finished,
    "wall_seconds": int(elapsed),
    "exit_code": int(rc),
}, sort_keys=True, indent=2) + "\n")
PY
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 124' TERM INT

"$PYTHON" - "$ROOT/started.json" "$STARTED_AT" <<'PY'
import json
import os
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema": "octopus-agent7-json134-transfer-start-v1",
    "started_at": sys.argv[2],
    "pid": os.getpid(),
    "source": "user@10.0.2.83",
    "received_asset_bytes_max": 16000000000,
    "wall_seconds_max": 3600,
    "planned_payload_bytes": 15242760997,
}, sort_keys=True, indent=2) + "\n")
PY
printf '{"event":"started","at":"%s","pid":%s}\n' "$STARTED_AT" "$$" >> "$ROOT/events.jsonl"

printf '%s\n' \
  config.json \
  model-00001-of-00004_4.safetensors \
  model-00002-of-00004_4.safetensors \
  model-00003-of-00004_4.safetensors \
  model-00004-of-00004_4.safetensors \
  | timeout --signal=TERM --kill-after=10 3500 \
    rsync -a --append-verify --files-from=- \
      "$SRC_HOST:$SRC_ACTOR/" "$DST_ACTOR/"
printf '{"event":"actor_received","at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/events.jsonl"

printf '%s\n' \
  config.json \
  generation_config.json \
  merges.txt \
  tokenizer.json \
  tokenizer_config.json \
  vocab.json \
  | timeout --signal=TERM --kill-after=10 3500 \
    rsync -a --append-verify --files-from=- \
      "$SRC_HOST:$SRC_TOKENIZER/" "$DST_TOKENIZER/"
printf '{"event":"tokenizer_received","at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/events.jsonl"

"$PYTHON" - "$ROOT/verification.json" "$DST_ACTOR" "$DST_TOKENIZER" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

output, actor_root, tokenizer_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
actor = [
    ("config.json", 606, "git_blob_sha1", "1bb26c94d1ba4e049ecc1a828df6bd7e77be1e5d"),
    ("model-00001-of-00004_4.safetensors", 4877660776, "sha256", "50483370092b6bf2a33e83150f6915a74cb12df870324b59acc5b87f62b726ca"),
    ("model-00002-of-00004_4.safetensors", 4932751008, "sha256", "782cca0cd9e57c25c7a4d62dc406e638c840053b7ec9bcb965c826f04b12e349"),
    ("model-00003-of-00004_4.safetensors", 4330865200, "sha256", "dd0ca030d6a4664d9f083b17d44439be5d79568101751c247e2eb70303dcb283"),
    ("model-00004-of-00004_4.safetensors", 1089994880, "sha256", "25380006a9b9e2aec6d14b7ae7a7cf718a9830bf5deb4d0f469350774432ad5d"),
]
tokenizer = [
    ("config.json", 663, "sha256", "c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28"),
    ("generation_config.json", 242, "sha256", "1a628a5775bc69cde01c6749a531150ca4d3189652c618a174f7077923acf3b1"),
    ("merges.txt", 1671839, "sha256", "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    ("tokenizer.json", 7031645, "sha256", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    ("tokenizer_config.json", 7305, "sha256", "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4"),
    ("vocab.json", 2776833, "sha256", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
]

def verify(root, rows):
    result = []
    for name, size, kind, expected in rows:
        path = root / name
        if kind == "git_blob_sha1":
            actual = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append({
            "path": name,
            "size": path.stat().st_size,
            "expected_size": size,
            "digest_kind": kind,
            "digest": actual,
            "expected_digest": expected,
            "match": path.stat().st_size == size and actual == expected,
        })
    return result

actor_result = verify(actor_root, actor)
tokenizer_result = verify(tokenizer_root, tokenizer)
payload = sum(row["size"] for row in actor_result + tokenizer_result)
value = {
    "schema": "octopus-agent7-json134-transfer-verification-v1",
    "actor": actor_result,
    "tokenizer": tokenizer_result,
    "payload_bytes": payload,
    "expected_payload_bytes": 15242760997,
    "all_match": all(row["match"] for row in actor_result + tokenizer_result),
}
output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
if not value["all_match"] or payload != value["expected_payload_bytes"]:
    raise SystemExit("asset verification failed")
PY

printf '{"event":"verified","at":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ROOT/events.jsonl"
STATUS=verified
DETAIL="actor and tokenizer payload transferred and verified"
