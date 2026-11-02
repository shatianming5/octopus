#!/bin/bash
# The Python environments of a deployment against what scripts/nvwulf/install_env.sh built: every name==version of
# ROOT's areal-req.txt / sglang-req.txt (and of each extra PY:REQ pair) in its venv, the Triton gdc.py fix, the AReaL
# patches in the trainer venv's source (ROOT/areal) and in the overlay (CURSOR/areal), check_areal_route.py on the
# overlay (PYTHONPATH=CURSOR/areal:CODE, as the trainer runs), the SGLang patches in CURSOR/sglang and its shims.
# One "ok"/"FAIL" line per check; exit 1 if any failed. Used by deploy.sh check and provision_newhost.sh check.
#
# usage: check_env.sh ROOT CODE CURSOR AREAL_PY SGLANG_PY [PY:REQ ...]
set -uo pipefail
ROOT=$1; CODE=$2; CURSOR=$3; AREAL_PY=$4; SGLANG_PY=$5; shift 5
O=$ROOT/check; mkdir -p "$O"; bad=0
ok() { echo "ok   $*"; }; no() { echo "FAIL $*"; bad=$((bad + 1)); }
vers() { "$1" - "$2" <<'PY'
import importlib.metadata as m, sys
bad = []
for line in open(sys.argv[1]):
    line = line.split(";")[0].strip()
    if "==" in line and not line.startswith(("#", "-")):
        name, want = line.split("==", 1)
        try:
            got = m.version(name)
        except m.PackageNotFoundError:
            got = None
        bad += [f"{name} {got} (want {want})"] if got != want and (got or "").split("+")[0] != want else []  # PEP 440: ==X takes X+local
print(", ".join(bad[:6]) + (f" and {len(bad) - 6} more" if len(bad) > 6 else ""))
sys.exit(bool(bad))
PY
}
for v in "$AREAL_PY:$ROOT/areal-req.txt" "$SGLANG_PY:$ROOT/sglang-req.txt" "$@"; do
  out=$(vers "${v%%:*}" "${v#*:}" 2>&1) && ok "versions ${v#*:}" || no "versions ${v#*:}: $out"; done
# patches: AReaL's (the trainer venv's source and the overlay) come off a copy in reverse order; re-running the SGLang
# patch scripts on a copy of the overlay changes nothing; the overlay's octopus_*.py are shims, no copies of CODE
GDC=$("$SGLANG_PY" -c 'import os, triton; print(os.path.dirname(triton.__file__))')/language/extra/cuda/gdc.py
[ -f "$GDC" ] && ! grep -q _builder "$GDC" && ok "Triton gdc.py fix" || no "Triton gdc.py _builder fix (install_env.sh) missing in $GDC"
if [ -d "$CURSOR/areal/areal" ] && [ -d "$CURSOR/sglang/sglang" ]; then
  for d in "$ROOT/areal" "$CURSOR/areal"; do  # trainer venv source, overlay
    rm -rf "$O/areal-copy"; mkdir -p "$O/areal-copy"; cp -r "$d/areal" "$O/areal-copy/"
    for p in train-rpc-timeout cpu-batch route-spread cursor logprobs; do patch -R -s -f -p1 -d "$O/areal-copy" < "$CODE/patches/areal-$p.patch" > /dev/null || break; p=; done
    [ -z "$p" ] && ok "AReaL patches in $d" || no "areal-$p.patch (or a later one) not applied in $d"
  done; rm -rf "$O/areal-copy"
  (cd "$CODE" && PYTHONPATH=$CURSOR/areal:$CODE "$AREAL_PY" patches/check_areal_route.py 2>&1 | tail -1 | grep -qx ok) && ok check_areal_route.py || no "check_areal_route.py on $CURSOR/areal"
  rm -rf "$O/sglang-copy"; cp -r "$CURSOR/sglang/sglang" "$O/sglang-copy"
  for p in lora_targets lora_abort_release top_k cursor mount; do "$SGLANG_PY" "$CODE/patches/apply_sglang_$p.py" "$O/sglang-copy" > /dev/null || no "apply_sglang_$p.py fails on the overlay"; done
  d=$(diff -rq -x __pycache__ "$CURSOR/sglang/sglang" "$O/sglang-copy" | sed "s|$O/sglang-copy/||; s|.* and ||" | tr '\n' ' ')
  [ -z "$d" ] && ok "SGLang patches in $CURSOR/sglang" || no "SGLang patches missing in $CURSOR/sglang (re-applying changes: $d)"
  rm -rf "$O/sglang-copy"
  copies=$(grep -L 'sys.modules\[__name__\]' "$CURSOR"/sglang/sglang/srt/octopus_*.py 2>&1)
  [ -z "$copies" ] && ok "overlay shims" || no "overlay has copies, not shims: $copies"
else no "no cursor overlay $CURSOR: install_env.sh builds it (CURSOR=...)"; fi
[ "$bad" = 0 ]
