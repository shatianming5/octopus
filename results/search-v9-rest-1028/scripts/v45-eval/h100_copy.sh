#!/bin/bash
# h100_copy.sh (node-138, evq-v45; run detached): once h100-1031/SYNC_DONE exists (the parent touches it when the H100's
# rows are synced and verified), copy val-sw-{linear,w1,w2,w5}-v45-fork-k1.jsonl from eval/h100-1031 to eval/, where the
# lab sweep's Val arms resume them (search_dev_eval keeps the scored rows, reruns infra and missing ones). A file the lab
# already has is not overwritten (its eval started first: logged). H100's default val-v45 rows stay in h100-1031 only.
# Provenance: notes-h100.txt here (per arm the H100 rows' count, infra, trial-path prefix and qids); later lab rows have
# lab trial paths (eval/jobs-val-sw-ARM-v45-fork-k1). Gives up when the chain and its sweep queue are both gone.
X=/mnt/ssd4/user/search-v9/logs/evq-v45; E=/mnt/ssd4/user/search-v9/eval; H=$E/h100-1031; N=$X/notes-h100.txt
log() { echo "$(date '+%F %T %Z') $*" >> $N; }
alive() { for f in $X/chain.pid $X/sweep.pid; do p=$(cat $f 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && return 0; done; return 1; }
log "waiting for $H/SYNC_DONE"
until [ -e $H/SYNC_DONE ]; do alive || { log "chain and sweep gone, no SYNC_DONE: nothing copied"; exit 1; }; sleep 60; done
for a in linear w1 w2 w5; do
  f=val-sw-$a-v45-fork-k1.jsonl
  if [ -e $E/$f ] || [ -e $E/launch-val-sw-$a-v45-fork-k1.log ]; then log "$f: the lab eval started first: not copied"; continue; fi
  [ -e $H/$f ] || { log "$f: not in h100-1031"; continue; }
  cp $H/$f $E/$f.tmp && mv -n $E/$f.tmp $E/$f && log "copied $f (sha256 $(sha256sum < $E/$f | cut -c1-16)): $(python3 - $E/$f <<'EOF'
import json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
err = [r for r in rows if "error" in r]
pre = sorted({os.path.dirname(os.path.dirname(r["trial"])) for r in rows if r.get("trial")})
print(f"{len(rows)} rows from the H100, {len(err)} infra; trial prefixes {pre}; scored qids "
      + ",".join(str(r["qid"]) for r in rows if "error" not in r))
EOF
)"
done
