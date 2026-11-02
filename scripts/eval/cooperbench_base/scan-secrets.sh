#!/bin/bash
# counts only: files with secret-like hits per pattern, and key names matched (no values)
cd /home/user/harbor-udocker
P="sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|token=[A-Za-z0-9._-]{12,}|[A-Z_]*_KEY=[^ ]{8,}|xai-[A-Za-z0-9]{20,}"
for d in s1b/jobs-train6 s1b/jobs-pal-134; do
  echo "[$d] files_with_hits=$(grep -rlE "$P" $d 2>/dev/null | wc -l)"
  grep -rhoE "[A-Z_]*_KEY=|token=|sk-[A-Za-z]{3}|ghp_|xai-" $d 2>/dev/null | sort | uniq -c
  grep -rlE "$P" $d 2>/dev/null | sed 's#/.*/##' | sort | uniq -c | head -5
done
echo SCAN_DONE
