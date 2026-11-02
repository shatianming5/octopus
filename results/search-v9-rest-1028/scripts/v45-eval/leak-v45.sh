# node-138: leak check after the v45 pilot (as evq/leak138.sh): 40-char slices of the pilot decrypted questions/answers in
# files written since its start, on 138 (python scan) and on 68/67/134 (grep, patterns over ssh stdin into /dev/shm)
D=/mnt/ssd4/user/search-v9; X=$D/logs/evq-v45; C=$D/runtime/code-local; PY=$D/areal-venv/.venv/bin/python
R=$D/eval/pilot-v45/test-v45-fork-k1.jsonl; E=$(cat $X/pilot.epoch)
echo "tmpfs left: $(ls -d /dev/shm/search-eval-* /dev/shm/bcp-trials-* 2>/dev/null | wc -l); bcp service: $(curl -s -m 3 127.0.0.1:32601/health || echo down)"
cd $C
find $D /tmp /dev/shm /home/user/search-v9-run/ /home/user/octo-env/ -xdev -type f -newermt @$E -size -4G -print0 2>/dev/null \
  | PYTHONPATH=$C $PY $X/leak.py $R $X/pilot.epoch files 2>&1 | tail -n +1
G=/home/user/octo-env/search-v9-rest/rgrep2.sh
for hd in "user@10.0.1.68 $G /home/user/octo-env/search-v9-rest /tmp /dev/shm /home/user/octo-env/models" \
          "user@198.51.100.67 $G /home/user/octo-env/search-v9-rest /tmp /dev/shm" \
          "user@198.51.100.134 /home/user/search-v9-134/eval-v45/rgrep2.sh /home/user/search-v9-134/ /home/user/octo-env/ /tmp /dev/shm /mnt/disk4/user/tmp"; do
  set -- $hd; h=$1; g=$2; shift 2
  PYTHONPATH=$C $PY $X/leak.py $R $X/pilot.epoch - | ssh -o BatchMode=yes $h "bash $g $E $*"
done
