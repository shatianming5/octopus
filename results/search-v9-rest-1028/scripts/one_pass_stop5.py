"""search-v9-rest (v5, 10-29 4-GPU relaunch from step 41: the pass start stays the 2-GPU trainer's, spooled qids and the
prompts the 09:16 process skipped count as covered; a qid whose group a later restart skips again never covers, so
the run then ends at total_train_steps 61 instead): stop the trainer once every one of the 281 rest2 questions has had its group (user choice A: one pass,
not global_step 61). Covered = each qid has >= 8 episode dirs since this trainer started; then wait until those
episodes all have result.json and one more weight_update_vN appears (the step that trains the tail), plus 3 min
for its checkpoint, and TERM the trainer's process group (watch-j.sh then releases every host).
Cap: 90 min after coverage. Run detached on node-138."""
import os, re, signal, sys, time
D = "/mnt/ssd4/user/search-v9"
J, CK = f"{D}/jobs-search-v9-rest", "/home/user/search-v9-run/areal-exp/checkpoints/user/rq1-rl/search-v9-rest/default"
start = 1793236560  # 10-29 09:16 CST: the 2-GPU trainer start (step 42 never checkpointed; the 4-GPU relaunch resumes at step 41 too)
spooled = {l.strip() for l in open(f"{D}/logs/rest2/spool-qids-1029.txt") if l.strip()}  # groups reused from before 09:16
# the 09:16 process resumed at the epoch start and its first 36 calls took spool groups instead of their prompts:
# those prompts are consumed for this pass (review-switch 10-29)
skipped = set("5256 7302 9167 7410 1083 9066 5050 388 8840 6290 8754 7299 8520 2192 7303 7979 5586 5607 5 8641 8534 8646 3923 5526 1108 4058".split())
restart = os.path.getmtime(f"{D}/logs/train-rest.pid")  # this trainer's launch: episodes a stop cut before it never get result.json
tp = int(open(f"{D}/logs/train-rest.pid").read())
qids = {l.strip() for l in open(f"{D}/runtime/search/rest2-qids.txt") if l.strip()}
pat = re.compile(r"^search-(\d+)-")
log = lambda m: print(time.strftime("%F %T %Z"), m, flush=True)
latest = lambda: max((int(x[len("weight_update_v"):]) for x in os.listdir(CK) if x.startswith("weight_update_v")), default=0)
t1 = v1 = None
while True:
    if not os.path.exists(f"/proc/{tp}"):
        log("trainer gone; exit"); sys.exit()
    eps = {}
    for n in os.listdir(J):
        m, p = pat.match(n), f"{J}/{n}"
        if m and m.group(1) in qids and os.path.isdir(p) and os.path.getmtime(p) >= start and (
                os.path.getmtime(p) >= restart or any(os.path.exists(f"{p}/{t}/result.json") for t in os.listdir(p))):
            eps.setdefault(m.group(1), []).append(p)
    covered = len({q for q, v in eps.items() if len(v) >= 8} | ((spooled | skipped) & qids))
    first8 = [p for v in eps.values() for p in sorted(v, key=os.path.getmtime)[:8]]
    open_ = sum(not any(os.path.exists(f"{p}/{t}/result.json") for t in os.listdir(p)) for p in first8)
    if t1 is None and covered == len(qids):
        t1, v1 = time.time(), latest(); log(f"all {covered} questions started at v{v1}")
    log(f"covered {covered}/{len(qids)} open_first8 {open_} latest v{latest()}")
    if t1 is not None and ((open_ == 0 and latest() > v1) or time.time() - t1 > 5400):
        log(f"one pass done (latest v{latest()}); TERM trainer {tp} in 180 s"); time.sleep(180)
        os.killpg(tp, signal.SIGTERM); log("TERM sent"); sys.exit()
    time.sleep(120)
