"""newhost.sh's one-pass stop: TERM the trainer's process group once the groups of the dataloader's current epoch have
been trained and checkpointed. Why this rule holds: README "New host (10-29)", Stop.
- trigger: a question has a second group (> 8 episode dirs) since this trainer started, or, after the first checkpoint
  whose dataloader has drawn the whole epoch, QUEUED groups (questions with an episode dir made since) have started: the prompts it
  had queued have (a restart near the epoch's end redraws few prompts, and those may take their spooled groups, which
  make no episode dirs);
- then the episodes open at the trigger end (their {job}.question.json is gone); cut = that moment + LAG;
- then no spool group with mtime <= cut is left in resume/, ready/, taken/ or trained/: TERM;
- or what is left of either wait has not shrunk for K steps of LATEST, at two polls in a row: TERM, naming it.
STATE keeps the pass's sampler epoch and the cut. A trainer resumed from a checkpoint whose dataloader has drawn the
whole epoch goes straight to the spool check (cut: the saved one, else its start); one resumed inside the epoch draws
the epoch's tail again, so the rule starts over.
Usage: pass_stop.py JOBS_DIR SPOOL_DIR TRAINER_PIDFILE CHECKPOINT_DIR STATE
(env: POLL s between checks, 60; K 3; LAG s for a group's spool save, 60; QUEUED 48: AReaL's pending limit,
(max_head_offpolicyness 2 + 1) x batch 16). Test: test_pass_stop.sh"""
import glob, json, os, pickle, re, signal, sys, time
from collections import Counter

jobs, spool, pidf, ck, statef = sys.argv[1:6]
POLL, K, LAG = float(os.environ.get("POLL", 60)), int(os.environ.get("K", 3)), float(os.environ.get("LAG", 60))
QUEUED = int(os.environ.get("QUEUED", 48))
tp, start = int(open(pidf).read()), os.path.getmtime(pidf)
log = lambda m: print(time.strftime("%F %T %Z"), m, flush=True)
pat = re.compile(r"^search-(\d+)-")


def mtime(p):  # -1: gone meanwhile (an episode ended, a spool group moved)
    try:
        return os.path.getmtime(p)
    except FileNotFoundError:
        return -1


def save(state):
    with open(statef + ".tmp", "w") as f:
        json.dump(state, f)
    os.replace(statef + ".tmp", statef)


def stop(why):
    log(f"{why}; TERM trainer {tp}")
    try:
        os.killpg(tp, signal.SIGTERM)
    except ProcessLookupError:
        pass
    sys.exit()


def latest():  # LATEST's global_step and whether its dataloader is past the pass (has drawn the whole epoch)
    gen = json.load(open(f"{ck}/LATEST"))["generation"]
    m = f"{ck}/checkpoint_generations/{gen}/manifest"
    loader = pickle.load(open(f"{m}/dataloader_info.pkl", "rb"))[0]
    info = json.load(open(f"{m}/step_info.json"))
    return info["global_step"], loader["_areal_distributed_sampler_epoch"], \
        loader["_num_yielded"] >= info["steps_per_epoch"]


step, epoch, drawn = latest()  # the resume's
state = json.load(open(statef)) if os.path.exists(statef) else {"epoch": epoch, "cut": None}
past = lambda e, d: e > state["epoch"] or d
if past(epoch, drawn):
    state["cut"] = state["cut"] or start  # the pass was drawn before this checkpoint: its spooled groups are left
else:
    state["cut"] = None  # resumed inside the pass: its tail is drawn again
save(state)
cut, snap, seen = state["cut"], None, None
log(f"trainer {tp} resumed at step {step}, sampler epoch {epoch}; pass epoch {state['epoch']}; "
    + ("spool check" if cut else "waiting for a second group"))
best = best_step = stuck = None  # the wait's progress: fewest left, the step it was reached at
last, said = None, 0
while os.path.exists(f"/proc/{tp}"):
    try:
        try:
            step, epoch, drawn = latest()
        except Exception as e:  # LATEST mid-rewrite or its generation pruned: the last read stands
            log(f"LATEST unreadable, step {step} kept: {e!r}")
        left = None
        if cut is None and snap is None:
            dirs = [p for p in glob.glob(f"{jobs}/search-*")
                    if pat.match(os.path.basename(p)) and os.path.isdir(p) and mtime(p) >= start]
            n = Counter(pat.match(os.path.basename(p)).group(1) for p in dirs)
            if seen is None and past(epoch, drawn):
                seen = set(n)  # questions started by then; the prompts drawn by then start before the later ones
            new = len(set(n) - seen) if seen is not None else 0
            q = max(n, key=n.get, default=None)
            if (q and n[q] > 8) or new >= QUEUED:
                snap = [p for p in glob.glob(f"{jobs}/search-*.question.json") if mtime(p) >= start]
                why = f"search-{q} has a second group" if q and n[q] > 8 else \
                    f"{new} groups started since a checkpoint's dataloader had drawn the whole epoch"
                msg = f"{why}: every prompt of the pass has started; {len(snap)} episodes open"
            else:
                msg = f"no second group yet: {len(n)} questions started since the trainer started" + \
                    (f"; {new} groups since the epoch was drawn" if seen is not None else "")
        elif cut is None:
            if left := [p for p in snap if os.path.exists(p)]:
                msg = f"{len(left)} episodes open at the trigger still run"
            else:
                cut = state["cut"] = time.time() + LAG
                save(state)
                msg = f"the episodes open at the trigger have ended (step {step})"
        elif time.time() >= cut:
            left = [p for d in ("resume", "ready", "taken", "trained") for p in glob.glob(f"{spool}/{d}/*.pt")
                    if 0 <= mtime(p) <= cut]
            if not left:
                stop("one pass done")
            msg = f"{len(left)} groups of the pass not yet trained and checkpointed"
        else:
            msg = "waiting for the cut"
        if not left:
            best = None  # a new wait
        elif best is None or len(left) < best:
            best, best_step, stuck = len(left), step, False
        elif step >= best_step + K:
            if stuck:  # also at the poll before: not a checkpoint whose commit had yet to empty trained/
                stop(f"{len(left)} left for {step - best_step} steps: " + " ".join(sorted(map(os.path.basename, left))))
            stuck = True
        if msg != last or time.time() - said > 600:
            log(msg)
            last, said = msg, time.time()
    except Exception as e:  # SystemExit (stop) is not one
        log(f"error, retried next poll: {e!r}")
    time.sleep(POLL)
log(f"trainer {tp} gone")
