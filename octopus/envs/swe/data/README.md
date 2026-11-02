# harbor-udocker (node-94, user user)

Deployed copy of this folder: the repo checkout `code/` under `/home/user/harbor-udocker`
on node-94 (`PYTHONPATH=$N/code`, `python -m octopus.envs.swe.data.<module>`) and `scripts/*.sh`.
The agent and `octopus/envs/swe/udocker.py` are that checkout's package, `code/octopus/`.

Harbor trials with udocker (the RL episodes' environment), all on the NVMe home disk. No Docker daemon.
Set up 2026-09-26; replaces `/mnt/disk3/user-harbor-udocker` (HDD was the bottleneck).
Other hosts: keep `~/harbor-udocker` a symlink to a persistent disk, never to `/tmp` (node-134's `/tmp` copy vanished on a
reboot on 2026-10-19; it now points at `/mnt/disk4/user/harbor-udocker`).

## Layout
| path | what |
|---|---|
| `uv-tools/harbor/` | venv: harbor 0.23.0 (pinned by `vendor/harbor-freeze.txt`) + udocker 1.3.17. `~/.local/bin/harbor` points here. |
| `bin/udocker` | udocker CLI wrapper using this venv and `udocker/` |
| `udocker/` | `UDOCKER_DIR`: udocker engines (`bin`, `lib`), image `repos/` and `layers/` |
| `golden/` | one pristine, never-run container per image (`UDOCKER_GOLDEN_DIR`); `.golden/<sha1(image)[:16]>.cid` maps image -> container id |
| `containers/` | per-trial copies of a golden (`UDOCKER_CONTAINERS`); removed when the trial stops |
| `ipc/` | index of live trial IPC dirs (debugging) |
| `code/` | repo checkout (CODE); `PYTHONPATH=$N/code` |
| `code/octopus/` | package: agent, `octopus/envs/swe/udocker.py`, data tools (`python -m octopus.envs.swe.data.<module>`, including `prebuild_goldens`, `mkremaining`, `status`) |
| `datasets/` | Harbor task dirs (`remain-n8`, `retry-async`, `retry-c`, ...) |
| `jobs/` | Harbor job outputs, incl. history copied from HDD3 |
| `scripts/` | entry points below; `env.sh` holds every env var |

## Use
```bash
# build a gradeable dataset from emitted tasks (no model calls): branch setup + hidden-test patch +
# reference fix, clean-main baseline, then Harbor oracle=1 / nop=0 selection
PYTHONPATH=~/harbor-udocker/code ~/harbor-udocker/uv-tools/harbor/bin/python -m octopus.envs.swe.data.regen_tasks tasks swesmith-v2
PYTHONPATH=~/harbor-udocker/code ~/harbor-udocker/uv-tools/harbor/bin/python -m octopus.envs.swe.data.baseline_tasks swesmith-v2 16
~/harbor-udocker/scripts/validate.sh swesmith-v2 swesmith-valid
~/harbor-udocker/scripts/prebuild.sh swesmith-valid 4          # optional: build missing goldens first (~15 s each)
~/harbor-udocker/scripts/status.sh <job>
~/harbor-udocker/scripts/remaining.sh swesmith-valid swesmith-r2 <job>   # still-unscored tasks -> next dataset
```
Workers inside containers exit by themselves once Harbor is gone; leftover `containers/*` can then be deleted.

## Notes
- Each golden/trial copy is ~3.3 GB. 64 concurrent trials ~ 210 GB in `containers/`. This is the
  system disk: keep an eye on `df -h /`.
- A trial copy takes ~10 s; at most `UDOCKER_CREATE_SLOTS` copies/creates run at once.
- udocker P1 has no pid namespace and runs as us: an agent's `pkill python` once killed every host
  python process of this user (SGLang servers, the proxy). Each trial preloads `octopus/envs/swe/udocker_nokill.c`
  (built with the host's gcc, listed in the trial's `/etc/ld.so.preload`): kill/killpg/tgkill/tkill,
  sigqueue, pidfd_send_signal and `syscall()` for those reach only processes traced by the caller's own PRoot,
  i.e. its own command (or zombies whose parent is); every other process, a concurrent command of the
  same trial included, acts as another user's: EPERM for a pid, skipped by a group kill. So `pkill`,
  `killall`, `kill`, `kill -9 0`, `os.kill` work inside and miss the host and sibling fork branches.
  Not covered: static or musl binaries, raw syscall instructions, and an agent that deletes
  `/etc/ld.so.preload`; e.g. a static busybox `kill -9 0` still kills PRoot, and the command's setsid'd
  processes outlive it. Host processes stay visible (`ps`, `/proc/<pid>/environ`, cmdlines): keep
  secrets out of the environment and command lines of processes on trial hosts. An exec timeout, stop() and a
  dead Harbor kill the command's first process, and its PRoot (`--kill-on-exit`) kills the rest,
  setsid'd ones included. (The earlier `unshare --user --pid` wrapper fails on Ubuntu 24.04, which
  blocks unprivileged user namespaces.)
- The same shim takes the network from the agent's tool calls (an agent fetched the tested library's upstream source,
  i.e. the fix, with `pip download`, `git clone`, `curl raw.githubusercontent.com`): `connect()` to an address off the
  host fails with EACCES ("Permission denied") when the command's PRoot has `OCTOPUS_NO_EGRESS=1` (exec env, set by
  `HarborWorkspace.exec` for every command of the agent's run; `unset` inside does not lift it). Loopback, 0.0.0.0 and
  unix sockets work, except the proxy ports 10808, 10809, 8118, 7890, 7891, 1080 (the hosts' xray/privoxy listen on
  loopback, which udocker shares with the host); DNS resolves. Healthcheck, verifier and setup keep the network. Same
  ceiling as above, plus sendto/TCP fast open, io_uring, ctypes through a libc handle (`CDLL("libc.so.6").connect`)
  and a proxy on another port; Singularity trials (NVwulf, Alpha) are not covered.
- Statement checks for bundle tasks (Grok via `acceptance_rewrite.ask`, no containers): `statement_coverage.py` asks
  whether every part of the reference fix is described by the statement; `solvability.py` asks the reverse, whether the
  graded F2P tests follow from the statement plus the bug tree (exact literals, reprs, messages, contradictions). Its
  labels are a shortlist: review each flagged task by hand before dropping it (coupled-a: 53 of 890 flagged, 23 of
  them solvable on review).
