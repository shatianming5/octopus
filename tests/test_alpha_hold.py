"""scripts/alpha/lib.sh hold: a queued task cut off by its job's TERM (the time limit) stays .running-<that job>, and the
next job's hold (another SLURM_JOB_ID) runs it again to .done once that job has ended (owner_running, stubbed); a task
.running under the current job id (a takeover in the same allocation) or under a job still running is left alone. Bash only, no Slurm."""
import subprocess
import time
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "scripts" / "alpha" / "lib.sh"


def hold(p: Path, job: str) -> subprocess.Popen:
    script = (f'source {LIB}; stop_stage() {{ :; }}; owner_running() {{ [ "$1" = 8 ]; }}; P={p}; SLURM_JOB_ID={job}; '
              "trap 'kill $BG; exit 143' TERM; hold")
    return subprocess.Popen(["bash", "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for(cond, secs=20):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


def test_cut_off_task_runs_again_in_the_next_job(tmp_path):
    q = tmp_path / "queue"
    q.mkdir()
    (q / "01-long.sh").write_text(f"[ -f {tmp_path}/second ] && exit 0; exec sleep 600\n")
    first = hold(tmp_path, "5")
    assert wait_for(lambda: (q / "01-long.sh.running-5").exists())
    first.terminate()
    first.wait(10)
    assert (q / "01-long.sh.running-5").exists() and not (q / "01-long.sh.done").exists()
    (tmp_path / "second").touch()
    (q / "02-mine.sh.running-7").write_text("exit 0\n")  # the current job's own (a takeover's): left alone
    (q / "03-live.sh.running-8").write_text("exit 0\n")  # another job's that still runs (job 8): left alone
    second = hold(tmp_path, "7")
    try:
        assert wait_for(lambda: (q / "01-long.sh.done").exists())
        assert (q / "02-mine.sh.running-7").exists() and (q / "03-live.sh.running-8").exists()
    finally:
        second.terminate()
        second.wait(10)
