# Teacher Ceiling v2 terminal report

- **scope_id:** `teacher-ceiling-v2-20260917`
- **status:** `blocked`
- **blocker:** The exact authorized cache `/mnt/ssd2/user/models/Qwen2.5-Coder-32B-Instruct` is absent on `.186`; an exact-name search under `/mnt` found zero matches. The freeze prohibits downloading weights, changing teacher/host, or substituting a model.
- **teacher/source/device/protocol:** Teacher `Qwen/Qwen2.5-Coder-32B-Instruct`; source `c91b921c1cfba18c5780124e8d2fbaa7759d256a`; no GPU selected or initialized; frozen `actor_reproduction_v2`, BF16, max output 4096, max model length 32768, max actions 100, max sequences 1, temperature 0, seed 17, episode/GPU/phase 1070/1170/1200 seconds, prefix caching disabled, stock in-process vLLM intended.
- **preflight:** Remote `/usr/bin/python3` is Python 3.12.3. No venv or source deployment was created because teacher config/tokenizer identity could not be resolved. Tokenizer/input preflight was therefore not runnable, with zero model requests preserved. The single launch-adjacent `nvidia-smi` conflict check was intentionally not consumed because launch was impossible.
- **episode/accounting:** 0 episodes started, 0 model requests, 0 actions, 0 GPU seconds, 0 phase seconds charged. No reproduction, patch, mechanism candidate, or official grade exists.
- **artifacts:** `terminal.json`, `preflight-evidence.txt`, unchanged `evaluator-freeze.json` (SHA-256 `e14174879b091d7248618f2763793919a5b78ff54e6b10495de4edbe8611dc9c`).
- **cleanup:** No local owned containers; remote task root absent; no engine PID, selected GPU, or live GPU allocation. Only local blocker evidence and the evaluator freeze are retained.
- **next decision:** Restore the exact authorized cached teacher directory on `.186`; then the parent can decide whether a fresh bounded teacher run is authorized.
