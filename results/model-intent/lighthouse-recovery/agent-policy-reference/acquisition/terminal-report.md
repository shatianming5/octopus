# SWE-Gym/OpenHands-7B-Agent acquisition

**Status:** Verified.

- Revision: `97054e70501d79c27499518af5ce613be6278bcb`
- Remote asset path: `/home/user/octopus-swe-gym-agent7-reference-v1-20260916/model/snapshots/97054e70501d79c27499518af5ce613be6278bcb`
- Installed and transferred bytes: `15,231,272,470` of the `16,000,000,000` byte cap
- Wall time: `3,132.877377477009` seconds of the `5,400` second cap
- Files: exact `config.json` plus four uploaded `*_4.safetensors` shards; all sizes and pinned Git/LFS digests match
- Headers: 339 tensors, 7,615,616,512 BF16 parameters, no duplicate keys, 7 repeated-shape groups, and normal Qwen Hugging Face parameter names
- Tokenizer: verified separately at `/home/user/octopus-qwen-coder-reference-v1-20260915/model/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242`; the weight repository does not ship a tokenizer or index
- Cleanup: no partial files remain, the owned terminal PID has exited, and verified assets were retained

Direct Hugging Face transport reset before transferring bytes. Three bounded public immutable-route recovery attempts retained completed ranges; the fourth completed the final shard. Exact official digests, rather than transport provenance alone, establish the verified payload. No GPU allocation, model inference, training, tokenizer overlay, invented index, or source/Git change occurred.

Evidence is recorded in `started.json`, `manifest-verification.json`, `terminal.json`, and `cleanup.json`. Detailed remote attempt logs remain under `/home/user/octopus-swe-gym-agent7-reference-v1-20260916/acquisition`.
