# Same-host Mooncake runtime diagnosis and component evidence

The supervisor's preserved `component-final.json` reports a same-host native
normal/provider-fault component pass. All three failed component attempts,
both model-free probes and cleanup remain charged: **773.159087425227 /
1,200 seconds**. Independent instances and shared real L3 satisfy the current
same-host component scope; a second physical compute host is not required.
No cross-machine effect, task-quality, performance or B8 claim is made.

`abi-mismatch.json`, `abi-coherent.json` and `source-evidence.txt` preserve the
actual CUDA-header, ELF, disassembly and `dladdr` evidence. On this `.138`
environment the CUDA-13 SGL kernel selected an eight-argument call while
`RTLD_DEFAULT` found preloaded CUDA-12.9's nine-argument function. The
CPU-verified CUDA-13-first/CUDA-12.9-second binding made that call coherent,
while Mooncake's versioned import remained CUDA 12.9. Do not copy this loader
order to `.68`/CUDA-12.9; inspect its own caller and function ABI.

`failed-default-stream.json` and `failed-default-stream-probe.py` preserve the
first standalone probe's error and exact script. The probe omitted the
nondefault stream required by CUDA batch memcpy and used by the real native
L2 transfer controller. `passed-explicit-stream.json` preserves the corrected
supervisor-run native D2H/H2D equality proof. No production kernel patch or
silent copy-backend fallback was applied.

The already-validated, byte-identical reusable probes are
`benchmarks.cuda_batch_abi_probe` (CPU inspection, CUDA hidden) and
`benchmarks.native_kv_copy_probe` (explicitly authorized supervisor GPU use
only). CPU ABI coherence is not GPU-copy safety; the copy probe is not a
normal/provider-fault or model-quality pass.

The fresh normal component completed 15 requests. The separate fault phase
persisted 512 tokens, stopped its exact owned provider, rejected the fresh
reader's 0/512 restore with no inference admitted, then recorded an explicit
cold rebind with zero cached tokens. `direct/page_first_direct`, runtime
binding and fresh cache identity were held fixed across those runs. Full
attempt logs remain in the supervisor artifact paths referenced by the final
report; this directory is the bounded diagnostic integration, not a rewrite
of those logs or historical file-L3 objects.
