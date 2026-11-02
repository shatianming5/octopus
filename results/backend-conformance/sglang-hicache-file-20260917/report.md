# Native HiCache file-L3 component: passed

**Real GPU L1/L2 movement and same-host cross-instance file-L3 restore passed
on .138. Distributed/cross-machine/provider-fault B5 remains open.** Application
source is `dce132b7dd5f8046faa06501e31899ba1196ecca`, pinned SGLang 0.5.19
at `0bcd822377da7b5718e674eaf9c870d349424dd1`, cumulative native patch
`27e200c425ddf58484b7181d48707f2255c8dc204cc26cc954526d1c64c480c8`.

The original OpenHands-7B/C03, BF16, context/KV limit 4096, root 512,
ratio-2 host cache, direct I/O, layer-first layout, write-through and file
storage configuration remained unchanged across the two failure corrections.
The three engine instances were strictly sequential, never three concurrent
model allocations. No global driver, OpenSSL, or shared Python environment
was modified.

| Native observation | Measured value |
| --- | ---: |
| Device KV capacity | 4096 tokens |
| Host KV capacity | 8193 tokens |
| Host bytes per token | 57,344 bytes |
| Actual host pool | 469,819,392 bytes |
| Checkpoint ready in L2 before L1 demotion | 520 tokens |
| L1 ready after demotion | 0 tokens |
| Actual L2 to L1 restore | 520 tokens / 29,818,880 bytes |
| Fresh reader's file-L3 to L2 payload | 29,818,880 bytes |
| Fresh reader's L2 to L1 restore | 520 tokens / 29,818,880 bytes |
| Good L3 objects retained | 520 files / 29,818,880 bytes |

The [successful conformance record](recovery-hash/extracted/recovery-hash/results/file-hicache-conformance.json)
also preserves shared-owner demand, consumer cancellation unblocking demotion,
identity rejection, and a real isolated zero-length short-read failure. On that
failure, L1/L2 ready counts and L3 confirmed tokens remained zero; only fault
objects were removed, leaving good shared content intact. File payload bytes
are not network-byte measurements or cross-machine evidence.

The initial tokenizer namespace error cost 29 seconds; the next run reached
native identity but exposed missing OpenSSL development headers after 62 seconds.
Task-local vendored headers/library enabled the unchanged native hash extension,
which matched independent hashlib in a 4.284-second CPU build/self-check.
The subsequent native flow passed in 115 seconds: **206/1200 seconds charged,
994 unused**, preserving both failed attempts. [Terminal evidence](recovery-hash/terminal-report.md)
and [cleanup](recovery-hash/post-terminal-cleanup.txt) confirm no compute process,
GPU memory back to 1 MiB, and no fault object. Bulk transport/source bundles
remain local and are not recommitted; exact extracted receipts and logs are retained.
