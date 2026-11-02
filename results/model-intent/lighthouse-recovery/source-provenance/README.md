# Lighthouse Recovery Historical Source Provenance

This directory preserves the exact historical Python sources named by the
attribution and runtime freeze files. Current benchmark sources are intentionally
not changed to make old hashes match.

Run the complete verification:

```bash
python3 results/model-intent/lighthouse-recovery/source-provenance/verify.py
```

The verifier hashes all four delivered files against their immutable freeze
entries, syntax-compiles their bytes without importing them, reconstructs the
attribution pair in a temporary directory from git base
`e42f7170b015e39f64612ae2731fd59edc59d5ef`, and requires byte equality with
the delivered attribution snapshots.

To reproduce only the attribution sources:

```bash
python3 \
  results/model-intent/lighthouse-recovery/source-provenance/reconstruct_attribution.py \
  --output /tmp/octopus-attribution-source
sha256sum /tmp/octopus-attribution-source/benchmarks/lighthouse_quality*.py
```

The runtime pair was copied byte-for-byte from the retained owned overlay
`/home/user/octopus-lighthouse-runtime-recovery-v1-20260915/benchmarks`.
The delivered snapshots make verification independent of continued remote
overlay availability.
