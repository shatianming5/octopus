# Lighthouse Recovery Historical Source Provenance

The final independent-review finding is resolved: all four historical source
files pinned by the attribution and runtime freezes are now delivered and
exactly hash-verifiable.

| Freeze | Source | Immutable SHA-256 | Recovery |
|---|---|---|---|
| attribution | `lighthouse_quality.py` | `c1d164647fd40c7bfbf18647056ceff8939934c83e55bda088bc0b62ec13ef23` | reconstructed from git base `e42f717` plus the recorded attribution-only change |
| attribution | `lighthouse_quality_solver.py` | `fe8e679f784621329c9ad20f7812352f92ac39b43b4a87ba388c3716c80b93e3` | reconstructed from git base `e42f717` plus the recorded attribution-only change |
| runtime | `lighthouse_quality.py` | `7f7bf0b6331491b840fd2e9786ea5e17d627d9e35f3ed1ec62417c8f31895f49` | copied byte-for-byte from the retained owned runtime overlay |
| runtime | `lighthouse_quality_solver.py` | `f2bbeb4eb6a908da4fbfd91a52c0f55554973b5f8ebf4c5ff3cfa0d765afd98a` | copied byte-for-byte from the retained owned runtime overlay |

`verify.py` hashes each delivered snapshot against its immutable freeze entry,
syntax-compiles the exact bytes, reconstructs the attribution pair in a
temporary directory, and requires byte equality with the delivered snapshots.
The verification passed for all four files.

Reproduce with:

```bash
python3 results/model-intent/lighthouse-recovery/source-provenance/verify.py
```

The manifest and detailed reconstruction instructions are in `manifest.json`
and `README.md`. No freeze, result, current policy/training/solver source,
model weight, Git index, or experiment state was modified.
