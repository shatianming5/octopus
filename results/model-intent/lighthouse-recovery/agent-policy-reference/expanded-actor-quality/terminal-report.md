# Expanded actor reference: invalid first launch

The first fresh4096/32768/100-profile attempt is infrastructure-invalid, not
quality evidence. The stock engine loaded the fixed actor, but its external
`nvidia-smi --query-gpu=uuid` identity query exceeded the hardcoded5seconds.
The actor factory closed the engine and raised before returning to the solver:
zero model requests, no actor output or patch, and no official grade.

The exact owned container is absent and GPU0 returned to0MiB/0%, with no compute
process. Conservatively charging the whole103.415402922s model relay leaves
1096.584597078s in the same1200sphase. This is not a new or reset budget.

The root fix reuses the in-process CUDA UUID lookup already used by Octopus:
`torch.cuda.get_device_properties(torch.cuda.current_device()).uuid`. It checks
the actual loaded device, still rejects a different frozen UUID and closes the
engine on mismatch; no silent identity fallback or larger query timeout exists.
The regression reproduces the original timeout and verifies both matching and
wrong-device paths; five affected factory/stock/CPU-check cases pass. The same
scientific profile requires one valid recovery run under its remaining budget.
