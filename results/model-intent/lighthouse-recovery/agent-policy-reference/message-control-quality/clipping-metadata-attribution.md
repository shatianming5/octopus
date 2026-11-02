# Bounded-view clipping metadata attribution

Actions 3 and 9 are identical for the relevant fields. Each requested `stack-packs/packs/react.js` from `start=1` with `count=200` and returned:

```text
view_kind=file
total_lines=47
returned_lines=47
more_lines_after=false
output_truncated=true
success=true
tool_error=false
error_type=null
action_submitted=true
action_rejected=false
affected_resource=stack-packs/packs/react.js
action_hash=8f92b054c83ea39ca94d1ce2d04ba0d13c5a7c7e1cdcab7a04bf417fe1c71dd7
```

There is no `returned_end` field in either observation.

The emitted `output` is exactly 6,000 characters, contains 21 newline characters, does not end in a newline, and `splitlines()` yields 22 entries. It therefore contains source lines 1-21 completely plus the first 280 characters of line 22; it does not emit lines 23-47.

The two metadata layers have different reference points:

- `benchmarks/lighthouse_tool_worker.py:128-131` constructs the full requested line-window output and computes `total_lines=47`, `returned_lines=47`, and `more_lines_after=false` before character clipping.
- `benchmarks/lighthouse_tool_worker.py:324-328` then compares that full text with the 6,000-character limit, sets `output_truncated=true`, and slices the emitted output to 6,000 characters.

Thus `returned_lines` and `more_lines_after` describe the pre-character-clipped requested line window, not the emitted source extent. `total_lines=47` remains the truthful file extent. The observation also explicitly and truthfully marks `output_truncated=true`, so it does not claim that all 47 selected lines were emitted. The contract is truthful when these fields are read with that distinction, although `returned_lines` is not an emitted-line count.
