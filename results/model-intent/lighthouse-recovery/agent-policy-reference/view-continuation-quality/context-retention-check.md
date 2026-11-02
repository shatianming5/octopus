# Narrow context-retention check

**Result:** The later successful `stack-packs/index.js` evidence is present in the exact decoded inputs for actions 11, 12, and 13. The actor's statements that direct index inspection was unavailable are contradictions of retained context, not a context-projection or state-loss defect. No repair is indicated.

| Compiled input | Prompt tokens | Dropped | Deduplicated | Effective journal actions retained |
|---|---:|---:|---:|---|
| Action 11 / context 11 | 12,386 | 1 | 2 | 1, 2, 3, 4, 5, 7, 8, 9, 10 |
| Action 12 / context 12 | 12,738 | 1 | 2 | 1, 2, 3, 4, 5, 7, 8, 9, 10, 11 |
| Action 13 / context 13 | 12,902 | 2 | 2 | 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 |

Action 0 is the one dropped journal action in contexts 11-12; actions 0-1 are dropped in context 13. Action 6's earlier successful index view is not retained as a separate assistant/tool pair because `deduplicate_v1` folds it into the later identical successful action 8 observation. The retained action 8 header is explicit:

> `"action_index": 8`, `"affected_resource": "stack-packs/index.js"`, `"success": true`, `"tool_error": false`, `"returned_lines": 18`, `"total_lines": 18`, `"runtime_repetition_count": 2`

It is immediately followed by the nonempty complete 869-character, 18-line source, including the `stackPacks` array with one `require('./packs/react.js')` and ending `module.exports = stackPacks;`. That exact successful header and source are present in all three decoded prompts. Thus both successful actions 6 and 8 are represented: action 8 is the retained refreshed result, and `runtime_repetition_count=2` records the deduplicated identical execution.

The action 7 out-of-bounds error is also retained in all three journal texts, including the exact path, failed `[19,36]` range, `success=false`, `error_type="view_range_out_of_bounds"`, and 18-line extent. It is followed by—not substituted for—the later action 8 successful full-file result. Therefore the prompts do not present the failure as the latest index evidence and do not create a misleading failure-only context.

**Journal versus TaskMemory:** The journal carries the ordered action 7 failure and refreshed action 8 success, including source text. Separately, each context's TaskMemory capsule contains only the latest predicate `tool_observation_10`, the clipped `stack-packs/packs/react.js` view with `next_view_range=[22,-1]`. The capsule does not claim that the index result is missing; it simply projects the latest tool predicate while the bounded journal retains the earlier index evidence.

**Conclusion:** Existing `deduplicate_v1` and journal trimming behaved correctly: obsolete oldest entries were dropped, the duplicate successful index view was compacted into the newer action 8 result, and the refreshed successful metadata plus full source remained model-visible alongside the older failure. The action 11-13 narrative that index inspection was unavailable is actor contradiction. There is no evidenced projection/size defect and therefore no source-backed protocol repair to propose.
