# Labeling specification

The labeling path uses task-specific scalar distances, not vector-state nearest
neighbors or visual embeddings. Fixed task definitions are packaged with the
code. VLM querying and configuration generation are outside this release.

| Paper procedure | Implementation | Verification |
| --- | --- | --- |
| Successful temporal labels | `temporal_progress` | Exact four-frame and single-frame cases |
| Stage-initial scalar reference selection | `label_profile` | Different references for different stages; deterministic ties |
| Monotone subsegments | `monotone_segments` | Turning points, plateaus, complete frame coverage |
| Candidate excess distance ≤ 0.001 | `match_stage` | Exact boundary and immediately beyond it |
| Sequential distance and continuity objective | `match_stage` | Continuity cannot retain an excluded candidate |
| Regressing failure progress | `label_profile` | `[0.5, 1.0, 0.25]`; stage re-entry |

For each contiguous occurrence of failure stage `k`, choose a successful
reference stage minimizing `abs(d_failure_start - d_reference_start)`. Restrict
the reference bank to successful trajectories of the same task and configuration.
Collection pairing identifies the replay prefix; it does **not** force the labeler
to use that reference. A paired reference remains eligible in the bank.

Within the selected reference stage, split the raw scalar curve at changes in
the sign of nonzero differences. Plateaus extend the preceding direction;
turning-point frames belong to both adjacent segments. Each segment contributes
its closest sampled frame. Keep candidates with distance excess at most `0.001`
(plus `1e-15` numerical comparison tolerance).

Stage-local reference progress is `j/m`, with `j=1..m`. The normalization scales
are mean absolute adjacent reference-distance changes and mean adjacent local
progress changes, each lower-bounded by `1e-12`. For the first failure frame,
choose the minimum distance. Later frames minimize
`distance/sigma_d + abs(local_progress - previous_local_progress)/sigma_p`.
Ties resolve by reference ID, then earliest reference frame/segment. Reset the
previous-label state on stage changes, including re-entry. Missing references,
non-finite values, and malformed profiles raise errors; no fabricated fallback
labels are emitted.

The task-level label is the matched frame's temporal progress in its entire
successful trajectory: `(reference_frame + 1)/reference_length`. Stage duration
therefore remains reference-specific; there is no invented equal-stage weighting.
The reset state is stored separately, outside the `T` post-action frames.

These explicit numerical conventions define this implementation. They are not
a claim of bitwise identity with all experimental labels. `alignment` records
the selected reference, frame, segment, local progress, distance, and scales.
No cumulative maximum is applied to the training label.

Task-specific physical stage rules can contain contact persistence or a latched
event (for example, the tool-pulling stage). These are distinct from forcing the
continuous progress target to increase. Success references must have contiguous
stage intervals. Failed trajectories may revisit stages.
