# Data format

```json
{
  "id": "failure",
  "task_id": "PushCube-v1",
  "success": false,
  "stage": ["push", "push", "push"],
  "distance": [0.2, 0.0, 0.3]
}
```

This numeric profile is the minimum offline labeling input. `stage` and
`distance` contain one item per frame. Distances retain the units defined by
the task configuration; they are not automatically rescaled to `[0,1]`.

A reference manifest has `{"schema_version": 1, "trajectories": ["ref/trajectory.json"]}`.
All resource paths are relative to the containing manifest or trajectory JSON.
Use `/` separators. Absolute paths (including Windows drive paths), `..`
components, and links escaping that directory are rejected.

Collected artifacts additionally contain:

| Field | Meaning |
| --- | --- |
| `artifact_kind` | `collected_trajectory` |
| `source` | `motion_plan` or `policy` |
| `success_once` | Whether the native task success predicate was true at any recorded step |
| `actions`, `control_modes` | Executed environment actions and matching controller names |
| `trace`, `frames` | Post-action physical observations and RGB frame paths |
| `states_file` | HDF5 `states/0` through `states/T`; reset state followed by post-action states |
| `video` | Video encoded from exactly the recorded frame sequence |
| `calibration` | Reference-derived task endpoints used during physical stage assignment |
| `config`, `config_sha256` | Task configuration and its original file hash |
| `handoff_step`, `policy_steps` | Number of prefix actions and learned-policy actions |
| `paired_collection_reference` | Reference used for initial state and motion-prefix replay |
| `policy` | Inference contract and checkpoint member checksums; no private checkpoint path |
| `sha256` | Integrity checksums of declared files |

The invariant is `actions[t] → states[t+1] ↔ trace[t] ↔ frames[t] ↔ distance[t]`.
The initial state is `states[0]` and has no executed action or progress label.
Failures require `success_once=false` and at least one policy action. Native
success is unrelated to the progress-sequence failure detector in the paper.

Labeling adds `progress` and an `alignment` record for each failure frame. Label
output copies only declared resources into its own directory, so moving it
does not require access to the input directory. Source artifacts are unchanged.
