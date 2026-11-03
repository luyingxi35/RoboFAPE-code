# Policy inference contract

The simulator and inference service may use separate virtual environments on
the same Linux host. They communicate through a Unix socket. No training runner,
Ray cluster, or remote service is started by the collector.

## Built-in Pi0.5 adapter

The adapter accepts a **standard OpenPI-exported checkpoint**, with complete model
weights and its original normalization statistics. It defines the two-camera
ManiSkill transforms locally; a private configuration registry is not required.
An arbitrary training-framework state dictionary is not automatically compatible.
Export it to the OpenPI format first, preserving the model architecture, image
slots, proprioception, and action conventions. Never rename a weights file and
assume compatibility.

Install OpenPI using its official installation instructions in a separate
Python 3.11 environment, then install this package in that environment. The
OpenPI source revision is fixed in `requirements/policy-source.txt`; its loading
API has been inspected. Installation, the resolved GPU dependency versions, and
checkpoint compatibility still require the release acceptance run.
OpenPI is an optional dependency and is never imported by offline labeling.

Copy `examples/policy-panda.json` outside the package and fill in the actual
checkpoint's `asset_id`, member files, state/action dimensions, state conditioning,
action horizon, supported tasks, controller, and executed chunk length. The
example describes Panda joint-position actions, not every policy architecture.

```bash
robofape policy-server \
  --checkpoint checkpoints/your-exported-checkpoint \
  --config examples/policy-panda.json \
  --socket robofape-policy.sock
```

Run the server and collector from the same repository root when using this
relative socket path. From different working directories, pass the same absolute
socket path to both commands. Paths containing spaces must be quoted.

The checkpoint path is deliberately a placeholder. Every `checkpoint_files`
entry must be relative to that checkpoint directory, use `/` separators, and
contain no `..` components; absolute member paths are rejected.
Missing files fail before the service
starts. File hashes are recorded without publishing the checkpoint path. The
provided adapter does not train a policy or generate fallback/random actions.

## Request and response

Input: HWC RGB `uint8` main and wrist images, one float proprioception vector,
task instruction, episode ID, and post-action frame count. Cameras are explicitly
selected by task configuration. The Pi0.5 transform uses `base_0_rgb` and
`left_wrist_0_rgb`, with the third image slot masked out.

Output: finite `[horizon, action_dim]` actions **in the environment controller's
units**. The service applies checkpoint normalization through OpenPI; the
collector does not guess gripper polarity, action scales, or EE/joint conversions.
Panda joint-position control normally uses seven joint targets and one gripper
command. The default proprioception is the full robot `qpos` vector.

The checkpoint contract must match the task's configured controller and action
dimension. An EE-delta checkpoint requires a validated controller/state adapter;
it must not be used with the provided joint-position contract. This compatibility
case is a pending collection acceptance item, not an implicit conversion.

The collector executes at most `execute_chunk` actions before requesting another
prediction and clips the final chunk to the remaining episode budget. Episode
context is passed on every request. Custom stateful policies must reset internal
history when the episode ID changes.

## Other compatible policies

Use `--factory your_installed_package.module:create_policy`. The factory receives
`(checkpoint_directory: Path, config: dict)` and returns an object whose
`infer(observation: dict)` returns the environment action chunk. This is a local
Python extension point, not an online API. Implement the same validation contract;
no changes to the collector are necessary for a checkpoint with matching I/O.

RPC messages are length-prefixed JSON with explicit array dtype/shape and base64
data. Partial reads, truncated messages, unsupported dtypes, excessive payloads,
timeouts, nonfinite actions, and request-ID mismatches are checked. The socket is
local and mode `0600`; stop the server with Ctrl-C, which removes its socket.
