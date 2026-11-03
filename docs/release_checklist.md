# Release checklist

Status: **development snapshot; not yet accepted for reproducible public release**.
Do not convert an unexecuted check into a pass. The machine-readable record is
[validation.json](validation.json).

## Implemented checks

- [x] Independent Python package and public CLI.
- [x] Fixed task configurations, package-local tasks and motion planners.
- [x] No scripted failure injection in the collection path.
- [x] Scalar stage-start reference selection and segment-aware labeling.
- [x] Explicit numerical conventions and per-frame alignment audit records.
- [x] Relative resources, path traversal checks, resource checksums.
- [x] Local policy service, input/action validation, partial-message handling.
- [x] Bounded collection, accepted takeover quotas, resume contract and recovery.
- [x] README overview image and concise paper/repository introduction.
- [x] Mandatory license notices retained.

## Offline acceptance

- [x] Analytical labels checked against independent expected values.
- [x] Algorithm, malformed input, RPC framing, relocation, and quota unit tests.
- [x] Relative CLI paths from an external directory with spaces; copied output
  validates after removing inputs and relocating it again.
- [x] Absolute checkpoint member paths, POSIX/Windows parent traversal, and
  escaping resource symlinks rejected.
- [x] Wheel/sdist builds and installed-wheel execution from an unrelated directory.
- [x] Full static file/link/import/privacy check after the final edit.
- [ ] GitHub CI on Python 3.10, 3.11, and 3.12 (workflow supplied; not run locally).

## Required Linux/GPU acceptance

- [ ] Fresh installation of pinned simulator, inference stack, and assets.
- [ ] All 20 task environments reset, render both required policy views, and step.
- [ ] Every motion planner produces valid successful reference trajectories.
- [ ] Real checkpoint inference, normalization, images, states, and actions verified.
- [ ] EE-delta checkpoint/controller compatibility validated separately from joint-position policies.
- [ ] Every task collects failed rollouts from the requested takeover stages.
- [ ] Initial-state restoration and prefix replay checks pass on the actual runtime.
- [ ] State/action/frame/label alignment and rendered videos inspected.
- [ ] Real success/failure sample bundle, expected labels, checksums, and redistribution review.
- [ ] Complete GPU dependency lock and hardware/driver/seed acceptance report.

No checkpoint is required to be published. The acceptance runner must nevertheless
use a real compatible learned policy locally. A dummy policy or numerical fixture
cannot satisfy collection or sample acceptance.
