# Reproduction

## Offline path

Install the package as shown in the README and run its analytical example. For
the exact locally tested offline dependency versions, install
`requirements/offline-tested.txt` using Python 3.12. Run checks with:

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
python -m pip wheel . --no-deps --wheel-dir dist
```

The repository can be installed from a wheel and the CLI run from any working
directory. Pass absolute CLI arguments or paths relative to your current
directory; paths *inside* manifests remain relative to the manifest.
Shell examples in this repository assume the repository root as the current
directory. Quote path arguments containing spaces. Task configurations are
loaded from the installed package, independent of the current directory.

## Linux collection environment

The collection target is Linux x86-64, Python 3.10/3.11, NVIDIA GPU, and a working
Vulkan driver. The implementation uses the ManiSkill 3 API, pinned initially to
`3.0.0b22`, with SAPIEN `3.0.1`. The paper names the ManiSkill2 benchmark family;
this runtime specification does not establish the exact original experiment
installation or change the paper. Full runtime acceptance is still pending.

```bash
python3.11 -m venv .venv-sim
source .venv-sim/bin/activate
python -m pip install '.[collection]'
python -m mani_skill.utils.download_asset --help
robofape tasks
```

Download the assets required by each task using the installed ManiSkill asset
utility. Custom scene tasks use AI2THOR or ReplicaCAD scenes and YCB objects;
Bridge tasks use `bridge_v2_real2sim`. Check the [task matrix](tasks.md).
Use the asset utility's documented destination or `MS_ASSET_DIR` to select your
local asset directory. Robot URDF and SRDF files used by MPLib must be present.
Installation never overwrites ManiSkill source files or user asset metadata.

Before a full run, collect two successful references for one task. Follow the
README to launch a compatible policy and collect two failures. Increase the
counts only after `robofape validate` passes and videos have been inspected.
The collector includes `reset` and observed stage boundaries as takeover points;
accepted failures are allocated evenly, with any remainder assigned in stage
order. The motion prefix must not have already succeeded.

`--max-attempts` bounds the total attempts across resumes. `--attempt-timeout`
bounds elapsed time at action boundaries; a hung native planner call cannot be
interrupted by that Python check and should be bounded by an external job timeout.
Failed attempts are logged. An unmet quota returns a nonzero exit code rather
than silently reporting completion.

## Resume and reproducibility records

Repeat the same command and output directory to resume. Seed start, task config,
policy weight hashes, and reference manifest contents must remain compatible.
Increase `--max-attempts` if necessary. Completed artifacts are verified before
reuse; `.partial` directories are never counted. A committed artifact is recovered
if its ledger write was interrupted.

Each artifact stores the configuration, checkpoint file hashes, software version,
seed, initial and per-step simulator states, executed actions, images, stage
distances, and calibration endpoints. Prefix replay checks state agreement before
policy takeover. Dynamics can differ across simulator/driver versions; a failed
state check is an error to investigate, not a reason to relax validation silently.

The 3,200 successful demonstrations per task described in the paper belong to
failure-generating policy training. That SFT pipeline is outside this release.
Do not mix its data with progress-estimator training or evaluation splits.
Changing a checkpoint can change the distribution and frequency of failures;
reproducing this pipeline does not imply reproducing the paper's numerical results.

## Real sample bundle

Real successful and failed samples have not yet been collected for this release.
After a validated GPU run, export a small set with:

```bash
python scripts/export_samples.py \
  --success outputs/push-success/manifest.json \
  --failure outputs/push-failure/manifest.json \
  --output outputs/real-samples
robofape label --input outputs/real-samples/failures.json \
  --references outputs/real-samples/references.json --output outputs/sample-labels
robofape validate outputs/sample-labels/manifest.json
```

Review asset/render redistribution permissions before adding the exported bundle
to a release. Preserve its checksums and newly-collected provenance. The numerical
example in the repository is not a substitute for this acceptance step.
