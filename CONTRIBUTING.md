# Contributing

Keep offline labeling independent of simulator and policy dependencies. Changes
to a task metric, stage rule, camera, action convention, or numerical matching
rule require a documented rationale and a focused regression test.

Run `python -m unittest discover -s tests -v` and
`python scripts/check_release.py`. For simulator changes, also run the relevant
Linux/GPU acceptance cases and attach commands, versions, seeds, and validation
reports. Do not mark unexecuted checks as passing.

Do not commit checkpoints, access keys, private paths, machine logs, or unreviewed
data. Preserve third-party notices and review asset licenses when contributing
rendered samples. See the release checklist before tagging a public version.
