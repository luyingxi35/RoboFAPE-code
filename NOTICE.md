# Third-party notices

New RoboFAPE code is licensed under the MIT License. Files carrying
`Copyright (c) 2025 MINT-SJTU` retain the MIT notice in
[licenses/MIT-MINT-SJTU.txt](licenses/MIT-MINT-SJTU.txt).

`src/robofape/openpi_adapter.py` includes adapted RLinf inference transforms,
Copyright 2025 The RLinf Authors, licensed under Apache-2.0. Its modifications
remove training orchestration and define a standalone two-camera interface.

Custom environments, scene builders, and motion-planning solutions incorporate
interfaces and portions of the ManiSkill project. Retain the Apache-2.0 terms
in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) for those portions. Package-local
imports, task selection, and collection interfaces have been modified.

Runtime dependencies (including ManiSkill, SAPIEN, MPLib, NumPy, and OpenPI)
retain their own licenses. Installing a dependency does not relicense it under
MIT. Simulator meshes, scenes, and robot assets are downloaded separately and
remain subject to their providers' terms. No simulator asset collection or
checkpoint is bundled here.

The paper overview image illustrates the associated research. The code license
does not grant rights to third-party artwork or assets depicted in that image.
