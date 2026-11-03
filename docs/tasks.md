# Task configurations

All 20 configurations are included. Every row still requires Linux/GPU collection acceptance.

| Task | Stages | Collection horizon | Scene/assets | Runtime |
| --- | ---: | ---: | --- | --- |
| PegInsertionSide-v1 | 3 | 350 | robot/task assets, ReplicaCAD | Unverified |
| PegInsertionVertical-v1 | 3 | 400 | robot/task assets | Unverified |
| PickCube-ball | 4 | 300 | robot/task assets, YCB, ArchitecTHOR | Unverified |
| PlaceSphere-v1 | 5 | 400 | robot/task assets | Unverified |
| PlugCharger-v1 | 3 | 350 | robot/task assets | Unverified |
| PokeCubeWristCam-v1 | 3 | 300 | robot/task assets | Unverified |
| PullCube-block | 2 | 250 | robot/task assets, YCB, ReplicaCAD | Unverified |
| PullCubeTool-golf | 3 | 400 | robot/task assets, YCB, ReplicaCAD | Unverified |
| PushCube-box | 2 | 180 | robot/task assets, YCB, ArchitecTHOR, iTHOR | Unverified |
| PushCube-cup | 2 | 180 | robot/task assets, YCB, ArchitecTHOR, iTHOR | Unverified |
| PushCube-toy | 3 | 180 | robot/task assets, YCB, ArchitecTHOR, iTHOR | Unverified |
| PushCube-v1 | 1 | 180 | robot/task assets | Unverified |
| PutEggplantInBasketScene-v1 | 4 | 300 | robot/task assets, bridge_v2_real2sim | Unverified |
| PutSpoonOnTableClothInScene-v1 | 4 | 300 | robot/task assets, bridge_v2_real2sim | Unverified |
| StackCube-can | 2 | 400 | robot/task assets, YCB, ReplicaCAD, ArchitecTHOR | Unverified |
| StackCube-dicebrick | 2 | 400 | robot/task assets, YCB, ReplicaCAD, ArchitecTHOR | Unverified |
| StackCube-orangebowl | 2 | 400 | robot/task assets, YCB, ReplicaCAD, ArchitecTHOR | Unverified |
| StackCube-v1 | 2 | 300 | robot/task assets | Unverified |
| StackPyramid-v1 | 8 | 600 | robot/task assets | Unverified |
| UprightStack-v1 | 6 | 650 | robot/task assets | Unverified |

The horizon is a collection budget, not a claim about the benchmark default.
Fixed task files are packaged under `robofape/configs`. Runtime defaults can be
replaced with `--config` only after validating checkpoint, camera, and controller
compatibility. The complete configuration hash is stored in every artifact.

Panda tasks default to wrist-camera robots and joint-position motion planning.
Bridge tasks use the native WidowX controller and third-person/wrist cameras.
EE-delta checkpoint integration is tracked separately in the release checklist.
