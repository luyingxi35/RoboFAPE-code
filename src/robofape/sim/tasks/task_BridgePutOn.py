# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import torch
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.put_on_in_scene import PutEggplantInBasketScene as _PutEggplantInBasketScene, PutSpoonOnTableClothInScene as _PutSpoonOnTableClothInScene
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry import rotation_conversions
_EGGPLANT_SEED0_Q = [-0.0212488133, 0.8413879499, -0.0332009615, 0.5389922571]
_SPOON_SEED0_Q = [0.9987334287, -0.04585043, -0.0207087252, 0.0006455602]

class _FixedSourceOrientation:
    source_orientation = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        source = self.objs[self.source_obj_name]
        q = torch.as_tensor(self.source_orientation, device=self.device, dtype=source.pose.q.dtype).expand(len(env_idx), -1)
        old_half = torch.abs(self.episode_source_obj_bbox_world) / 2
        model_bbox = self.episode_model_bbox_sizes[self.source_obj_name].float()
        new_bbox = (rotation_conversions.quaternion_to_matrix(q) @ model_bbox[..., None])[0, :, 0]
        new_half = torch.abs(new_bbox) / 2
        p = source.pose.p[env_idx].clone()
        p[..., 2] += new_half[..., 2] - old_half[..., 2]
        source.set_pose(Pose.create_from_pq(p, q))
        self.episode_source_obj_xyz_after_settle = source.pose.p
        self.episode_source_obj_bbox_world = new_bbox

@register_env('PutEggplantInBasketScene-v1', max_episode_steps=120, asset_download_ids=['bridge_v2_real2sim'], override=True)
class PutEggplantInBasketScene(_FixedSourceOrientation, _PutEggplantInBasketScene):
    source_orientation = _EGGPLANT_SEED0_Q

@register_env('PutSpoonOnTableClothInScene-v1', max_episode_steps=60, asset_download_ids=['bridge_v2_real2sim'], override=True)
class PutSpoonOnTableClothInScene(_FixedSourceOrientation, _PutSpoonOnTableClothInScene):
    source_orientation = _SPOON_SEED0_Q
