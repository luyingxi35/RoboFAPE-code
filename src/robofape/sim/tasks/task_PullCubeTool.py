# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from robofape.sim.scene_builder import noTableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder

@register_env('PullCubeTool-golf', max_episode_steps=100)
class PullCubeToolEnv_golf(BaseEnv):
    SUPPORTED_ROBOTS = ['panda_wristcam', 'panda', 'fetch']
    SUPPORTED_REWARD_MODES = ('normalized_dense', 'dense', 'sparse', 'none')
    agent: Union[PandaWristCam, Panda, Fetch]
    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.25
    hook_length = 0.05
    width = 0.04
    height = 0.03
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids='panda_wristcam', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2 ** 25, max_rigid_patch_count=2 ** 18))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [CameraConfig('base_camera', pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.85, -0.5, 0.6], [0.0, 0.0, 0.35])
        return [CameraConfig('render_camera', pose=pose, width=512, height=512, fov=1, near=0.01, far=100)]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([handle_length / 2, 0, 0]), [handle_length / 2, width / 2, height / 2], density=500)
        builder.add_box_visual(sapien.Pose([handle_length / 2, 0, 0]), [handle_length / 2, width / 2, height / 2], material=mat)
        builder.add_box_collision(sapien.Pose([handle_length - hook_length / 2, width, 0]), [hook_length / 2, width, height / 2])
        builder.add_box_visual(sapien.Pose([handle_length - hook_length / 2, width, 0]), [hook_length / 2, width, height / 2], material=mat)
        return builder.build(name='l_shape_tool')

    def _load_scene(self, options: dict):
        self.scene_builder = noTableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.scene_builder.build()
        self.replicaCAD_scene = ReplicaCADSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.replicaCAD_scene.build(1)
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'058_golf_ball'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cube = builder.build(name='cube')
        self.l_shape_tool = self._build_l_shaped_tool(handle_length=self.handle_length, hook_length=self.hook_length, width=self.width, height=self.height)

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = self.height / 2
            tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)
            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)
            cube_xyz = torch.zeros((b, 3), device=self.device)
            cube_xyz[..., 0] = self.arm_reach + torch.rand(b, device=self.device) * self.handle_length - 0.3
            cube_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
            cube_xyz[..., 2] = self.cube_size / 2 + 0.015
            cube_q = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-np.pi / 6, np.pi / 6), device=self.device)
            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ['state', 'state_dict']:
            obs.update(cube_pose=self.cube.pose.raw_pose, tool_pose=self.l_shape_tool.pose.raw_pose)
        return obs

    def evaluate(self):
        cube_pos = self.cube.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p
        cube_to_base_dist = torch.linalg.norm(cube_pos[:, :2] - robot_base_pos[:, :2], dim=1)
        cube_pulled_close = cube_to_base_dist < 0.6
        workspace_center = robot_base_pos.clone()
        workspace_center[:, 0] += self.arm_reach * 0.1
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
        progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)
        return {'success': cube_pulled_close, 'success_once': cube_pulled_close, 'success_at_end': cube_pulled_close, 'cube_progress': progress.mean(), 'cube_distance': cube_to_workspace_dist.mean(), 'reward': self.compute_normalized_dense_reward(None, None, {'success': cube_pulled_close})}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))
        is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
        grasping_reward = 2.0 * is_grasping
        ideal_hook_pos = cube_pos + torch.tensor([-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device)
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05
        workspace_target = robot_base_pos + torch.tensor([0.05, 0, 0], device=self.device)
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(torch.tensor([self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device) - workspace_target, dim=1)
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping
        cube_pushed_away = cube_pos[:, 0] > self.arm_reach + 0.15
        reward[cube_pushed_away] -= 2.0
        if 'success' in info:
            reward[info['success']] += 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
