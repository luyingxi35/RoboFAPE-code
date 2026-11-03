# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from robofape.sim.scene_builder import noTableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.scene_builder.replicacad import ReplicaCADSceneBuilder
from mani_skill.utils.scene_builder.ai2thor import ArchitecTHORSceneBuilder

@register_env('StackCube-can', max_episode_steps=50)
class StackCubeEnv_can(BaseEnv):
    """
    # Replace with stacking a can on another can, using the ReplicaCAD dataset, seed(1), and random lighting
    """
    SUPPORTED_ROBOTS = ['panda_wristcam', 'panda', 'fetch']
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids='panda_wristcam', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig('base_camera', pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.4, -0.6, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig('render_camera', pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = noTableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.replicaCAD_scene = ReplicaCADSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.replicaCAD_scene.build(1)
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'005_tomato_soup_can'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name='cubeA')
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'007_tuna_fish_can'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name='cubeB')

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= torch.linalg.norm(self.cube_half_size[:2]) + 0.01
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.05
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        is_cubeA_static = self.cubeA.is_static(lin_thresh=0.05, ang_thresh=1)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * ~is_cubeA_grasped
        return {'is_cubeA_grasped': is_cubeA_grasped, 'is_cubeA_on_cubeB': is_cubeA_on_cubeB, 'is_cubeA_static': is_cubeA_static, 'success': success.bool()}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if 'state' in self.obs_mode:
            obs.update(cubeA_pose=self.cubeA.pose.raw_pose, cubeB_pose=self.cubeB.pose.raw_pose, tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p, tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p, cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack([cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]])
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)
        reward[info['is_cubeA_grasped']] = (4 + place_reward)[info['is_cubeA_grasped']]
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_cubeA_grasped = info['is_cubeA_grasped']
        ungrasp_reward = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info['is_cubeA_on_cubeB']] = (6 + (ungrasp_reward + static_reward) / 2.0)[info['is_cubeA_on_cubeB']]
        reward[info['success']] = 8
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

@register_env('StackCube-orangebowl', max_episode_steps=50)
class StackCubeEnv_orangebowl(BaseEnv):
    """
    # Replace with placing an orange on a bowl, using ArchitecTHORSceneBuilder, seed([2]), and random lighting
    """
    SUPPORTED_ROBOTS = ['panda_wristcam', 'panda', 'fetch']
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids='panda_wristcam', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig('base_camera', pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.4, -0.6, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig('render_camera', pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = noTableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.ai2thor_scene.build([2])
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'017_orange'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name='cubeA')
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'024_bowl'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name='cubeB')

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= torch.linalg.norm(self.cube_half_size[:2]) + 0.01
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.03
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        is_cubeA_static = self.cubeA.is_static(lin_thresh=0.1, ang_thresh=5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * ~is_cubeA_grasped
        return {'is_cubeA_grasped': is_cubeA_grasped, 'is_cubeA_on_cubeB': is_cubeA_on_cubeB, 'is_cubeA_static': is_cubeA_static, 'success': success.bool()}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if 'state' in self.obs_mode:
            obs.update(cubeA_pose=self.cubeA.pose.raw_pose, cubeB_pose=self.cubeB.pose.raw_pose, tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p, tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p, cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack([cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]])
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)
        reward[info['is_cubeA_grasped']] = (4 + place_reward)[info['is_cubeA_grasped']]
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_cubeA_grasped = info['is_cubeA_grasped']
        ungrasp_reward = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info['is_cubeA_on_cubeB']] = (6 + (ungrasp_reward + static_reward) / 2.0)[info['is_cubeA_on_cubeB']]
        reward[info['success']] = 8
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

@register_env('StackCube-dicebrick', max_episode_steps=50)
class StackCubeEnv_dicebrick(BaseEnv):
    """
    # Replace with placing an orange on a bowl, using ArchitecTHORSceneBuilder, seed([5]), and random lighting
    """
    SUPPORTED_ROBOTS = ['panda_wristcam', 'panda', 'fetch']
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids='panda_wristcam', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig('base_camera', pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.4, -0.6, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig('render_camera', pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = noTableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.ai2thor_scene = ArchitecTHORSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.ai2thor_scene.build([5])
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'062_dice'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeA = builder.build(name='cubeA')
        builder = actors.get_actor_builder(self.scene, id=f"ycb:{'061_foam_brick'}")
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.5])
        self.cubeB = builder.build(name='cubeB')

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= torch.linalg.norm(self.cube_half_size[:2]) + 0.01
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.05
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        is_cubeA_static = self.cubeA.is_static(lin_thresh=0.05, ang_thresh=1)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * ~is_cubeA_grasped
        return {'is_cubeA_grasped': is_cubeA_grasped, 'is_cubeA_on_cubeB': is_cubeA_on_cubeB, 'is_cubeA_static': is_cubeA_static, 'success': success.bool()}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if 'state' in self.obs_mode:
            obs.update(cubeA_pose=self.cubeA.pose.raw_pose, cubeB_pose=self.cubeB.pose.raw_pose, tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p, tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p, cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack([cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]])
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)
        reward[info['is_cubeA_grasped']] = (4 + place_reward)[info['is_cubeA_grasped']]
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
        is_cubeA_grasped = info['is_cubeA_grasped']
        ungrasp_reward = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info['is_cubeA_on_cubeB']] = (6 + (ungrasp_reward + static_reward) / 2.0)[info['is_cubeA_on_cubeB']]
        reward[info['success']] = 8
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])
