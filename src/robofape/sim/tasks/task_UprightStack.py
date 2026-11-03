# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import random
from mani_skill.agents.robots import Fetch, Panda, PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from robofape.sim.scene_builder import TableSceneBuilder
CUBE_HALF_SIZE = 0.02
BRICK_LENGTH = 0.15
BRICK_WIDTH = 0.04
BRICK_HEIGHT = 0.03
BRICK_UPRIGHT_COSINE = 0.95
STATIC_LINEAR_THRESHOLD = 0.1
STATIC_ANGULAR_THRESHOLD = 1.0

@register_env('UprightStack-v1', max_episode_steps=200)
class UprightStackEnv(BaseEnv):
    """
    Task Description
    ----------------
    Upright + Stack(choose which one)

    Randomizations
    --------------
    Random Position
    """
    SUPPORTED_ROBOTS = ['panda_wristcam', 'panda', 'fetch']
    agent: Union[PandaWristCam, Panda, Fetch]

    def __init__(self, *args, robot_uids='panda_wristcam', robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig('base_camera', pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.8, 0.3], [0.0, 0.0, 0.15])
        return [CameraConfig('render_camera', pose, 512, 512, 1, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0]))

    def _build_brick(self, length, width, height, color=[0.7, 0.2, 0.2, 1], density=2000):
        builder = self.scene.create_actor_builder()
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(color)
        mat.metallic = 0.0
        mat.roughness = 0.6
        mat.specular = 0.3
        builder.add_box_collision(sapien.Pose(), [length / 2, width / 2, height / 2], density=density)
        builder.add_box_visual(sapien.Pose(), [length / 2, width / 2, height / 2], material=mat)
        return builder.build(name='brick')

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cubeA = actors.build_cube(self.scene, half_size=CUBE_HALF_SIZE, color=[1, 0, 0, 1], name='cubeA', initial_pose=sapien.Pose(p=[0, 0, 0.1]))
        self.cubeB = actors.build_cube(self.scene, half_size=0.02, color=[0, 1, 0, 1], name='cubeB', initial_pose=sapien.Pose(p=[0.1, 0, 0.1]))
        self.brick = self._build_brick(length=BRICK_LENGTH, width=BRICK_WIDTH, height=BRICK_HEIGHT, color=[random.random(), random.random(), random.random(), 1])

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _generate_separated_y_positions(self, batch_size: int, count: int, y_range: tuple, min_distance: float):
        """
        Generate random y positions with minimum separation distance

        Args:
            batch_size: Number of environments
            count: Number of positions to generate
            y_range: (min_y, max_y) tuple for position range
            min_distance: Minimum distance between positions
        """
        with torch.device(self.device):
            y_positions = torch.zeros((batch_size, count), device=self.device)
            for b in range(batch_size):
                available_range = y_range[1] - y_range[0] - (count - 1) * min_distance
                if available_range < 0:
                    raise ValueError('Range too small for minimum distance requirement')
                positions = torch.rand(count, device=self.device) * available_range + y_range[0]
                positions = torch.sort(positions)[0]
                for i in range(1, count):
                    positions[i:] += min_distance
                y_positions[b] = positions
            return y_positions

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            object_y = self._generate_separated_y_positions(b, 3, (-0.3, 0.3), 0.1)
            object_indices = torch.randperm(3, device=self.device)
            objects = [self.cubeA, self.brick, self.cubeB]
            object_rotations = [torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device), torch.tensor(euler2quat(0, 0, 0), device=self.device)]
            for (i, object_idx) in enumerate(object_indices):
                object_xyz = torch.zeros((b, 3), device=self.device)
                object_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.2 - 0.2
                object_xyz[..., 1] = object_y[:, object_idx]
                object_xyz[..., 2] = 0.015
                object_q = object_rotations[i].expand(b, 4)
                tool_pose = Pose.create_from_pq(p=object_xyz, q=object_q)
                objects[i].set_pose(tool_pose)

    def evaluate(self):
        is_cubeA_static = self.cubeA.is_static(lin_thresh=STATIC_LINEAR_THRESHOLD, ang_thresh=STATIC_ANGULAR_THRESHOLD)
        is_cubeB_static = self.cubeB.is_static(lin_thresh=STATIC_LINEAR_THRESHOLD, ang_thresh=STATIC_ANGULAR_THRESHOLD)
        brick_to_cube = self.brick.pose.p - self.cubeA.pose.p
        brick_xy_aligned = torch.linalg.norm(brick_to_cube[..., :2], axis=1) <= CUBE_HALF_SIZE
        brick_z_aligned = torch.abs(brick_to_cube[..., 2] - (CUBE_HALF_SIZE + BRICK_LENGTH / 2)) <= 0.02
        brick_axis = self.brick.pose.to_transformation_matrix()[..., :3, 0]
        is_brick_upright = torch.abs(brick_axis[..., 2]) >= BRICK_UPRIGHT_COSINE
        is_brick_on_cubeA = brick_xy_aligned & brick_z_aligned & is_brick_upright
        is_brick_static = self.brick.is_static(lin_thresh=STATIC_LINEAR_THRESHOLD, ang_thresh=STATIC_ANGULAR_THRESHOLD)
        is_brick_grasped = self.agent.is_grasping(self.brick)
        success = is_cubeA_static & is_cubeB_static & is_brick_on_cubeA & is_brick_static & ~is_brick_grasped
        return {'is_cubeA_static': is_cubeA_static, 'is_cubeB_static': is_cubeB_static, 'is_brick_on_cubeA': is_brick_on_cubeA, 'is_brick_static': is_brick_static, 'is_brick_upright': is_brick_upright, 'is_brick_grasped': is_brick_grasped, 'success': success.bool()}

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        state = super().get_state_dict()
        return state

    def set_state_dict(self, state):
        if 'goal_pos' in state:
            self.goal_pos = state['goal_pos']
        super().set_state_dict(state)
