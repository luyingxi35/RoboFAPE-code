# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
BRIDGE_CONTROL_MODE = 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
BRIDGE_SPEED_SCALE = 0.5
BRIDGE_WARMUP_STEPS = 20

class BridgeWaypointSolver:

    def __init__(self, env, *, vis=False):
        self.env = env
        self.base = env.unwrapped
        if self.base.num_envs != 1:
            raise ValueError('Bridge waypoint solver requires num_envs=1')
        if self.base.control_mode != BRIDGE_CONTROL_MODE:
            raise ValueError(f'Bridge task requires {BRIDGE_CONTROL_MODE}')
        self.arm = self.base.agent.controller.controllers['arm']
        self.vis = vis
        self.gripper = 1.0
        self.result = None

    def step(self, position_delta=(0, 0, 0)):
        action = np.hstack([np.asarray(position_delta, dtype=np.float32), np.zeros(3), self.gripper]).astype(np.float32)
        self.result = self.env.step(action)
        if self.vis:
            self.base.render_human()
        return self.result

    @property
    def ee_world(self):
        return self.arm.ee_pose.sp

    def move_world(self, target_world, *, max_steps=24, tolerance=0.012):
        """Plan a straight Cartesian waypoint, then track it in the root frame."""
        root = self.arm.root_link.pose.sp
        target_at_root = (root.inv() * sapien.Pose(target_world)).p
        for _ in range(max_steps):
            actual = self.arm.ee_pose_at_base.sp.p
            if np.linalg.norm(actual - target_at_root) < tolerance:
                return True
            commanded = self.arm._target_pose.sp.p
            delta = np.asarray(target_at_root - commanded)
            length = np.linalg.norm(delta)
            max_delta = 0.025 * BRIDGE_SPEED_SCALE
            if length > max_delta:
                delta *= max_delta / length
            self.step(delta)
        return False

    def grip(self, closed, *, steps=5):
        self.gripper = -1.0 if closed else 1.0
        for _ in range(steps):
            self.step()
        return self.result

    def settle(self, steps=6):
        for _ in range(steps):
            self.step()
        return self.result

def _solve_put_on(env, *, seed, vis, is_eggplant):
    env.reset(seed=seed)
    base = env.unwrapped
    planner = BridgeWaypointSolver(env, vis=vis)
    planner.settle(steps=BRIDGE_WARMUP_STEPS)
    if hasattr(env, 'flush_video'):
        env.flush_video(save=False)
    if hasattr(env, 'flush_trajectory'):
        env.flush_trajectory(save=False)
    source = base.objs[base.source_obj_name]
    target = base.objs[base.target_obj_name]
    source_p = np.asarray(source.pose.sp.p)
    target_p = np.asarray(target.pose.sp.p)
    source_half = np.abs(base.episode_source_obj_bbox_world.cpu().numpy()) / 2
    target_half = np.abs(base.episode_target_obj_bbox_world.cpu().numpy()) / 2
    pregrasp = source_p + np.array([0, 0, max(0.09, source_half[2] + 0.06)])
    grasp = source_p + np.array([0, 0, 0.005])
    if not planner.move_world(pregrasp):
        return -1
    if not planner.move_world(grasp):
        return -1
    planner.grip(closed=True)
    if not bool(base.agent.is_grasping(source)[0].item()):
        return -1
    lift_ee = np.asarray(planner.ee_world.p) + np.array([0, 0, 0.14])
    if not planner.move_world(lift_ee):
        return -1
    held_offset = source.pose.sp.p - planner.ee_world.p
    clear_z = max(0.07, source_half[2] + target_half[2] + 0.015) if is_eggplant else source_half[2] + target_half[2] + 0.012
    desired_source = target_p + np.array([0, 0, clear_z])
    place_ee = desired_source - held_offset
    high_place = place_ee + np.array([0, 0, 0.1])
    if not planner.move_world(high_place):
        return -1
    if not planner.move_world(place_ee):
        return -1
    planner.grip(closed=False)
    planner.settle(steps=8)
    return planner.result

def solve_put_eggplant_in_basket(env, seed=None, debug=False, vis=False):
    return _solve_put_on(env, seed=seed, vis=vis, is_eggplant=True)

def solve_put_spoon_on_cloth(env, seed=None, debug=False, vis=False):
    return _solve_put_on(env, seed=seed, vis=vis, is_eggplant=False)
