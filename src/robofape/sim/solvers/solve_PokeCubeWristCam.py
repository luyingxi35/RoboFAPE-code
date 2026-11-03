# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
from robofape.sim.solvers.new_task_common import actor_pose, move, settle, target_tcp_for_object, translated, wrist_planner
from robofape.sim.tasks.task_PokeCubeWristCam import PokeCubeWristCamEnv

def solve_pokecube_wristcam(env: PokeCubeWristCamEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    base = env.unwrapped
    planner = wrist_planner(env, debug=debug, vis=vis)
    stages = []
    try:
        grasp = base.agent.build_grasp_pose(np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.asarray(base.peg.pose.sp.p) + np.array([0.0, 0.0, 0.015]))
        for pose in (translated(grasp, [0, 0, 0.07]), grasp):
            if move(planner, pose, allow_rrt=True) == -1:
                return -1
        planner.close_gripper(t=5)
        if not bool(base.agent.is_grasping(base.peg)[0].item()):
            return -1
        if move(planner, translated(base.agent.tcp.pose.sp, [0, 0, 0.07]), allow_rrt=True) == -1:
            return -1
        peg = actor_pose(base.peg)
        tcp = base.agent.tcp.pose.sp
        align_start = int(getattr(base, 'elapsed_steps', 0))
        cube_p = actor_pose(base.cube).p
        peg_target = sapien.Pose([cube_p[0] - base.peg_half_length - base.cube_half_size - 0.015, cube_p[1], base.peg_half_width + 0.006], [1, 0, 0, 0])
        pre_contact = target_tcp_for_object(peg, tcp, peg_target)
        if move(planner, translated(pre_contact, [0, 0, 0.07]), allow_rrt=True) == -1:
            return -1
        if move(planner, pre_contact) == -1:
            return -1
        align_end = int(getattr(base, 'elapsed_steps', 0)) - 1
        goal_p = actor_pose(base.goal_region).p
        push_distance = float(goal_p[0] - cube_p[0]) + 0.02
        if move(planner, translated(pre_contact, [push_distance, 0, 0])) == -1:
            return -1
        result = settle(planner)
        end = int(getattr(base, 'elapsed_steps', 0)) - 1
        return {'final_success': bool(base.evaluate()['success'][0].item()), 'stage_records': [{'stage': 'approach', 'start_timestep': 0, 'end_timestep': max(0, align_start - 1)}, {'stage': 'align', 'start_timestep': align_start, 'end_timestep': max(align_start, align_end)}, {'stage': 'push', 'start_timestep': align_end + 1, 'end_timestep': max(align_end + 1, end)}]}
    finally:
        planner.close()
