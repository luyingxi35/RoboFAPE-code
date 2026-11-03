# Copyright (c) 2025 MINT-SJTU
# SPDX-License-Identifier: MIT
# Modified for package-local imports and standalone collection.
import numpy as np
import sapien
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from robofape.sim.tasks.task_PegInsertionVertical import PegInsertionVerticalEnv

def solve_peginsertionvertical(env: PegInsertionVerticalEnv, seed=None, debug=False, vis=False, reset_options=None):
    env.reset(seed=seed, options=reset_options or {})
    assert env.unwrapped.control_mode in ['pd_joint_pos', 'pd_joint_pos_vel'], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose, visualize_target_grasp_pose=vis, print_env_info=False, joint_vel_limits=0.75, joint_acc_limits=0.75)
    env = env.unwrapped
    n_arm_joints = len(planner.planner.joint_vel_limits)
    stage_records = []

    def pose_to_record(pose):
        if pose is None:
            return None
        return np.r_[np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)].astype(float).tolist()

    def current_qpos():
        return planner.robot.get_qpos().cpu().numpy()[0]

    def pose_for_planning(pose):
        pose = sapien.Pose(pose.p, pose.q)
        current_q = env.agent.tcp.pose.q[0].detach().cpu().numpy()
        target_q = np.asarray(pose.q, dtype=np.float64)
        if np.dot(current_q, target_q) < 0:
            pose.q = -target_q
        return planner._transform_pose_for_planning(pose)

    def path_score(result, fallback_penalty=0.0):
        qpos_path = result['position']
        if len(qpos_path) <= 1:
            return fallback_penalty
        diffs = np.diff(qpos_path, axis=0)
        joint_length = np.linalg.norm(diffs, axis=1).sum()
        max_step = np.linalg.norm(diffs, axis=1).max()
        duration = float(result.get('duration', 0.0))
        return fallback_penalty + joint_length + 2.0 * max_step + 0.05 * duration

    def plan_with_screw(pose):
        pose = pose_for_planning(pose)
        planner._update_grasp_visual(pose)
        result = planner.planner.plan_screw(np.concatenate([pose.p, pose.q]), current_qpos(), time_step=planner.base_env.control_timestep, use_point_cloud=planner.use_point_cloud)
        if result['status'] == 'Success':
            return result
        return None

    def plan_with_best_sampling(pose, attempts=8):
        pose = pose_for_planning(pose)
        planner._update_grasp_visual(pose)
        goal_pose = np.concatenate([pose.p, pose.q])
        start_qpos = current_qpos()
        candidates = []
        rrt_ranges = [0.03, 0.05, 0.08, 0.1]
        for i in range(attempts):
            result = planner.planner.plan_qpos_to_pose(goal_pose, start_qpos.copy(), time_step=planner.base_env.control_timestep, use_point_cloud=planner.use_point_cloud, wrt_world=True, rrt_range=rrt_ranges[i % len(rrt_ranges)], planning_time=0.6, planner_name='RRTConnect')
            if result['status'] == 'Success':
                candidates.append(result)
        result = planner.planner.plan_qpos_to_pose(goal_pose, start_qpos.copy(), time_step=planner.base_env.control_timestep, use_point_cloud=planner.use_point_cloud, wrt_world=True, rrt_range=0.05, planning_time=1.2, planner_name='RRTstar')
        if result['status'] == 'Success':
            candidates.append(result)
        if not candidates:
            return None
        return min(candidates, key=lambda result: path_score(result, fallback_penalty=5.0))

    def plan_to_pose(pose, sampling_attempts=8):
        result = plan_with_screw(pose)
        if result is not None:
            return result
        return plan_with_best_sampling(pose, attempts=sampling_attempts)

    def record_stage(stage, target_waypoint, start_timestep, result, extra=None):
        if result == -1:
            return
        record = {'stage': stage, 'target_waypoint': pose_to_record(target_waypoint), 'start_timestep': int(start_timestep), 'end_timestep': int(planner.elapsed_steps - 1), 'elapsed_steps': int(planner.elapsed_steps)}
        if extra:
            record.update(extra)
        stage_records.append(record)

    def move_to_pose(pose, stage, refine_steps=0):
        start_timestep = int(planner.elapsed_steps)
        result = plan_to_pose(pose)
        if result is None:
            return -1
        res = planner.follow_path(result, refine_steps=refine_steps)
        record_stage(stage, pose, start_timestep, res)
        return res

    def move_to_pose_origin(pose, stage, refine_steps=0):
        start_timestep = int(planner.elapsed_steps)
        res = planner.move_to_pose_with_screw(pose, refine_steps=refine_steps)
        planner_name = 'screw'
        if res == -1:
            res = planner.move_to_pose_with_RRTConnect(pose, refine_steps=refine_steps)
            planner_name = 'RRTConnect'
        record_stage(stage, pose, start_timestep, res, {'planner': planner_name})
        return res

    def peg_axis_lateral_correction():
        head_error = (env.box_hole_pose.inv() * env.peg_head_pose).p[0].detach().cpu().numpy()
        body_error = (env.goal_pose.inv() * env.peg.pose).p[0].detach().cpu().numpy()
        lateral_error = 0.7 * head_error[1:3] + 0.3 * body_error[1:3]
        correction_hole = np.array([0.0, -lateral_error[0], -lateral_error[1]])
        hole_rotation = env.box_hole_pose.sp.to_transformation_matrix()[:3, :3]
        return np.clip(hole_rotation @ correction_hole, -0.018, 0.018)
    peg_init_pose = env.peg.pose
    peg_pos = peg_init_pose.p[0].cpu().numpy()
    grasp_height = env.table_top_z + env.peg_half_length * 1.7
    grasp_center = [peg_pos[0], peg_pos[1], grasp_height]
    grasp_candidates = []
    for (approaching, closing) in [([0, 0, -1], [0, 1, 0]), ([0, 0, -1], [1, 0, 0]), ([0, 0, -1], [0, -1, 0]), ([0, 0, -1], [-1, 0, 0])]:
        candidate = env.agent.build_grasp_pose(approaching=np.array(approaching), closing=np.array(closing), center=grasp_center)
        reach_candidate = candidate * sapien.Pose([0, 0, -0.06])
        plan = plan_to_pose(reach_candidate, sampling_attempts=4)
        if plan is not None:
            grasp_candidates.append((path_score(plan), candidate))
    if not grasp_candidates:
        planner.close()
        return -1
    grasp_pose = min(grasp_candidates, key=lambda item: item[0])[1]
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.06])
    res = move_to_pose(reach_pose, 'reach')
    if res == -1:
        planner.close()
        return res
    res = move_to_pose(grasp_pose, 'grasp')
    if res == -1:
        planner.close()
        return res
    start_timestep = int(planner.elapsed_steps)
    res = planner.close_gripper(t=10, gripper_state=-0.45)
    record_stage('close_gripper', None, start_timestep, res, {'gripper_state': -0.45})
    lifted_pose = sapien.Pose([0, 0, 0.12]) * grasp_pose
    res = move_to_pose_origin(lifted_pose, 'lift')
    if res == -1:
        planner.close()
        return res
    hole_pos = env.box_hole_pose.p[0].cpu().numpy()
    insert_tcp_z = env.table_top_z + env.hole_half_depth + env.peg_half_length * 1.7
    transfer_pose = sapien.Pose(p=[(peg_pos[0] + hole_pos[0]) / 2, (peg_pos[1] + hole_pos[1]) / 2, insert_tcp_z + 0.16], q=grasp_pose.q)
    res = move_to_pose_origin(transfer_pose, 'transfer')
    if res == -1:
        planner.close()
        return res
    pre_insert_pose = sapien.Pose(p=[hole_pos[0], hole_pos[1], insert_tcp_z + 0.12], q=grasp_pose.q)
    res = move_to_pose_origin(pre_insert_pose, 'pre_insert')
    if res == -1:
        planner.close()
        return res
    insert_correction = peg_axis_lateral_correction()
    if np.linalg.norm(insert_correction) > 0.0001:
        corrected_pre_insert_pose = sapien.Pose(p=(env.agent.tcp.pose.p[0].detach().cpu().numpy() + insert_correction).tolist(), q=grasp_pose.q)
        res = move_to_pose_origin(corrected_pre_insert_pose, 'correction')
        if res == -1:
            planner.close()
            return res
        insert_correction = peg_axis_lateral_correction()
    insert_pose = sapien.Pose(p=(np.array([hole_pos[0], hole_pos[1], insert_tcp_z]) + insert_correction).tolist(), q=grasp_pose.q)
    res = move_to_pose_origin(insert_pose, 'insert', refine_steps=8)
    planner.close()
    return (res, stage_records)
