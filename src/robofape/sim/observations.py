"""Read task state during collection; never reconstruct contacts by replay."""
from __future__ import annotations
import numpy as np
from .grasp_confirmation import GraspPersistence, observe_grasp, update_grasp


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def one(value):
    a = array(value)
    return a[0] if a.ndim > 1 else a


def scalar(value):
    return bool(array(value).reshape(-1)[0])


def resolve(base, source, info):
    if source.startswith("env.evaluate."):
        return info[source[len("env.evaluate."):]]
    obj = base.agent if source.startswith("agent.") else base
    for component in source.split(".")[1:]:
        if component.startswith("_"):
            raise ValueError("private attributes are not valid observation sources")
        obj = getattr(obj, component)
    return obj


def transform(value, kind):
    if kind in ("position", "position_z", "quaternion", "pose", "axis_x", "axis_y", "axis_z"):
        if kind == "position": return one(value.p).astype(float).tolist()
        if kind == "position_z": return float(one(value.p)[2])
        if kind == "quaternion": return one(value.q).astype(float).tolist()
        if kind == "pose": return one(value.raw_pose).astype(float).tolist()
        return one(value.to_transformation_matrix())[:3, {"axis_x":0,"axis_y":1,"axis_z":2}[kind]].astype(float).tolist()
    if kind == "boolean": return scalar(value)
    if kind in ("float", "scalar"): return float(array(value).reshape(-1)[0])
    if kind == "sum": return float(array(value).sum())
    if kind == "norm": return float(np.linalg.norm(array(value)))
    if kind in ("vector", "axis"): return one(value).astype(float).tolist()
    raise ValueError(f"unsupported observation transform {kind}")


class Observer:
    def __init__(self, base, config):
        self.base, self.config = base, config
        self.grasp = GraspPersistence()
        name = config["runtime"]["object"]
        self.obj = base.objs[base.source_obj_name] if name == "source" else getattr(base, name)
        self.initial_z = float(one(self.obj.pose.p)[2])

    def snapshot(self, index):
        base, config = self.base, self.config
        info = base.evaluate()
        qpos = one(base.agent.robot.get_qpos()).astype(float)
        grasped = scalar(base.agent.is_grasping(self.obj))
        row = {"tcp": one(base.agent.tcp.pose.p).astype(float).tolist(),
               "object": one(self.obj.pose.p).astype(float).tolist(), "qpos": qpos.tolist(),
               "width": float(qpos[-2:].sum()), "grasped": grasped, "success": scalar(info["success"])}
        if config["runtime"]["robot"] == "panda_wristcam":
            contact = observe_grasp(base, self.obj)
            confirmation = update_grasp(self.grasp, index, contact["left_ball_force"], contact["right_ball_force"],
                                       contact["left_ball_angle"], contact["right_ball_angle"],
                                       gripper_width=contact["gripper_width"],
                                       relative_position=contact["sphere_to_tcp_relative_position"],
                                       relative_quaternion=contact["sphere_to_tcp_relative_quaternion"])
            row["contacts"], row["grasp_confirmed"] = contact, bool(confirmation["grasp_confirmed"])
        else:
            row["grasp_confirmed"] = grasped
        row["lift_confirmed"] = row["object"][2] - self.initial_z >= 0.02
        for name in ("cubeA", "cubeB", "cubeC", "goal_region", "goal_site", "tool", "cube", "brick", "bin"):
            if hasattr(base, name):
                actor = getattr(base, name)
                if hasattr(actor, "pose"):
                    row[name] = one(actor.pose.p).astype(float).tolist()
                    row[name + "_axis"] = one(actor.pose.to_transformation_matrix())[:3, 0].astype(float).tolist()
        if hasattr(base, "peg_head_pos"):
            row["peg_head"] = one(base.peg_head_pos).astype(float).tolist()
        if config["profile_adapter"] == "pyramid":
            for name in ("cubeA", "cubeC"):
                actor = getattr(base, name)
                row["grasp_" + name[-1] + "_raw"] = scalar(base.agent.is_grasping(actor))
                row[name + "_static"] = scalar(actor.is_static(lin_thresh=1e-2, ang_thresh=.5))
            row["gripper_width"] = row["width"]
        if config["profile_adapter"] == "bridge":
            row["target"] = one(base.objs[base.target_obj_name].pose.p).astype(float).tolist()
        row["info"] = {k: array(v).tolist() for k, v in info.items()}
        context = {}
        for obs in config["observations"]:
            source = obs["source"]
            if source.startswith("derived."):
                continue
            if source == "agent.strict_is_grasping":
                if resolve(base, obs["argument"], info) is not self.obj:
                    raise ValueError("strict grasp source must match the configured grasp object")
                value = row["grasp_confirmed"]
            elif source in ("env.grasp_confirmed", "env.lift_confirmed"):
                value = row[source.split('.')[-1]]
            else:
                value = resolve(base, source, info)
                if obs.get("argument"):
                    value = value(resolve(base, obs["argument"], info))
            context[obs["name"]] = transform(value, obs["transform"])
        row["context"] = context
        return row
