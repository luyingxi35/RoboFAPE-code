"""Apply fixed task stages to recorded physical observations."""
from __future__ import annotations
import numpy as np
from .expressions import SafeExpression


def make_profile(trace, initial, config, calibration=None, stage_records=None):
    stages, distances, calibration = _make_profile(trace, initial, config, calibration, stage_records)
    if any(r["success"] for r in trace):
        # Successful references describe ordered completed stages. A release
        # must not create a late reach segment simply because contact ended.
        order = {name:i for i,name in enumerate(config["stage_order"])}
        if config["profile_adapter"] == "schema":
            highest = 0
            expr = {s["stage"]:SafeExpression(s["metric"], {o["name"] for o in config["observations"]}) for s in config["stages"]}
            for i,stage in enumerate(stages):
                highest = max(highest, order[stage])
                if trace[i]["success"]: highest = len(order)-1
                stages[i] = config["stage_order"][highest]
                distances[i] = expr[stages[i]].evaluate({**trace[i]["context"], **calibration})
    return stages, distances, calibration


def _make_profile(trace, initial, config, calibration=None, stage_records=None):
    if not trace:
        raise ValueError("cannot profile an empty trajectory")
    adapter = config["profile_adapter"]
    calibration = dict(calibration or {})
    if adapter == "schema":
        names = {o["name"] for o in config["observations"]}
        if any(o["source"] == "derived.terminal_separation" for o in config["observations"]):
            calibration.setdefault("terminal_separation", float(np.linalg.norm(np.asarray(trace[-1]["cubeA"]) - trace[-1]["cubeB"])))
        expr = [SafeExpression(s["metric"], names) for s in config["stages"]]
        gates = [SafeExpression(s["end_condition"], names, allow_condition=True) for s in config["stages"]]
        stages, distances = [], []
        for row in trace:
            context = {**row["context"], **calibration}
            flags = [g.evaluate(context) for g in gates]
            # Current conditions determine the stage; there is no cumulative maximum.
            active = min(len(flags)-1,max([0] + [i + 1 for i, flag in enumerate(flags) if flag]))
            stages.append(config["stages"][active]["stage"])
            distances.append(expr[active].evaluate(context))
        return stages, distances, calibration
    if adapter == "pullcube":
        from .sim.label_pullcube_natural import phase_metrics
        steps = [{"obj_pos": np.asarray(r["object"]), "eef_pos": np.asarray(r["tcp"])} for r in trace]
        stages, approach, goal, _ = phase_metrics(np.asarray(initial["object"]), np.asarray(initial["goal_region"]), steps)
        return stages, [a if s == "approach_pull_point" else g for s, a, g in zip(stages, approach, goal)], calibration
    if adapter == "bridge":
        from .sim.label_putspoon_natural import classify
        rows = [{"source_obj_pos": r["object"], "target_obj_pos": r["target"], "eef_pos": r["tcp"], "source_obj_grasped": r["grasped"]} for r in trace]
        stages, metrics, _ = classify(rows)
        return stages, metrics.tolist(), calibration
    if adapter == "poke":
        stages, distances = [], []
        for row in trace:
            if not row["grasp_confirmed"]:
                stage, value = "approach", np.linalg.norm(np.asarray(row["tcp"]) - row["object"])
            elif np.linalg.norm(np.asarray(row["peg_head"])[:2] - np.asarray(row["cube"])[:2]) > .0406835:
                stage, value = "align", np.linalg.norm(np.asarray(row["peg_head"])[:2] - np.asarray(row["cube"])[:2])
            else:
                stage, value = "push", np.linalg.norm(np.asarray(row["cube"])[:2] - np.asarray(row["goal_region"])[:2])
            stages.append(stage); distances.append(float(value))
        return stages, distances, calibration
    if adapter == "golf":
        from .sim.label_pullcubetool_golf_natural import classify_trace, _metric
        rows = []
        previous = np.asarray(initial["cube"])
        for row in trace:
            current = np.asarray(row["cube"])
            forward = -previous[:2] / max(np.linalg.norm(previous[:2]), 1e-12)
            rows.append({"eef_pos": row["tcp"], "tool_pos": row["tool"], "cube_pos": row["cube"],
                         "tool_grasp_confirmed": row["grasp_confirmed"], "pull_success": row["success"],
                         "ball_forward_velocity": float(np.dot(current[:2]-previous[:2], forward) / config["control_dt"])})
            previous = current
        stages = classify_trace(rows)
        metrics = {s: _metric(rows, s) for s in config["stage_order"]}
        return stages, [float(metrics[s][i]) for i, s in enumerate(stages)], calibration
    if adapter == "pyramid":
        from .sim.hybrid_stackpyramid_pipeline import endpoints, classify, metric
        if not calibration:
            calibration = endpoints([initial])
        stages = [classify(r, calibration) for r in trace]
        return [config["stage_order"][s] for s in stages], [metric(r, s, calibration) for r,s in zip(trace,stages)], calibration
    if adapter == "sphere":
        from .sim.hybrid_placesphere import PlaceSphereEndpoints, PlaceSphereState, label_state
        if not calibration:
            grasp = next((r for r in trace if r["grasp_confirmed"]), None)
            lift = next((r for r in trace if r["grasp_confirmed"] and r["object"][2] - initial["object"][2] >= .02), None)
            if grasp is None or lift is None:
                raise ValueError("successful sphere reference needs grasp and lift evidence")
            calibration = {"initial_sphere": initial["object"], "grasp_width": grasp["width"],
                           "lift_z": lift["object"][2], "final_sphere": trace[-1]["object"], "target": trace[-1]["object"]}
        endpoint = PlaceSphereEndpoints(**{k: np.asarray(v) if isinstance(v,list) else v for k,v in calibration.items()})
        labels = [label_state(PlaceSphereState(np.asarray(r["tcp"]), np.asarray(r["object"]), r["width"], r["grasp_confirmed"], r["success"]), endpoint) for r in trace]
        return [x.stage.name.lower() for x in labels], [x.metric_value for x in labels], calibration
    if adapter == "upright":
        from .sim.hybrid_uprightstack import StageEndpoints, UprightState, label_state
        if not calibration:
            records = {r["stage"]: r["end_timestep"] for r in (stage_records or [])}
            if any(s not in records for s in config["stage_order"]):
                raise ValueError("upright reference requires all planner stage endpoints")
            ends = [trace[records[s]] for s in config["stage_order"]]
            calibration = {"pregrasp_tcp": ends[0]["tcp"], "grasp_width": ends[1]["width"], "initial_brick": initial["object"],
                           "lift_height": ends[2]["object"][2], "upright_axis": ends[3]["brick_axis"],
                           "pre_place_brick": ends[4]["object"], "final_brick": ends[5]["object"]}
        endpoint = StageEndpoints(**{k: np.asarray(v) if isinstance(v,list) else v for k,v in calibration.items()})
        labels=[]
        for r in trace:
            labels.append(label_state(UprightState(tcp_position=np.asarray(r["tcp"]), gripper_width=r["width"],
                brick_position=np.asarray(r["object"]), brick_axis=np.asarray(r["brick_axis"]), is_grasping=r["grasped"],
                cube_a_static=bool(np.asarray(r["info"]["is_cubeA_static"]).all()), cube_b_static=bool(np.asarray(r["info"]["is_cubeB_static"]).all()),
                brick_static=bool(np.asarray(r["info"]["is_brick_static"]).all()), brick_on_target=bool(np.asarray(r["info"]["is_brick_on_cubeA"]).all())), endpoint))
        return [x.stage.name.lower() for x in labels], [x.metric_value for x in labels], calibration
    raise ValueError(f"unknown profile adapter {adapter}")
