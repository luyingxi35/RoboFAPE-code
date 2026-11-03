"""Bounded motion-plan and learned-policy collection with aligned artifacts."""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import gymnasium as gym
import h5py
import torch
from PIL import Image
import imageio.v2 as imageio

from .. import __version__
from ..artifacts import digest, read_json, resource, validate_trajectory, write_json
from ..labeling import label_profile, validate_profile
from ..policy import PolicyClient, valid_actions
from ..profiles import make_profile
from ..collection_state import next_handoff
from .observations import Observer, array, one, scalar


class AttemptLimit(RuntimeError):
    pass


def copy_state(state):
    return {k: copy_state(v) for k,v in state.items()} if isinstance(state, dict) else array(state).copy()


def state_to_torch(state, device):
    return {k: state_to_torch(v, device) for k,v in state.items()} if isinstance(state, dict) else torch.as_tensor(state, device=device)


def write_state(group, state):
    for key, value in state.items():
        if isinstance(value, dict): write_state(group.create_group(key), value)
        else: group.create_dataset(key, data=value)


def read_state(group):
    return {k: read_state(v) if isinstance(v, h5py.Group) else v[()] for k,v in group.items()}


def states_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(states_equal(a[k], b[k]) for k in a)
    a,b = array(a),array(b)
    return a.shape == b.shape and np.allclose(a,b,atol=1e-5,rtol=1e-5)


def make_env(config):
    import mani_skill.envs  # noqa: F401
    spec=config["runtime"]
    if spec["task_module"]:
        importlib.import_module("robofape.sim.tasks.task_"+spec["task_module"])
    kwargs=dict(num_envs=1,robot_uids=spec["robot"],obs_mode="rgb",control_mode=spec["control_mode"],
                sim_backend=spec["sim_backend"],reward_mode="none",render_mode="rgb_array",
                max_episode_steps=spec["max_episode_steps"],sensor_configs={"width":224,"height":224})
    if spec["main_camera"] == "render_camera":
        kwargs["human_render_camera_configs"]={"render_camera":{"width":512,"height":512}}
    return gym.make(config["task_id"],**kwargs)


def solver(config):
    module,name=config["runtime"]["solver"].split(":",1)
    if module == "upstream":
        from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
        if name not in MP_SOLUTIONS:
            raise RuntimeError(f"installed ManiSkill has no motion planner for {name}")
        return MP_SOLUTIONS[name]
    return getattr(importlib.import_module("robofape.sim.solvers.solve_"+module), name)


class Recorder(gym.Wrapper):
    def __init__(self,env,config,seconds=300):
        super().__init__(env)
        self.config=config
        self.seconds=seconds
        self.attempt_start=time.monotonic()
        self.trace=[];self.actions=[];self.frames=[];self.states=[];self.modes=[]

    def reset(self,**kwargs):
        result=self.env.reset(**kwargs)
        self.observer=Observer(self.unwrapped,self.config)
        self.initial=self.observer.snapshot(-1)
        self.trace=[];self.actions=[];self.frames=[];self.modes=[]
        self.states=[copy_state(self.unwrapped.get_state_dict())]
        return result

    def sync_initial(self):
        self.observer=Observer(self.unwrapped,self.config)
        self.initial=self.observer.snapshot(-1)
        self.states=[copy_state(self.unwrapped.get_state_dict())]

    def images(self):
        base=self.unwrapped
        base.scene.update_render()
        obs=base.get_obs()["sensor_data"]
        runtime=self.config["runtime"]
        main = (base.render_rgb_array(camera_name="render_camera") if runtime["main_camera"]=="render_camera"
                else obs[runtime["main_camera"]]["rgb"])
        wrist=obs[runtime["wrist_camera"]]["rgb"]
        return one(main)[...,:3].astype(np.uint8),one(wrist)[...,:3].astype(np.uint8)

    def step(self,action):
        if len(self.trace)>=self.config["runtime"]["max_episode_steps"] or time.monotonic()-self.attempt_start>self.seconds:
            raise AttemptLimit("collection attempt exceeded its step/time budget")
        result=self.env.step(action)
        self.actions.append(array(action).reshape(-1).astype(float).tolist())
        self.modes.append(self.unwrapped.control_mode)
        self.trace.append(self.observer.snapshot(len(self.trace)))
        self.states.append(copy_state(self.unwrapped.get_state_dict()))
        self.frames.append(self.images()[0])
        return result


def save(recorder,path,identifier,seed,source,calibration=None,records=None,metadata=None):
    config=recorder.config
    config={**config,"control_dt":1/float(recorder.unwrapped.control_freq)}
    stages,distances,calibration=make_profile(recorder.trace,recorder.initial,config,calibration,records)
    success=any(r["success"] for r in recorder.trace)
    payload={"schema_version":1,"artifact_kind":"collected_trajectory","id":identifier,"task_id":config["task_id"],
             "config_sha256":config["sha256"],"config":config,"seed":seed,"source":source,"success":success,"success_once":success,
             "stage":stages,"distance":distances,"actions":recorder.actions,"control_modes":recorder.modes,
             "trace":recorder.trace,"initial_observation":recorder.initial,"calibration":calibration,"stage_records":records or [],
             "states_file":"states.h5","video":"trajectory.mp4","frame_alignment":"actions[t] -> states[t+1], trace[t], frames[t], distance[t]",
             "software":{"robofape":__version__,"numpy":np.__version__},**(metadata or {})}
    validate_profile(payload)
    # Ensure accepted success references can be used by the stage-bank matcher.
    if success:
        for stage in set(stages):
            idx=np.flatnonzero(np.asarray(stages)==stage)
            if len(idx)>1 and np.any(np.diff(idx)!=1):
                raise ValueError("success stage re-entry is not a valid contiguous reference")
    path=Path(path)
    temporary=path.with_name(path.name+".partial")
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    (temporary/"frames").mkdir()
    with h5py.File(temporary/"states.h5","w") as handle:
        states=handle.create_group("states")
        for i,state in enumerate(recorder.states): write_state(states.create_group(str(i)),state)
    payload["frames"]=[]
    for i,frame in enumerate(recorder.frames):
        name=f"frames/{i:06d}.png"
        Image.fromarray(frame).save(temporary/name)
        payload["frames"].append(name)
    imageio.mimwrite(temporary/"trajectory.mp4",recorder.frames,fps=int(recorder.unwrapped.control_freq),macro_block_size=1)
    payload["sha256"]={name:digest(temporary/name) for name in ["states.h5","trajectory.mp4",*payload["frames"]]}
    if success: payload=label_profile(payload,[])
    write_json(temporary/"trajectory.json",payload)
    validate_trajectory(temporary/"trajectory.json")
    if path.exists(): raise FileExistsError(path)
    temporary.rename(path)
    return payload


def collect(config,output,count,seed_start,max_attempts,policy_socket=None,reference_manifest=None,timeout=300):
    if count<1 or max_attempts<1 or seed_start<0:
        raise ValueError("count/max-attempts must be positive; seed must be nonnegative")
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    source="policy" if policy_socket else "motion_plan"
    client=PolicyClient(policy_socket) if policy_socket else None
    references=[];health={}
    if client:
        health=client.health()
        if config["task_id"] not in health["task_ids"]:
            raise ValueError("policy checkpoint is not declared compatible with this task")
        if health["control_mode"] != config["runtime"]["control_mode"] or health["action_dim"] != config["runtime"]["action_dim"]:
            raise ValueError("checkpoint controller differs from task config; use a validated matching config")
        manifest=Path(reference_manifest)
        for relative in read_json(manifest)["trajectories"]:
            p=resource(manifest.parent,relative);validate_trajectory(p);ref=read_json(p)
            if ref["success"] and ref["task_id"]==config["task_id"] and ref["config_sha256"]==config["sha256"]:
                references.append((p,ref))
        if not references: raise ValueError("no compatible successful references")
    contract={"task_id":config["task_id"],"config_sha256":config["sha256"],"source":source,"seed_start":seed_start,
              "policy":health.get("checkpoint_sha256",{}),"references":[digest(p) for p,_ in references]}
    contract_path=output/"contract.json"
    if contract_path.exists() and read_json(contract_path)!=contract:
        raise ValueError("output directory belongs to a different collection contract")
    write_json(contract_path,contract)
    ledger_path=output/"attempts.json"
    ledger=read_json(ledger_path) if ledger_path.exists() else []
    # Recover a fully committed artifact if the process died before its ledger write.
    indexed={row["seed"]:row for row in ledger}
    for finished in sorted(output.glob(f"{config['task_id']}_{source}_*/trajectory.json")):
        saved=read_json(finished)
        desired=saved.get("success") if source=="motion_plan" else saved.get("success") is False
        if not desired or saved.get("config_sha256")!=config["sha256"] or saved.get("source")!=source:
            raise ValueError("completed artifact differs from collection contract")
        if client and saved.get("policy",{}).get("checkpoint_sha256")!=health["checkpoint_sha256"]:
            raise ValueError("completed artifact uses a different checkpoint")
        validate_trajectory(finished)
        indexed[saved["seed"]]={"seed":saved["seed"],"accepted":True,"trajectory":finished.relative_to(output).as_posix(),
                                "handoff_stage":saved.get("handoff_stage","reset")}
    ledger=sorted(indexed.values(),key=lambda row:row["seed"])
    if any(row["seed"]!=seed_start+i for i,row in enumerate(ledger)):
        raise ValueError("collection ledger has a gap or an incompatible seed range")
    write_json(ledger_path,ledger)
    accepted=[x["trajectory"] for x in ledger if x.get("accepted")]
    if len(accepted)>count: raise ValueError("count is smaller than the existing accepted collection")
    for p in accepted: validate_trajectory(resource(output,p))
    for attempt in range(len(ledger),max_attempts):
        if len(accepted)>=count: break
        seed=seed_start+attempt
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        env=Recorder(make_env(config),config,timeout)
        identifier=f"{config['task_id']}_{source}_{seed}"
        entry={"seed":seed,"accepted":False}
        try:
            records=[];calibration=None;metadata={}
            if client:
                selected=next_handoff([r for _,r in references],[r for r in ledger if r.get("accepted")],count,attempt)
                if selected is None: break
                reference_index,handoff_stage,prefix,quota=selected
                reference_path,ref=references[reference_index]
                env.reset(seed=ref["seed"])
                with h5py.File(resource(reference_path.parent,ref["states_file"]),"r") as handle:
                    initial=read_state(handle["states/0"])
                env.unwrapped.set_state_dict(state_to_torch(initial,env.unwrapped.device))
                if not states_equal(initial,env.unwrapped.get_state_dict()):
                    raise ValueError("reference initial-state restoration mismatch")
                env.sync_initial()
                # Take over at stage boundaries, including the empty motion prefix.
                if prefix>=config["runtime"]["max_episode_steps"]: raise ValueError("no policy budget after prefix")
                for action in ref["actions"][:prefix]: env.step(np.asarray(action,dtype=np.float32))
                with h5py.File(resource(reference_path.parent,ref["states_file"]),"r") as handle:
                    expected=read_state(handle[f"states/{prefix}"])
                if not states_equal(expected,env.unwrapped.get_state_dict()):
                    raise ValueError("motion-prefix replay diverged from reference")
                if any(r["success"] for r in env.trace): raise ValueError("motion prefix already completed the task")
                calibration=ref["calibration"]
                episode_id=identifier
                while len(env.trace)<config["runtime"]["max_episode_steps"]:
                    main,wrist=env.images()
                    state=one(env.unwrapped.agent.robot.get_qpos()).astype(np.float32)
                    if len(state)!=health["state_dim"]: raise ValueError("checkpoint state dimension mismatch")
                    actions=valid_actions(client.infer(main,wrist,state,episode_id,len(env.trace),config["task_description"]),health["action_dim"])
                    remaining=config["runtime"]["max_episode_steps"]-len(env.trace)
                    execute=min(len(actions),int(health["execute_chunk"]),remaining)
                    if execute<1: raise ValueError("execute_chunk must be positive")
                    for action in actions[:execute]: env.step(action)
                    if any(r["success"] for r in env.trace): break
                metadata={"paired_collection_reference":ref["id"],"handoff_step":prefix,"handoff_stage":handoff_stage,
                          "handoff_quotas":quota,"policy_steps":len(env.trace)-prefix,
                          "policy":{k:v for k,v in health.items() if k not in ("ok","request_id")}}
                wanted=bool(env.trace) and not any(r["success"] for r in env.trace)
            else:
                result=solver(config)(env,seed=seed,debug=False,vis=False)
                if isinstance(result,tuple) and len(result)==2 and isinstance(result[1],list): records=result[1]
                wanted=bool(env.trace) and any(r["success"] for r in env.trace)
            if wanted:
                target=output/identifier
                payload=save(env,target,identifier,seed,source,calibration,records,metadata)
                entry.update(accepted=True,trajectory=f"{identifier}/trajectory.json",frames=len(payload["trace"]),
                             handoff_stage=payload.get("handoff_stage","reset"))
                accepted.append(entry["trajectory"])
            else:
                entry["reason"]="policy_success" if client else "motion_plan_unsuccessful"
        except (AttemptLimit,ValueError,RuntimeError) as error:
            entry["reason"]=str(error)
        finally:
            env.close()
        ledger.append(entry);write_json(ledger_path,ledger)
        write_json(output/"manifest.json",{"schema_version":1,"task_id":config["task_id"],"trajectories":accepted})
        print(json.dumps({"attempt":attempt+1,"accepted":len(accepted),"target":count,"reason":entry.get("reason")} ),flush=True)
    report={"requested":count,"accepted":len(accepted),"attempts":len(ledger),"complete":len(accepted)>=count}
    write_json(output/"manifest.json",{"schema_version":1,"task_id":config["task_id"],"trajectories":accepted})
    write_json(output/"collection_report.json",report)
    if not report["complete"]: raise RuntimeError(f"collection quota incomplete: {len(accepted)}/{count}; see attempts.json")
    return report
