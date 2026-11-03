"""Public command line interface; simulator imports are lazy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys

from .artifacts import read_json, relative_path, resource, validate_trajectory, write_json
from .config import load_config, task_ids
from .labeling import label_profile


def inputs(path):
    path=Path(path)
    data=read_json(path)
    if "trajectories" in data:
        return [resource(path.parent,p) for p in data["trajectories"]]
    return [path]


def label_command(input_path, references, output):
    paths=inputs(input_path)
    reference_paths=inputs(references)
    for path in reference_paths:
        validate_trajectory(path)
    bank=[read_json(p) for p in reference_paths]
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("label output must be a new or empty directory")
    prepared=[];ids=set()
    for path in paths:
        validate_trajectory(path)
        data=read_json(path)
        identifier=data["id"]
        directory=relative_path(identifier)
        if identifier in ids or len(directory.parts) != 1 or directory.name != identifier:
            raise ValueError("trajectory ids must be unique safe directory names")
        ids.add(identifier)
        refs=[r for r in bank if r["task_id"]==data["task_id"]]
        labeled=label_profile(data,refs)
        prepared.append((path,labeled))
    result=[]
    for path,labeled in prepared:
        target=output/labeled["id"]
        target.mkdir(parents=True)
        # Copy only declared resources; never private sidecars or an entire dataset tree.
        resources=set(labeled.get("frames",[]))|set(labeled.get("sha256",{}))
        resources|={labeled[k] for k in ("states_file","video") if labeled.get(k)}
        for relative in resources:
            src=resource(path.parent,relative)
            destination=target/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,destination)
        write_json(target/"trajectory.json",labeled)
        result.append(f"{labeled['id']}/trajectory.json")
    write_json(output/"manifest.json",{"schema_version":1,"trajectories":result})
    return {"labeled":len(result),"manifest":str(output/"manifest.json")}


def parser():
    p=argparse.ArgumentParser(description="RoboFAPE trajectory collection and progress labeling")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("tasks",help="list packaged task configurations")
    label=sub.add_parser("label",help="label a trajectory or manifest using a successful reference bank")
    label.add_argument("--input",required=True,type=Path);label.add_argument("--references",required=True,type=Path);label.add_argument("--output",required=True,type=Path)
    validate=sub.add_parser("validate",help="validate portable trajectory artifacts")
    validate.add_argument("input",type=Path)
    for name in ("collect-success","collect-failure"):
        c=sub.add_parser(name)
        c.add_argument("--task",required=True,choices=task_ids());c.add_argument("--config",type=Path)
        c.add_argument("--output",required=True,type=Path);c.add_argument("--count",type=int,default=2)
        c.add_argument("--seed-start",type=int,default=0);c.add_argument("--max-attempts",type=int,default=100)
        c.add_argument("--attempt-timeout",type=float,default=300)
        if name=="collect-failure":
            c.add_argument("--policy-socket",required=True,type=Path);c.add_argument("--references",required=True,type=Path)
    server=sub.add_parser("policy-server")
    server.add_argument("--socket",required=True,type=Path);server.add_argument("--checkpoint",required=True,type=Path)
    server.add_argument("--config",required=True,type=Path)
    server.add_argument("--factory",default="robofape.policy:openpi_factory")
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=="tasks":
            result=[{"task":t,"stages":load_config(t)["stage_order"],"runtime":"unverified"} for t in task_ids()]
        elif args.command=="label":
            result=label_command(args.input,args.references,args.output)
        elif args.command=="validate":
            result=[validate_trajectory(path) for path in inputs(args.input)]
        elif args.command=="policy-server":
            from .policy import serve
            serve(args.socket,args.checkpoint,read_json(args.config),args.factory)
            return
        else:
            from .sim.collection import collect
            result=collect(load_config(args.task,args.config),args.output,args.count,args.seed_start,args.max_attempts,
                           getattr(args,"policy_socket",None),getattr(args,"references",None),args.attempt_timeout)
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,KeyError,OSError,RuntimeError,ImportError) as error:
        print(f"robofape: {error}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    sys.exit(main())
