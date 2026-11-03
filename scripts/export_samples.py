"""Export verified collected samples; never fabricate demonstration trajectories."""
import argparse
from pathlib import Path
import shutil
from robofape.artifacts import read_json,resource,validate_trajectory,write_json

p=argparse.ArgumentParser()
p.add_argument("--success",type=Path,required=True);p.add_argument("--failure",type=Path,required=True)
p.add_argument("--output",type=Path,required=True);p.add_argument("--count",type=int,default=2)
a=p.parse_args()
if a.count<1: raise ValueError("count must be positive")
if a.output.exists(): raise FileExistsError(a.output)
selected={}
for name,manifest,wanted in [("references",a.success,True),("failures",a.failure,False)]:
    candidates=[resource(manifest.parent,x) for x in read_json(manifest)["trajectories"]]
    if len(candidates)<a.count: raise ValueError("not enough collected samples")
    selected[name]=[]
    for source in candidates[:a.count]:
        validate_trajectory(source);data=read_json(source)
        if data.get("artifact_kind")!="collected_trajectory" or data["success"]!=wanted:
            raise ValueError("sample must be a collected trajectory with the requested outcome")
        if not wanted and (data.get("source")!="policy" or data.get("policy_steps",0)<1):
            raise ValueError("failed sample must contain learned-policy execution")
        selected[name].append((source,data))
for name,rows in selected.items():
    relative=[]
    for index,(source,data) in enumerate(rows):
        target=a.output/name/str(index);target.mkdir(parents=True)
        members=set(data.get("sha256",{}))|set(data.get("frames",[]))|{data["states_file"],data["video"]}
        for member in members:
            destination=target/member;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(resource(source.parent,member),destination)
        write_json(target/"trajectory.json",data)
        relative.append((target/"trajectory.json").relative_to(a.output).as_posix())
    write_json(a.output/(name+".json"),{"schema_version":1,"trajectories":relative})
write_json(a.output/"provenance.json",{"kind":"newly_collected_reproduction_samples","paper_original_data":False,
    "redistribution_review":"required before public release"})
print(a.output)
