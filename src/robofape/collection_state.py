"""Pure collection bookkeeping, also usable without a simulator."""
from __future__ import annotations
from collections import Counter


def handoff_boundaries(reference):
    stages=reference["stage"]
    return {"reset":0, **{stages[i]:i for i in range(1,len(stages)) if stages[i]!=stages[i-1]}}


def next_handoff(references,accepted,count,attempt):
    boundaries=[handoff_boundaries(ref) for ref in references]
    names=list(dict.fromkeys(name for mapping in boundaries for name in mapping))
    targets={name:count//len(names)+(i<count%len(names)) for i,name in enumerate(names)}
    current=Counter(item["handoff_stage"] for item in accepted)
    pending=[name for name in names if current[name]<targets[name]]
    if not pending: return None
    name=pending[0]
    eligible=[i for i,mapping in enumerate(boundaries) if name in mapping]
    index=eligible[attempt%len(eligible)]
    return index,name,boundaries[index][name],targets
