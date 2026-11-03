"""Deterministic scalar, stage-aware reference alignment.

This module has no simulator, policy, image-model, or network dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

EPSILON = 0.001
SCALE_FLOOR = 1e-12


def vector(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def temporal_progress(length):
    """Post-action frames are indexed 1..T; the reset state is not a frame."""
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer")
    return np.arange(1, length + 1, dtype=np.float64) / length


def monotone_segments(values):
    """Inclusive turning points; plateaus extend the preceding direction.

    Every returned segment is weakly monotone in the original scalar values.
    No smoothing or amplitude threshold changes those values.
    """
    values = vector(values, "reference distances")
    cuts, direction = [0], 0
    for index, delta in enumerate(np.diff(values)):
        sign = int(np.sign(delta))
        if sign and direction and sign != direction:
            cuts.append(index)
        if sign:
            direction = sign
    cuts.append(len(values) - 1)
    return [np.arange(lo, hi + 1) for lo, hi in zip(cuts[:-1], cuts[1:])]


@dataclass(frozen=True)
class Match:
    reference_index: int
    segment_index: int
    distance: float
    local_progress: float


def match_stage(query, reference):
    query, reference = vector(query, "query"), vector(reference, "reference")
    progress = temporal_progress(len(reference))
    segments = monotone_segments(reference)
    sigma_d = max(float(np.mean(np.abs(np.diff(reference)))) if len(reference) > 1 else 0, SCALE_FLOOR)
    sigma_p = max(float(np.mean(np.diff(progress))) if len(progress) > 1 else 0, SCALE_FLOOR)
    previous = None
    matches = []
    for value in query:
        candidates = []
        for sid, indices in enumerate(segments):
            idx = int(indices[np.argmin(np.abs(reference[indices] - value))])
            candidates.append(Match(idx, sid, abs(float(reference[idx] - value)), float(progress[idx])))
        best_distance = min(c.distance for c in candidates)
        valid = [c for c in candidates if c.distance - best_distance <= EPSILON + 1e-15]
        def cost(candidate):
            score = candidate.distance if previous is None else (
                candidate.distance / sigma_d + abs(candidate.local_progress - previous) / sigma_p
            )
            return score, candidate.reference_index, candidate.segment_index
        chosen = min(valid, key=cost)
        matches.append(chosen)
        previous = chosen.local_progress
    return matches, {"sigma_d": sigma_d, "sigma_p": sigma_p, "epsilon": EPSILON}


def validate_profile(profile):
    if not isinstance(profile.get("id"), str) or not profile["id"]:
        raise ValueError("trajectory id is required")
    if not isinstance(profile.get("task_id"), str) or not profile["task_id"]:
        raise ValueError("task_id is required")
    values = vector(profile.get("distance", []), "distance")
    stages = profile.get("stage", [])
    if len(stages) != len(values) or any(not isinstance(s, str) or not s for s in stages):
        raise ValueError("stage and distance must have equal lengths and named stages")
    if type(profile.get("success")) is not bool:
        raise ValueError("success must be boolean")
    if "success_once" in profile and profile["success_once"] != profile["success"]:
        raise ValueError("success must agree with success_once")
    return values, np.asarray(stages, dtype=object)


def label_profile(profile, success_bank):
    values, stages = validate_profile(profile)
    if profile["success"]:
        return {**profile, "label_method": "temporal", "progress": temporal_progress(len(values)).tolist()}
    references = []
    ids = set()
    for ref in success_bank:
        rv, rs = validate_profile(ref)
        if not ref["success"] or ref["task_id"] != profile["task_id"]:
            raise ValueError("reference bank must contain successes of the same task")
        if ref["id"] in ids:
            raise ValueError("reference ids must be unique")
        ids.add(ref["id"])
        if profile.get("config_sha256") != ref.get("config_sha256"):
            raise ValueError("query/reference task configurations differ")
        # Successful reference stages must each occupy a contiguous interval.
        for stage in set(rs):
            idx = np.flatnonzero(rs == stage)
            if len(idx) > 1 and np.any(np.diff(idx) != 1):
                raise ValueError(f"reference {ref['id']} revisits stage {stage}")
        references.append((ref, rv, rs))
    progress, audit = np.zeros(len(values)), []
    starts = [0] + (np.flatnonzero(stages[1:] != stages[:-1]) + 1).tolist() + [len(values)]
    for start, end in zip(starts[:-1], starts[1:]):
        stage = stages[start]
        candidates = []
        for ref, rv, rs in references:
            indices = np.flatnonzero(rs == stage)
            if len(indices):
                candidates.append((abs(float(values[start] - rv[indices[0]])), ref["id"], ref, rv, indices))
        if not candidates:
            raise ValueError(f"no successful reference for stage {stage!r}")
        _, _, ref, rv, indices = min(candidates, key=lambda x: (x[0], x[1]))
        matches, scales = match_stage(values[start:end], rv[indices])
        reference_progress = temporal_progress(len(rv))
        for offset, match in enumerate(matches):
            source_index = int(indices[match.reference_index])
            progress[start + offset] = reference_progress[source_index]
            audit.append({"frame": start + offset, "stage": str(stage), "reference_id": ref["id"],
                          "reference_frame": source_index, "segment": match.segment_index,
                          "distance": match.distance, "local_progress": match.local_progress, **scales})
    return {**profile, "label_method": "stage_scalar_segment_aware_v1", "progress": progress.tolist(), "alignment": audit}
