from typing import Optional

ABSOLUTE_JOINT = "absolute_joint"
ABSOLUTE_POSE = "absolute_pose"
DELTA_POSE = "delta_pose"
DELTA_JOINT = "delta_joint"
RELATIVE_POSE_CHUNK = "relative_pose_chunk"

_ALIASES = {
    "absolute_joint": ABSOLUTE_JOINT,
    "joint": ABSOLUTE_JOINT,
    "joint_pos": ABSOLUTE_JOINT,
    "absolute_pose": ABSOLUTE_POSE,
    "pose": ABSOLUTE_POSE,
    "delta_pose": DELTA_POSE,
    "delta_ee_pose": DELTA_POSE,
    "pd_ee_delta_pose": DELTA_POSE,
    "delta_joint": DELTA_JOINT,
    "relative_pose_chunk": RELATIVE_POSE_CHUNK,
}

_LEGACY_MODE_MAP = {
    ABSOLUTE_JOINT: "joint",
    ABSOLUTE_POSE: "pose",
    DELTA_POSE: "delta_pose",
    RELATIVE_POSE_CHUNK: "relative_pose_chunk",
}


def canonicalize_control_mode(mode: Optional[str], default: str = ABSOLUTE_POSE) -> str:
    if mode is None:
        return default
    normalized = str(mode).strip().lower()
    if not normalized:
        return default
    return _ALIASES.get(normalized, normalized)


def to_legacy_control_mode(mode: Optional[str], default: Optional[str] = None) -> Optional[str]:
    canonical = canonicalize_control_mode(mode, default=default or ABSOLUTE_POSE)
    return _LEGACY_MODE_MAP.get(canonical, default)


def is_joint_mode(mode: Optional[str]) -> bool:
    return canonicalize_control_mode(mode) in {ABSOLUTE_JOINT, DELTA_JOINT}


def is_pose_mode(mode: Optional[str]) -> bool:
    return canonicalize_control_mode(mode) in {
        ABSOLUTE_POSE,
        DELTA_POSE,
        RELATIVE_POSE_CHUNK,
    }


def is_absolute_mode(mode: Optional[str]) -> bool:
    return canonicalize_control_mode(mode) in {ABSOLUTE_JOINT, ABSOLUTE_POSE}


def is_delta_mode(mode: Optional[str]) -> bool:
    return canonicalize_control_mode(mode) in {DELTA_POSE, DELTA_JOINT}
