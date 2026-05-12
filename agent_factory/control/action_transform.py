from typing import Any, Dict, List, Optional

import numpy as np

from agent_factory.control.modes import (
    ABSOLUTE_JOINT,
    ABSOLUTE_POSE,
    DELTA_POSE,
    canonicalize_control_mode,
)


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float32)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quat(lhs)
    x2, y2, z2, w2 = _normalize_quat(rhs)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = rotvec / angle
    half = angle * 0.5
    sin_half = float(np.sin(half))
    return _normalize_quat(
        np.array(
            [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)],
            dtype=np.float32,
        )
    )


def quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    xyz = quat[:3]
    w = float(np.clip(quat[3], -1.0, 1.0))
    norm_xyz = float(np.linalg.norm(xyz))
    if norm_xyz <= 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * float(np.arctan2(norm_xyz, w))
    axis = xyz / norm_xyz
    return (axis * angle).astype(np.float32)


def pose_from_position_quat(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float32).reshape(3)
    return np.concatenate([position, quat_to_rotvec(quat)], axis=0).astype(np.float32)


def pose_to_quat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(6)
    return rotvec_to_quat(pose[3:])


def apply_pose_delta(base_pose: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    base_pose = np.asarray(base_pose, dtype=np.float32).reshape(6)
    delta_pose = np.asarray(delta_pose, dtype=np.float32).reshape(6)
    next_position = base_pose[:3] + delta_pose[:3]
    delta_quat = rotvec_to_quat(delta_pose[3:])
    base_quat = pose_to_quat(base_pose)
    next_quat = quat_multiply(delta_quat, base_quat)
    return pose_from_position_quat(next_position, next_quat)


def compute_pose_delta(target_pose: np.ndarray, source_pose: np.ndarray) -> np.ndarray:
    target_pose = np.asarray(target_pose, dtype=np.float32).reshape(6)
    source_pose = np.asarray(source_pose, dtype=np.float32).reshape(6)
    delta_position = target_pose[:3] - source_pose[:3]
    target_quat = pose_to_quat(target_pose)
    source_quat = pose_to_quat(source_pose)
    delta_quat = quat_multiply(target_quat, quat_conjugate(source_quat))
    return np.concatenate([delta_position, quat_to_rotvec(delta_quat)], axis=0).astype(np.float32)


def _resolve_meta_slices(meta_node: Dict[str, Any]) -> Dict[str, slice]:
    offset = 0
    slices: Dict[str, slice] = {}
    for key in sorted((meta_node or {}).keys()):
        shape = tuple(meta_node[key])
        length = int(np.prod(shape, dtype=np.int64))
        slices[key] = slice(offset, offset + length)
        offset += length
    return slices


def resolve_state_slices_from_meta(env_meta: Optional[Dict[str, Any]]) -> Dict[str, slice]:
    state_meta = ((env_meta or {}).get("obs", {}) or {}).get("state", {}) or {}
    return _resolve_meta_slices(state_meta)


def resolve_action_slices_from_meta(env_meta: Optional[Dict[str, Any]]) -> Dict[str, slice]:
    action_meta = (env_meta or {}).get("action", {}) or {}
    return _resolve_meta_slices(action_meta)


def _default_arm_entries(env_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    env_meta = env_meta or {}
    control_meta = dict(env_meta.get("control", {}) or {})
    action_meta = dict(env_meta.get("action", {}) or {})
    state_meta = ((env_meta.get("obs", {}) or {}).get("state", {}) or {})
    action_slices = resolve_action_slices_from_meta(env_meta)

    raw_entries = control_meta.get("arm_entries")
    entries: List[Dict[str, Any]] = []
    if isinstance(raw_entries, list):
        for raw in raw_entries:
            if not isinstance(raw, dict):
                continue
            action_key = raw.get("action_key")
            if action_key not in action_slices:
                continue
            entry = dict(raw)
            entry["action_key"] = action_key
            entry["slice"] = action_slices[action_key]
            if entry.get("arm_action_dim") is None:
                entry["arm_action_dim"] = int(action_slices[action_key].stop - action_slices[action_key].start)
            entries.append(entry)
    if entries:
        return entries

    for action_key in sorted(action_meta.keys()):
        if not action_key.endswith("arm"):
            continue
        action_slice = action_slices[action_key]
        prefix = action_key[:-3]
        state_pose_key = None
        for candidate in (f"{prefix}ee_pose", f"{prefix}eef_pose"):
            if candidate in state_meta:
                state_pose_key = candidate
                break
        entry = {
            "action_key": action_key,
            "slice": action_slice,
            "arm_action_dim": int(action_slice.stop - action_slice.start),
            "gripper_action_key": f"{prefix}gripper" if f"{prefix}gripper" in action_meta else None,
        }
        if state_pose_key is not None:
            entry["state_pose_key"] = state_pose_key
        entries.append(entry)

    if entries:
        return entries

    arm_action_dim = control_meta.get("arm_action_dim")
    if arm_action_dim is None:
        return []
    return [
        {
            "action_key": "arm",
            "slice": slice(0, int(arm_action_dim)),
            "arm_action_dim": int(arm_action_dim),
            "gripper_action_key": "gripper",
            "state_pose_key": control_meta.get("eef_pose_key"),
            "eef_pos_key": control_meta.get("eef_pos_key"),
            "eef_quat_key": control_meta.get("eef_quat_key"),
        }
    ]


def _extract_pose_by_entry(state: np.ndarray, env_meta: Optional[Dict[str, Any]], entry: Dict[str, Any]) -> np.ndarray:
    state_slices = resolve_state_slices_from_meta(env_meta)
    pose_key = entry.get("state_pose_key")
    if pose_key and pose_key in state_slices:
        return state[state_slices[pose_key]].astype(np.float32)

    pos_key = entry.get("eef_pos_key")
    quat_key = entry.get("eef_quat_key")
    if pos_key and quat_key and pos_key in state_slices and quat_key in state_slices:
        position = state[state_slices[pos_key]]
        quat = state[state_slices[quat_key]]
        return pose_from_position_quat(position, quat)

    raise KeyError(
        f"Cannot recover eef pose from state for action_key={entry.get('action_key')!r}. "
        "Missing state_pose_key or eef_pos/eef_quat metadata."
    )


def extract_arm_pose_map_from_flat_state(state: np.ndarray, env_meta: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    entries = _default_arm_entries(env_meta)
    if not entries:
        raise KeyError("Cannot recover arm poses from state. Missing action/control metadata.")
    return {
        str(entry["action_key"]): _extract_pose_by_entry(state, env_meta, entry)
        for entry in entries
    }


def extract_eef_pose_from_flat_state(state: np.ndarray, env_meta: Optional[Dict[str, Any]]) -> np.ndarray:
    pose_map = extract_arm_pose_map_from_flat_state(state, env_meta)
    if len(pose_map) != 1:
        raise KeyError(
            "extract_eef_pose_from_flat_state only supports single-arm metadata. "
            "Use extract_arm_pose_map_from_flat_state for multi-arm envs."
        )
    return next(iter(pose_map.values()))


def build_action_transform_meta(
    current_pose: Optional[np.ndarray] = None,
    current_poses: Optional[Dict[str, np.ndarray]] = None,
    env_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if env_meta is not None:
        meta["env_meta"] = env_meta
    if current_pose is not None:
        meta["current_pose"] = np.asarray(current_pose, dtype=np.float32).reshape(6)
    if current_poses is not None:
        meta["current_poses"] = {
            str(key): np.asarray(value, dtype=np.float32).reshape(6)
            for key, value in current_poses.items()
        }
    return meta


def _get_arm_entries(meta: Optional[Dict[str, Any]], action: np.ndarray) -> List[Dict[str, Any]]:
    env_meta = meta.get("env_meta") if meta else None
    entries = _default_arm_entries(env_meta)
    if entries:
        return entries
    if action.shape[-1] < 6:
        raise ValueError(f"Pose transform expects action_dim >= 6, got {action.shape[-1]}.")
    return [
        {
            "action_key": "arm",
            "slice": slice(0, 6),
            "arm_action_dim": 6,
        }
    ]


def _get_current_poses(obs: Optional[Any], meta: Optional[Dict[str, Any]], entries: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if meta and meta.get("current_poses") is not None:
        current_poses = {
            str(key): np.asarray(value, dtype=np.float32).reshape(6)
            for key, value in meta["current_poses"].items()
        }
        return {
            str(entry["action_key"]): current_poses[str(entry["action_key"])]
            for entry in entries
        }

    if meta and meta.get("current_pose") is not None:
        if len(entries) != 1:
            raise ValueError("Multi-arm pose transform requires meta['current_poses'] or obs.")
        return {
            str(entries[0]["action_key"]): np.asarray(meta["current_pose"], dtype=np.float32).reshape(6)
        }

    if obs is None:
        raise ValueError("Pose transform requires obs or meta['current_pose(s)'].")
    if isinstance(obs, dict) and "state" in obs:
        state = obs["state"]
    else:
        state = obs
    env_meta = meta.get("env_meta") if meta else None
    return extract_arm_pose_map_from_flat_state(np.asarray(state, dtype=np.float32), env_meta)


def _reshape_action(action: np.ndarray):
    original_shape = tuple(action.shape)
    if action.ndim == 1:
        return action.reshape(1, 1, -1), "vector", original_shape
    if action.ndim == 2:
        return action.reshape(1, action.shape[0], action.shape[1]), "sequence", original_shape
    if action.ndim == 3:
        return action, "batched_sequence", original_shape
    raise ValueError(f"Unsupported action shape {original_shape}; expected [A], [T,A], or [B,T,A].")


def _restore_action_shape(action: np.ndarray, kind: str, original_shape):
    if kind == "vector":
        return action.reshape(original_shape[-1]).astype(np.float32)
    if kind == "sequence":
        return action.reshape(original_shape).astype(np.float32)
    return action.astype(np.float32)


def _require_supported(agent_mode: str, env_mode: str):
    supported = {
        (ABSOLUTE_JOINT, ABSOLUTE_JOINT),
        (ABSOLUTE_POSE, ABSOLUTE_POSE),
        (DELTA_POSE, ABSOLUTE_POSE),
    }
    if (agent_mode, env_mode) not in supported:
        raise NotImplementedError(
            f"Unsupported control-mode transform: agent_control_mode={agent_mode!r}, "
            f"env_control_mode={env_mode!r}."
        )


def forward_transform_action(
    obs: Optional[Any],
    agent_action: np.ndarray,
    agent_control_mode: str,
    env_control_mode: str,
    meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    agent_mode = canonicalize_control_mode(agent_control_mode)
    env_mode = canonicalize_control_mode(env_control_mode)
    _require_supported(agent_mode, env_mode)

    action = np.asarray(agent_action, dtype=np.float32)
    if agent_mode == env_mode:
        return action.copy()

    reshaped, kind, original_shape = _reshape_action(action)
    entries = _get_arm_entries(meta, action)
    current_poses = _get_current_poses(obs, meta, entries)

    transformed = reshaped.copy()
    for batch_idx in range(transformed.shape[0]):
        prev_pose_map = {
            str(key): value.copy() for key, value in current_poses.items()
        }
        for step_idx in range(transformed.shape[1]):
            current = transformed[batch_idx, step_idx]
            for entry in entries:
                action_key = str(entry["action_key"])
                action_slice = entry["slice"]
                arm_dim = int(entry["arm_action_dim"])
                prev_pose = prev_pose_map[action_key]
                next_pose = apply_pose_delta(prev_pose, current[action_slice][:arm_dim])
                current[action_slice.start: action_slice.start + arm_dim] = next_pose
                prev_pose_map[action_key] = next_pose

    return _restore_action_shape(transformed, kind, original_shape)


def inverse_transform_action(
    obs: Optional[Any],
    next_obs: Optional[Any],
    env_action: np.ndarray,
    env_control_mode: str,
    agent_control_mode: str,
    meta: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    del next_obs
    env_mode = canonicalize_control_mode(env_control_mode)
    agent_mode = canonicalize_control_mode(agent_control_mode)
    _require_supported(agent_mode, env_mode)

    action = np.asarray(env_action, dtype=np.float32)
    if env_mode == agent_mode:
        return action.copy()

    reshaped, kind, original_shape = _reshape_action(action)
    entries = _get_arm_entries(meta, action)
    current_poses = _get_current_poses(obs, meta, entries)

    transformed = reshaped.copy()
    for batch_idx in range(transformed.shape[0]):
        prev_pose_map = {
            str(key): value.copy() for key, value in current_poses.items()
        }
        for step_idx in range(transformed.shape[1]):
            current = reshaped[batch_idx, step_idx]
            for entry in entries:
                action_key = str(entry["action_key"])
                action_slice = entry["slice"]
                arm_dim = int(entry["arm_action_dim"])
                prev_pose = prev_pose_map[action_key]
                target_pose = current[action_slice][:arm_dim]
                transformed[batch_idx, step_idx, action_slice.start: action_slice.start + arm_dim] = compute_pose_delta(
                    target_pose,
                    prev_pose,
                )
                prev_pose_map[action_key] = target_pose.copy()

    return _restore_action_shape(transformed, kind, original_shape)
