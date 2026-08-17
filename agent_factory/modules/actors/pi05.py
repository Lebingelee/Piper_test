"""LeRobot π₀.₅ adapter used for deterministic VLM-prefix feature export."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn


PI05_STATE_KEY = "observation.state"
PI05_ACTION_KEY = "action"
PI05_TASK_KEY = "task"
PI05_IMAGE_PREFIX = "observation.images."
PI05_PREPROCESSOR_NAME = "policy_preprocessor.json"
PI05_POSTPROCESSOR_NAME = "policy_postprocessor.json"


class Pi05DependencyError(ImportError):
    """Raised when the optional LeRobot π₀.₅ stack is unavailable."""


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)


def _resolve_pretrained_path(cfg: Any) -> str:
    path = str(_cfg_get(cfg, "pretrained_path", "") or "").strip()
    if not path:
        raise ValueError("Pi05ActorConfig requires actor.pretrained_path when mock_mode=False.")
    return path


def _validate_pretrained_dir(path_text: str, *, local_files_only: bool) -> None:
    if not local_files_only and not path_text.startswith(("/", ".", "~")):
        return
    path = Path(path_text).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"π₀.₅ pretrained directory does not exist: {path_text}")
    missing = [
        name
        for name in ("config.json", "model.safetensors", PI05_PREPROCESSOR_NAME, PI05_POSTPROCESSOR_NAME)
        if not (path / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "π₀.₅ pretrained directory is missing required LeRobot artifacts "
            f"{missing}: {path_text}"
        )


def _as_frame_tensor(value: Any, frame_idx: int, batch_size: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.ndim > 0 and int(tensor.shape[0]) == batch_size:
        return tensor[frame_idx]
    if batch_size != 1:
        raise ValueError("π₀.₅ observation value is missing its leading batch dimension.")
    return tensor


def _state_batch_size(obs: Dict[str, Any], state_key: str) -> int:
    source = state_key if state_key in obs else PI05_STATE_KEY if PI05_STATE_KEY in obs else "state"
    if source not in obs:
        raise KeyError("π₀.₅ observations require 'observation.state' or 'state'.")
    state = obs[source] if isinstance(obs[source], torch.Tensor) else torch.as_tensor(obs[source])
    return 1 if state.ndim == 1 else int(state.shape[0])


class Pi05PolicyWrapper(nn.Module):
    """Thin adapter around LeRobot ``PI05Policy``.

    Feature export deliberately stops after the VLM prefix forward.  The
    LeRobot preprocessor is applied per frame because its public pipeline adds
    a batch dimension for a single environment observation; processed frames
    are then concatenated for efficient prefix inference.
    """

    def __init__(self, cfg: Any, action_dim: int, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.feature_dim = int(_cfg_get(cfg, "feature_dim", 2048))
        self.mock_mode = bool(_cfg_get(cfg, "mock_mode", True))
        self.image_keys = list(_cfg_get(cfg, "image_keys", []) or [])
        self.state_key = str(_cfg_get(cfg, "state_key", PI05_STATE_KEY))
        self.prompt_key = str(_cfg_get(cfg, "prompt_key", "prompt"))
        self.feature_pooling = str(_cfg_get(cfg, "feature_pooling", "last_language_token"))
        if self.feature_pooling != "last_language_token":
            raise ValueError("π₀.₅ supports only feature_pooling='last_language_token'.")
        max_state_dim = int(_cfg_get(cfg, "max_state_dim", 32))
        max_action_dim = int(_cfg_get(cfg, "max_action_dim", 32))
        if self.state_dim > max_state_dim:
            raise ValueError(
                f"actor.state_dim={self.state_dim} exceeds actor.max_state_dim={max_state_dim}."
            )
        if self.action_dim > max_action_dim:
            raise ValueError(
                f"actor.action_dim={self.action_dim} exceeds actor.max_action_dim={max_action_dim}."
            )

        self.pretrained_path = "" if self.mock_mode else _resolve_pretrained_path(cfg)
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self._processor_device: Optional[str] = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        if not self.mock_mode:
            self.policy = self._load_policy()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _load_policy(self):
        try:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        except ImportError as exc:
            raise Pi05DependencyError(
                "Pi05PolicyWrapper requires lerobot[pi] with π₀.₅ support."
            ) from exc
        local_files_only = bool(_cfg_get(self.cfg, "local_files_only", True))
        _validate_pretrained_dir(self.pretrained_path, local_files_only=local_files_only)
        return PI05Policy.from_pretrained(
            self.pretrained_path,
            revision=_cfg_get(self.cfg, "pretrained_revision", None),
            local_files_only=local_files_only,
            strict=bool(_cfg_get(self.cfg, "strict_load", True)),
        )

    def _ensure_processors(self) -> None:
        if self.mock_mode:
            return
        device_name = str(self.device)
        if self.preprocessor is not None and self._processor_device == device_name:
            return
        try:
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise Pi05DependencyError("LeRobot policy processor factory is unavailable.") from exc
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.pretrained_path,
            pretrained_revision=_cfg_get(self.cfg, "pretrained_revision", None),
            preprocessor_overrides={"device_processor": {"device": device_name}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._processor_device = device_name

    def _prompt_at(self, prompt: Any, index: int, batch_size: int) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray)):
            if len(prompt) != batch_size:
                raise ValueError("π₀.₅ prompt count does not match observation batch size.")
            return str(prompt[index])
        raise ValueError("π₀.₅ observations require a prompt/task string per frame.")

    def _raw_frame(self, obs: Dict[str, Any], prompt: Any, index: int, batch_size: int) -> Dict[str, Any]:
        state_source = self.state_key if self.state_key in obs else PI05_STATE_KEY if PI05_STATE_KEY in obs else "state"
        state = _as_frame_tensor(obs[state_source], index, batch_size).float()
        if state.ndim != 1:
            raise ValueError(f"π₀.₅ state frame must be rank 1, got {tuple(state.shape)}.")
        if self.state_dim > 0 and int(state.numel()) != self.state_dim:
            raise ValueError(
                f"π₀.₅ state dim {state.numel()} does not match configured actor.state_dim={self.state_dim}."
            )
        images_source = obs.get("images", obs.get("rgb"))
        if not isinstance(images_source, dict):
            raise ValueError("π₀.₅ requires RGB observations as a camera-name dictionary.")
        if not self.image_keys:
            raise ValueError("Pi05ActorConfig.image_keys must explicitly define the multi-task camera schema.")
        raw: Dict[str, Any] = {PI05_STATE_KEY: state, PI05_TASK_KEY: self._prompt_at(prompt, index, batch_size)}
        for camera_name in self.image_keys:
            if camera_name not in images_source:
                raise KeyError(f"π₀.₅ observation is missing configured camera {camera_name!r}.")
            image = _as_frame_tensor(images_source[camera_name], index, batch_size)
            if image.ndim != 3:
                raise ValueError(f"π₀.₅ image {camera_name!r} must be [C,H,W] or [H,W,C].")
            raw[f"{PI05_IMAGE_PREFIX}{camera_name}"] = image
        return raw

    @staticmethod
    def _concatenate_processed_frames(frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not frames:
            raise ValueError("No π₀.₅ frames were supplied for prefix extraction.")
        shared = set(frames[0])
        for frame in frames[1:]:
            shared &= set(frame)
        batch: Dict[str, Any] = {}
        for key in shared:
            values = [frame[key] for frame in frames]
            if all(isinstance(value, torch.Tensor) for value in values):
                batch[key] = torch.cat(values, dim=0)
        return batch

    def _processed_batch(self, obs: Dict[str, Any], prompt: Any) -> Dict[str, Any]:
        self._ensure_processors()
        if self.preprocessor is None:
            raise RuntimeError("π₀.₅ preprocessor was not loaded.")
        batch_size = _state_batch_size(obs, self.state_key)
        frames = [self.preprocessor(self._raw_frame(obs, prompt, idx, batch_size)) for idx in range(batch_size)]
        return self._concatenate_processed_frames(frames)

    @torch.no_grad()
    def extract_vlm_prefix_features(self, obs: Dict[str, Any], prompt: Any) -> Dict[str, torch.Tensor]:
        """Return final VLM prefix hidden states and the selected language token."""
        batch_size = _state_batch_size(obs, self.state_key)
        if self.mock_mode:
            tokens = torch.zeros((batch_size, 1, self.feature_dim), dtype=torch.float32, device=self.device)
            return {
                "prefix_tokens": tokens,
                "prefix_valid_mask": torch.ones((batch_size, 1), dtype=torch.bool, device=self.device),
                "state_token": tokens[:, 0],
                "state_token_index": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            }
        try:
            from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        except ImportError as exc:
            raise Pi05DependencyError("LeRobot π₀.₅ prefix helpers are unavailable.") from exc

        batch = self._processed_batch(obs, prompt)
        images, image_masks = self.policy._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_valid_mask, prefix_att_mask = self.policy.model.embed_prefix(
            images, image_masks, tokens, language_masks
        )
        prefix_att_2d_mask = make_att_2d_masks(prefix_valid_mask, prefix_att_mask)
        prefix_position_ids = torch.cumsum(prefix_valid_mask, dim=1) - 1
        attention_mask = self.policy.model._prepare_attention_masks_4d(prefix_att_2d_mask)
        outputs, _ = self.policy.model.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
        prefix_tokens = outputs[0]
        indices = prefix_valid_mask.to(dtype=torch.long).sum(dim=1) - 1
        if torch.any(indices < 0):
            raise ValueError("π₀.₅ prefix contains no valid token for at least one frame.")
        selected = prefix_tokens[torch.arange(prefix_tokens.shape[0], device=prefix_tokens.device), indices]
        return {
            "prefix_tokens": prefix_tokens,
            "prefix_valid_mask": prefix_valid_mask.bool(),
            "state_token": selected,
            "state_token_index": indices.to(dtype=torch.long),
        }
