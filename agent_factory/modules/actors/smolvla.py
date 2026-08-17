from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from unittest.mock import patch

import torch
import torch.nn as nn


SMOLVLA_STATE_KEY = "observation.state"
SMOLVLA_ACTION_KEY = "action"
SMOLVLA_TASK_KEY = "task"
SMOLVLA_IMAGE_PREFIX = "observation.images."
SMOLVLA_LANGUAGE_TOKENS_KEY = "observation.language.tokens"
SMOLVLA_LANGUAGE_ATTENTION_MASK_KEY = "observation.language.attention_mask"
SMOLVLA_NORMALIZATION_SOURCE_AGENT_FACTORY = "agent_factory"
SMOLVLA_NORMALIZATION_SOURCE_NONE = "none"
SMOLVLA_NORMALIZATION_SOURCES = {
    SMOLVLA_NORMALIZATION_SOURCE_AGENT_FACTORY,
    SMOLVLA_NORMALIZATION_SOURCE_NONE,
}


class SmolVLADependencyError(ImportError):
    """Raised when LeRobot/SmolVLA dependencies are needed but unavailable."""


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _looks_like_local_path(path: str) -> bool:
    return path.startswith(("/", ".", "~")) or Path(path).exists()


def _resolve_pretrained_path(cfg: Any) -> str:
    pretrained_path = str(_cfg_get(cfg, "pretrained_path", "") or "").strip()
    if not pretrained_path:
        raise ValueError("SmolVLAActorConfig requires actor.pretrained_path.")
    return pretrained_path


def _validate_pretrained_path(pretrained_path: str, *, local_files_only: bool) -> None:
    if not local_files_only and not _looks_like_local_path(pretrained_path):
        return
    path = Path(pretrained_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"SmolVLA pretrained path does not exist: {pretrained_path}")
    if not path.is_dir():
        raise ValueError(f"SmolVLA actor.pretrained_path must point at a directory: {pretrained_path}")
    missing = [filename for filename in ("config.json", "model.safetensors") if not (path / filename).exists()]
    if missing:
        raise FileNotFoundError(f"SmolVLA pretrained path is missing {missing}: {pretrained_path}")


def resolve_smolvla_normalization_source(cfg: Any) -> str:
    source = str(_cfg_get(cfg, "normalization_source", SMOLVLA_NORMALIZATION_SOURCE_AGENT_FACTORY) or "")
    source = source.strip().lower()
    if source not in SMOLVLA_NORMALIZATION_SOURCES:
        raise ValueError(
            "SmolVLAActorConfig.normalization_source must be one of "
            f"{sorted(SMOLVLA_NORMALIZATION_SOURCES)}, got {source!r}."
        )
    return source


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _as_batched_state(value: Any, *, single_prompt: bool = False) -> torch.Tensor:
    state = _to_tensor(value).float()
    if state.ndim == 1:
        return state.unsqueeze(0)
    if state.ndim == 2:
        if single_prompt and state.shape[0] != 1:
            return state[-1:].contiguous()
        return state
    if state.ndim == 3:
        return state[:, -1, :]
    raise ValueError(f"SmolVLA state must have shape [D], [B,D], or [B,T,D], got {tuple(state.shape)}.")


def _as_batched_image(value: Any, *, batch_size: int) -> torch.Tensor:
    image = _to_tensor(value)
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    else:
        image = image.float()

    if image.ndim == 5:
        image = image[:, -1]
    elif image.ndim == 4 and batch_size == 1 and image.shape[0] != 1 and (
        image.shape[1] == 3 or image.shape[-1] == 3
    ):
        image = image[-1]
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(
            "SmolVLA image observations must have shape [C,H,W], [H,W,C], [B,C,H,W], "
            f"or [B,H,W,C], got {tuple(image.shape)}."
        )
    if image.shape[1] != 3 and image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2).contiguous()
    if image.shape[1] != 3:
        raise ValueError(f"SmolVLA image observations require RGB channels, got {tuple(image.shape)}.")
    if int(image.shape[0]) != batch_size:
        raise ValueError(f"Image batch size {image.shape[0]} does not match state batch size {batch_size}.")
    return image


def _coerce_task(prompt: Optional[Any], batch_size: int) -> Any:
    if prompt is None:
        raise ValueError("SmolVLA observations require a prompt/task string.")
    if isinstance(prompt, str):
        return prompt if batch_size == 1 else [prompt for _ in range(batch_size)]
    if isinstance(prompt, Sequence) and not isinstance(prompt, (bytes, bytearray)):
        values = list(prompt)
        if len(values) != batch_size:
            raise ValueError(f"SmolVLA prompt list length {len(values)} does not match batch size {batch_size}.")
        return values
    return prompt


def _to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class SmolVLAPolicyWrapper(nn.Module):
    """
    Thin adapter around LeRobot SmolVLAPolicy.

    The wrapper owns config adaptation and tokenization only. State/action
    normalization is intentionally left to agent_factory dataset/mixins so we do
    not accidentally apply LeRobot processor normalization on top.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        action_dim: int,
        pred_horizon: int,
        state_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.pred_horizon = int(pred_horizon)
        self.state_dim = int(state_dim)
        self.pretrained_path = _resolve_pretrained_path(cfg)
        self.local_files_only = bool(_cfg_get(cfg, "local_files_only", True))
        self.strict_load = bool(_cfg_get(cfg, "strict_load", False))
        self.image_keys = list(_cfg_get(cfg, "image_keys", []) or [])
        self.image_shape = tuple(int(v) for v in (_cfg_get(cfg, "image_shape", [3, 256, 256]) or [3, 256, 256]))
        self.prompt_key = str(_cfg_get(cfg, "prompt_key", SMOLVLA_TASK_KEY))
        self.state_key = str(_cfg_get(cfg, "state_key", SMOLVLA_STATE_KEY))
        self.normalization_source = resolve_smolvla_normalization_source(cfg)
        self.policy = None
        self.tokenizer = None
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        _validate_pretrained_path(self.pretrained_path, local_files_only=self.local_files_only)
        self.policy = self._load_lerobot_policy()

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _make_policy_config(self):
        try:
            from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
        except ImportError as exc:
            raise SmolVLADependencyError("SmolVLA requires LeRobot with smolvla support.") from exc

        config = PreTrainedConfig.from_pretrained(
            self.pretrained_path,
            revision=_cfg_get(self.cfg, "pretrained_revision", None),
            local_files_only=self.local_files_only,
        )
        config.device = str(self.device)
        config.chunk_size = int(_cfg_get(self.cfg, "chunk_size", 50))
        config.n_action_steps = int(_cfg_get(self.cfg, "n_action_steps", self.pred_horizon))
        config.max_state_dim = int(_cfg_get(self.cfg, "max_state_dim", 32))
        config.max_action_dim = int(_cfg_get(self.cfg, "max_action_dim", 32))
        config.num_steps = int(_cfg_get(self.cfg, "num_inference_steps", 10))
        config.tokenizer_max_length = int(_cfg_get(self.cfg, "tokenizer_max_length", 48))
        config.freeze_vision_encoder = bool(_cfg_get(self.cfg, "freeze_vision_encoder", True))
        config.train_expert_only = bool(_cfg_get(self.cfg, "train_expert_only", True))
        config.train_state_proj = bool(_cfg_get(self.cfg, "train_state_proj", True))
        config.load_vlm_weights = bool(_cfg_get(self.cfg, "load_vlm_weights", True))

        image_keys = self.image_keys
        if not image_keys:
            raise ValueError("SmolVLAActorConfig.image_keys must name at least one camera.")
        config.input_features = {
            SMOLVLA_STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
        }
        for camera_name in image_keys:
            config.input_features[f"{SMOLVLA_IMAGE_PREFIX}{camera_name}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=self.image_shape,
            )
        config.output_features = {
            SMOLVLA_ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }
        return config

    def _load_lerobot_policy(self):
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise SmolVLADependencyError("SmolVLA requires LeRobot with smolvla support.") from exc

        policy_config = self._make_policy_config()
        original_processor_from_pretrained = AutoProcessor.from_pretrained
        original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

        def _offline_processor_from_pretrained(*args, **kwargs):
            kwargs.setdefault("local_files_only", self.local_files_only)
            return original_processor_from_pretrained(*args, **kwargs)

        def _offline_tokenizer_from_pretrained(*args, **kwargs):
            kwargs.setdefault("local_files_only", self.local_files_only)
            return original_tokenizer_from_pretrained(*args, **kwargs)

        with patch.object(AutoProcessor, "from_pretrained", _offline_processor_from_pretrained), patch.object(
            AutoTokenizer, "from_pretrained", _offline_tokenizer_from_pretrained
        ):
            return SmolVLAPolicy.from_pretrained(
                self.pretrained_path,
                config=policy_config,
                revision=_cfg_get(self.cfg, "pretrained_revision", None),
                local_files_only=self.local_files_only,
                strict=self.strict_load,
            )

    def _ensure_tokenizer(self):
        if self.tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SmolVLADependencyError("SmolVLA tokenization requires transformers.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.policy.config.vlm_model_name,
            local_files_only=self.local_files_only,
        )

    def _tokenize_task(self, task: Any) -> Dict[str, torch.Tensor]:
        self._ensure_tokenizer()
        if isinstance(task, str):
            text = task if task.endswith("\n") else f"{task}\n"
        else:
            text = [t if str(t).endswith("\n") else f"{t}\n" for t in task]
        tokenized = self.tokenizer(
            text,
            max_length=int(self.policy.config.tokenizer_max_length),
            truncation=True,
            padding=self.policy.config.pad_language_to,
            padding_side="right",
            return_tensors="pt",
        )
        return {
            SMOLVLA_LANGUAGE_TOKENS_KEY: tokenized["input_ids"],
            SMOLVLA_LANGUAGE_ATTENTION_MASK_KEY: tokenized["attention_mask"].bool(),
        }

    def preprocess_agent_factory_obs(self, obs: Dict[str, Any], prompt: Optional[Any] = None) -> Dict[str, Any]:
        raw_prompt = prompt if prompt is not None else obs.get(SMOLVLA_TASK_KEY, obs.get(self.prompt_key, None))
        single_prompt = isinstance(raw_prompt, str)
        source_state_key = self.state_key if self.state_key in obs else SMOLVLA_STATE_KEY if SMOLVLA_STATE_KEY in obs else "state"
        if source_state_key not in obs:
            raise KeyError("SmolVLA observations require 'observation.state' or 'state'.")

        batch: Dict[str, Any] = {SMOLVLA_STATE_KEY: _as_batched_state(obs[source_state_key], single_prompt=single_prompt)}
        batch_size = int(batch[SMOLVLA_STATE_KEY].shape[0])

        existing_image_keys = [key for key in obs if str(key).startswith(SMOLVLA_IMAGE_PREFIX)]
        if existing_image_keys:
            selected = [f"{SMOLVLA_IMAGE_PREFIX}{name}" for name in self.image_keys] or sorted(existing_image_keys)
            for key in selected:
                if key not in obs:
                    raise KeyError(f"SmolVLA observation is missing image key {key!r}.")
                batch[key] = _as_batched_image(obs[key], batch_size=batch_size)
        else:
            image_source = obs.get("images", obs.get("rgb"))
            if not isinstance(image_source, dict):
                raise ValueError("SmolVLA requires images as a camera-name dict or observation.images.<camera> keys.")
            for camera_name in self.image_keys:
                if camera_name not in image_source:
                    raise KeyError(f"SmolVLA image source is missing camera {camera_name!r}.")
                batch[f"{SMOLVLA_IMAGE_PREFIX}{camera_name}"] = _as_batched_image(
                    image_source[camera_name],
                    batch_size=batch_size,
                )

        task = _coerce_task(raw_prompt, batch_size)
        batch[SMOLVLA_TASK_KEY] = task
        batch.update(self._tokenize_task(task))
        return _to_device_batch(batch, self.device)

    def _ensure_policy_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(batch)
        if SMOLVLA_LANGUAGE_TOKENS_KEY not in out or SMOLVLA_LANGUAGE_ATTENTION_MASK_KEY not in out:
            task = out.get(SMOLVLA_TASK_KEY, out.get(self.prompt_key, None))
            out.update(self._tokenize_task(task))
        return _to_device_batch(out, self.device)

    def forward(self, batch: Dict[str, Any]):
        if self.policy is None:
            raise RuntimeError("SmolVLA policy is not loaded.")
        return self.policy(self._ensure_policy_batch(batch))

    @torch.no_grad()
    def sample_action(self, obs: Dict[str, Any], prompt: Optional[Any] = None, **kwargs) -> torch.Tensor:
        if self.policy is None:
            raise RuntimeError("SmolVLA policy is not loaded.")
        batch = self.preprocess_agent_factory_obs(obs, prompt=prompt)
        if hasattr(self.policy, "predict_action_chunk"):
            action_chunk = self.policy.predict_action_chunk(batch, **kwargs)
        else:
            action = self.policy.select_action(batch, **kwargs)
            action_chunk = action[:, None, :].expand(-1, self.pred_horizon, -1)
        return self._align_action_chunk(action_chunk)

    @torch.no_grad()
    def extract_vlm_prefix_features(self, obs: Dict[str, Any], prompt: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """
        Return deterministic final VLM prefix features for normalized observations.

        This mirrors SmolVLA inference up to the prefix KV-cache fill and stops
        before action suffix embedding or denoising. The returned state token is
        selected from the final prefix hidden states, not from the raw state
        projection.
        """
        if self.policy is None:
            raise RuntimeError("SmolVLA policy is not loaded.")
        batch = self.preprocess_agent_factory_obs(obs, prompt=prompt)
        policy_model = self.policy.model

        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = policy_model.embed_prefix(
            images,
            img_masks,
            batch[SMOLVLA_LANGUAGE_TOKENS_KEY],
            batch[SMOLVLA_LANGUAGE_ATTENTION_MASK_KEY],
            state=state,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        outputs_embeds, _past_key_values = policy_model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.policy.config.use_cache,
            fill_kv_cache=True,
        )
        prefix_tokens = outputs_embeds[0]
        if prefix_tokens is None:
            raise RuntimeError("SmolVLA VLM prefix forward returned no prefix tokens.")

        state_mask = prefix_att_masks.bool() & prefix_pad_masks.bool()
        state_token_indices = []
        state_tokens = []
        for batch_idx in range(state_mask.shape[0]):
            indices = torch.nonzero(state_mask[batch_idx], as_tuple=False).reshape(-1)
            if indices.numel() != 1:
                raise ValueError(
                    "Expected exactly one SmolVLA state token in the prefix, "
                    f"got {indices.numel()} for batch item {batch_idx}."
                )
            index = indices[0]
            state_token_indices.append(index)
            state_tokens.append(prefix_tokens[batch_idx, index, :])

        return {
            "prefix_tokens": prefix_tokens,
            "prefix_valid_mask": prefix_pad_masks.bool(),
            "state_token": torch.stack(state_tokens, dim=0),
            "state_token_index": torch.stack(state_token_indices, dim=0).to(dtype=torch.long),
        }

    def _align_action_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[:, None, :]
        if action_chunk.ndim != 3:
            raise ValueError(f"SmolVLA sample_action must return [B,T,D], got {tuple(action_chunk.shape)}.")
        action_chunk = action_chunk.to(device=self.device, dtype=torch.float32)
        if int(action_chunk.shape[-1]) != self.action_dim:
            raise ValueError(
                "SmolVLA action dimension does not match env action_dim: "
                f"policy returned {action_chunk.shape[-1]}, env expects {self.action_dim}."
            )
        if int(action_chunk.shape[1]) < self.pred_horizon:
            repeat = action_chunk[:, -1:, :].expand(-1, self.pred_horizon - int(action_chunk.shape[1]), -1)
            action_chunk = torch.cat([action_chunk, repeat], dim=1)
        elif int(action_chunk.shape[1]) > self.pred_horizon:
            action_chunk = action_chunk[:, : self.pred_horizon, :]
        return action_chunk
