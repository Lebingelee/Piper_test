# π₀.₅ Feature Export: Memory Anchor

## Frozen first-stage objective

Replace the existing `agent_factory` π₀ integration with a LeRobot π₀.₅
(`pi05`) integration.  Given a complete LeRobot-compatible π₀.₅ weight
directory, convert raw agent_factory H5 trajectories into feature H5
trajectories, following the storage behaviour of
`agent_factory/script/export_smolvla_features.py`.

The first stage is intentionally **feature-export only**.  It must not
download weights, run a real π₀.₅ inference smoke test, or implement
fine-tuning/rollout on this local machine.  Those checks belong to the server
handoff stage.

## Model and feature decision

- Runtime backend: LeRobot `PI05Policy` from the local `src` environment
  (`lerobot==0.6.0`).
- Weight input: a complete LeRobot pretrained/checkpoint directory, configured
  through `actor.pretrained_path`.  The directory is expected to provide the
  model config, `model.safetensors`, and serialized pre/post-processors.
- The LeRobot π₀.₅ preprocessor normalizes state, discretizes it, and inserts
  it into the task text before tokenization.  Therefore VLM prefix tokens are
  state-aware as well as image- and prompt-aware.
- Feature source: the hidden states returned by the VLM prefix forward,
  before action-expert denoising.  No action noise or denoising step is used.
- Pooling: select the final valid language token from the prefix.  It is the
  last valid prefix token after image tokens and the state-aware task prompt.

## H5 input and output contract

Supported raw input layouts:

1. A single root trajectory, such as `data/SmolVLA/<task>/traj_*.h5`.
2. A merged H5 file containing `traj_xxx` groups, such as the Robosuite stack
   merged demos.

Input observations may use either a single state dataset or a state group.
State fields are flattened in the ordering declared by `meta/env_meta`.
RGB must remain a camera-name group; camera names are configured in
`actor.image_keys`.

The exporter copies every trajectory node except `obs`, then writes:

```text
obs/feature/prefix_valid_mask  [T_obs, L] bool
obs/feature/state_token        [T_obs, D] float16|float32
obs/feature/state_token_index  [T_obs] int64
```

`state_token` remains the compatibility field used by `VLAFeatureDataset`.
It is semantically a selected π₀.₅ prefix language token, recorded with:

```text
obs/feature attrs:
  feature_source = "pi05_vlm_prefix_last_language_token"
```

Feature length follows raw observation length.  Thus rollout H5 files may
produce `T_action + 1` features, while merged demos may produce `T_action`
features; the existing feature reader already supports both.

## Configuration ownership

Do not add π₀.₅-specific fields to public `env` or `runner` configuration,
and do not modify `critic` configuration.

`Pi05ActorConfig` owns π₀.₅ details:

- `pretrained_path`, `pretrained_revision`, `local_files_only`;
- `image_keys`, `state_key`, `action_key`, `prompt_key`;
- state/action schema constraints resolved consistently with the existing
  public environment dimensions;
- prefix pooling and feature-export options.

`agent_sp` is limited to agent-specific run metadata such as its experiment
name and save directory.  A multi-task run may use any state/action sizes, but
they must be uniform within that resolved configuration and compatible with
the selected checkpoint processor.

## Migration boundary

All old π₀-specific actor, agent, runner, config, script, registry, and test
paths under `agent_factory` are to be removed or replaced by `pi05` paths.

The former `data/impl/pi0/policy_dataset.py` also contains H5 helpers used by
SmolVLA.  Move those generic helpers to a neutral module before deleting the
π₀ package.  The generic `convert_h5_to_lerobot.py` converter remains, with
its imports pointed at the neutral module.

## Local validation and server handoff

Local acceptance is structural only:

- imports and resolved config construction succeed without weights;
- single and merged H5 schemas are accepted by the exporter helpers;
- produced feature H5 files match the reader contract;
- no old π₀ implementation imports remain in `agent_factory`.

The server must validate real checkpoint loading, preprocessor compatibility,
GPU prefix forward, and one-trajectory end-to-end export before any rollout
or fine-tuning work.
