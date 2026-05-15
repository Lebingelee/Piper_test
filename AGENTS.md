# Repository Guidelines

## Project Structure & Module Organization
This repository is Python-first and centers on robot environment integration.

- `agent_infra/`: production-style environments and tooling.
- `agent_infra/Piper_Env/`: Piper arm envs, camera wrappers, recorder/replay tools, and scripts.
- `agent_infra/Realman_Env/`: Realman reference implementation; use it as the design baseline for multi-arm patterns and `meta_keys`.
- `piper_infra/`: earlier Piper prototype code kept for migration/reference.
- `pyAgxArm/`: hardware SDK dependency vendored in-tree.
- `trial/`: ad hoc experiments and hardware probes; do not treat as production code.
- `gpt_log/`: planning notes and prompts.
- `data/`: recorded trajectories and replay assets.

## Build, Test, and Development Commands
- `python3 -m py_compile agent_infra/Piper_Env/...`: quick syntax check before committing.
- `python3 agent_infra/Piper_Env/Script/test_piper_env.py`: basic Piper env smoke test.
- `python3 agent_infra/Piper_Env/Script/test_teleop_obs_action.py`: inspect teleop-time `obs` and executed `action`.
- `bash agent_infra/Piper_Env/Script/single/collect_h5.sh`: collect single-arm Piper trajectories to H5.
- `bash agent_infra/Piper_Env/Script/dual/collect_h5.sh`: collect dual-arm Piper trajectories to H5.
- `bash agent_infra/Piper_Env/Script/single/collect_lerobot.sh`: collect single-arm Piper trajectories to LeRobot format.
- `bash agent_infra/Piper_Env/Script/dual/collect_lerobot.sh`: collect dual-arm Piper trajectories to LeRobot format.
- `bash agent_infra/Piper_Env/Script/single/replay.sh -i <path> -f h5|lerobot`: replay single-arm trajectories.
- `bash agent_infra/Piper_Env/Script/dual/replay.sh -i <path> -f h5|lerobot`: replay dual-arm trajectories.
- `python3 agent_infra/Piper_Env/Script/test_piper_pipeline.py --arm-mode single --camera-mode without --function teleop`: Piper env smoke test.

## Coding Style & Naming Conventions
Use 4-space indentation and follow existing Python style. Prefer small modules with clear hardware/env/record separation. Use `snake_case` for files, functions, and config keys; use `PascalCase` for classes such as `PiperEnv`. Keep all Piper changes inside `agent_infra/Piper_Env/` unless the task explicitly requires broader refactoring.

## Testing Guidelines
There is no centralized `pytest` suite yet; validation is script-driven. Add focused test scripts under `agent_infra/Piper_Env/Script/` named `test_*.py`. For changes to env I/O, verify `reset()`, `step()`, `meta_keys`, and recorder/replay compatibility. Run `py_compile` plus at least one relevant script before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short subjects (`0326`, `Resolve conflicts by favoring local changes`), but contributors should prefer concise imperative messages, e.g. `Add Piper reset-to-state support`. Keep commits scoped to one change. PRs should include: purpose, affected modules, test commands run, hardware assumptions, and sample output or screenshots when changing teleop/recording flows.

## Security & Configuration Tips
Do not hardcode machine-specific CAN, IP, or camera values outside config files. Keep environment-specific settings in `agent_infra/*/Config/*.yaml`. Never commit recorded data, secrets, or destructive hardware scripts without explicit review.

## Current Working Knowledge

### `agent_factory` / Runner Pipeline
- `agent_factory` owns config loading, agent construction, wrappers, runner control loops, trajectory saving, datasets, and training entrypoints.
- `script/test_base_runner.py` and `script/test_hitl_runner.py` are the current hardware rollout smoke tests. They default to `run_results/piper_dual_merged_cpiql_dac/model_config.yaml` and load a checkpoint via `--checkpoint`; both support `--skip-load`, `--device`, `--max-steps`, `--control-hz`, and `--save-dir`.
- `BaseRunner` should save root `action` as the env-native executed action. Prefer `info["actual_action"]` from the env when available; this is the source of truth for rollout H5.
- The current rollout H5 schema should include `obs/*`, root `action`, `success`, `terminated`, `truncated`, `intervention`, optional `action_type`, and `meta/env_cfg` plus `meta/env_meta`. Do not split primary behavior into a separate `policy_action` truth unless explicitly needed for debugging.
- `HITLRunner` should treat Piper teleop as an env-side intervention source. It reads `env.unwrapped.get_env_state("teleop")` before each step, clears stale action chunks during intervention, and replans after release.
- Runner-side keyboard override should use `o`. Teleop hotkeys such as `t` belong in runner/recorder layers and should call the env teleop interface; PiperEnv itself should not own keyboard listeners for teleop.
- `agent_factory/data/utils.py::preprocess_obs` must keep `depth` as a separate modality. Do not concatenate depth into `rgb`; current vision policies trained on RGB expect `rgb` to remain `3 * num_cameras` channels.
- CPIQL-DAC checkpoints saved before actor training may contain invalid `action_normalizer` buffers (`initialized=False`, `min=inf`, `max=-inf`). `agent_factory/runner/checkpoint_utils.py` now checks this and can refit the normalizer from the configured expert H5 before inference.
- `agent_factory/agents/impl/diffusion_cpiql_dac.py` should fit the action normalizer before critic training/saving, so both critic and actor checkpoints carry finite `action_normalizer.min_val/max_val`.
- `agent_factory/script/train_universal.py` is the preferred command-line training entrypoint. Use `python -m agent_factory.script.train_universal` from the repo root, not direct file execution, unless `PYTHONPATH` is set.

### `agent_infra` / Piper Environment
- `agent_infra/Piper_Env` owns hardware-facing Piper behavior: arm wrappers, camera wrappers, teleop, safety locks, reset/mode switching, and config-driven CAN/camera settings.
- Piper hardware settings belong in `agent_infra/Piper_Env/Config/*.yaml`, especially CAN channels, camera serials, and `common.master_follow`.
- PiperEnv exposes `switch_passive("true"|"false")` as the hardware action-dispatch safety lock. Default should be safe/off; tests explicitly enable it before rollout.
- PiperEnv exposes `switch_master_follow(mode="toggle")` and reads default `common.master_follow` from `dual_piper_config.yaml`.
- Environment-owned states are managed by `switch_*` plus `get_env_state()`: use `switch_tele()`, `switch_passive()`, or `switch_master_follow()` to mutate state, and use `get_env_state("teleop"|"passive"|"master_follow")` to read state. Keyboard listeners must live above PiperEnv, such as in `Record/recorder.py` or `agent_factory/runner`.
- `begin_master_follow()` / `end_master_follow()` must not run inside the high-frequency `step()` loop. Mode switches should happen only on state boundaries: `switch_passive`, `switch_master_follow`, teleop toggle, or close/reset paths.
- During policy control with `master_follow=True`, the master arm may mirror follower state, but mode switching must remain low frequency to avoid CAN buffer saturation.
- During teleop, the master arm should be released to draggable leader mode and the follower should track fresh leader readings. While `get_env_state("teleop")` is true but leader readings are not ready, PiperEnv must hold with `get_safe_action()` rather than executing stale policy chunks.
- After `reset(options={"sync_master": True})`, PiperEnv should re-enter policy master-follow only when `get_env_state("passive")` and `get_env_state("master_follow")` are true while teleop is false; when `master_follow` is false, the master should remain in draggable leader mode for expert collection compatibility.
- `PiperEnv.step()` should report `info["actual_action"]`, `info["policy_action"]`, `info["intervened_map"]`, `info["action_source_map"]`, `info["syncing_map"]`, `info["teleop_enabled"]`, and `info["master_follow_enabled"]` when possible.
- `MetadataAdapterWrapper` flattens observations according to `meta_keys` sorted by key. For structured dual-Piper actions, expected action order is lexicographic over action leaves, e.g. `left_arm`, `left_gripper`, `right_arm`, `right_gripper`.

### Identity Agent And Hardware Smoke Tests
- `agent_factory/agents/impl/identity.py` is a deterministic hardware test agent. It can either use env safe actions or emit a smooth absolute-joint action chunk for motion checks.
- For safety tests, `uses_env_safe_action=True` should hold the robot using env-native safe actions.
- For motion tests, `uses_env_safe_action=False` emits a `[B, 64, 14]` absolute-joint chunk with smooth movement in the configured joint dimensions. Use this only when Piper `switch_passive("true")` is intentional and the workspace is clear.

### Useful Validation Commands
- Syntax checks:
  - `python3 -m py_compile agent_factory/runner/base_runner.py agent_factory/runner/hitl_runner.py`
  - `python3 -m py_compile agent_infra/Piper_Env/Env/utils/piper_base_env.py agent_infra/Piper_Env/Env/utils/piper_arm.py`
  - `python3 -m py_compile agent_factory/data/utils.py agent_factory/data/normalization/min_max.py`
- Runner help checks:
  - `/home/lebinge/miniforge3/envs/SL/bin/python script/test_base_runner.py --help`
  - `/home/lebinge/miniforge3/envs/SL/bin/python script/test_hitl_runner.py --help`
- Minimal CPIQL-DAC training smoke test:
  - `/home/lebinge/miniforge3/envs/SL/bin/python -m agent_factory.script.train_universal --config run_results/towel_cpiql_dac/model_config.yaml --train-object critic_then_actor --critic-iters 1 --actor-iters 1 --batch-size 2 --num-workers 0 --save-interval 1 --exp-name towel_cpiql_dac_smoke_iter1_normfix`
