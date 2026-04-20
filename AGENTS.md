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
