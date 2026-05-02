# Agent Factory Data Flow Summary

## Background

This round focused on the boundary between `agent_infra` robot data and
`agent_factory` training data.

The key design decision is to keep robot-side data structured for replay,
debugging, and format conversion, while keeping algorithm-side data flattened
and tensor-friendly for training efficiency.

## Current Data Layers

### 1. Raw H5

Produced by `agent_infra/Piper_Env/Record/recorder.py`.

Typical structure:

```text
obs/state/<key>
obs/rgb/<role>
obs/depth/<role>
obs/under_control/<arm>
action/<key>
meta/env_meta
attrs["success"]
```

Purpose:

- Preserve the original robot trajectory.
- Support replay, inspection, and debugging.
- Allow later conversion to different control modes.

### 2. Merged Structured H5

Produced by `agent_infra/Piper_Env/Record/postprocess.py merge_h5`.

Typical structure:

```text
meta/env_meta
meta/merge_info
traj_x/obs/...
traj_x/action/...
traj_x/success
traj_x/terminated
traj_x/truncated
traj_x.attrs["success"]
```

Purpose:

- Merge per-trajectory raw H5 files.
- Optionally convert action control mode.
- Keep structured obs/action layout for infra tools.
- Preserve compatibility with replay and LeRobot conversion.

Episode signal semantics:

- `success`: per-frame bool array. If synthesized, only the last frame mirrors
  the trajectory-level success flag.
- `terminated`: per-frame bool array. If synthesized, the last frame is true.
- `truncated`: per-frame bool array. If synthesized, all values are false.

### 3. Flattened Training H5

Produced by `agent_factory/script/convert.sh` and
`agent_factory/data/converter.py`.

Typical structure:

```text
meta/env_meta
traj_x/obs/rgb
traj_x/obs/state
traj_x/actions
traj_x/terminated
traj_x/truncated
traj_x/success
```

Purpose:

- Provide model-ready data for `ExpertDataset`.
- Avoid repeated flattening during training.
- Keep algorithm input stable:

```text
observations["rgb"]    -> [T, C, H, W]
observations["state"]  -> [T, D]
actions                -> [T, A]
```

## Completed Changes

### ExpertDataset Input Formats

`agent_factory/data/dataset.py` now supports:

```text
flat
structured
auto
```

The default remains `flat`, preserving the previous behavior.

Configuration:

```yaml
dataset:
  expert:
    demo_path: path/to/data.h5
    format: flat        # flat | structured | auto
```

Runtime override:

```python
ExpertDataset(cfg, dataset_format="structured")
```

Behavior:

- `flat`: reads `traj_x/obs/rgb`, `traj_x/obs/state`, and `traj_x/actions`.
- `structured`: reads `traj_x/obs/rgb/<role>`,
  `traj_x/obs/state/<key>`, and `traj_x/action/<key>`, then flattens once
  during initialization.
- `auto`: detects the format from the H5 structure.

Flattening for structured data happens during dataset initialization, not in
`__getitem__`, so training-time sampling remains lightweight.

### Merge Episode Signals

`agent_infra/Piper_Env/Record/postprocess.py merge_h5` now writes the following
datasets for each merged trajectory:

```text
success
terminated
truncated
```

Existing `traj_x.attrs["success"]` is still preserved for backward
compatibility.

## Design Decisions

### Keep Algorithm Inputs Flat

The algorithm stack currently expects:

```text
obs:    {"rgb": tensor, "state": tensor}
action: tensor with final dimension A
```

Diffusion actors, critics, normalizers, and datasets all assume this contract.
Changing algorithms to consume structured robot dictionaries would create a
large cross-module refactor, so that is intentionally avoided for now.

### Do Not Modify Existing Merge Semantics Too Aggressively

`merge_h5` remains an infra-level structured-data operation. It should not be
converted into a training-only flattening tool because it also supports replay,
inspection, and LeRobot conversion.

### Prefer Flattened Data for Formal Training

Structured input support in `ExpertDataset` is mainly for convenience and
debugging. For repeated or large training runs, flattened training H5 remains
the recommended format because it avoids repeated preprocessing across runs.

