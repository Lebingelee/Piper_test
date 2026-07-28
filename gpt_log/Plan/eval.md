# Eval Universal Plan

## Background

The current goal is to evaluate trained critic behavior on an existing replay trajectory and visualize critic-related outputs along the whole trajectory timeline.

The first target is the trained `CPIQL-DAC` critic on the current replay `.h5` single-trajectory format. The design should stay extensible so future critic algorithms can reuse the same evaluation pipeline with minimal extra work.

This document summarizes the agreed scope, constraints, risks, and implementation plan from the discussion.

## Agreed Scope

### Primary Goal

Implement a universal evaluation script that:

1. Loads one replay `.h5` trajectory.
2. Expands the trajectory into model-compatible evaluation windows.
3. Runs critic-side evaluation step by step along the whole trajectory.
4. Draws line plots for returned metrics.
5. Saves figures to a user-specified output directory.

### First-Version Scope

Only support:

- replay single-file `.h5` trajectory input
- critic evaluation
- `CPIQL` critic implementation first
- figure saving first

Not in first version:

- expert trajectory evaluation
- folder-based multi-file replay browsing
- mandatory `npz` saving
- support for every critic algorithm on day one

## Confirmed Product Decisions

### 1. Input Source

First version only supports current replay `.h5` single-file format.

Priority:

1. `--replay_h5_path` if user provides it
2. otherwise use `cfg.dataset.replaybuffer.replaybuffer_path`

`folder_path` should be ignored in this evaluation path.

If neither is provided, evaluation should raise a clear error instead of silently falling back to another dataset source.

### 2. Checkpoint Resolution

Priority:

1. `--ckpt_path`
2. otherwise `cfg.train.ckpt_path`

If both are empty, evaluation should raise a clear error.

### 3. Evaluation Unit

Evaluation is **not** single-frame evaluation.

Each evaluation point is:

- an anchor timestep `t`
- expanded into a legal model input window

The window must satisfy current config requirements:

- `env.obs_horizon`
- `env.pred_horizon`

This is necessary because critic input shapes must match training/inference expectations.

### 4. Window Construction Rule

For each anchor timestep `t`:

- `observations`: build a history window of length `obs_horizon`
- if the left side is insufficient, pad by repeating the first observation
- `action`: build a forward chunk of length `pred_horizon`
- if the right side is insufficient, pad by repeating the last action

The time axis should be aligned to action timesteps:

- evaluate only over `t = 0 ... T-1`
- where `T = len(action)`

### 5. Required Batch Semantics

Each evaluation batch must contain at least:

- `observations`
- `action`

Even when `only_obs=True`, `action` is still required so batch semantics remain complete and future-compatible.

### 6. Critic Eval API

`eval_batch()` should be implemented primarily in critic mixins.

First implementation target:

- `CPIQLCriticMixin`

Future algorithms can implement the same interface in their own critic mixins.

`impl` level is allowed to adjust downstream mixin methods if needed, but the evaluation capability should conceptually live in the critic mixin.

### 7. CPIQL First-Version Metrics

When `only_obs=True`, return:

- `V(k=0)`
- `V(k=1)`
- `critic_gap`

When `only_obs=False`, additionally return:

- `Q(k=0)`
- `Q(k=1)`
- `adv(k=0)`

`adv(k=1)` is not needed in first version.

### 8. Plot Output Convention

The evaluation script should build line plots based on returned dict keys.

Recommended dict convention:

- `frame`
- `figure:value/V(k=0)`
- `figure:value/V(k=1)`
- `figure:value/critic_gap`
- `figure:action_value/Q(k=0)`
- `figure:action_value/Q(k=1)`
- `figure:action_value/adv(k=0)`

Use semantic figure names such as `value` and `action_value` instead of `figure1`, `figure2` to keep future extension cleaner.

### 9. Save Policy

Priority is saving figures.

Optional `npz` saving can exist behind a global flag in `eval_universal`, but it is not required to be enabled by default in first version.

## Architecture Decision

### Core Principle

Do **not** overload existing training dataset `__getitem__` logic for this evaluation flow.

The current training datasets are sample-oriented and tightly coupled to:

- training slicing
- next observation logic
- reward/discount targets
- intervention segment semantics

For evaluation we want a separate, trajectory-level view.

### Planned Layering

#### A. Universal Data-Side Window Builder

Add a new narrow utility module under `agent_factory/data` for replay evaluation windows.

Recommended path:

- `agent_factory/data/eval/window_builder.py`

Responsibility:

- load one replay trajectory from a single `.h5`
- convert it into a unified raw trajectory dict
- build anchor-based legal window batches

This layer must remain algorithm-agnostic.

#### B. Critic-Side Eval Interface

Introduce a lightweight critic evaluation base interface.

Suggested path:

- `agent_factory/agents/mixins/critic/base_eval.py`

Responsibility:

- define shared helper methods or a default `eval_batch()` that raises `NotImplementedError`

Then let concrete critic mixins implement:

- `CPIQLCriticMixin.eval_batch(...)`
- future `IQLCriticMixin.eval_batch(...)`
- future `ITQCCriticMixin.eval_batch(...)`

#### C. Script Layer

Add universal entry script:

- `agent_factory/script/eval_universal.py`

Responsibility:

- parse args
- load config
- resolve checkpoint
- resolve replay `.h5`
- build agent
- iterate all anchor windows
- call `critic.eval_batch`
- aggregate arrays
- draw and save plots
- optionally save `npz`

## File-Level Implementation Plan

### 1. Add Critic Eval Base

Create:

- `agent_factory/agents/mixins/critic/base_eval.py`

Provide:

- a lightweight `CriticEvalMixinBase`
- default `eval_batch()` that raises `NotImplementedError`
- optional small shared helpers for result formatting if helpful

This base should be non-invasive and should not disturb current config MRO behavior.

### 2. Add Replay Trajectory Window Builder

Create:

- `agent_factory/data/eval/window_builder.py`

Functions to include:

- `load_replay_eval_trajectory(h5_path, include_rgb=True) -> dict`
- `build_eval_window_batches(trajectory, obs_horizon, pred_horizon) -> List[dict]`

Expected raw trajectory dict fields:

- `obs`
- `action`
- optional metadata like `success`, `terminated`, `truncated`, `intervention`, `env_meta`

The batch builder should:

- iterate anchor timesteps over action length
- construct legal observation window
- construct legal action chunk
- return one batch per anchor timestep

### 3. Implement CPIQL Critic Eval

Modify:

- `agent_factory/agents/mixins/critic/CPIQL.py`

Add:

- `eval_batch(batch, only_obs=True) -> Dict[str, torch.Tensor or np.ndarray]`

Behavior:

- preprocess `observations`
- consume `action`
- compute `V(k=0)`, `V(k=1)`, `critic_gap`
- if `only_obs=False`, compute `Q(k=0)`, `Q(k=1)`, `adv(k=0)`

Output shape for each returned metric should be batch-aligned and easy for script aggregation.

### 4. Add Universal Eval Script

Create:

- `agent_factory/script/eval_universal.py`

Required CLI args:

- `--config`
- `--ckpt_path`
- `--save_dir`
- `--replay_h5_path`
- `--only_obs`

Optional or defaulted args:

- if `--ckpt_path` absent, read from `cfg.train.ckpt_path`
- if `--replay_h5_path` absent, read from `cfg.dataset.replaybuffer.replaybuffer_path`
- optional `--save_npz`

Flow:

1. load config
2. resolve replay single-file path
3. resolve checkpoint path
4. build agent and load ckpt
5. load raw replay trajectory
6. build window batches
7. loop over all anchor batches and call `agent.eval_batch(...)` or critic-side eval entry
8. aggregate returned arrays
9. plot by `figure:*/*` key convention
10. save figures
11. optionally save `npz`

### 5. Optional Script Export Hook

If needed, update:

- `agent_factory/script/__init__.py`

Only if we want the new script helpers to be importable from the package level.

## Key Risks And Mitigations

### Risk 1: Mixing Training Dataset Semantics With Eval Semantics

If evaluation is implemented by hacking current training dataset `__getitem__`, logic will become hard to reason about and fragile.

Mitigation:

- keep eval trajectory/window builder separate from training sample dataset logic

### Risk 2: Shape Drift Across Algorithms

Different critics may expect different input structure, even if they all use `obs` and `action`.

Mitigation:

- centralize legal window construction in one place
- keep `eval_batch()` algorithm-specific
- keep universal script dependent only on returned dict keys

### Risk 3: Ambiguous Time Axis

If `obs` length and `action` length are treated inconsistently, figures become misleading.

Mitigation:

- align first version strictly to action timesteps
- document `frame` as anchor timestep

### Risk 4: Hidden Failure When Replay Path Is Missing

Evaluation should never silently fall back to expert dataset or other sources.

Mitigation:

- explicit validation and fail-fast error messages

### Risk 5: Large RGB Replay Files Slow Evaluation

Visual replay trajectories can be expensive.

Mitigation:

- first version only load modalities required by current config
- keep `include_rgb` behavior aligned with model config

## First-Version Acceptance Criteria

Implementation is considered complete when:

1. `eval_universal.py` can evaluate one replay `.h5` trajectory end-to-end.
2. It correctly uses `--replay_h5_path` or falls back to `cfg.dataset.replaybuffer.replaybuffer_path`.
3. It correctly uses `--ckpt_path` or falls back to `cfg.train.ckpt_path`.
4. It builds legal windows using `obs_horizon` and `pred_horizon`.
5. `CPIQLCriticMixin.eval_batch()` returns expected value metrics.
6. Figures are saved successfully under the chosen output directory.
7. The script does not depend on folder-based replay browsing.

## Explicit Non-Goals For This Round

- expert trajectory evaluation
- folder-level replay selection UX
- dataset-wide aggregate statistics
- multi-critic implementation in one pass
- mandatory raw metric dump
- intervention-aware segmented plotting

## Recommended Build Order

1. Add replay single-file trajectory loader.
2. Add legal window batch builder.
3. Add critic eval base mixin.
4. Implement `CPIQLCriticMixin.eval_batch()`.
5. Implement `eval_universal.py`.
6. Test on one known replay `.h5`.
7. Verify saved figures and metric curves manually.
