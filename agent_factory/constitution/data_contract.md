# Data Contract Constitution

本文件约束 `agent_factory/data/`、converter、runner 保存轨迹和算法 batch 之间的数据流。任何修改数据逻辑的 code agent **MUST** 说明自己处理的是持久化环境动作，还是已转换到策略控制空间的训练动作。

## 1. 数据层级

当前 H5 主路径为：

```text
raw / structured trajectory
  -> converter 或 MetadataAdapter/runner
  -> flattened H5 trajectory / deployment session
  -> data.registry 选择算法 dataset
  -> trajectory slicing + control inverse transform + algorithm labels
  -> canonical training batch
```

`raw_structure_traj`、`merge_structure_trajs` 与 `merge_flatten_trajs` 可作为采集或离线转换来源；当前训练入口实际以 registry builder 可读取的 H5 为准。

## 2. 元数据与确定性顺序

凡是 structured obs/action 被展平，或环境动作需要转换到策略控制模式，数据 **MUST** 保存 `meta/env_meta`，或在单轨迹 group 内保存等价 metadata。

最低结构为：

```python
{
    "obs": {
        "rgb": {"camera_role": shape, ...},
        "state": {"state_key": shape, ...},
        "depth": {"camera_role": shape, ...},  # optional
    },
    "action": {"action_key": shape, ...},
    "env_control_mode": "...",                 # required when modes may differ
    "controller_backend": "...",               # optional traceability
    "control": {
        "arm_entries": [...],                  # recommended for multi-arm transforms
    },
}
```

展平顺序当前由 key 的排序顺序决定。converter、wrapper 和 dataset **MUST** 对同一 metadata 使用相同顺序，不得按运行时字典插入顺序改变维度含义。

若执行动作与策略动作的 control mode 不同，metadata **MUST** 指明或可推导：

- 每个 arm action 的扁平切片。
- 用于恢复当前末端位姿的 `state_pose_key`，或 `eef_pos_key` 与 `eef_quat_key`。
- 多臂下每条 arm entry 对应的 action key。

缺少这些信息时，dataset 或 runner **MUST** 明确报错，不得猜测姿态切片。

## 3. 持久化 H5 结构

### 3.1 Structured 输入

通用 `data.impl.ExpertDataset` 支持 structured 输入，典型结构为：

```text
meta/env_meta
traj_<id>/
  obs/rgb/<camera_role>
  obs/state/<state_key>
  action/<action_key>
  terminated | done | success
```

structured 输入必须携带 metadata，以确定 obs/action 拼接顺序。

### 3.2 Flattened 单轨迹或多轨迹 H5

flattened trajectory 的通用形态为：

```text
meta/env_meta
traj_<id>/                       # 或由文件根直接表示单条轨迹
  obs/rgb                        # [T or T+1, C, H, W], optional
  obs/state                      # [T or T+1, D], optional
  obs/depth                      # optional
  action                         # [T, A], executed env-native action
  rewards                        # optional, runner trace
  success                        # protocol-specific
  terminated
  truncated
  intervention                   # online CPIQL replay required
```

`converter.flatten_raw_h5()` **MUST**：

- 写入全局 `meta/env_meta`。
- 按 metadata 展平 `action` 与 `state`。
- 将 RGB 从 HWC 转为 CHW 后按通道拼接。
- 保留存在的 `rewards`、`success`、`terminated`、`truncated` 与 attrs。

converter 当前以 H5 写模式打开 `output_path`。调用方 **MUST** 使用新的输出文件路径，不得把 raw 输入或已有唯一副本作为输出目标；后续 converter 改造 **SHOULD** 在覆盖前显式拒绝或要求确认。

converter 不会替算法凭空生成 CPIQL 必需信号。转换后若将文件交给 CPIQL builder，调用方 **MUST** 确认 `success`、`terminated` 与 `truncated` 均已存在。

### 3.3 Runner 输出

`BaseRunner` 保存一个 root-level flattened trajectory；`HITLDeployRunner` 保存一个包含多个 `traj_<id>` 子轨迹的 session。两者保存的 `action` 均是执行后的 env-native action，并保存 `meta/env_meta`。

详细字段及边界切分规则见 [runner_bridge.md](runner_bridge.md)。

## 4. Dataset Registry 契约

通用训练脚本 **MUST** 通过 `data.registry.build_training_bundle()` 构造数据，不能根据 agent 名称在脚本内硬编码 dataset class。

当前选择规则：

```text
agent type 含 "itqc"   -> dataset_type = "diffusion_itqc"
agent type 含 "cpiql"  -> dataset_type = "cpiql"
其他未显式配置类型      -> dataset_type = "cpiql"
```

用户显式配置的 `cfg.dataset.dataset_type` 优先。

`cfg.train.dataset_key` 只允许：

```text
expert_dataset
replaybuffer
expert_dataset+replaybuffer
```

选择 replay 相关 view 时 **MUST** 提供 `dataset.replaybuffer.folder_path` 或 `dataset.replaybuffer.replaybuffer_path`。

## 5. `diffusion_itqc` 数据协议

`DiffusionITQCDataset` 当前复用通用 `data.impl.ExpertDataset`：

- expert 输入支持 `flat`、`structured` 或 `auto`。
- obs/action 会缓存在内存中。
- 切片输出 `observations`、`next_observations` 与 action chunk。
- RL 信号由 `compute_rl_signals()` 与 `compute_n_step_signals()` 构造。
- `cond` 初始为零，并由 ITQC critic 的 `relabel_data()` 更新。

其 replay builder 使用 `DiffusionITQCReplayBuffer`，从文件目录读取 flattened trajectory，并兼容 `action` / `actions` 名称。当前传入单个 replay 文件时，builder 会选择该文件所在目录作为 replay 来源；调用方 **MUST** 避免目录中混入非本轮数据。

ITQC 所需 batch 字段为：

```python
{
    "observations": {...},
    "next_observations": {...},
    "action": Tensor[B, pred_horizon, A],
    "reward": Tensor[B, 1],
    "terminated": Tensor[B, 1],
    "discount": Tensor[B, 1],
    "value": Tensor[B, 1],
    "cond": Tensor[B, 1],
}
```

## 6. `cpiql` 数据协议

### 6.1 Expert 输入

`CPIQLExpertDataset` 读取 flattened H5。每条轨迹 **MUST** 包含：

```text
obs
action
success
terminated
truncated
```

`obs/rgb`、`obs/state` 与 `action` 可已是扁平 dataset；实现也可按 `env_meta` 读取 grouped leaves。离线 expert 不要求 `intervention`，每条来源轨迹按整体标记为 `success` 或 `failure` segment。

当前 CPIQL 公共实现会预载 state、action、metadata 与标签；当 `include_rgb=True` 且内部加载模式为 `lazy` 时，RGB 只在 `__getitem__()` 请求观测窗口时从 H5 读取。新增并行加载或缓存策略 **MUST** 保持样本字段和窗口索引不变。

### 6.2 Replay 输入与切分

`CPIQLFileReplayBuffer` 的在线 replay trajectory **MUST** 额外包含：

```text
intervention       # [T] bool
```

有 intervention 的轨迹会按连续来源切分为子轨迹：

```text
autonomous -> intervention -> autonomous -> ...
```

当前 segment 类型包括：

```text
success
intervention
rollout_failure
boundary_failure
post_intervention_failure
failure
unknown
```

自主段若紧邻专家接管，或来源 group 的 `boundary_reason` 为 `teleop_start`，应标记 `segment_end_is_intervention_boundary=True`，并以可配置伪终止奖励初始化。critic **MAY** 使用 `refresh_replay_intervention_rewards()` 根据价值估计更新这些边界奖励。

### 6.3 Progress 与分布字段

CPIQL dataset **MUST** 构造 critic 所需字段：

```python
{
    "progress_return": Tensor[B, 1],
    "progress_mask": Tensor[B, 1],
    "progress_weight": Tensor[B, 1],
    "failure_rank": Tensor[B, 1],
    "is_success_segment": Tensor[B, 1],
    "is_failure_segment": Tensor[B, 1],
    "segment_end_is_intervention_boundary": Tensor[B, 1],
}
```

可选诊断/训练字段为：

```python
"segment_type"
"segment_terminal_reward"
"success"
"truncated"
"intervention"
"intervention_segment"
"cond"
```

dataset 只提供 segment 与 progress 的语义事实；不同 `k` 下的分布 mask、anchor 开关及 loss 权重属于 `CPIQLCriticMixin`，不得移入 recorder 或 env。

## 7. Canonical Training Batch

所有 dataset builder 输出给 agent 的观测和动作必须遵循：

```python
{
    "observations": {
        "rgb": Tensor[B, obs_horizon, 3 * num_cameras, H, W],  # optional
        "state": Tensor[B, obs_horizon, proprio_dim],            # optional
        "depth": Tensor[B, obs_horizon, C, H, W],                # optional
    },
    "next_observations": {...},                                  # critic required
    "action": Tensor[B, pred_horizon, action_dim],
}
```

时间边界 padding 规则 **MUST** 可复现：

- 观测窗口在轨迹开头/末尾重复可用边界帧。
- action chunk 不足 `pred_horizon` 时重复末动作，或在明确支持的增量协议下使用已定义 padding。
- `next_observations` 按 `act_horizon` 前移。

## 8. 控制空间与归一化

H5 中 runner 记录的：

```text
action = executed action in env_control_mode
```

dataset 向 agent 输出的：

```text
batch["action"] = training action in agent_control_mode
```

当二者不同，dataset **MUST** 在 slicing 后通过 `inverse_transform_action()` 转换 action chunk；runner **MUST** 在执行前通过 `forward_transform_action()` 转换 policy chunk。

动作 normalizer 的顺序为：

```text
持久化 env action
  -> inverse control transform
  -> agent-control action
  -> actor normalizer.normalize()
  -> network
```

推理时顺序反向：

```text
network
  -> actor normalizer.denormalize()
  -> agent-control action
  -> forward control transform
  -> env action
```

normalizer **MUST NOT** 被用来掩盖 control mode 或动作维度不一致。

## 9. Eval Window 契约

`data/eval/window_builder.py` 当前复用 CPIQL flattened reader 从一个 replay H5 读取轨迹，因此输入仍 **MUST** 带有 `success`、`terminated` 与 `truncated`。它按每个 action timestep 构造：

```python
{
    "frame": Tensor[],
    "observations": {...},
    "action": Tensor[pred_horizon, action_dim],
}
```

若 rollout 保存了 `T+1` 个 observations 对应 `T` 个 actions，评估窗口只以 `obs[:T]` 对齐 action 帧。critic `eval_batch()` **MUST** 接受这种只用于推理可视化的窗口，而不得要求训练专用 label。

当前 eval window builder 不执行 `inverse_transform_action()`。仅评估 observation value 时不受 action 坐标系影响；评估 Q/adv action metric 时，输入 H5 的动作 **MUST** 已处于 agent control mode，或调用方必须先提供等价转换后的评估路径。

## 10. 非 H5 扩展

LeRobot 或其他原生数据格式当前未接入上述 registry 主链。若新增此类数据源：

- dataset wrapper **MUST** 输出相同 canonical training batch。
- 原生数据语义 **MUST** 保留，不得先强制伪装为 H5 再宣称为原生读取。
- 若需要在线 recorder，必须单独声明其持久化格式与控制空间归属。

## 11. 数据转换禁止项

数据处理代码 **MUST NOT**：

- 无提示覆盖 raw trajectory。
- 无 metadata 仍改变 action/state 拼接顺序。
- 无提示丢弃相机、夹爪、终止或接管信号。
- 将执行动作误标为策略控制空间动作。
- 把 CPIQL segment/reward 语义硬编码进环境 recorder。
