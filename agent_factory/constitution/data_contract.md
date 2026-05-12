# Data Contract Constitution

本文件约束 `agent_factory/data/`、`agent_infra/*/Record/` 与 runner 保存轨迹之间的数据交互。

## 1. 数据层级

本项目数据分为四层：

```text
raw_structure_traj
  -> merge_structure_trajs
  -> merge_flatten_trajs / flatten_traj
  -> canonical training batch
```

任何 code agent 修改数据逻辑时 **MUST** 明确自己正在处理哪一层。

## 2. raw_structure_traj

`raw_structure_traj` 属于 `agent_infra`。

结构：

```text
obs/state/<state_key>
obs/rgb/<camera_role>
obs/depth/<camera_role>
obs/under_control/<arm_name>
action/<action_key>
meta/env_meta
attrs/success
```

raw trajectory **MUST** 尽量保留真实环境结构。它不是算法 batch。

## 3. merge_structure_trajs

`merge_structure_trajs` **MAY** 由后处理生成，用于把多条 raw structure trajectory 汇总成一个 H5。

它 **MUST** 保留嵌套结构和 `meta/env_meta`。

它 **MAY** 增加：

```text
success
terminated
truncated
```

## 4. merge_flatten_trajs

`merge_flatten_trajs` **MUST** 将 obs/action 按 `meta_keys` 确定性拼接。

推荐结构：

```text
traj_<id>/
  obs/rgb      # [T, C, H, W]
  obs/state    # [T, D]
  action       # [T, A]
  success
  terminated
  truncated
meta/env_meta
```

若保留 `obs/depth`，必须说明其是否独立字段还是已拼入 `rgb`。

## 5. flatten_traj

`flatten_traj` 是 replay buffer 默认读取单位。

每条 `flatten_traj` **MUST** 至少包含：

```text
obs/state or obs/rgb
action
terminated
truncated
```

若算法需要 success、intervention、action_type，缺失字段必须由 dataset 明确降级或报错。

## 6. canonical training batch

canonical training batch 属于 `agent_factory/data` 输出，不得由 `agent_infra` 直接保存。

基础行为克隆 batch：

```python
{
    "observations": {
        "rgb": Tensor[B, T, C, H, W],
        "state": Tensor[B, T, D],
    },
    "action": Tensor[B, pred_horizon, A],
}
```

critic / RL batch：

```python
{
    "observations": {...},
    "next_observations": {...},
    "action": Tensor[B, pred_horizon, A],
    "reward": Tensor[B, 1],
    "discount": Tensor[B, 1],
    "terminated": Tensor[B, 1],
}
```

算法扩展 batch 必须由对应 dataset 明确构造。

## 7. LeRobot 数据流

LeRobot 数据流 **MAY** 绕过 H5 canonical path。

若使用 LeRobot，默认原则是：

- `agent_infra` 可采集 LeRobot 数据。
- `agent_factory` 可新增 LeRobot dataset wrapper。
- LeRobot wrapper 输出 canonical training batch。
- 不在 `agent_infra` 对 LeRobot 做算法预处理。

## 8. 数据转换规则

转换函数 **MUST** 满足：

- 输入路径和输出路径显式。
- 不覆盖原始文件。
- 保留或生成 `env_meta`。
- 明确图像 layout：`CHW` 或 `HWC`。
- 明确动作拼接顺序。
- 明确 success/terminated/truncated 的生成规则。

转换函数 **MUST NOT**：

- 在无提示下丢弃相机。
- 在无提示下丢弃 gripper。
- 在无提示下改变动作量纲。
- 在无提示下覆盖 raw trajectory。

