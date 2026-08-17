# Runner Bridge Constitution

本文件约束 `agent_factory/runner/` 如何连接 agent、control transform、wrapper 后的环境与在线 H5 回流数据。runner 是执行动作归属、接管标签和训练复用最容易混淆的位置，因此必须单独规定。

## 1. Runner 的使命

runner **MUST** 负责：

- 持有已构建 agent 与 env。
- 维持控制循环并异步获取 action chunk。
- 在无可执行 policy chunk 或风险/重规划阶段使用 safe action 或明确暂停。
- 在策略控制模式与环境控制模式不一致时执行正向转换。
- 记录实际执行动作、状态转移、接管和必要 metadata。

runner **MUST NOT** 负责：

- actor/critic loss 或 dataset slicing。
- CPIQL progress label 或 ITQC cond 重标注。
- 私有机器人 SDK、相机 SDK 或硬件 action key 解析。
- 在缺少 metadata 的情况下猜测控制模式转换。

## 2. Env 与 Agent 接口

runner 接受的 env **MUST** 满足 Gymnasium 风格接口：

```python
obs, info = env.reset(...)
next_obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

用于真实环境或 metadata-driven action 的 wrapper/env **SHOULD** 提供：

```python
env.get_safe_action()                  # 或 wrapper get_safe_flatten_action()
env.flatten_action(dict_action)
env.unwrapped.meta_keys
env.unwrapped.get_env_metadata()
env.unwrapped.get_control_state()      # control mode 需要转换时
```

支持在线执行的 agent **MUST** 提供：

```python
action_chunk = agent.sample_action(obs_batch, ...)
# action_chunk: [B, pred_horizon, action_dim] in agent_control_mode
```

部署 runner 使用风险判断时，agent **MAY** 提供：

```python
risky = agent.get_risk(obs_batch, action_batch)
```

## 3. 观测流

环境 wrapper 向 runner 输出 canonical online obs：

```python
{
    "rgb": Tensor[T, 3 * num_cameras, H, W],  # optional
    "state": Tensor[T, proprio_dim],            # optional
    "depth": Tensor[T, C, H, W],                # optional
}
```

`UnifiedFrameStackWrapper` 负责时间维；runner 推理线程只增加 batch 维，再传给 agent：

```text
[T, ...] -> [1, T, ...]
```

runner **MUST NOT** 读取 raw nested obs 的机器人私有路径。保存 H5 时，runner 当前取 frame stack 的最新帧，使 `obs` 序列成为可被 replay dataset 重建窗口的逐步记录。

## 4. 动作流与控制模式

动作流必须按以下顺序执行：

```text
agent.sample_action()
  -> denormalized policy chunk in agent_control_mode
  -> forward_transform_action() when modes differ
  -> flat env action
  -> MetadataAdapterWrapper.step()
  -> nested env-native action
  -> env executes/arbitrates action
  -> info["actual_action"] preferred for storage
```

runner 保存：

```text
action        = actual executed flat action in env_control_mode
policy_action = planned action trace when the runner supports it
```

`BaseRunner._transform_policy_chunk_to_env_chunk()` 当前只支持 `control/` 已声明的转换。若需要 `delta_joint`、`relative_pose_chunk` 或其他新组合，必须先扩展 `control/action_transform.py` 与数据逆转换，再允许 runner 使用。

当 control mode 不一致时，env **MUST** 公开当前位姿与 metadata；正向转换不能仅凭扁平 action 猜测基准姿态。

## 5. Safe Action 与动作类型

safe action **MUST** 优先从 wrapper 链获取扁平动作，以保持与 `MetadataAdapterWrapper` 的 action 顺序一致。底层环境仅在 wrapper 不提供接口时作为回退。

当前 runner 本地 fallback 类型含义为：

```text
0 = policy chunk
1 = safe / replanning / waiting action
2 = human override request
3 = reserved or environment-defined extension
```

环境在 `info` 中返回 `action_type` 或 `action_type_map` 时，其结果优先作为 `action_type` / `env_action_type` 保存；runner 自己选择的来源可另存为 `runner_action_type`。

HITL 环境 **SHOULD** 返回：

```python
info["actual_action"]
info["policy_action"]          # optional when environment owns arbitration
info["intervened"]
info["intervened_map"]         # multi-arm optional
info["action_type"]
info["action_type_map"]        # multi-arm optional
```

接管发生时，保存的 `action` **MUST** 使用 actual action，不得将原计划 policy action 冒充为专家动作。

## 6. `BaseRunner`

`BaseRunner` 当前实现：

- 用独立推理线程消费最新 obs，产出 action chunk。
- 每次最多执行 `act_horizon` 个 policy 动作，然后重新规划。
- `runner.no_safe_action_gap=False` 时，在等待新 chunk 期间 step safe action。
- `runner.no_safe_action_gap=True` 时，在等待新 chunk 期间不调用 `env.step()`。
- 达到 episode 终止/截断后保存 root-level 单轨迹 H5。
- 通过 `buffer_capacity + redundancy_margin` 管理保存目录的 FIFO 文件数。

`BaseRunner` H5 **SHOULD** 包含：

```text
action                  # [T, A], actual env-native action
action_type             # [T]
runner_action_type      # optional
env_action_type         # optional
obs/rgb                  # [T+1, C, H, W], optional
obs/state                # [T+1, D], optional
obs/depth                # [T+1, C, H, W], optional
rewards                  # [T]
success                  # [T]
terminated               # [T]
truncated                # [T]
intervention             # [T]
meta/env_cfg
meta/env_meta
attrs/success
attrs/length
```

## 7. `HITLRunner`

`HITLRunner` 在 `BaseRunner` 上增加状态机：

```text
POLICY -> HUMAN_OVERRIDE -> REPLAN -> POLICY
```

它 **MUST**：

- 接管进入时清空 stale policy chunk。
- 接管退出时以最新 obs 强制重新规划。
- 同时记录 actual action 与 policy action。
- 对齐 `obs` 长度为 `T+1`、其他字段长度为 `T`。

其内存轨迹扩展字段为：

```text
policy_action
runner_action_type
env_action_type
```

当前 `HITLRunner.run()` 在 episode 完成后调用 `_prepare_traj_for_save()`，但未自动调用 `_save_trajectory()`。依赖在线文件回流的任务 **MUST** 使用显式保存流程或 `HITLDeployRunner`，不得假定 generic HITL 已写盘。

## 8. `HITLDeployRunner`

`HITLDeployRunner` 是当前用于部署采集并落盘的 HITL runner。它在 `HITLRunner` 上增加：

- 独立风险检查线程，调用 `agent.get_risk()`。
- 初始化、开始、结束、继续、退出与遥操操作员控制。
- risk trigger、teleop boundary 与 manual/max-step 边界记录。
- 单 session 文件中的多个 `traj_<id>` 子轨迹。

部署 session 根级字段：

```text
meta/env_cfg
meta/env_meta
attrs/success
attrs/length
attrs/num_traj
```

每个子轨迹 group **MUST** 保存：

```text
traj_<id>/
  action
  policy_action
  action_type
  runner_action_type
  env_action_type
  obs/rgb | obs/state | obs/depth
  rewards
  success
  terminated
  truncated
  intervention
  risk
  attrs/success
  attrs/length
  attrs/boundary_reason
```

`boundary_reason` 当前可包括：

```text
teleop_start
teleop_end
risk_trigger
risk_continue
manual_save
max_step_save
max_step_tail
episode_end
```

其中 `teleop_start` 边界会被 CPIQL replay 读取为 intervention boundary 候选。新增边界原因 **SHOULD** 保持可解释，并明确其是否影响 dataset segment 语义。

## 9. 在线数据回流

runner 到训练数据的回流契约为：

```text
runner H5 action in env_control_mode
  -> data.registry builder
  -> optional segment splitting / signal construction
  -> inverse_transform_action() when required
  -> batch["action"] in agent_control_mode
```

将部署 session 用作 CPIQL replay 时：

- 每个子轨迹 **MUST** 有 `success`、`terminated`、`truncated`、`intervention`。
- root `meta/env_meta` **MUST** 足够支持 action/state 顺序及必要的控制逆变换。
- risk 与 boundary reason **MAY** 作为诊断或 segment 解释信息保留。

将 runner 输出用作 ITQC replay 时，dataset 必须仍能构造 `value`、`discount` 与 `cond` 工作流。

## 10. Checkpoint 与失败处理

部署前，checkpoint 中的动作 normalizer **MUST** 有效。`ensure_action_normalizer_ready()` 可从配置指定的 expert H5 直接读取 action 重新拟合；失败时再回退到 dataset builder。

直接 H5 拟合路径不会应用 dataset 的 `inverse_transform_action()`。当 expert H5 保存的是 `env_control_mode` 动作且与 `agent_control_mode` 不同时，现有 helper 不得原样用于部署恢复；实现 **MUST** 改为通过可执行控制转换的 dataset 路径拟合，或使用已处于策略控制空间的训练 H5。

runner 推理失败时 **MUST NOT** 发送未初始化动作。实现应降级到安全动作、保持等待，或明确终止 rollout。

runner 结束时 **MUST** 停止推理、风险和监听线程。

runner action 维度与环境不一致时，当前实现可在执行/保存边界裁剪或补零；新的生产路径 **SHOULD** 在配置或 metadata 校验阶段提前消除这种不一致，并清晰报告降级。
