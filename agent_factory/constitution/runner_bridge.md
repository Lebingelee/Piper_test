# Runner Bridge Constitution

本文件约束 `agent_factory/runner/` 如何连接 `agent_factory` 算法和 `agent_infra` 环境。runner 是两套宪法之间最容易产生跨级耦合的位置，因此必须单独规定。

## 1. Runner 的使命

runner **MUST** 负责：

- 加载或持有已构建 agent。
- 持有已构建 env。
- 控制执行频率。
- 组织推理线程。
- 将 agent 输出动作发送给 env。
- 处理 safe action。
- 记录在线轨迹。
- 在 HITL 场景中处理接管状态。

runner **MUST NOT** 负责：

- 机器人 SDK 调用。
- 相机 SDK 调用。
- 机械臂物理 reset 细节。
- 算法 loss。
- dataset slicing。
- 第三方环境内部字段解析。

## 2. Env 接口要求

runner 可接受的 env **MUST** 满足：

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

env 或其 wrapper 链 **SHOULD** 提供：

```python
env.get_safe_action()
env.flatten_action(dict_action)
env.unwrapped.meta_keys
```

runner **MUST** 优先使用 wrapper 暴露的 `get_safe_action` 和 `flatten_action`。只有在 wrapper 不存在时，才可回退到 `env.unwrapped.get_safe_action()`。

## 3. 动作流

agent 输出的部署动作默认是 flattened action：

```text
np.ndarray[action_dim]
```

若底层 env 使用 dict action，转换 **MUST** 由 `MetadataAdapterWrapper` 或同等 wrapper 完成。

runner **MUST NOT** 自己根据机器人名称手写动作切片规则。动作切片规则属于 `meta_keys` 和 wrapper。

## 4. 观测流

runner 接收的 obs **MUST** 是经过 wrapper 后的 canonical online obs：

```python
{
    "rgb": Tensor or ndarray[T, C, H, W],   # if visual
    "state": Tensor or ndarray[T, D],       # if proprio
}
```

runner **MUST NOT** 访问 raw env obs 的 `obs/state/<key>`、`obs/rgb/<role>` 细节，除非该 runner 明确标记为 infra-side recorder，而不是 algorithm runner。

## 5. HITL 信号

HITL runner **MUST** 支持环境提供的接管信号：

```python
info["actual_action"]
info["policy_action"]
info["intervened"]
info["intervened_map"]
info["action_type"]
info["action_type_map"]
```

字段含义：

- `actual_action`：真实下发到环境的动作。若人类接管，它不同于 policy action。
- `policy_action`：agent 或 runner 原本计划下发的动作。
- `intervened`：单臂或聚合接管标记。
- `intervened_map`：多臂逐臂接管标记。
- `action_type`：单值或聚合动作类型。
- `action_type_map`：多臂逐臂动作类型。

动作类型编码 **SHOULD** 使用：

```text
0 = policy
1 = expert / human
2 = sync / safe / replanning
```

如果某个环境需要不同编码，必须在对应 env README 和 runner 适配代码中说明。

## 6. 在线轨迹保存

runner 保存在线轨迹时 **MUST** 明确其输出类型是 `flatten_traj` 或可被转换成 `flatten_traj` 的格式。

runner 保存的在线 H5 **SHOULD** 包含：

```text
obs/rgb
obs/state
action
policy_action          # optional
action_type
runner_action_type     # optional
env_action_type        # optional
rewards
terminated
truncated
meta/env_cfg
meta/env_meta
attrs/success
attrs/length
```

runner **MUST NOT** 覆盖 `agent_infra` 采集得到的 raw trajectory。

## 7. LeRobot 与 runner

runner 当前 **SHOULD NOT** 直接写 LeRobot 原生数据集，除非新增专门 LeRobot runner 或 recorder。

若新增 LeRobot runner，必须保持 LeRobot dataset 原生语义，不得先强制转换成 H5 再伪装为 LeRobot。

## 8. 失败处理

runner 推理失败时 **MUST** 降级到 safe action 或跳过动作，不得发送未初始化动作。

runner action 维度不匹配时 **SHOULD** 报告清晰 warning，并只在明确安全的场景下裁剪或补零。

runner 结束时 **MUST** 停止推理线程和监听线程。

