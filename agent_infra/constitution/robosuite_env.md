# robosuite_env Constitution

本文件约束 `agent_infra/robosuite_env/`。该目录尚未实现时，本文件作为 Robosuite 仿真器接入前的设计基线。

## 1. 定位

Robosuite 是基于 MuJoCo 的仿真环境库。robomimic 集成在 Robosuite 等仿真任务之上，并提供模仿学习、强化学习、部署算法和数据生态。

`agent_infra/robosuite_env` **MUST** 接入 Robosuite task env，使其符合 `agent_infra` 环境宪法和 `agent_factory` 算法测试接口。

`agent_infra/robosuite_env` **MUST NOT** 适配 robomimic 算法。robomimic 算法、checkpoint、policy wrapper 或数据集读取若后续需要，属于 `agent_factory/integrations/robomimic/` 或专门实验层。

## 2. 当前依赖状态

当前项目环境尚未安装：

```text
robosuite
robomimic
mujoco
```

在依赖安装前，本目录 **SHOULD** 先完成接口设计、配置 schema、文档和可选 mock smoke test。

任何 import `robosuite`、`mujoco` 或 `robomimic` 的代码 **MUST** 延迟到函数内部或构造器内部，并在缺失依赖时给出清晰错误。顶层 import **MUST NOT** 让整个 `agent_infra` 或 `agent_factory` 无法导入。

## 3. 接入目标

Robosuite 接入层 **MUST** 输出本项目标准环境接口：

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
env.unwrapped.meta_keys
env.get_safe_action()
```

Robosuite 原始 observation **MUST** 被适配为：

```python
obs = {
    "state": {
        "<state_key>": np.ndarray,
    },
    "rgb": {
        "<camera_role>": np.ndarray,
    },
    "depth": {
        "<camera_role>": np.ndarray,
    },
}
```

`depth` 可选。若 Robosuite task 不启用视觉，`obs/rgb` 可缺失，但 `meta_keys` 必须与实际输出一致。

## 4. Human In The Loop

Robosuite 仿真测试阶段 **SHOULD NOT** 默认实现 human in the loop。

Robosuite env **MAY** 不提供：

- `obs/under_control`
- `info["intervened"]`
- `info["intervened_map"]`
- `info["actual_action"]` 中的专家覆盖动作

若 env 不支持 HITL，runner **MUST** 按纯 policy 执行路径处理。

若未来需要 scripted expert、teleop expert 或 dataset replay expert，必须作为单独 wrapper 或 runner mode 实现，不得污染 Robosuite 基础 env。

## 5. `meta_keys` 规则

Robosuite 接入层 **MUST** 构建单帧 `meta_keys`：

```python
meta_keys = {
    "obs": {
        "state": {
            "robot0_eef_pos": (3,),
            "robot0_eef_quat": (4,),
            "...": (...,),
        },
        "rgb": {
            "agentview": (3, H, W),
            "robot0_eye_in_hand": (3, H, W),
        },
    },
    "action": {
        "arm": (A_arm,),
        "gripper": (A_gripper,),
    },
}
```

实际 key **MUST** 由 Robosuite task config 决定，不得在算法层硬编码。

图像 layout **SHOULD** 在 Robosuite env adapter 中转换为 `CHW uint8`，以对齐现有 `MetadataAdapterWrapper`、H5 和 `agent_factory` encoder 约定。

若保留 Robosuite 原生 `HWC`，则 adapter **MUST** 在 README 中明确，并在 `agent_factory/env` wrapper 中完成转换。

## 6. 动作规则

Robosuite 接入层 **MUST** 显式区分：

- `env_control_mode`：对上暴露给 `agent_factory` 的动作语义。
- `controller_backend`：Robosuite 内部具体 controller 名，例如 `BASIC`。

`controller_backend` **MUST NOT** 直接替代 `env_control_mode`。

Robosuite 原始动作通常是连续向量。接入层 **MUST** 明确：

- 原始 action dim。
- 本项目 action key 划分。
- gripper 是否单独 key。
- action 是否已经归一化。
- action_space 的 low/high。

基础策略：

```python
action = {
    "arm": np.ndarray,
    "gripper": np.ndarray,
}
```

若某个 Robosuite controller 使用单一整向量且无法可靠拆分，**MAY** 使用：

```python
action = {
    "action": np.ndarray,
}
```

但必须在 `meta_keys["action"]` 和配置中明确。

第一阶段 Robosuite **MUST** 对上稳定暴露：

- `env_control_mode = absolute_pose`
- `controller_backend = BASIC`

若底层 backend 本质仍是 delta pose，Robosuite env **MUST** 在 env 内部完成：

```text
absolute_pose (external env action)
  -> backend delta pose
  -> robosuite controller step
```

`get_safe_action()` **MUST** 返回 `env_control_mode` 语义动作，而不是 backend 语义动作。

## 7. 配置规则

Robosuite 接入 **SHOULD** 使用统一配置入口：

```text
env.library = "robosuite"
env.env_id = "<task_name>"
env.env_config_path = "agent_infra/robosuite_env/Config/<task>.yaml"
```

配置文件 **SHOULD** 描述：

- task name。
- robot name。
- controller config。
- camera names。
- image size。
- whether use_object_obs。
- whether use_camera_obs。
- reward shaping。
- horizon / max episode steps。
- action key schema。
- state key allowlist。

## 8. 数据规则

Robosuite 仿真 rollout **MAY** 写入 H5 主数据流。

若使用 scripted / TAMP 风格采集，robosuite 侧 **SHOULD** 通过独立 `Solution` 基类及其 task solver 子类产出动作，而不是把任务求解逻辑硬编码进 recorder 或 env。

该 `Solution` 层 **SHOULD** 输出 `env_control_mode` 语义动作；对于当前 Panda `BASIC` backend，这意味着 solver 输出 `absolute_pose`，再由 env 内部转换为 backend delta pose。

若写 H5，输出 **SHOULD** 优先是 `flatten_traj` 或可直接 merge 的 structure traj，因为仿真环境通常用于算法快速验证，不需要保留真实硬件完整 raw signal。

若为了和真实机器人数据统一，Robosuite env **MAY** 保存 `raw_structure_traj`，但必须保留 `meta/env_meta`。

若需要可复现实验回放，raw trajectory **SHOULD** 同时保存 reset 所需的 simulator state，例如 `meta/reset_state`。

## 9. 与 agent_factory 的连接

`agent_factory/env/env_factories.py` **MAY** 新增：

```python
library == "robosuite"
```

该分支 **MUST** 调用 `agent_infra.robosuite_env` 的最终 env entry，再通过 `MetadataAdapterWrapper` 和 `UnifiedFrameStackWrapper` 对齐算法接口。

算法层 **MUST NOT** 直接 import robosuite task env。

## 10. 最小验收

Robosuite 接入第一阶段完成后 **MUST** 通过：

- 依赖缺失时，导入 `agent_infra` 和 `agent_factory` 不失败。
- 安装依赖后，可创建一个 Robosuite task env。
- `reset()` 返回符合 `meta_keys` 的 obs。
- `get_safe_action()` 返回合法 action。
- `step(get_safe_action())` 成功执行。
- `MetadataAdapterWrapper` 可 flatten obs/action。
- `agent_factory` 中至少一个 BC/Diffusion 类算法可拿该 env 的 observation/action dim 做 smoke test。
