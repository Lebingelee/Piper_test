# robomimic_env Constitution

本文件约束 `agent_infra/robomimic_env/`。该目录尚未实现时，本文件作为接入前设计基线。

## 1. 定位

`agent_infra/robomimic_env` **MUST** 把 robomimic task env 接入本项目环境体系，让本项目的 `agent_factory` 算法能够在 robomimic 任务环境中测试。

`agent_infra/robomimic_env` **MUST NOT** 让本项目去适配 robomimic 的算法框架。robomimic 在这里是任务环境来源，不是上层算法库。

## 2. 允许实现的功能

该目录 **MAY** 实现：

- robomimic task env 创建器。
- robomimic observation 到本项目 `obs/state`、`obs/rgb`、`obs/depth` 的映射。
- robomimic action space 到本项目 `action/<key>` 的映射。
- `meta_keys` 构建。
- Gymnasium 风格 `reset/step/close` 适配。
- success、reward、terminated、truncated 信号映射。
- 用于 smoke test 的脚本。
- 用于 `agent_factory.env.env_factories` 调用的最终 env entry。

## 3. 禁止实现的功能

该目录 **MUST NOT** 实现：

- robomimic BC、IQL、CQL、IRIS、HBC 等算法训练逻辑。
- robomimic checkpoint 到本项目 agent checkpoint 的转换。
- 本项目 actor/critic 模块。
- 本项目 replay buffer。
- torch DataLoader 或算法采样器。
- robot-specific hardware SDK 逻辑。

## 4. 适配输出契约

robomimic task env 适配后 **MUST** 满足 `agent_infra` 通用环境契约：

```python
obs = {
    "state": {
        "<state_key>": np.ndarray,
    },
    "rgb": {
        "<camera_role>": np.ndarray,  # CHW uint8 preferred
    },
}

action = {
    "<action_key>": np.ndarray,
}
```

`meta_keys` **MUST** 精确描述单帧 obs/action。

若 robomimic 原始图像为 `HWC`，适配层 **SHOULD** 转为本项目 env raw 约定的 `CHW uint8`。若暂时不转换，必须在该 env 的 README 中明确说明，并在 `agent_factory/env` wrapper 中完成转换。

## 5. 与 agent_factory 的连接

`agent_factory/env/env_factories.py` **MAY** 新增 `library == "robomimic"` 分支来创建 robomimic task env。

该分支 **MUST** 只依赖 `agent_infra.robomimic_env` 的最终 env entry 和 `MetadataAdapterWrapper` 等通用 wrapper，不得绕过 `meta_keys` 直接拼接 robomimic 私有字段。

## 6. 配置要求

robomimic task env 接入配置 **SHOULD** 通过 `env.env_config_path` 或等价统一配置入口提供。

配置 **MUST** 至少能够表达：

- task/env name。
- observation modality。
- camera roles。
- state keys。
- action keys 和维度。
- max episode steps。
- reward/success 信号来源。

## 7. 测试要求

最小接入完成后 **MUST** 提供 smoke test，覆盖：

- 构造 env。
- 打印 `meta_keys`。
- `reset()`。
- `get_safe_action()`。
- 至少一次 `step()`。
- 通过 `MetadataAdapterWrapper` 后 action flatten/unflatten 正常。

