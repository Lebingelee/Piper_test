# agent_factory Constitution

本宪法约束 `agent_factory/` 内所有算法、模型模块、数据集、replay buffer、runner、配置系统、环境 wrapper 和第三方算法/环境集成入口。它面向多 code agent 并行开发，目标是让算法库和环境库通过稳定契约协作，而不是互相穿透。

本文使用强约束词：

- **MUST**：必须遵守。
- **MUST NOT**：禁止行为。
- **SHOULD**：默认应遵守，偏离时必须说明理由。
- **MAY**：允许行为。

## 1. 核心使命

`agent_factory` 的核心使命是提供算法构建、训练、推理、数据读取、runner 执行和第三方任务环境测试能力。

`agent_factory` **MUST** 以算法契约和数据契约为中心。它可以消费 `agent_infra` 提供的环境，但不得把机器人硬件细节写入算法实现。

`agent_factory` **MUST NOT** 直接管理真实硬件 SDK、CAN、IP、相机设备句柄或机械臂 reset 物理过程。这些属于 `agent_infra`。

## 2. 目录职责

### 2.1 `agent_factory/agents/`

`agents/` **MUST** 负责 agent 生命周期、算法组合、actor/critic mixin、训练入口和 checkpoint 保存加载。

`agents/base_agent.py` **MUST** 定义所有 agent 的最低接口：

- `_init_components()`
- `_init_optimizers()`
- `start_train(dataset, additional_args=None)`
- `sample_action(obs, ...)`，若 agent 支持部署
- `save(path, meta=None)`
- `load(path)`
- `required_keys`

`agents/impl/` **MUST** 只放具体算法组合类，例如 Diffusion + CPIQL。组合类负责选择 mixin、组织训练 loop、注册 agent。

`agents/mixins/` **MUST** 放可组合的 actor/critic/normalization 能力。Mixin **MUST** 声明自身 dataset 需求：

```python
REQUIRED_KEYS = {"observations", "action", ...}
```

算法实现 **MUST NOT** import `agent_infra.Piper_Env`、`agent_infra.Realman_Env` 或任何具体硬件 env。

### 2.2 `agent_factory/modules/`

`modules/` **MUST** 只放纯神经网络、编码器、actor policy、critic network 和通用数学模块。

`modules/` **MUST NOT** 读取 H5 文件、创建 Gym env、调用 runner、调用机器人 SDK、保存轨迹或解析任务目录。

网络模块 **SHOULD** 只依赖 tensor shape 和 config，不依赖具体 robot name。

### 2.3 `agent_factory/data/`

`data/` **MUST** 负责算法训练数据读取、trajectory slicing、reward/progress 信号构造、normalization 和 replay buffer。

`data/base.py` **MUST** 只负责通用 trajectory discovery 和 H5 handle 管理，不应包含算法私有 slicing。

`data/impl/` **MUST** 放算法或数据协议专用 dataset/replay buffer。

`data/normalization/` **MUST** 放动作或观测 normalization 工具。

`data/` **MUST NOT** 连接真实机器人、启动相机、读取 CAN/IP、执行 env.step。

### 2.4 `agent_factory/env/`

`env/` **MAY** 依赖 `agent_infra` 来创建环境，这是 `agent_factory` 中允许跨入 infra 的主要位置。

`env/wrappers.py` **MUST** 提供通用 wrapper，例如：

- `MetadataAdapterWrapper`
- `UnifiedFrameStackWrapper`
- 第三方环境 observation adapter

`MetadataAdapterWrapper` **MUST** 通过 `env.unwrapped.meta_keys` 展平 obs/action，不得为某个机器人硬编码 action key。

`env/env_factories.py` **MUST** 是根据配置创建环境的统一入口。新增环境来源时，应优先新增 `library == "<name>"` 分支，并调用对应 infra final env entry。

`env/` **MUST NOT** 实现算法训练逻辑。

### 2.5 `agent_factory/runner/`

`runner/` **MUST** 负责在线部署、策略执行、HITL 状态机、推理线程、控制频率、在线轨迹保存。

runner **MUST** 只通过环境公开接口交互：

- `reset`
- `step`
- `close`
- `get_safe_action`
- `action_space`
- `observation_space`
- wrapper 暴露的 `flatten_action`
- `info["actual_action"]` 等标准字段

runner **MUST NOT** 调用机器人 SDK、相机 SDK、私有 arm handle 或私有 camera handle。

runner 是连接 `agent_factory` 与 `agent_infra` 的关键桥梁，详细规则见 [runner_bridge.md](runner_bridge.md)。

### 2.6 `agent_factory/config/`

`config/` **MUST** 定义全局 schema、默认值、配置加载、配置一致性校验和环境默认值推断。

配置系统 **MAY** 从 `agent_infra` 环境 YAML 推断维度，但 **MUST** 保持用户显式配置优先。

配置系统 **MUST NOT** 启动真实硬件或 import 重型硬件 SDK。

### 2.7 `agent_factory/integrations/`

若需要接入第三方训练框架、策略格式或任务生态，**SHOULD** 新增 `agent_factory/integrations/<name>/`。

注意：robomimic task env 本身属于 `agent_infra/robomimic_env`；只有当我们需要处理 robomimic 算法、checkpoint、policy adapter 或 dataset converter 时，才应放入 `agent_factory/integrations/robomimic/`。

## 3. 与 agent_infra 的边界

`agent_factory` **MAY** 通过以下位置依赖 `agent_infra`：

- `agent_factory/env/env_factories.py`
- `agent_factory/env/wrappers.py`
- `agent_factory/runner/`，仅通过 env 对象公开接口
- 明确命名的 `agent_factory/integrations/`

`agent_factory` 其他位置 **MUST NOT** 依赖具体机器人环境。

允许：

```python
# env factory 中允许
from agent_infra.Piper_Env.Env.single_piper_env import SinglePiperEnv
```

禁止：

```python
# agent 或 module 中禁止
from agent_infra.Piper_Env.Env.utils.piper_arm import PiperArm
from agent_infra.Realman_Env.Env.realman_env import RealManEnv
```

算法代码 **MUST** 只依赖 canonical batch：

```python
batch["observations"]
batch["next_observations"]
batch["action"]
batch["reward"]
batch["terminated"]
batch["discount"]
```

算法代码 **MUST NOT** 依赖 raw H5 内的 `obs/state/<key>` 或 `action/<key>` 路径。

## 4. 数据宪法

### 4.1 H5 主数据流

本项目当前训练主数据流 **MUST** 以 `.h5` 为中心。

`agent_infra` 负责采集 `raw_structure_traj`。`agent_factory` 负责读取或构造训练 dataset。

`agent_factory` 中：

- `expert_dataset` **MUST** 读取 `merge_structure_trajs` 或 `merge_flatten_trajs`。
- `replaybuffer` **MUST** 读取每一条 `flatten_traj`。
- 算法私有字段 **MUST** 在 `agent_factory/data` 中构造。

### 4.2 canonical training batch

所有算法训练 **MUST** 消费 canonical batch，而不是 raw env obs。

基础字段：

```python
{
    "observations": {
        "rgb": Tensor[B, T, C, H, W],      # optional
        "state": Tensor[B, T, D],          # optional
        "depth": Tensor[B, T, C, H, W],    # optional
    },
    "action": Tensor[B, pred_horizon, A],
}
```

RL/critic 字段：

```python
{
    "next_observations": {...},
    "reward": Tensor[B, 1],
    "discount": Tensor[B, 1],
    "terminated": Tensor[B, 1],
    "truncated": Tensor[B, 1],
    "success": Tensor[B, 1],
}
```

算法扩展字段 **MAY** 包括：

```python
"k"
"cond"
"progress_return"
"progress_mask"
"progress_weight"
"intervention"
"segment_type"
"segment_terminal_reward"
```

新增算法 mixin **MUST** 通过 `REQUIRED_KEYS` 声明所需字段。

### 4.3 raw / merged / flattened 的所有权

`raw_structure_traj`：

- 由 `agent_infra` 采集。
- 保留嵌套 obs/action。
- 每条轨迹单独保存。
- 不应被算法直接修改。

`merge_structure_trajs`：

- 可由 `agent_infra` 后处理生成。
- 用于统一任务数据和保留结构。
- 可被 `agent_factory` expert dataset 读取。

`merge_flatten_trajs`：

- 可由后处理生成。
- 供 `agent_factory` 快速读取。
- obs/action 已按 `meta_keys` 确定性拼接。

`flatten_traj`：

- 每条轨迹一个扁平文件。
- replay buffer 默认读取单位。
- 必须包含训练所需基础信号或足够构造基础信号的信息。

### 4.4 LeRobot 数据流

LeRobot **MAY** 作为与 H5 并行的数据流。

当前原则：LeRobot 数据流 **SHOULD** 交给 LeRobot dataset 原生读取和采样，`agent_factory` 不应预先强制转换其内部格式。

若算法需要 LeRobot 数据，**SHOULD** 新增专门 dataset wrapper，且 wrapper **MUST** 输出 canonical training batch。

## 5. 算法实现规则

新增算法 **MUST** 遵循：

1. 在 `agents/mixins/actor` 或 `agents/mixins/critic` 实现可复用能力。
2. 在 `agents/impl` 组合 mixin 和 `BaseAgent`。
3. 在 `agents/registry.py` 注册。
4. 在 config schema 中提供 dataclass 配置。
5. 在 dataset 中声明和验证所需字段。

算法 **MUST NOT**：

- 在 `impl` 中解析 raw H5 路径。
- 在 `modules` 中访问 config 文件系统。
- 在 loss 中读取 env 对象。
- 在 actor/critic 中调用 `env.step`。
- 在训练 loop 中启动相机或真实机器人。

## 6. 观测与动作规则

`agent_factory` canonical obs **MUST** 使用 tensor。

当前视觉默认约定：

```text
rgb: [B, T, 3*num_cameras, H, W]
state: [B, T, proprio_dim]
```

`BaseStateEncoder` 当前假设 RGB 以 `3*k` 通道拼接多视角。新增多模态 encoder 时 **MUST** 明确更新本宪法或补充专章。

动作默认约定：

```text
action: [B, pred_horizon, action_dim]
```

动作 normalization **MUST** 在 agent 或 data normalization 层处理。环境层不得为了某个算法改变物理动作量纲。

## 7. 配置规则

`GlobalConfig` **MUST** 是算法、环境、dataset、runner 的共同顶层配置。

环境维度，如 `action_dim`、`proprio_dim`、`num_cameras`，**MAY** 从 `env.env_config_path` 指向的 infra YAML 推断。

推断值 **MUST NOT** 覆盖用户显式配置。

新增 config 字段 **SHOULD** 放在明确 dataclass 中，不应散落在代码里用魔法字符串读取。

## 8. 禁止项

`agent_factory` 中 **MUST NOT** 出现以下行为：

- 具体算法 import 具体机器人 env。
- 神经网络模块读取 H5 或启动 env。
- dataset 启动真实硬件。
- runner 调用私有硬件 SDK。
- env wrapper 写入算法 loss。
- config loader 启动 env 或硬件。
- 为某个机器人在 actor/critic 中硬编码 action key。
- 在 LeRobot 数据流中强制预处理破坏其原生 dataset 语义。

## 9. 新功能检查表

新增算法、dataset、runner 或 env integration 前，开发者或 code agent **MUST** 检查：

- 是否放在正确目录。
- 是否违反 `agent_infra` 与 `agent_factory` 依赖方向。
- 是否声明 `REQUIRED_KEYS`。
- 是否输出 canonical batch。
- 是否通过 config schema 暴露参数。
- 是否保留用户显式配置优先。
- 是否不依赖具体机器人私有字段。
- 是否有最小 smoke test 或 py_compile 验证。

