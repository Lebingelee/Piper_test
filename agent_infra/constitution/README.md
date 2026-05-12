# agent_infra Constitution

本宪法约束 `agent_infra/` 内所有环境库、硬件接口、任务环境接入、数据采集、轨迹回放和后处理代码。它面向人类开发者和多个 code agent 协同开发场景，优先保证边界清晰、硬件安全、数据契约稳定、跨库协作可推理。

本文使用强约束词：

- **MUST**：必须遵守，违反即视为架构错误。
- **MUST NOT**：禁止行为，除非先修改宪法并说明迁移策略。
- **SHOULD**：默认应遵守，只有在明确记录理由时可以偏离。
- **MAY**：允许行为，但不构成默认推荐。

## 1. 核心使命

`agent_infra` 的唯一核心使命是提供可复用的机器人环境、任务环境、数据采集、数据回放和硬件抽象。

`agent_infra` **MUST** 以环境契约为中心，而不是以某个算法为中心。Piper、Realman、robomimic task env 或后续任何机器人环境，都必须通过统一的 Gymnasium 风格接口、`meta_keys`、raw trajectory 数据格式和可选 wrapper 与上层算法库连接。

`agent_infra` **MUST NOT** 实现算法训练循环、神经网络模块、actor/critic 更新、RL 损失函数、模型 checkpoint 管理或算法专用 replay buffer。这些属于 `agent_factory`。

## 2. 目录职责

### 2.1 `agent_infra/base_robot_env.py`

`base_robot_env.py` **MUST** 定义所有机器人环境的最低接口：

- `reset(seed=None, options=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `_setup_hardware()`
- `_get_obs()`
- `_apply_action(action)`
- `close()`
- `get_safe_action()`
- `meta_keys`

`BaseRobotEnv` **MUST NOT** 依赖任何具体硬件 SDK、相机 SDK、算法库、数据集库或训练框架。

### 2.2 `agent_infra/<Robot>_Env/Env/`

`Env/` **MUST** 只放环境实现和环境 wrapper。

环境实现 **MUST** 输出嵌套结构：

```text
obs/state/<state_key>
obs/rgb/<camera_role>        # optional
obs/depth/<camera_role>      # optional
obs/under_control/<arm_name> # optional, HITL only
action/<action_key>
```

环境实现 **MUST** 保持 `meta_keys` 与实际 `obs/action` 一致。任何新增观测或动作字段，都必须同步更新 `meta_keys`。

环境实现 **MUST NOT** 为某个算法硬编码 batch shape、obs horizon、pred horizon、normalization、reward shaping 或 dataset slicing。

### 2.3 `agent_infra/<Robot>_Env/Env/utils/`

`Env/utils/` **MUST** 放机器人底层逻辑、机械臂封装、相机 wrapper 基类、位姿工具和环境内部辅助函数。

硬件 SDK 直接调用 **SHOULD** 限定在这里或 `Camera/` 下。最终用户入口层不应散落 SDK 调用。

### 2.4 `agent_infra/<Robot>_Env/Camera/`

`Camera/` **MUST** 只负责相机设备发现、启动、停止、取帧和设备级配置。

`Camera/` **MUST NOT** 处理算法需要的时序堆叠、训练 batch、action、reward 或 replay buffer。

相机输出的原始颜色图像 **SHOULD** 在环境 wrapper 中转成 `CHW uint8`，并通过 `obs/rgb/<role>` 暴露。若接入第三方环境天然使用 `HWC`，必须在该环境接入文档中明确说明转换位置。

### 2.5 `agent_infra/<Robot>_Env/Record/`

`Record/` **MUST** 负责 raw trajectory 采集、raw trajectory 后处理、raw trajectory 回放，以及 H5 和 LeRobot 之间的工程转换。

`Record/` **MUST NOT** 实现算法训练 dataset、torch sampler、Q/V/actor 更新逻辑。这些属于 `agent_factory/data` 和 `agent_factory/agents`。

### 2.6 `agent_infra/<Robot>_Env/Script/`

`Script/` **MUST** 放可执行 smoke test、采集、回放、转换、硬件探测脚本。

脚本 **MUST** 尽量通过配置文件和命令行参数选择硬件，不得把机器特定 CAN、IP、serial number 硬编码进 Python 逻辑。

### 2.7 `agent_infra/robomimic_env/`

`agent_infra/robomimic_env/` 的定位是：把 robomimic 中的任务环境接入本项目的环境构建体系，用本项目设计的算法在 robomimic task env 中测试。

该目录 **MUST** 实现 robomimic task env 到 `BaseRobotEnv` / Gymnasium 风格接口 / `meta_keys` 的适配。

该目录 **MUST NOT** 适配 robomimic 算法到本项目。robomimic 算法训练、robomimic checkpoint、robomimic policy wrapper 若后续需要，属于 `agent_factory/integrations/` 或单独实验目录。

该目录 **MUST** 把 robomimic task env 当作一种环境来源，而不是把整个项目迁移到 robomimic 的算法体系。

## 3. 环境分层规则

环境分层 **MUST** 遵循以下方向：

```text
hardware/device wrapper
  -> robot core env
  -> optional teleop/HITL env
  -> optional camera/vision wrapper
  -> final user env entry
```

以 Piper 为目标范式：

```text
PiperArm
  -> PiperBaseEnv
  -> PiperEnv
  -> PiperCameraWrapper
  -> SinglePiperEnv / DualPiperEnv
```

机械臂 core env **MUST** 负责：

- 连接和关闭机械臂。
- 读取本体状态。
- 构建 `obs/state` 与 `action` 的 `meta_keys`。
- 执行动作。
- 提供 `reset_to_state` 和 `get_safe_action`。

机械臂 core env **MUST NOT** 负责：

- 相机启动和图像处理。
- 算法训练。
- replay buffer。
- 第三方数据集读取。

相机 wrapper **MUST** 负责：

- 基于配置装配相机。
- 维护最新图像缓存。
- 注入 `obs/rgb` 和可选 `obs/depth`。
- 扩展视觉 `meta_keys`。

teleop/HITL env **MUST** 负责：

- 人类接管状态。
- `under_control` 观测。
- `info["actual_action"]`。
- `info["policy_action"]`。
- `info["intervened"]` 与 `info["intervened_map"]`。
- `info["action_type"]` 与 `info["action_type_map"]`。

teleop/HITL env **MUST NOT** 把算法策略、训练 loop 或 loss 写进环境层。

## 4. `meta_keys` 契约

所有可被上层算法消费的环境 **MUST** 提供 `env.unwrapped.meta_keys`：

```python
meta_keys = {
    "obs": {
        "state": {
            "<state_key>": (<dim>,),
        },
        "rgb": {
            "<camera_role>": (3, H, W),
        },
        "depth": {
            "<camera_role>": (1, H, W),
        },
        "under_control": {
            "<arm_name>": (1,),
        },
    },
    "action": {
        "<action_key>": (<dim>,),
    },
}
```

`meta_keys` **MUST** 描述单帧、未堆叠、未 batch 的环境原始结构。

多臂命名 **MUST** 使用 prefix 规则：

- 单臂：`joint_pos`、`arm`、`gripper`
- 双臂或多臂：`left_joint_pos`、`right_joint_pos`、`left_arm`、`right_arm`

key ordering **MUST** 由消费者根据 `meta_keys` 做确定性排序或按文档约定读取。环境层不得依赖 Python dict 插入顺序作为唯一契约。

`meta_keys` **MUST NOT** 描述算法 batch 字段，如 `observations`、`next_observations`、`k`、`progress_return`。这些属于 `agent_factory` dataset。

## 5. 数据宪法

### 5.1 H5 是主数据流

本项目当前主数据流 **MUST** 以 `.h5` 为中心。

`agent_infra` 采集侧 **MUST** 以 `raw_structure_traj` 为主。每条轨迹文件 **MUST** 保存在任务路径下，保留嵌套结构和原始 `meta/env_meta`。

raw structure trajectory 的推荐结构：

```text
<task_root>/
  h5_raw/
    traj_<id>_<time>.h5

traj file:
  obs/
    state/<state_key>
    rgb/<camera_role>
    depth/<camera_role>        # optional
    under_control/<arm_name>   # optional
  action/<action_key>
  meta/env_meta
  attrs:
    success
```

`agent_infra` **MAY** 提供 merge 和 flatten 后处理工具，但后处理工具必须明确输入和输出，不得覆盖 raw trajectory。

### 5.2 `agent_factory` 读取规则

`agent_factory` 的 expert dataset **MUST** 读取：

- `merge_structure_trajs`
- 或 `merge_flatten_trajs`

`agent_factory` 的 replay buffer **MUST** 读取每一条 `flatten_traj`。

`agent_infra` **MUST NOT** 为某个具体算法生成算法私有字段，例如 `k`、`progress_return`、`segment_type`。这些字段由 `agent_factory/data` 根据轨迹信号构建。

### 5.3 LeRobot 数据流

LeRobot **MAY** 作为并行数据流。

当前原则：LeRobot 数据流 **SHOULD NOT** 在 `agent_infra` 中做算法预处理。若以 LeRobot 采集，默认交给 LeRobot dataset 负责读取和采样。

LeRobot 与 H5 之间的转换 **MAY** 存在于 `Record/postprocess.py`，但转换结果必须保留 `env_meta` 或足够推断 `env_meta` 的 metadata。

### 5.4 第三方环境数据

robomimic task env、仿真环境或后续第三方任务环境的数据 **MUST** 先适配到本项目环境契约。是否写入 H5 由实验需要决定。

若第三方环境原生数据格式与本项目不同，适配层 **MUST** 明确：

- 原始 observation key。
- 本项目 `obs/state` 映射。
- 本项目 `obs/rgb` 映射。
- 本项目 `action` 映射。
- reward、terminated、truncated、success 的来源。

## 6. 与 `agent_factory` 的边界

`agent_infra` **MAY** 被 `agent_factory/env`、`agent_factory/runner` 或明确标记的 integration 层调用。

`agent_infra` **MUST NOT** import：

- `agent_factory.agents`
- `agent_factory.modules`
- `agent_factory.data.impl`
- `agent_factory.runner`
- 任何具体算法实现

`agent_infra` **MAY** import 通用第三方库，如 `gymnasium`、`numpy`、`h5py`、相机 SDK、机器人 SDK、LeRobot dataset 工具。但依赖必须位于功能边界内，并在缺失时给出清晰错误或降级路径。

## 7. Runner 连接规则

`runner` 是两套宪法之间最容易产生耦合的连接点。

环境为 runner 提供的最小接口 **MUST** 是：

- `reset`
- `step`
- `close`
- `action_space`
- `observation_space`
- `meta_keys`
- `get_safe_action`
- `switch_passive(mode: str)` for real hardware envs

HITL 环境 **MUST** 在 `info` 中提供执行动作和接管信号：

```python
info["actual_action"]
info["policy_action"]
info["intervened"]
info["intervened_map"]
info["action_type"]
info["action_type_map"]
```

runner **MUST NOT** 调用机器人私有 SDK、私有 CAN/IP 逻辑或相机私有句柄。若 runner 需要动作扁平化，必须通过 wrapper 暴露的 `flatten_action` 或 `meta_keys`。

## 8. 硬件安全原则

硬件安全规则 **SHOULD** 由具体硬件 env 实现，但以下原则为全局要求：

- 所有真实硬件 env **MUST** 提供 `get_safe_action`。
- 所有真实硬件 env **MUST** 提供动作下发安全锁 `switch_passive(mode: str)`。
- `switch_passive("true")` **MUST** 表示允许 `step()` 中的动作真正下发到硬件。
- `switch_passive("false")` 或默认未开启状态 **MUST** 表示 `step()` 仍可执行频率控制、观测读取和 `info` 构建，但不得调用底层硬件动作下发接口。
- 动作下发安全锁 **MUST** 位于环境内部最终硬件 dispatch 边界，例如 `_apply_action()`，并覆盖 policy、safe action、teleop/HITL 覆盖后的动作等所有 `step()` 动作来源。
- 动作下发安全锁 **MUST NOT** 被 runner、agent、dataset 或算法层绕过；上层只能通过 `env.unwrapped.switch_passive(...)` 显式开启或关闭。
- replay 脚本 **MUST** 支持从首帧状态 reset 或显式声明不 reset。
- 真实硬件动作下发 **SHOULD** 有控制频率限制。
- 真实硬件 reset **SHOULD** 支持配置化初始位姿。
- 任何 destructive 或不可逆硬件动作 **MUST** 通过脚本名、参数名或日志明确提示。

不同硬件构型差异很大，宪法不强行规定统一的物理安全策略；但任何硬件 env 都必须把自身安全假设写入对应 README 或 Config 注释。

## 9. 禁止项

`agent_infra` 中 **MUST NOT** 出现以下行为：

- 在环境代码中训练神经网络。
- 在环境代码中实现算法 loss。
- 在机器人 env 中直接依赖某个 agent class。
- 为某个算法硬编码 `obs_horizon`、`pred_horizon`、`batch_size`。
- 把机器特定 CAN、IP、camera serial 硬编码到生产 Python 模块。
- 覆盖 raw trajectory。
- 在无 `meta_keys` 更新的情况下新增 obs/action 字段。
- 在 `robomimic_env` 中实现 robomimic 算法适配。

## 10. 新环境接入检查表

新增环境或任务环境接入前，开发者或 code agent **MUST** 检查：

- 是否有最终用户入口 env。
- 是否有 `meta_keys`。
- `reset/step/close/get_safe_action` 是否可用。
- obs/action 是否与 `meta_keys` 一致。
- 单臂/多臂命名是否稳定。
- 是否有最小 smoke test。
- 是否说明 raw trajectory 是否可采集。
- 是否说明 LeRobot 数据流是否支持。
- 是否说明与 runner 的交互方式。
- 是否未引入算法层依赖。
