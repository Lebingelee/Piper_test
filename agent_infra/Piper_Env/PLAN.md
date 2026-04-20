# Piper Env Reconstruction Plan

## 1. Plan Purpose

本计划用于指导 `agent_infra/Piper_Env` 重构为一个更适合 VLA 算法训练、部署、数据采集与轨迹回放的 infra 库。

本计划面向的主要读者是：

- 接手 `infra` 的开发者
- 调用 `Piper` env API 做策略训练与部署的算法开发者

本计划不会要求用户频繁修改 infra 内部代码，而是强调：

- 清晰稳定的 env 分层
- 配置驱动的硬件与相机装配
- 一致的 `meta_keys`
- 可复用的数据采集、处理与回放链路

## 2. Confirmed Goals

当前已经确认的目标如下：

- `piper_base_env.py` 只负责机械臂核心逻辑，不负责视觉。
- `piper_base_env.py` 负责机械臂连接、`obs/state/...`、`action/...`、`meta_keys`、`reset/reset_to_state`、动作下发等能力。
- 相机走 `wrapper` 路线，由 wrapper 负责扩展 `obs/rgb/...` 相关 `meta_keys`。
- 第一优先支持绝对 `joint` 控制模式，并保留 `pose` 接口。
- `gripper` 保持 SDK 原始物理量纲，不做归一化；Piper `AgxGripper.move_gripper(width)` 的真实范围为 `0.0 ~ 0.1 m`。
- 控制模式优先从 config 读取；若用户在创建环境时传入参数，则覆盖 config。
- 单臂与双臂都要支持。
- 双臂场景下有两对 `leader-follower`，当前采用 Linux ifname 友好的短名接入：`can_ml` / `can_sl` / `can_mr` / `can_sr`。
- `action` 仍然只面向 `follower`；`leader` 的控制通过 env 内部模式开关触发。
- `under_control` 要保留，用于标识当前哪个机械臂正在被人类接管。
- `tele_enabled` 通过 `T` 键切换。
- 第一阶段相机只先落地 `RealSense`，配置字段至少包含 `camera_type`、`role`、`serial_number`。
- 奥比中光后续在独立对话或分支中实现，当前只需要为其预留可扩展架构。
- 数据链路需要覆盖：
  - 数据采集
  - 数据后处理
  - 轨迹回放
- 数据后处理至少包含：
  - 多条 `.h5` 轨迹合并成一条 `.h5`
  - `.h5` 控制模式切换
  - `.h5` 与 `LeRobot` 的转换
- 数据格式同时考虑 `.h5` 与 `LeRobot`，但第一优先级更偏 `LeRobot`。
- `Script` 目录需要拆为 `single/` 和 `dual/`。
- `.sh` 脚本需要预填参数，用户主要通过修改 `.sh` 中参数完成自定义。
- 需要新增 `agent_infra/Piper_Env/README.md`，当前已完成第一版并需要随实机验证持续维护。
- 需要新增本计划文件 `agent_infra/Piper_Env/PLAN.md`。

## 3. Target Architecture

### 3.1 Target Layering

目标分层暂定如下：

1. `agent_infra/base_robot_env.py`
   - 提供通用 env 抽象接口。

2. `agent_infra/Piper_Env/Env/utils/piper_base_env.py`
   - Piper 机械臂核心基类。
   - 仅处理机械臂本体与本体感知。
   - 仅构建 `obs/state/...` 与 `action/...` 的 `meta_keys`。

3. Piper 机械臂核心 env 层
   - 单臂核心 env
   - 双臂核心 env
   - 负责 teleop、`under_control`、`step` 行为、机械臂 reset 与 SDK 装配。

4. 相机 wrapper 层
   - 负责配置驱动的相机装配。
   - 负责将 `obs/rgb/...` 注入 obs。
   - 负责扩展 `meta_keys["obs"]["rgb"]`。
   - 后续可同时支持 `RealSense` 与奥比中光。

5. 最终用户使用层
   - 单臂最终环境
   - 双臂最终环境
   - 对外暴露稳定 API，用于 VLA 训练、部署、录制与回放。

### 3.2 Naming Status

当前文件命名与最终目标之间还存在一个待收敛点：

- 当前已有 `piper_env.py`
- 后续目标中，用户也提到了 `single_piper_env.py` 与 `dual_piper_env.py`

因此实现时需要明确：

- `piper_env.py` 是否保留为“单臂核心 env”
- `single_piper_env.py` 是否作为“camera-wrapped 的最终单臂环境”
- 或者统一改为更清晰的命名体系

该问题在正式实施前需要先收敛，但不会影响本计划的大方向。

### 3.3 Confirmed Wrapper Chain

当前已经确认的包装关系如下：

`single_piper_env / dual_piper_env -> camera_wrapper -> PiperEnv -> PiperBaseEnv -> BaseRobotEnv`

这条链路意味着：

- `single_piper_env.py` 和 `dual_piper_env.py` 是最终用户入口层。
- `camera_wrapper` 是视觉扩展层，不承载机械臂核心逻辑。
- `PiperEnv` 是机器人 teleop / 专家介入逻辑层。
- `PiperBaseEnv` 是机械臂核心抽象层。
- `BaseRobotEnv` 继续作为最底层通用机器人 env 基类。

### 3.4 Class Responsibility Draft

本节用于明确后续改代码时每一层“负责什么、不负责什么”。

#### 3.4.1 `BaseRobotEnv`

职责：

- 提供统一 env 抽象接口。
- 约束 `_setup_hardware()`、`_get_obs()`、`_apply_action()`、`close()`。
- 提供基础 `reset()` / `step()` 调用框架。

不负责：

- 任何 Piper 专用逻辑
- 任何相机逻辑
- 任何 teleop / 专家介入逻辑

#### 3.4.2 `PiperBaseEnv`

职责：

- 管理 Piper 机械臂硬件连接与释放。
- 管理单臂 / 双臂 follower 侧核心动作执行。
- 管理本体观测读取：
  - `joint_pos`
  - `joint_vel`
  - `ee_pose`
  - `gripper_pos`
- 构建纯机械臂层的 `meta_keys`：
  - `obs/state/...`
  - `action/...`
- 提供 `reset()`、`reset_to_state()`、`get_safe_action()`。
- 统一多臂 prefix 规则。

不负责：

- 相机初始化与视觉 obs
- `obs/rgb/...` 与视觉 `meta_keys`
- teleop 状态机
- `under_control`
- 键盘监听
- 专家介入覆盖 action 的逻辑

设计约束：

- `PiperBaseEnv` 必须保持“无视觉依赖”。
- `PiperBaseEnv` 必须尽量保持对 VLA 算法透明，即仅暴露稳定的机器人状态与动作契约。

#### 3.4.3 `PiperEnv`

职责：

- 在 `PiperBaseEnv` 之上增加 teleop / 专家介入能力。
- 管理 leader 读取线程。
- 管理 `tele_enabled` 开关与 `T` 键切换。
- 维护 `under_control` 状态。
- 在 `step()` 中根据当前介入状态决定：
  - 正常执行传入 action
  - 或用专家动作覆写 follower action
- 提供面向录制与训练的 `actual_action` / `intervened` 信息。

不负责：

- 相机实例化
- 图像采集线程
- `obs/rgb/...` 的实际注入
- 最终用户脚本入口组织

设计约束：

- `PiperEnv` 仍然是“机器人层 env”，不是最终视觉完整 env。
- `PiperEnv` 必须能在没有相机 wrapper 的情况下独立工作。

#### 3.4.4 `camera_wrapper`

职责：

- 基于 config 装配具体相机实现。
- 管理多品牌相机抽象与后续扩展。
- 管理图像采集线程 / 缓存。
- 在 obs 中注入：
  - `obs/rgb/...`
  - 后续可能扩展 `obs/depth/...`
- 追加视觉相关 `meta_keys`。

不负责：

- 机械臂 SDK 初始化
- follower 动作下发
- teleop 状态机
- `under_control`

设计约束：

- `camera_wrapper` 必须是“可插拔”的。
- 第一阶段优先支持 `RealSense`，但结构上不能绑死单一品牌。
- wrapper 与 core env 之间应只通过 obs / meta_keys / config 契约交互。

#### 3.4.5 `single_piper_env.py`

职责：

- 作为单臂最终用户入口。
- 负责读取单臂 config。
- 负责实例化：
  - 单臂 `PiperEnv`
  - 相机 wrapper
- 对外暴露“开箱可用”的单臂环境。

不负责：

- 重复实现机械臂核心逻辑
- 重复实现相机线程逻辑

设计约束：

- `single_piper_env.py` 更偏“装配层”而不是“逻辑层”。
- 用户应优先通过这个入口创建单臂最终环境，而不是手工拼装底层类。

#### 3.4.6 `dual_piper_env.py`

职责：

- 作为双臂最终用户入口。
- 负责读取双臂 config。
- 负责实例化：
  - 双臂 `PiperEnv`
  - 相机 wrapper
- 对外暴露“开箱可用”的双臂环境。

不负责：

- 重复实现双臂机械臂核心逻辑
- 重复实现相机线程逻辑

设计约束：

- 双臂最终 env 仍然遵循相同包装链。
- 双臂 action 默认仍只面向左右 follower。
- leader 相关逻辑继续收敛在 `PiperEnv` 内部。

### 3.5 Assembly Flow Draft

后续推荐的装配流程如下。

单臂：

1. 读取单臂 config
2. 构建单臂 `PiperEnv`
3. 根据 config 构建 `camera_wrapper`
4. 返回最终单臂 env

双臂：

1. 读取双臂 config
2. 构建双臂 `PiperEnv`
3. 根据 config 构建 `camera_wrapper`
4. 返回最终双臂 env

这意味着：

- 单双臂入口文件只负责装配，不重复造轮子。
- `PiperEnv` 是机械臂 teleop 核心层。
- 视觉层可替换。

## 4. Meta Keys Direction

根据当前讨论，目标 `meta_keys` 方向如下：

- `PiperBaseEnv` 负责：
  - `obs/state/...`
  - `action/...`
- `PiperEnv` 负责追加：
  - `obs/under_control/...`
- 相机 wrapper 负责追加：
  - `obs/rgb/...`
  - 后续可扩展 `obs/depth/...`

### 4.1 State Keys

单臂场景至少包含：

- `joint_pos`
- `joint_vel`
- `ee_pose`
- `gripper_pos`

双臂场景按前缀展开，例如：

- `left_joint_pos`
- `left_joint_vel`
- `left_ee_pose`
- `left_gripper_pos`
- `right_joint_pos`
- `right_joint_vel`
- `right_ee_pose`
- `right_gripper_pos`

### 4.2 Under Control Keys

保留 `under_control`，并按机械臂区分，例如：

- 单臂：`under_control.arm`
- 双臂：`under_control.left`、`under_control.right`

### 4.3 Action Keys

第一阶段以 `follower` 为唯一动作对象，例如：

- 单臂：`arm`、`gripper`
- 双臂：`left_arm`、`left_gripper`、`right_arm`、`right_gripper`

### 4.4 Vision Keys

由 wrapper 注入，例如：

- `obs/rgb/base_camera`
- `obs/rgb/left_hand_camera`
- `obs/rgb/right_hand_camera`

当前 `meta_key_example.png` 也说明了后续 `.h5` 产物需要能稳定表达：

- `obs/state/...`
- `obs/rgb/...`
- `meta/env_meta`
- `meta/env_kwargs`

## 5. Priority Roadmap

下列任务按优先级划分为 `P0 / P1 / P2`。

- `P0`：必须优先完成，决定整体架构是否成立。
- `P1`：强相关增强项，在 `P0` 稳定后尽快推进。
- `P2`：保留扩展项，可在独立对话或后续分支实现。

## 6. Work Categories

为了方便后续拆分对话与分支开发，本项目重构分为 8 个大类。

### Category A. Core Env Architecture

优先级：`P0`

目标：

- 收敛 `Piper` env 的核心分层。
- 明确 `base -> core env -> camera wrapper -> final env` 的包装关系。
- 让单臂与双臂在结构上保持统一。

子目标：

1. 明确核心 env 文件命名与职责边界。
2. 稳定 `piper_base_env.py` 的机械臂核心职责。
3. 规划单臂核心 env。
4. 规划双臂核心 env。
5. 明确最终用户入口类与 wrapper 组合方式。

预期产物：

- 清晰稳定的 env 类图
- 单臂 / 双臂统一的接口约定
- README 中的分层说明

### Category B. Robot Control Semantics

优先级：`P0`

目标：

- 固定机械臂动作与观测语义。
- 使 teleop、专家介入与普通 action 执行具有一致可解释性。

子目标：

1. 固定第一阶段 `joint` 为主、`pose` 预留的控制模式策略。
2. 固定 `gripper` 的原始物理量纲表达。
3. 固定 `under_control` 语义。
4. 明确 `tele_enabled` 行为：
   - `T` 键切换
   - 主臂同步逻辑
   - 专家手动移动主臂后的接管逻辑
5. 保证 `action` 默认只面向 `follower`。

预期产物：

- 稳定的 action/obs 行为定义
- teleop 行为说明
- 对 VLA 算法清晰的动作数据语义

### Category C. Meta Keys and Config Contract

优先级：`P0`

目标：

- 将 `meta_keys` 变成整个 infra 的主契约。
- 将配置变成 env 与 wrapper 的标准装配入口。

子目标：

1. 固定单臂与双臂 `meta_keys` 结构。
2. 明确哪些 keys 由 core env 负责，哪些由 camera wrapper 追加。
3. 明确 config 覆盖优先级：
   - 环境构造参数
   - YAML config
   - 默认值
4. 明确双臂 CAN 命名约定与配置格式。
5. 为后续 `camera_type` 扩展保留兼容结构。

预期产物：

- 统一的 `meta_keys` 规范
- 统一的 `piper_config.yaml` / `dual_piper_config.yaml` 结构
- 统一的环境构造参数优先级约定

### Category D. Camera Wrapper System

优先级：`P1`

目标：

- 保持相机系统与机械臂核心解耦。
- 支持后续引入不同品牌相机。

子目标：

1. 设计通用 camera wrapper 抽象层。
2. 明确 wrapper 与 core env 的交互方式。
3. 第一阶段完成 `RealSense` wrapper 的配置驱动装配。
4. 支持基于 `camera_type / role / serial_number` 构建相机实例。
5. 为奥比中光预留接口但不在本阶段实现其 SDK。

预期产物：

- 可扩展的 camera wrapper 架构
- `RealSense` 首版稳定接入
- 后续新增相机品牌时无需大改核心 env

### Category E. Data Pipeline

优先级：`P1`

目标：

- 打通采集、转换、合并、回放的数据流。
- 兼容 `.h5` 与 `LeRobot`。

子目标：

1. 定义 `.h5` 目标结构。
2. 优先对齐 `LeRobot` 输出流程。
3. 实现多条 `.h5` 合并为一条 `.h5`。
4. 实现 `.h5` 控制模式切换。
5. 实现 `.h5 <-> LeRobot` 转换。
6. 让 replay 依赖 `meta_keys` 自动还原 state/action。

预期产物：

- 统一数据格式约定
- 面向训练和部署的稳定数据流
- 可复用的数据处理工具链

### Category F. Script Layout

优先级：`P1`

目标：

- 将脚本入口按单臂 / 双臂拆分。
- 让用户主要通过修改 `.sh` 中参数来使用。

子目标：

1. 新建 `Script/single/`。
2. 新建 `Script/dual/`。
3. 每个目录至少覆盖：
   - 数据收集
   - 数据后处理
   - 轨迹回放
4. 统一脚本命名与参数风格。

预期产物：

- 更清晰的脚本组织结构
- 更适合外部用户直接使用的入口

### Category G. Documentation

优先级：`P1`

目标：

- 新增开发者导向的 `README.md`。
- 明确每一层的功能与包装关系。

子目标：

1. 说明项目目录职责。
2. 说明各层类关系。
3. 说明 `meta_keys` 与 config 的关系。
4. 说明 single / dual / camera wrapper 的组合方式。
5. 说明数据采集、处理与回放路径。

预期产物：

- `agent_infra/Piper_Env/README.md`

### Category H. Future Extensions

优先级：`P2`

目标：

- 为后续独立对话或分支留好扩展空间。

子目标：

1. 奥比中光 RGB 接入与 V4L2 fallback 已完成；SDK depth 仍需实机稳定性验证。
2. `delta_pose` 已接入 env / recorder / smoke test CLI。
3. `relative_pose_chunk` 已接入 env / recorder / smoke test CLI。
4. depth obs/H5 写入已接入，默认开启；LeRobot depth 写入暂缓。
5. 更完整的离线 env / mock env。
6. 更丰富的测试与验证脚本。

## 7. Recommended Implementation Order

建议按以下顺序推进：

1. `Category A` Core Env Architecture
2. `Category B` Robot Control Semantics
3. `Category C` Meta Keys and Config Contract
4. `Category D` Camera Wrapper System
5. `Category E` Data Pipeline
6. `Category F` Script Layout
7. `Category G` Documentation
8. `Category H` Future Extensions

## 8. Suggested Conversation / Branch Split

你后续如果准备拆分对话或分支，建议以以下粒度拆：

- 分支 1：Core Env 架构与命名收敛
- 分支 2：单臂 / 双臂机械臂控制语义与 `under_control`
- 分支 3：`meta_keys` 与 config 规范
- 分支 4：相机 wrapper 与 `RealSense` 接入
- 分支 5：数据采集 / H5 / LeRobot / 回放
- 分支 6：`Script/single` 与 `Script/dual`
- 分支 7：README 文档整理

## 9. Current Open Questions

虽然已经足够开始实施，但仍有几个待后续实现前确认的问题：

1. `piper_env.py`、`single_piper_env.py`、`dual_piper_env.py` 的最终命名方案。
2. 双臂最终用户入口类是否直接返回 camera-wrapped env，还是提供 core env + wrapper 两种入口。
3. `.h5` 最终字段命名是否完全比照当前 `meta_key_example` 风格。
4. 第二品牌相机接入时，`camera_type` 抽象层是否需要更早落地成 factory。

这些问题不会阻止 `P0` 设计推进，但在正式改代码前应先逐步收敛。
