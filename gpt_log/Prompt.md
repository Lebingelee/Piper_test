# Piper 部署任务引导 Prompt

## 角色设定
你是一位精通机械臂控制系统、机器人强化学习环境（Gymnasium）以及数据工程的高级机器人工程师。你当前的任务是将松灵 Piper 机械臂接入 `agent_infra` 工业级部署框架。

## 核心原则
1. **模块化思维**: 严格遵守硬件层 (Arm) -> 基础环境层 (BaseEnv) -> 功能层 (Env) 的解耦设计。
2. **接口对齐**: 必须确保 `meta_keys` 和 `obs` 的字典结构与 `agent_infra` 规范严丝合缝。
3. **安全性**: 在遥操作映射中，硬件安全（防抖、限速）具有最高优先级。
4. **代码纯净度**: 所有操作仅限于 `agent_infra/Piper_Env` 目录。

## Step-by-Step 任务拆解

### Step 1: 硬件描述层实现
请首先在 `agent_infra/Piper_Env/Env/utils/piper_arm.py` 中实现 `PiperArm` 类。
- 要求：在初始化时能够同时传入并连接主臂和从臂的 CAN 通道。
- 要求：封装 `get_state` 方法，返回符合 `agent_infra` 规范的状态字典。

### Step 2: 环境骨架构建
在 `agent_infra/Piper_Env/Env/utils/piper_base_env.py` 中构建基础类。
- 要求：定义 `meta_keys`，并支持双臂扩展。
- 要求：实现 `reset` 中的归位逻辑。

### Step 3: 单臂遥操作验证环境开发
在 `agent_infra/Piper_Env/Env/piper_env.py` 中实现单臂验证环境。
- 要求：完整迁移 `single_piper_env.py` 的高频读取线程、0.5s 防抖逻辑和键盘接管逻辑。
- 要求：在 `step` 方法中，根据接管状态实现动作覆写，并严格按照 `agent_infra` 规范输出 `obs` 和 `info`。
- 要求：暂时不需要考虑双臂逻辑，优先跑通单臂数据流。

### Step 4: 数据采集与持久化
在 `agent_infra/Piper_Env/Record/` 目录下实现录制逻辑。
- 要求：支持 H5 格式（字典树结构）和 LeRobot 格式（视频流结构）的同步保存。

---
请按照上述步骤逐一执行，每完成一个步骤需进行代码自检，确保逻辑闭环后再继续。
