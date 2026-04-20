# Piper 机械臂环境部署与数据采集优化计划表

## 1. 核心目标
在 `agent_infra` 框架下建立 `Piper_Env` 库，实现高度模块化的 Piper 机械臂交互环境。支持单/双臂遥操作映射、标准化的 `meta_keys` 观测结构、以及 H5 和 LeRobot 双格式的数据持久化。

## 2. 框架搭建流程与工作路径

### 第一阶段：基础配置与硬件抽象 (Foundation & Hardware)
1. **目录结构创建**:
   - `agent_infra/Piper_Env/Env/utils/`
   - `agent_infra/Piper_Env/Camera/` (复用或适配 RealSense)
   - `agent_infra/Piper_Env/Record/`
   - `agent_infra/Piper_Env/Script/`
2. **硬件包装 (`Env/utils/piper_arm.py`)**:
   - 实现 `PiperArm` 类。
   - **关键点**: 构造函数需接受 `master_can` 和 `follower_can` 参数，实现主从臂 ID 的录入与连接。
   - 提供 `get_state()` (返回关节、位姿、速度、夹爪) 和 `apply_action()` (支持透传与规划模式)。
3. **配置驱动 (`Env/piper_config.yaml`)**:
   - 定义机械臂 CAN 通道、初始位姿、相机 SN 序列号及角色挂载点（如 `base_camera`, `hand_camera`）。

### 第二阶段：单臂环境验证开发 (Single-Arm Validation)
1. **基础环境类 (`Env/utils/piper_base_env.py`)**:
   - 继承 `BaseRobotEnv`。
   - **状态空间定义**: 严格在 `self.meta_keys["obs"]["state"]` 下定义单臂 key：
     - `joint_pos`: (6,)
     - `joint_vel`: (6,)
     - `ee_pose`: (6,)
     - `gripper_pos`: (1,)
   - **动作空间定义**: `self.meta_keys["action"]` 定义为 `arm`: (6,) 和 `gripper`: (1,)。
   - 管理单个 `PiperArm` 实例的生命周期。
2. **单臂综合环境类 (`Env/piper_env.py`)**:
   - **逻辑迁移**: 完整迁移 `piper_infra/Env/single_piper_env.py` 中的以下核心逻辑：
     - **主臂高频读取线程**: 200Hz 后台读取 `mja`, `ok`, `gcs`。
     - **安全防抖机制**: 0.5s 的 `_last_ok_time` 硬件连接判定。
     - **键盘监听**: 按 'T' 键切换 `tele_enabled` 状态。
   - **观测聚合**: 将视觉数据（wrist/front camera）与硬件状态聚合。
     - 严格遵循二级 Key：`obs["rgb"]["wrist_camera"]`, `obs["state"]["joint_pos"]` 等。
   - **动作接管 (`.step`)**:
     - 判定 `is_intervened`。
     - 如果接管，将 `action` 覆写为 `expert_action`。
     - 在 `info` 中返回 `actual_action` (7D 数组) 和 `intervened` (bool)。


### 第三阶段：数据采集与回放系统 (Data Pipeline)
1. **双格式录制器 (`Record/recorder.py`)**:
   - 编写 `DataRecorder` 类。
   - **H5 格式**: 参考 `agent_infra` 实现递归字典保存，便于通用工具链合并。
   - **LeRobot 格式**: 参考 `piper_infra` 实现，集成 `LeRobotDataset` API，支持视频流编码。
2. **回放工具 (`Record/replay.py`)**:
   - 实现轨迹回放脚本，支持从 H5 或数据集读取动作序列。

### 第四阶段：验证与自动化脚本 (Validation)
1. **测试脚本 (`Script/test_env.py`)**: 验证单臂遥操作下的观测对齐与动作下发。
2. **启动脚本 (`Script/collect_data.sh`, `Script/replay_data.sh`)**: 封装命令行调用。

## 3. 警示与约束
- **严禁修改** `agent_infra/Realman_Env` 或 `piper_infra` 中的任何现有代码，确保所有开发均在 `agent_infra/Piper_Env` 内进行。
- **状态一致性**: 确保 `.reset` 和 `.step` 返回的字典结构完全一致。
- **安全防抖**: 必须保留 `single_piper_env.py` 中的 0.5s 通信防抖逻辑。
