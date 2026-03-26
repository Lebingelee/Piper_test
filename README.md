# Piper HITL 遥操作与数据录制系统 (LeRobot v3.0)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/LeRobot-v3.0-orange.svg)](https://github.com/huggingface/lerobot)
[![Hardware](https://img.shields.io/badge/Hardware-Agilex_Piper-green.svg)](https://agilex.ai/)

本项目是为 **Agilex Piper** 机械臂打造的自适应遥操作 (Teleoperation) 环境与数据采集系统。它深度集成了 Hugging Face 的 **LeRobot v3.0** 框架，支持多视角视觉流的异步采集以及 Human-in-the-Loop (HITL) 的实时接管录制。

---

## 🚀 核心特性

### 1. 混合控制模式 (Dual Control Modes)
- **`joint` 模式**: 直接控制 6 自由度关节角度。支持高频透传 (`move_js`) 与梯形规划平滑移动 (`move_j`) 的自动切换。
- **`pose` 模式**: 控制末端 6D 位姿 [x, y, z, roll, pitch, yaw]。调用硬件级逆解接口 (`move_p`)。

### 2. 自适应路径管理 (Smart Path Handling)
- **模式感知命名**: 录制数据会自动根据控制模式命名（如 `data/piper_joint_recording` 或 `data/piper_pose_recording`）。
- **自动冲突处理**: 系统会自动检测路径是否存在，并生成 `_000`, `_001` 等对齐后缀，确保数据不会被覆盖。

### 3. HITL 实时接管机制
- **平滑覆盖**: 当人类操作物理主臂时，系统自动拦截并覆盖模型 Action，将真实的专家轨迹实时记录至数据集。
- **断连防抖**: 内置 0.5s 硬件状态防抖，确保控制过程不会因瞬间丢包而中断。

### 4. 稳健的轨迹重播 (Robust Replay)
- **自动使能**: 重播开始前，系统会自动使能从臂并慢速引导至轨迹起点，防止电机未上电或瞬时冲击导致的任务失败。
- **参数智能合并**: `replay.sh` 脚本支持智能参数合并，优先响应命令行输入的路径和模式，极大提升了调试效率。

---

## 📂 项目结构

```text
Api_test/
├── piper_infra/
│   ├── Env/
│   │   └── single_piper_env.py      # 核心 Gym 环境：处理主从联动、接管逻辑与模式切换
│   ├── Record/
│   │   ├── recoder.py               # 异步相机包装器：视觉流软同步
│   │   ├── record_piper_dataset.py  # 录制主程序：对接 LeRobot v3.0 Dataset API
│   │   ├── replay.py                # 轨迹回放工具：支持数值数据提取与平滑对齐
│   │   └── utils.py                 # 工具库：自适应路径冲突与模式感知的命名处理
│   └── script/
│       ├── collect.sh               # 数据采集一键启动脚本
│       └── replay.sh                # 轨迹重播一键启动脚本
├── README.md                        # 本文档
└── can_activate.sh                  # SocketCAN 激活脚本
```

---

## 🛠️ 快速开始

### 1. 硬件连接与配置
激活主从臂的 CAN 通讯（确保比特率为 1M）：
```bash
sudo bash can_activate.sh can_master 1000000
sudo bash can_activate.sh can_slave 1000000
```

### 2. 录制数据 (Collect)
运行一键启动脚本进行数据采集：
```bash
bash piper_infra/script/collect.sh
```
*注：你可以修改 `collect.sh` 中的 `CTRL_MODE`（默认为 `pose`）来切换录制模式。数据将自动保存至 `data/piper_{mode}_recording/`。*

### 3. 验证回放 (Replay)
回放录制好的轨迹以验证质量（脚本会自动处理路径补全）：
```bash
# 示例：回放 pose 模式下的最新录制 (episode 0)
bash piper_infra/script/replay.sh --path data/piper_pose_recording_002 --episode 0 --ctrl_mode pose

# 示例：回放 joint 模式下的轨迹
bash piper_infra/script/replay.sh --path data/piper_joint_recording --episode 0 --ctrl_mode joint
```

---

## 💡 技术细节
- **数据格式**: LeRobot v3.0 采用 Parquet 格式存储。回放模块优化了数据读取，直接提取数值，**无视 FFmpeg 环境报错**。
- **控制逻辑**: 重播时，系统会先调用 `follower.enable()` 确保从臂上电，随后通过 `move_j` 实现平滑对齐。
- **命名规范**: `utils.py` 内部实现了严谨的路径递增逻辑，最高支持到 `999` 个目录，方便进行大规模数据集采集。

---

## 📬 维护
Developed by **Lebingelee (lebinge@163.com)**.
