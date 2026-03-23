# Piper HITL 遥操作与数据录制系统 (LeRobot v3.0)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LeRobot](https://img.shields.io/badge/LeRobot-v3.0-orange.svg)](https://github.com/huggingface/lerobot)
[![Hardware](https://img.shields.io/badge/Hardware-Agilex_Piper-green.svg)](https://agilex.ai/)

本项目是为 **Agilex Piper** 机械臂打造的自适应遥操作 (Teleoperation) 环境与数据采集系统。它深度集成了 Hugging Face 的 **LeRobot v3.0** 框架，支持多视角视觉流的异步采集以及 Human-in-the-Loop (HITL) 的实时接管录制。

---

## 🚀 核心架构与特性

### 1. 混合控制模式 (Dual Control Modes)
- **`joint` 模式**: 直接控制 6 自由度关节角度。支持高频透传 (`move_js`) 与 梯形规划平滑移动 (`move_j`) 的自动切换。
- **`pose` 模式**: 控制末端 6D 位姿 [x, y, z, roll, pitch, yaw]。调用硬件级逆解接口 (`move_p`)。

### 2. HITL 实时接管机制
- 采用**读写分离**的多线程架构，主臂状态以 200Hz 频率在后台同步。
- **平滑覆盖逻辑**: 当人类操作物理主臂时，系统自动拦截并覆盖模型 Action，将真实的专家轨迹实时记录至数据集。
- **断连防抖**: 内置 0.5s 硬件状态防抖，确保控制过程不会因瞬间丢包而中断。

### 3. 异步感知流 (Soft-sync Vision)
- 视觉采集频率与控制频率解耦。
- 采用 `AbstractCameraWrapper` 实现非阻塞读取，确保主控循环不会被相机的 I/O 阻塞。

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
│   │   └── utils.py                 # 工具库：自适应路径冲突处理
│   └── script/
│       ├── collect.sh               # 数据采集一键启动脚本
│       └── replay.sh                # 轨迹重播一键启动脚本
├── README.md                        # 本文档
└── can_activate.sh                  # SocketCAN 激活与重命名脚本
```

---

## 🛠️ 快速开始

### 1. 硬件连接与配置
使用配套脚本激活主从臂的 CAN 通讯（确保比特率为 1M）：
```bash
sudo bash can_activate.sh can_master 1000000
sudo bash can_activate.sh can_slave 1000000
```

### 2. 录制数据 (Collect)
编辑 `piper_infra/script/collect.sh` 设置 `CTRL_MODE`（`joint` 或 `pose`），然后运行：
```bash
bash piper_infra/script/collect.sh
```
- **[T] 键**: 切换“遥控”状态。
- **[R] 键**: 开启/停止当前 Episode 录制。
- **[Q] 键**: 安全保存并退出。

### 3. 验证回放 (Replay)
回放录制好的轨迹以验证质量：
```bash
# 示例：回放 Episode 0
bash piper_infra/script/replay.sh --path data/piper_recording_000 --episode 0 --ctrl_mode joint
```

---

## 💡 技术细节
- **数据存储**: LeRobot v3.0 采用 Parquet 格式打包存储。回放模块已优化，直接读取数值数据，**无视 FFmpeg 环境报错**。
- **对齐逻辑**: 重播开始前，系统会调用 `reset_to_state` 引导机械臂从当前姿态慢速移动至轨迹起点，防止瞬时冲击。
- **物理映射**: 统一了夹爪在不同模式下的物理量程映射，确保录制数据与模型输出分布一致。

---

## 📬 维护
Developed by **Lebingelee (lebinge@163.com)**.
