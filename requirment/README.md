# Requirment 依赖说明

本目录用于给项目克隆后的环境安装提供可选依赖清单。

按你的当前仓库结构，依赖拆成 4 个方向，用户可以按需选择安装：

1. `agent_infra_base.txt`
   适合只想运行 `agent_infra` 通用环境基类、基础 env 结构与通用脚本的场景。

2. `agent_infra_server_train.txt`
   适合在服务器上做训练、数据读取、离线处理的场景。
   这份会补充 `torch`、`diffusers`、`omegaconf` 等训练栈。

3. `agent_infra_piper_env.txt`
   适合运行 `agent_infra/Piper_Env/`。
   这份包含基础 Python 依赖，但硬件 SDK 仍需按机器环境手动安装。

4. `agent_factory.txt`
   适合只安装 `agent_factory` 训练与推理侧依赖。

常用安装示例：

```bash
pip install -r requirment/agent_infra_base.txt
pip install -r requirment/agent_infra_server_train.txt
pip install -r requirment/agent_infra_piper_env.txt
pip install -r requirment/agent_factory.txt
```

如果你是典型的 Piper 数据采集 + 训练场景，推荐顺序：

```bash
pip install -r requirment/agent_infra_piper_env.txt
pip install -r requirment/agent_factory.txt
```

注意事项：

- `pyAgxArm` 是 Piper 机械臂控制 SDK，当前仓库根目录下没有对应可直接安装的 Python 包，需要按实际 SDK 来源单独安装。
- `pyrealsense2` 依赖 Intel RealSense 本机环境，部分系统需要先装系统级驱动。
- `pyorbbecsdk` 在当前仓库中带有源码目录 `pyorbbecsdk/`，但其 `setup.py` 依赖已经准备好的原生 `install/lib` 产物，因此更适合作为手动步骤处理。
- 如果只在服务器做离线训练，不需要安装 RealSense、Orbbec、CAN、Piper 硬件相关 SDK。
