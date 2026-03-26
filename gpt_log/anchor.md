# 角色与当前进度同步 (Context)
你是一名资深的机器人系统架构师与具身智能开发工程师。
在上一阶段，我们已经成功基于 `pyagxarm` API 封装了一个名为 `PiperTeleopEnv` 的 Gym 环境，实现了松灵 Piper 机械臂主从遥操的 Human-in-the-Loop (HITL) 接管机制，并使用了内部多线程读写分离与防抖逻辑。每次调用 `env.step(action)` 时，如果有物理主臂的干预，环境会在内部将真实手部动作覆盖进去，并通过 `info['actual_action']` 和 `info['intervened']` 返回实际执行的 7D 动作与干预标签。

# 当前任务目标 (Phase 2)
本阶段的核心目标是：**对接 LeRobot 采集框架，实现多视角视觉感知流的软同步，并录制带有干预标签的脱机数据集（Offline Dataset）。**
为了保证工程的可维护性和组件的极致解耦，请严格按照以下架构需求编写代码：

## 核心开发需求 1：多视角抽象视觉包装器 (`AbstractCameraWrapper`)
请基于 `gym.Wrapper` 开发一个相机包装器，用于软同步（Soft-sync）多路视觉流：
1. **多线程异步读取**：预留 `start_cameras()` 和 `stop_cameras()` 接口。定义后台读取线程（例如 `_camera_thread_loop`），在当前抽象类中，该线程使用 `time.sleep(1/30)` 模拟相机 30Hz 获取数据的耗时阻塞。
2. **防阻塞状态锁**：使用 `threading.Lock` 维护一个 `self._latest_frames` 字典。外部主控循环（Env 的 `step`）只能从这里拿“最新一帧”，绝不能被相机的 I/O 阻塞。
3. **Dummy 数据产出**：作为占位和逻辑跑通的 MVP 版本，更新到 `_latest_frames` 中的数据应为全零矩阵（Dummy zeros）。对齐 LeRobot 的标准：RGB (`observation.images.wrist` 和 `front`) 形状为 `(3, 224, 224)` 类型 `uint8`；Depth 形状为 `(1, 224, 224)` 类型 `float32`。
4. **接口覆写**：重写 `reset` 和 `step`，在调用原 Env 的对应方法后，将 `_latest_frames` 注入到返回的 `obs` 字典中。

## 核心开发需求 2：解耦的数据录制脚本 (`record_piper_dataset.py`)
请编写一个完全独立的脚本，使用 `LeRobotDataset` API 进行数据采集：
1. **环境挂载**：实例化 `PiperTeleopEnv`，套上 `AbstractCameraWrapper`，并启动相机线程。
2. **Dataset 特征修改 (Scheme Registration)**：初始化 `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`。**关键要求**：除了基础的 state、images、action 等特征外，必须在数据集的 features 中额外注册一个 1D 数组特征 `"meta.intervened": {"dtype": "bool", "shape": (1,)}`。
3. **精确的 50Hz 录制主循环**：
   - 使用 `time.perf_counter()` 配合 `time.sleep()` 实现精确的 50Hz 阻塞。
   - 每次下发 Dummy Action（全零数组，因为是遥操演示阶段）给 `env.step()`。
   - **拦截逻辑**：提取 `info['actual_action']` 作为真实录制的 Action，提取 `info['intervened']` 存入 `"meta.intervened"` 字段。
   - 调用 `dataset.add_frame(...)` 保存数据。
4. **安全与优雅退出**：捕获 `KeyboardInterrupt`，退出前依次安全停止相机线程、关闭机械臂连接并调用 `dataset.save()` 序列化数据。

# 你的目标
1.  提供上述两个核心模块的完整 Python 代码。
2.  代码必须具备高度的工业级规范，包含类型提示（Type Hints）以及详尽的中文注释（特别要对“软同步状态锁”和“Dataset 拦截注入逻辑”进行原理解释）。
3.  这一份prompt是作为一份锚点，保证每次思考时不会脱离主线。接下来我会将1这个任务分成三大块，每一块独立设置一prompt，引导你完成代码。
