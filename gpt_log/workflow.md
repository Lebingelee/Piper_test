在开始任务之前，我们先给出相关附件和参考文档，这是lerobot的项目地址https://github.com/huggingface/lerobot。 对于数据采集，env的包装和遥操作并行化的相关需求和设定如下：1）action为关节角度控制，当env未处于遥操状态时，env收到一个action并以控制频率执行该action，而env处于遥操状态下，env.step()中的action转为获取主臂的关节角度信息并执行。因此，env中要提前设置好一个under_control变量，该变量会实时检测并引导env.step()切换模式；2）env.step()的返回值(state,action)中，state需要额外保存一个under_control的key，来显示当前状态是自主推理还是遥操控制, 并保证state的返回值能对齐lerobot的数据采集接口3）在pyagxarm中对主臂和从臂初始化后，从臂全程会保持通讯（即.is_ok()的api会固定返回true）,而主臂只有我们拖动时才会返回true，因此基于该设定，我们可以额外设置一个遥操判断bool变量tele，当tele且主臂的is_ok()均为true后under_control才会为true

你的主要任务：请务必进行周密且细致的思考，仔细阅读相关附件中的文档，并结合我给出的需求，思考该设置需要如何进一步完善。你在思考过程中如果遇到了任何模棱两可的地方或无法把握的地方，请务必记录下来向我提问和咨询。我们最终的目标是生成一个详细且点面结合的workflow，作为prompt传入到gemini-cli中去实现

总体大纲
# 角色与任务
你是一名资深的机器人硬件工程师与具身智能研究员。你的任务是基于松灵机器人 `pyagxarm` API，为 Piper 机械臂开发一个兼容 `LeRobot` 和 `Gymnasium` 标准的自适应遥操作强化学习环境（Environment），用于 Human-in-the-Loop (HITL) 的数据采集与模型训练。

# 核心背景与需求
我们需要实现主臂（Master）与从臂（Follower/Slave）的联动，并将其封装为一个 `gym.Env`。该 Env 必须支持自适应接管：在未受到人类干预时，执行传入的策略模型 Action；当人类接管主臂时，平滑覆盖模型 Action，执行主臂的真实指令，并在返回结果中打上“人类接管”的标签。

# 技术规范与架构设计 (请严格按此实现)

## 1. 空间定义 (Spaces)
* **Action Space**: `Box`，7维 (6个关节角度 + 1个夹爪宽度)。
* **Observation Space**: `Dict`，严格对齐 LeRobot 规范，包含：
    * `observation.state`: `Box`，1D 数组（至少包含6个关节角度、6个关节速度、法兰6D位姿[x,y,z,r,p,y]、1个夹爪宽度）。
    * `observation.images.wrist`: `Box`，占位图像 (RGB, 例如 shape 3x224x224，全0)。
    * `observation.images.front`: `Box`，占位图像 (RGB, 同上)。
    * `observation.images.wrist_depth`: `Box`，占位深度图 (单通道，全0)。
    * `observation.images.front_depth`: `Box`，占位深度图 (单通道，全0)。
    * `under_control`: `Discrete(2)` 或 `Box`，0表示模型自主，1表示人类遥操接管。

## 2. 并发与多线程设计 (核心)
为了避免 I/O 阻塞并保证控制丝滑，必须采用**读写分离**的多线程架构：
* **后台主臂读取线程 (200Hz)**：
    * 专门负责调用 `master.get_leader_joint_angles()` 和获取夹爪状态。
    * 将获取到的 [6关节 + 1夹爪] 最新状态存入一个全局共享变量（必须使用 `threading.Lock` 保证线程安全）。
* **主线程 Env.step 循环 (例如 30Hz 或 50Hz)**：
    * 只负责读取共享变量并向从臂下发控制指令 (`follower.move_j` 或 `follower.move_js`)。
* **键盘监听线程**：
    * 使用 `pynput` 或类似库，监听特定按键（如 'T' 键），用于切换布尔变量 `tele` 的状态。

## 3. 接管逻辑与防抖机制 (Debounce)
* 定义接管状态：`under_control = tele and master.is_ok()`
* **防抖机制**：由于主臂拖动时可能会有微小的数据包丢失导致 `master.is_ok()` 瞬间变为 False。请在 Env 内部实现一个防抖计时器：当 `master.is_ok()` 变为 False 时，不要立即解除 `under_control`，而是等待 **0.5秒**。如果 0.5 秒内 `is_ok()` 恢复为 True，则视为未断开；若持续 0.5 秒为 False，才真正将 `under_control` 设为 False。

## 4. Env.step(action) 内部逻辑 (Human-in-the-Loop 方案)
当 Env 外部调用 `step(action)` 时（`action` 为策略模型的输出）：
1.  **判断模式**：检查防抖后的 `under_control` 标志位。
2.  **动作覆盖**：
    * 如果 `under_control == True`：忽略传入的 `action`，从线程锁共享变量中提取人类遥操的真实动作（专家动作 `expert_action`）。将 `expert_action` 转换为对应格式下发给从臂。
    * 如果 `under_control == False`：直接将传入的 `action` 下发给从臂。
3.  **模式自适应下发**：参考原遥操代码逻辑，如果真实目标角度与当前从臂角度差异较大 (`sum(abs(target - current)) > 0.7`)，使用 `move_j` (平滑插值)；否则使用 `move_js` (高频透传)。夹爪映射逻辑：`np.clip(width * 0.5, 0.01, 0.1)`。
4.  **返回值封装**：
    * 返回标准的 `obs, reward, terminated, truncated, info`。
    * **关键点**：由于 LeRobot 的外层数据记录器只会记录它传给 `step` 的那个 action，为了保存真实的专家动作，**你必须在 `info` 字典中返回实际执行的动作和接管标签**。例如：`info['actual_action'] = executed_action`，`info['intervened'] = under_control`。
    * `obs` 字典中的 `under_control` 键也应同步更新。

# 你的目标
请在多次对话后给出完整的 Python 代码实现，包括：
1.  依赖导入 (`pyAgxArm`, `gymnasium`, `threading`, `pynput` 等)。
2.  Env 类的完整实现（包含 `__init__`, `reset`, `step`, `_get_obs`, 内部线程方法等）。
3.  主程序的简单测试代码（实例化 Env 并进行一个 Dummy Loop）。
4.  代码需具备完善的中文注释，特别是关于多线程锁、防抖机制、以及 `info['actual_action']` 的处理逻辑。
5.  这一份prompt是作为一份锚点，保证每次思考时不会脱离主线。接下来我会将1这个任务分成三大块，每一块独立设置一prompt，引导你完成代码。

# 你的输出
请你充分了解代码目标后，回答“我准备好了”

子任务 1.1：基础框架与空间定义 (Spaces)

    目标：搭建符合标准的 gym.Env 骨架，精准定义 7D Action Space 和复杂的 Observation Space 字典结构。

    输入附件：无（依赖通用强化学习知识即可）。

    Agent Prompt：

        "你正在为一个双臂协同遥操作项目开发基于 gymnasium 的环境（Environment）。
        请帮我搭建一个名为 PiperTeleopEnv 的类，继承自 gym.Env。
        要求：

            在 __init__ 中定义 action_space 为 gym.spaces.Box，维度为 7（6 个关节角度 + 1 个夹爪宽度）。

            定义 observation_space 为 gym.spaces.Dict，严格包含以下键：

                observation.state: Box, 1D 数组（6个关节角度、6个关节速度、末端6D位姿[x,y,z,r,p,y]、1个夹爪宽度）。

                observation.images.wrist: Box, 占位 RGB 图像 (shape: 3x224x224，uint8全0)。

                observation.images.front: Box, 占位 RGB 图像 (同上)。

                under_control: Discrete(2)，用于标记当前是模型自主(0)还是人类接管(1)。

            提供 reset 和 step 的空函数占位符（只需返回符合空间格式的 dummy 数据即可）。
            请输出结构清晰、包含类型提示（Type Hints）的 Python 代码。"


子任务 1.2：多线程硬件通信与状态同步

    目标：实现读写分离，后台 200Hz 读主臂，提供线程安全的全局状态读取。

    输入附件：提供 tele.py、general.py、joint.py 的代码片段，以及 piper_api.md 中关于 get_leader_joint_angles 和 is_ok() 的说明。

    Agent Prompt：

        "基于松灵机器人的 pyagxarm API，我们需要在刚才的 Env 内部实现高频硬件通信。
        需求：

            在 Env 中引入 threading。初始化主臂（Leader）和从臂（Follower）的 CAN 连接。

            创建一个后台线程 _master_read_thread，以 200Hz (time.sleep(0.005)) 的频率循环运行。

            该线程负责读取主臂的关节角度（get_leader_joint_angles）和通信状态（is_ok()）。

            使用 threading.Lock() 将读取到的最新数据更新到类的实例属性中（如 self._latest_master_state），确保主线程读取时的线程安全。

            补充从臂的安全退出逻辑（在 close() 方法中调用 disable()）。
            请仅输出这一部分的扩展代码，并详细注释多线程锁的使用机制。"


目标：融合前两个步骤，实现最核心的防抖判定与动作替换。

输入附件：整合 1.1 和 1.2 的生成代码，以及描述防抖逻辑的需求说明。

Agent Prompt：

"现在我们来完善 PiperTeleopEnv 的核心 step(action) 逻辑。
控制逻辑设定：

    键盘监听：使用 pynput 开启一个监听线程，当按下 'T' 键时，翻转布尔变量 self.tele_enabled。

    防抖接管判定：计算状态 under_control = self.tele_enabled and master.is_ok()。关键要求：如果 is_ok() 变为 False，请实现一个 0.5 秒的防抖延迟，0.5 秒内持续 False 才真正退出 under_control 状态，防止通信微小丢包导致的抽搐。

    动作执行与覆写：

        若 under_control == True：忽略传入的 action，从共享锁中获取主臂的实时关节数据作为目标动作下发给从臂。根据差异大小动态调用 move_j 或 move_js（参考此前的 tele 逻辑）。

        若 under_control == False：直接将传入的 action（从 numpy 数组解析出 6 关节 + 夹爪）下发给从臂。

    LeRobot 对齐要求：在 step 返回的 info 字典中，必须注入 info['actual_action'] = 实际执行的7D动作，以及 info['intervened'] = 当前的接管状态(bool)。
    请输出完整合并后的最终 Env 代码，并在 step 方法处保留详尽的中文注释。"




针对问题1，我们考虑使用相关api接口，写一个完全独立且解耦的 record_piper_dataset.py 脚本；针对问题2，采用方案A;针对问题3，我们可以使用wrapper格式，但是我们也需要先实现一个abstract_wrapper, 其传入的图像为none或者为0矩阵。我们要优先跑通整个逻辑链条。针对问题4，我们确实需要让每一个相机拥有独立线程，主控制循环（30/50Hz）在读取时只去拿各个相机锁定的“最新一帧”（Latest Frame），从而实现软同步（Soft-sync）。但是我们目前不需要实现它，在abstract_wrapper上保留接口或相关方法即可



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

# 你的输出
请你充分了解代码目标后，回答“我准备好了”


2.1 任务目标
# 角色与任务
你是一名资深的机器人与具身智能开发工程师。在之前的任务中，我们已经完成了一个名为 `PiperTeleopEnv` 的 Gym 环境，用于松灵 Piper 机械臂的遥操作控制。
现在的任务是：基于 `gym.Wrapper` 开发一个抽象的多视角相机包装器 `AbstractCameraWrapper`。

# 架构与设计需求 (请严格遵守)
1. **Wrapper 模式**：
   - 继承自 `gym.Wrapper`，接收一个 `env` 实例。
   - 在 `__init__` 中保留对 `env.observation_space` 的兼容，特别是 `observation.images.wrist` 和 `observation.images.front`（以及对应的 depth）。
2. **多线程异步预留 (Soft-sync 思想)**：
   - 明确需要为每个相机预留独立的后台读取线程。
   - 定义类似 `start_cameras()` 和 `stop_cameras()` 的方法。
   - 定义占位的后台线程循环方法（如 `_camera_thread_loop(camera_name)`），但在当前抽象类中，这些线程内部只需要 `time.sleep(1/30)` 模拟耗时即可。
   - 使用线程锁（`threading.Lock`）和字典（如 `self._latest_frames`）来保存每个相机的“最新一帧”。
3. **Dummy 数据产出**：
   - 因为当前是抽象基类（用于优先跑通逻辑链），后台线程更新到 `self._latest_frames` 中的图像数据应该全是 **全 0 矩阵 (Dummy zeros)**，数据形状需对齐 LeRobot 的标准（如 RGB 为 `(3, 224, 224)` uint8，Depth 为 `(1, 224, 224)` float32）。
4. **重写 step 和 reset**：
   - `reset(**kwargs)`：调用 `self.env.reset()`，然后将 `self._latest_frames` 中的占位图像覆盖到返回的 `obs` 字典中。
   - `step(action)`：调用 `self.env.step(action)`，同样将 `self._latest_frames` 中的图像覆盖到返回的 `obs` 字典中。

# 输出要求
请输出完整的 `AbstractCameraWrapper` Python 代码，包含详细的中文注释，特别是关于多线程异步更新“最新一帧”以实现防阻塞（Non-blocking）软同步的设计理念。