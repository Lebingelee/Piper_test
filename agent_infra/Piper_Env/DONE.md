# Piper Env Progress Summary

## 1. Purpose

本文档用于记录 `agent_infra/Piper_Env` 相对于 [PLAN.md](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/PLAN.md) 的当前完成进度。

文档目标不是替代 `PLAN.md`，而是回答两个问题：

- 目前已经做完了哪些部分
- 这些改动已经达成了什么效果

## 2. Current Overall Status

截至当前，`Piper_Env` 已经完成了第一阶段的核心架构收敛，并跑通了以 `RealSense` 为主的主 pipeline 骨架；同时已经补齐奥比中光相机的首版类封装与 wrapper 路由骨架。

当前已经具备的能力包括：

- `PiperBaseEnv -> PiperEnv -> camera_wrapper -> Single/DualPiperEnv` 的分层落地
- 单臂 / 双臂最终入口类
- `RealSense` 相机包装与基础 corner case 处理
- `Orbbec` 相机类与 camera_type 路由骨架
- 单臂 / 双臂采集与回放脚本入口
- `.h5` 合并
- `.h5` 控制模式切换
- `.h5 <-> LeRobot` 转换工具

当前还没有完成的内容包括：

- 更完整的后处理能力说明
- 奥比中光 SDK 安装后的实机联调与 round-trip 验证记录
- 更丰富的测试与 mock/offline 能力

## 3. Progress By Plan Category

### Category A. Core Env Architecture

状态：`已部分完成，达到第一阶段目标`

已实现内容：

- 已确认并落地包装链：
  - `SinglePiperEnv / DualPiperEnv -> PiperCameraWrapper -> PiperEnv -> PiperBaseEnv -> BaseRobotEnv`
- `PiperBaseEnv` 已收敛为机械臂核心层：
  - 管理 follower 动作下发
  - 读取本体状态
  - 构建 `obs/state/...` 与 `action/...` 的 `meta_keys`
- `PiperEnv` 已收敛为 teleop / 专家介入层：
  - 管理 `tele_enabled`
  - 管理 leader 读取线程
  - 维护 `under_control`
  - 在 `step()` 中覆写实际执行动作
- 新增最终用户入口类：
  - [single_piper_env.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/single_piper_env.py)
  - [dual_piper_env.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/dual_piper_env.py)

当前效果：

- 分层职责已经从“计划描述”变成“代码现实”
- 单臂和双臂入口不再需要用户手工拼装底层 env 和 wrapper

对应文件：

- [piper_base_env.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/utils/piper_base_env.py)
- [single_piper_env.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/single_piper_env.py)
- [dual_piper_env.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/dual_piper_env.py)

### Category B. Robot Control Semantics

状态：`已部分完成，absolute joint 与 absolute pose 主路径已具备`

已实现内容：

- 第一阶段控制模式仍以绝对 `joint` 为主
- `pose` 已接入主控模式和 follower 下发接口：
  - `PiperBaseEnv.get_safe_action()` 在 `pose` 模式下返回 `ee_pose`
  - `PiperEnv.step()` 在专家接管时读取 master `ee_pose`
  - `PiperArm.apply_action()` 在 `pose` 模式下调用 `move_p`
- 已新增 `control_mode` 与 action 维度校验
- `action` 仍只面向 follower
- `under_control` 已保留并由 `PiperEnv` 维护
- `tele_enabled` 继续由 `T` 键切换
- `step()` 中继续输出：
  - `info["actual_action"]`
  - `info["intervened"]`

当前效果：

- 训练和录制侧仍然能够拿到“模型动作”和“实际执行动作”的差异信息
- 单臂 / 双臂都可以沿用同样的 `under_control` 语义

当前边界：

- 还没有对更细的 teleop 行为做额外重构
- `delta_pose` 与 `relative_pose_chunk` 已接入基础执行链路，仍需要实机专项验证与更完整的后处理覆盖

对应文件：

- [piper_base_env.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/utils/piper_base_env.py)
- [piper_arm.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/utils/piper_arm.py)

### Category C. Meta Keys and Config Contract

状态：`已部分完成`

已实现内容：

- `PiperBaseEnv` 只构建：
  - `obs/state/...`
  - `action/...`
- `PiperEnv` 追加：
  - `obs/under_control/...`
- `PiperCameraWrapper` 追加：
  - `obs/rgb/...`
- 已补齐单臂 / 双臂配置入口：
  - [piper_config.yaml](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Config/piper_config.yaml)
  - [dual_piper_config.yaml](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Config/dual_piper_config.yaml)
- 配置已开始显式包含 `camera_type`

当前效果：

- `meta_keys` 的来源层级更清晰
- 单臂 / 双臂配置都能作为最终 env 的构造入口

当前边界：

- config 结构还没有写成正式文档
- 多品牌相机 factory 还没有完全抽象出来

### Category D. Camera Wrapper System

状态：`已完成 RealSense 第一阶段目标，并补齐 Orbbec 首版接入骨架`

已实现内容：

- `PiperCameraWrapper` 已基于 config 装配 `RealSense`
- `PiperCameraWrapper` 已可按 `camera_type` 路由到：
  - `realsense`
  - `orbbec`
- 已支持读取：
  - `camera_type`
  - `role`
  - `serial_number`
  - `source_resolution`
  - `crop_resolution`
- 已新增：
  - [orbbec_camera.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Camera/orbbec_camera.py)
  - `OrbbecCamera`
  - `OrbbecCameraGroup`
- 当本机未安装 `pyorbbecsdk` 时，Orbbec 路线会自动降级为 fallback 图像，不阻塞整体 obs 结构
- 当本机未安装 `pyrealsense2` 时，RealSense 路线也会自动降级为 fallback 图像，不阻塞 wrapper 导入
- 已补充 corner case 处理：
  - config 中没有 `cameras.nodes` 时可进入 dummy/fallback
  - 某路相机初始化失败时，对应 `role` 回退为纯色图
  - 支持 `fallback_color`
  - `source_resolution` 与 `crop_resolution` 比例不一致时，先中心裁切再 resize

当前效果：

- `RealSense` 路线的最终 env 可以更稳地跑通
- `camera_wrapper` 已不再绑死单一品牌
- 奥比中光后续只需补 SDK 安装、参数调优和实机验证，不需要再从零改 wrapper 架构
- 图像不会因为比例不一致而被直接拉伸
- 单路相机失败不会拖垮整个 obs 结构

当前边界：

- 当前只有 `RealSense` 做过完整主链路验证
- `Orbbec` 当前是首版实现，尚未在本机完成 SDK 安装与实机验证

对应文件：

- [piper_camera_wrapper.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Env/piper_camera_wrapper.py)
- [realsense_camera.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Camera/realsense_camera.py)
- [orbbec_camera.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Camera/orbbec_camera.py)

### Category E. Data Pipeline

状态：`已部分完成，主链路已打通`

已实现内容：

- 录制：
  - `.h5`
  - `LeRobot`
- 回放：
  - `.h5`
  - `LeRobot`
- 新增后处理工具：
  - 多条 `.h5` 合并为一个 merged `.h5`
  - merged `.h5 -> LeRobot`
  - `LeRobot -> merged .h5`
  - `.h5` 合并时支持控制模式切换：
    - `joint`
    - `pose`

当前效果：

- 采集 -> 合并 -> 格式转换 -> 回放 这条主 pipeline 已经有代码入口
- `.h5` 合并时可以顺手完成 action 语义切换

当前边界：

- 目前是工具和脚本已实现，尚未记录完整实机 round-trip 结果
- `.h5` 控制模式切换当前只支持绝对 `joint` / `pose`

对应文件：

- [recorder.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Record/recorder.py)
- [replay.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Record/replay.py)
- [postprocess.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Record/postprocess.py)

### Category F. Script Layout

状态：`已完成第一阶段重组，入口进一步收敛`

已实现内容：

- 已新增：
  - [Script/single](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Script/single)
  - [Script/dual](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Script/dual)
- single / dual 下目前都已有：
  - 数据采集脚本
  - 轨迹回放脚本
  - H5 合并脚本
  - H5 -> LeRobot 转换脚本
  - LeRobot -> H5 转换脚本
- Piper 专用 CAN 脚本已迁移到 `agent_infra/Piper_Env/Script/`
- 已新增统一硬件 smoke test：
  - [test_piper_pipeline.py](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/Script/test_piper_pipeline.py)
- 已删除旧顶层脚本：
  - `Script/collect_h5.sh`
  - `Script/collect_lerobot.sh`
  - `Script/replay.sh`

当前效果：

- 用户可以主要通过修改 `.sh` 中变量来使用 pipeline
- single / dual 两套流程已经有了清晰入口
- 旧入口不再与 single / dual 正式入口产生歧义

### Category G. Documentation

状态：`部分完成，README 已补齐第一版`

已实现内容：

- 已完成规划文档：
  - [PLAN.md](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/PLAN.md)
- 已新增当前进度文档：
  - [DONE.md](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/DONE.md)
- 已新增使用与结构说明：
  - [README.md](/home/lebinge/Desktop/Piper_test/agent_infra/Piper_Env/README.md)
- 已新增根目录结构整理提案：
  - [STRUCTURE_REORG_PROPOSAL.md](/home/lebinge/Desktop/Piper_test/STRUCTURE_REORG_PROPOSAL.md)

当前边界：

- README 还需要随着实机验证持续补充 sample output 与故障排查记录

### Category H. Future Extensions

状态：`部分完成`

未完成内容：

- 更完整的 offline/mock env
- 更丰富的测试覆盖

已完成内容：

- `delta_pose` 控制模式已接入：action 为 6D 末端相对增量。
- `relative_pose_chunk` 控制模式已接入：action 为展平的多步 6D 末端相对增量，chunk size 从 config 的 `relative_pose_chunk_size` 读取。
- depth 默认开启：wrapper 会注入 `obs/depth/<role>`，Orbbec `auto` 后端在 depth 开启时优先尝试 SDK，缺失 depth 时使用零值 fallback，可通过 `enable_depth: false` 关闭。

## 4. Validation Completed

截至当前，已经反复做过的静态检查包括：

- `agent_infra/Piper_Env` 全部 Python 文件的 `py_compile`
- 新增 shell 脚本的 `bash -n`

当前可以确认的是：

- 语法层面通过
- 入口链路已经接通

当前还不能宣称的是：

- 所有新工具都已经在真实数据与真实硬件环境中完成端到端验证

## 4.1 Latest Maintenance Notes

本轮新增或调整：

- 根目录旧 `README.md` 已删除，避免继续指向早期 `piper_infra` 结构。
- Piper 专用 CAN 脚本已移动到 `agent_infra/Piper_Env/Script/`。
- `can_config.sh` 的四路默认映射已更新为：
  - `can_ml -> 1-3:1.0`
  - `can_sl -> 1-1:1.0`
  - `can_mr -> 1-10:1.0`
  - `can_sr -> 1-9:1.0`
- 单臂默认配置已改为左侧主从臂：
  - `master_can: can_ml`
  - `follower_can: can_sl`
- 已补充统一测试脚本，可组合测试：
  - 单臂 / 双臂
  - 带相机 / 不带相机
  - teleop / collect / replay
- 已补充相机枚举和数据采集接入说明。
- 已修复 teleop smoke test 中 safe hold action 反复触发 SDK joint limit warning 的问题：当 joint action 与当前关节位置几乎一致时不再重复下发关节运动指令。
- Orbbec 枚举函数已增加 USB fallback 诊断：SDK 取不到序列号时，会继续检查 `lsusb` 是否能看到 Orbbec/`2bc5` 设备。
- `orbbec_camera.py` 已改为 lazy import `pyorbbecsdk`，普通导入 Piper camera wrapper 时不会立刻触发 Orbbec SDK 原生枚举或 extension 加载。
- Orbbec USB fallback 已过滤 `bytewerk candleLight USB to CAN adapter` 这类 `2bc5` 非相机设备，避免把 Piper 的 USB-CAN 适配器误判为 Orbbec 相机。
- Orbbec 序列号枚举已改为子进程隔离执行，避免 `pyorbbecsdk/libusb` native 枚举失败时直接终止主进程或污染主进程输出。
- `test_teleop_obs_action.py` 已支持单臂/双臂切换，并支持通过 `--master` / `--slave` 直接覆盖 CAN 接口，双臂参数按 `left right` 顺序对应。
- Orbbec USB fallback 已增加 `/sys/bus/usb/devices` 读取路径，在 `lsusb` 不可用时也能检测系统 USB 层是否看到 Orbbec RGB/Depth 设备。
- Orbbec 角色标定结果已写入配置：`CH2592100GJ` 确认为右臂相机，`CH28822007C` 暂作为左臂相机；`calibrate_orbbec_roles.py` 默认每个序列号只采一张图，避免重复打开同一物理相机的多个 video 节点。
- Orbbec 相机类已支持 SDK 优先、V4L2 fallback：当前实测 `CH28822007C -> /dev/video2`、`CH2592100GJ -> /dev/video0`，均可返回 `(480, 640, 3) uint8` RGB 帧。
- Orbbec 默认启动后端已改为 `v4l2`，避免每次采集先触发 `pyorbbecsdk/libusb` 枚举报错；如需 SDK 可通过 `camera_backend: "sdk"` 显式切换。
- 数据采集预览默认关闭，避免 OpenCV Qt 字体 warning；需要预览时在 recorder 命令中显式传入 `--preview`。
- `T` 键遥操开关由 `PiperEnv` 统一处理；数据采集器只保留 `I/S/E/F/D/Q` 录制按键，避免 recorder 与 env 双监听 `T` 导致一次按键被切换两次。
- LeRobot recorder 已适配当前 LeRobot API：每帧写入必需的 `task` 字段；single/dual LeRobot 脚本新增可编辑 `TASK_DESCRIPTION`。
- 修复 Orbbec V4L2 黑帧问题：wrapper 不再默认传 `exposure=100`，Orbbec V4L2 默认使用 `MJPG + auto exposure`；当前两路相机均值恢复到约 `214~219`。
- Orbbec V4L2 支持配置 `auto_exposure / exposure / brightness / gain`；当前配置使用自动曝光并将 `gain` 降到 `0`，缓解默认增益过高导致的过曝。
- 统一相机颜色约定：RealSense 与 Orbbec 相机类输出给 wrapper 的 `color` 均为 RGB/HWC，wrapper 再转为 `obs/rgb/<role>` 的 CHW，避免 BGR 被当作 RGB 录入导致红蓝通道错位。
- LeRobot 视频默认编码改为 `h264`，避免默认 AV1/libsvtav1 在部分播放器或 OpenCV 环境中显示黑屏/解码失败；脚本暴露 `VCODEC` 可编辑。
- Piper gripper 语义已统一为 SDK 原始物理宽度，单位 m，范围 `0.0 ~ 0.1`；已移除 follower/master reset 与 step 中的 `0.5/0.8` 隐式缩放。
- 数据采集按 `I` 复位时会调用 `env.reset(options={"sync_master": True})`，要求 follower 与 leader 一起回到初始/目标位姿；leader 同步后会恢复零力拖动模式。
- 采集任务名已补充 `control_mode`：默认脚本会生成 `piper_single_h5_joint_task`、`piper_dual_h5_joint_task`、`piper_single_lerobot_joint_task`、`piper_dual_lerobot_joint_task`；recorder 也会自动给缺失控制模式的任务名补 `joint/pose`。
- `PiperCameraWrapper` 已支持可选 `obs/depth/<role>`，配置字段为 `enable_depth`。当前 Orbbec V4L2 稳定路径只使用 RGB；若切换到 `camera_backend: "sdk"` 且 SDK depth 可用，则 depth 会以 `(1, H, W)` `uint16` 注入 obs。
- 结构整理提案第一批已执行：CAN 脚本移至 `Script/can/`，Orbbec 标定/SDK 探测脚本移至 `Script/camera/`，Orbbec SDK 日志归档到 `Log/hardware_debug/`。
- 默认 config 路径解析已修正为 `agent_infra/Piper_Env/Config/`，避免不传 `config_path` 时误找 `Env/Config/`。
- `piper_infra/` 与 `trial/` 已分别新增 README，明确标记为 legacy/reference 与 experiments，避免新开发者误用旧入口。
- 数据采集复位增加并发保护：按 `I` 时主循环会暂停 step，避免 listener 线程 reset 与主循环 action 下发同时发生。
- 新增 `Script/camera/check_orbbec_depth.py`，用于单独验证 Orbbec `v4l2/sdk/auto` 后端是否能返回 RGB/depth 帧。
- LeRobot recorder 遇到 `obs/depth` 时会提示当前 LeRobot 路径暂不保存 depth；depth 第一阶段建议使用 H5。
- H5/LeRobot 后处理脚本已支持命令行参数覆盖默认 `-i/-o`，不再强制使用脚本内默认 merged 路径。
- `h5_to_lerobot` 已支持 raw H5 或 merged H5，写入每帧必需的 `task` 字段，默认使用 `h264`，并在输出目录已存在时自动切换到 `_001` 这类后缀目录。
- `lerobot_to_h5` 已支持 LeRobot 根目录、任务目录或误传的 `meta/` 子目录作为输入；新录制的 LeRobot 数据集会写入 `env_meta.json`，旧数据缺失时可从 `meta/info.json` features 推断 Piper single/dual env_meta。
- 已实测 `piper_dual_lerobot_joint_task/lerobot/meta -> dual_from_lerobot_smoke.h5` 转换链路成功：70 帧、1 个 episode、state/action/rgb 结构恢复正常。
- 已实测 `traj_0_174122.h5 -> lerobot_converted_001` 转换成功：956 帧、1 个 episode、双路视频 key、state 维度 38、action 维度 14。
- 针对左臂夹爪遥操时偶发突然开合的问题，已确认历史 H5 中存在 `left_gripper` 单帧 `0.1274 -> 0.0 -> 0.1274` 类跳变，且部分 master 读数超过 follower SDK 真实上限 `0.1m`。当前修复为：master gripper 读数先裁剪到 `0.0~0.1m`，再经短时跳变确认；未读到有效 gripper 帧时不再默认下发 `0.0`。
- 修复主臂不动时 follower 夹爪可能直接归零闭合的问题：夹爪状态读取优先使用 `get_gripper_status()` 实际反馈，并维护上一条已下发的 follower gripper 命令；master gripper cache 会先用 follower hold 宽度初始化，明显打开状态下收到精确 `0.0` 会保持当前夹爪，真实逐步闭合仍可下发到 `0.0`。
- 新增 `Script/test_master_gripper_signal.py`，用于实机打印 master gripper 原始读数、过滤后读数与候选跳变计数。
- single/dual 采集脚本均新增 `--master` / `--slave` 参数，可在采集时直接覆盖 CAN 接口；默认不传 `--teleop-on-start`，需要进入采集界面后按 `T` 开启遥操。
- Piper SDK joint5 校验上限已从截断的 `1.221730` 放宽到 `70deg + 1e-6rad`，避免 `1.2217304706573486` 这类合法边界值反复触发 `must be within [-1.22173, 1.22173]` warning；真实越界仍由 SDK validator 保留提示。
- 主臂同步初始位姿已改为优先调用 SDK 的 `move_leader_to_home()`，避免把 leader 主臂误走 follower 的 `move_j()` 路径导致打印同步但主臂实际不动。
- `delta_pose` / `relative_pose_chunk` 已接入 env、recorder CLI 与 smoke test CLI；teleop 下会把 master 绝对末端位姿转换为 follower 当前末端位姿的相对增量。
- depth 默认开启并进入 `obs/depth/<role>` / H5；LeRobot recorder 仍暂不写 depth，会给出提示并跳过 depth 字段。

## 5. Key Effects Achieved

如果只看效果层面，当前阶段最重要的成果有 5 个：

1. `Piper_Env` 的类分层已经从“讨论方案”变成了“代码现实”。
2. `RealSense` 路线已经能作为优先 pipeline 继续推进。
3. 单臂 / 双臂最终 env 入口已经出现，不再需要临时拼装。
4. 数据 pipeline 已具备录制、合并、转换、回放的骨架。
5. `.h5` 合并阶段已经支持控制模式切换。

## 6. Recommended Next Steps

结合 `PLAN.md` 与当前进度，后续建议优先顺序如下：

1. 对 single / dual 的真实数据流程做一轮实测记录
2. 对 `test_piper_pipeline.py` 的真实硬件输出做记录
3. 补充后处理工具的使用说明与示例
4. 开始下一阶段功能扩展：
   - 奥比中光
   - depth
   - 更丰富的控制模式
